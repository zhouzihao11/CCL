# This code is constructed based on Pytorch Implementation of FixMatch(https://github.com/kekmodel/FixMatch-pytorch)
import argparse
import logging
import math
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from utils import Logger
from progress.bar import Bar
import loss.semiConLoss as scl

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_b = 0


def make_imb_data(max_num, class_num, gamma, flag=1, flag_LT=0):
    mu = np.power(1 / gamma, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))

    if flag == 0 and flag_LT == 1:
        class_num_list = list(reversed(class_num_list))
    return list(class_num_list)


def compute_adjustment_list(label_list, tro, args):
    label_freq_array = np.array(label_list)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments


def compute_py(train_loader, args):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell) in enumerate(train_loader):
        labell = labell.to(args.device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.to(args.device)
    return label_freq_array


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', epoch_p=1):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def compute_adjustment_by_py(py, tro, args):
    adjustments = torch.log(py ** tro + 1e-12)
    adjustments = adjustments.to(args.device)
    return adjustments


def sharp(a, T):
    a = a ** T
    a_sum = torch.sum(a, dim=1, keepdim=True)
    a = a / a_sum
    return a.detach()


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='1', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'stl10', 'smallimagenet'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnet'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=250000, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=500, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=1, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument('--num-max', default=500, type=int,
                        help='the max number of the labeled data')
    parser.add_argument('--num-max-u', default=4000, type=int,
                        help='the max number of the unlabeled data')
    parser.add_argument('--imb-ratio-label', default=1, type=int,
                        help='the imbalanced ratio of the labelled data')
    parser.add_argument('--imb-ratio-unlabel', default=1, type=int,
                        help='the imbalanced ratio of the unlabeled data')
    parser.add_argument('--flag-reverse-LT', default=0, type=int,
                        help='whether to reverse the distribution of the unlabeled data')
    parser.add_argument('--ema-mu', default=0.99, type=float,
                        help='mu when ema')

    parser.add_argument('--tau', default=2.0, type=float,
                        help='tau for head consistency')
    parser.add_argument('--est-epoch', default=5, type=int,
                        help='the start step to estimate the distribution')
    parser.add_argument('--img-size', default=32, type=int,
                        help='image size for small imagenet')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='ema ratio for estimating distribution of the unlabeled data')
    parser.add_argument('--beta', default=0.5, type=float,
                        help='ema ratio for estimating distribution of the all data')
    parser.add_argument('--lambda1', default=0.7, type=float,
                        help='coefficient of final loss')
    parser.add_argument('--lambda2', default=1.0, type=float,
                        help='coefficient of final loss')

    args = parser.parse_args()
    global best_acc
    global best_acc_b

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)

        elif args.arch == 'resnet':
            import models.resnet_ori as models
            model = models.ResNet50(num_classes=args.num_classes, rotation=True, classifier_bias=True)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank},"
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.dataset_name = 'cifar10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.dataset_name = 'cifar100'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2

    elif args.dataset == 'stl10':
        args.num_classes = 10
        args.dataset_name = 'stl10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2


    elif args.dataset == 'smallimagenet':
        args.num_classes = 127
        if args.img_size == 32:
            args.dataset_name = 'imagenet32'
        elif args.img_size == 64:
            args.dataset_name = 'imagenet64'

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, 'datasets/' + args.dataset_name)

    if args.local_rank == 0:
        torch.distributed.barrier()

    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    args.est_step = 0

    args.py_con = compute_py(labeled_trainloader, args)
    args.py_uni = torch.ones(args.num_classes) / args.num_classes
    # args.py_uni = args.py_uni.to(args.device)

    args.py_all = args.py_con
    args.py_unlabeled = args.py_uni

    class_list = []
    for i in range(args.num_classes):
        class_list.append(str(i))

    title = 'FixMatch-' + args.dataset
    args.logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
    args.logger.set_names(
        ['Top1_co acc', 'Top5_co acc', 'Best Top1_co acc', 'Top1_b acc', 'Top5_b acc', 'Best Top1_b acc'])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.py_unlabeled = checkpoint['py_unlabeled']
        args.py_all = checkpoint['py_all']

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()

    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)

    args.logger.close()


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    global best_acc_b
    test_accs = []
    avg_time = []
    end = time.time()
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
    logits_la_s = compute_adjustment_by_py(args.py_con, args.tau, args)
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    semiConLoss = scl.SemiConLoss(args.batch_size, args.batch_size, args.num_classes, args)
    semiConLoss2 = scl.softConLoss(args.batch_size, args.batch_size, args.num_classes, args)
    model.train()
    lbs = args.batch_size
    ubs = args.batch_size * args.mu
    py_labeled = args.py_con.to(args.device)
    py_unlabeled = args.py_uni.to(args.device)
    py_all = args.py_all.to(args.device)
    cut1 = lbs + 3 * ubs
    pro = ubs / (ubs + lbs)
    for epoch in range(args.start_epoch, args.epochs):
        print('current epoch: ', epoch + 1)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_con = AverageMeter()
        losses_cls = AverageMeter()
        losses_con2 = AverageMeter()

        bar = Bar('Training', max=args.eval_step)

        num_unlabeled = torch.ones(args.num_classes).to(args.device)
        num_all = torch.ones(args.num_classes).to(args.device)
        for batch_idx in range(args.eval_step):
            try:
                (inputs_x, inputs_x_s, inputs_x_s1), targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_s, inputs_x_s1), targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = next(unlabeled_iter)
            u_real = u_real.to(args.device)
            mask_l = (u_real != -2).float().unsqueeze(1).to(args.device)
            data_time.update(time.time() - end)
            inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1, inputs_x_s, inputs_x_s1], dim=0).to(
                args.device)
            targets_x = targets_x.to(args.device)
            feat, feat_mlp, center_feat = model(inputs)
            # -----------------------------------------------------------------------------------------------------------
            logits = model.classify(feat[:cut1])
            logits_b = model.classify1(feat[:cut1])
            logits_x = logits[:lbs]
            logits_x_w, logits_x_s, logits_x_s1 = logits[lbs:].chunk(3)
            logits_x_b = logits_b[:lbs]
            # logits LA
            logits_x_b_w, logits_x_b_s, logits_x_b_s1 = logits_b[lbs:].chunk(3)
            del logits, logits_b
            l_u_s = F.cross_entropy(logits_x, targets_x, reduction='mean')
            l_b_s = F.cross_entropy(logits_x_b + logits_la_s, targets_x, reduction='mean')
            logits_la_u = (- compute_adjustment_by_py((1 - pro) * py_labeled + pro * py_all, 1.0, args) +
                           compute_adjustment_by_py(py_unlabeled, 1 + args.tau / 2, args))
            logits_co = 1 / 2 * (logits_x_w + logits_la_u) + 1 / 2 * logits_x_b_w
            energy = -torch.logsumexp((logits_co.detach()) / args.T, dim=1)
            pseudo_label_co = F.softmax((logits_co.detach()) / args.T, dim=1)
            pseudo_label_con = sharp(F.softmax((logits_co.detach()) / args.T, dim=1), 4.0)

            prob_co, targets_co = torch.max(pseudo_label_co, dim=-1)
            mask = prob_co.ge(args.threshold)
            mask = mask.float()

            targets_co = torch.cat([targets_co, targets_co], dim=0).to(args.device)
            logits_b_s = torch.cat([logits_x_b_s, logits_x_b_s1], dim=0).to(args.device)
            logits_la_u_b = compute_adjustment_by_py(py_all, args.tau, args)
            mask_twice = torch.cat([mask, mask], dim=0)
            l_u_b = (F.cross_entropy(logits_b_s + logits_la_u_b, targets_co,
                                     reduction='none') * mask_twice).mean()

            logits_u_s = torch.cat([logits_x_s, logits_x_s1], dim=0).to(args.device)
            l_u_u = (F.cross_entropy(logits_u_s, targets_co,
                                     reduction='none') * mask_twice).mean()

            loss_u = max(1.5, args.mu) * l_u_u + l_u_s
            loss_b = max(1.5, args.mu) * l_u_b + l_b_s
            loss_cls = loss_u + loss_b
            # ----------------------------------------------------------------------------------------------------------
            feat_mlp = feat_mlp[lbs:]
            f3, f4 = feat_mlp[ubs:3 * ubs, :].chunk(2)
            f1, f2 = feat_mlp[3 * ubs:, :].chunk(2)

            # ----------------------------------------------------------------------------------------------------------
            feat_mlp = torch.cat([center_feat, feat_mlp[3 * ubs:, :], feat_mlp[:3 * ubs, :]], dim=0)
            center_label = torch.ones(args.num_classes, args.num_classes).to(args.device)
            one_hot_targets = F.one_hot(targets_x, num_classes=args.num_classes)
            one_hot_targets = torch.cat([one_hot_targets, one_hot_targets], dim=0).to(args.device)
            label_contrac = torch.cat([center_label, one_hot_targets], dim=0).to(args.device)
            # la = compute_adjustment_by_py(py_all, 1.0, args)
            contrac_loss = semiConLoss(feat_mlp, label_contrac)

            # ----------------------------------------------------------------------------------------------------------
            maskcon = energy.le(-8.75)
            idx = torch.nonzero(maskcon).squeeze()
            f3 = torch.reshape(f3[idx, :], (-1, f1.shape[1]))
            f4 = torch.reshape(f4[idx, :], (-1, f1.shape[1]))
            pseudo_label_con = torch.reshape(pseudo_label_con[idx, :], (-1, args.num_classes))

            label_contrac = torch.cat([center_label, one_hot_targets, pseudo_label_con, pseudo_label_con], dim=0).to(
                args.device)
            feat_all = torch.cat([center_feat, f1, f2, f3, f4], dim=0)
            contrac_loss2 = semiConLoss2(label_contrac, feat_all, args.device)

            loss = args.lambda1 * loss_cls + args.lambda2 * contrac_loss + (1 - args.lambda1) * contrac_loss2

            loss.backward()
            losses.update(loss.item())
            losses_cls.update(loss_cls.item())
            losses_con.update(contrac_loss.item())
            losses_con2.update(contrac_loss2.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            mask = mask.unsqueeze(1).to(args.device)
            maskcon = maskcon.float().unsqueeze(1).to(args.device)
            num_all += torch.sum(pseudo_label_co * mask, dim=0)
            # num_unlabeled += torch.sum(pseudo_label_co * mask_l * mask, dim=0)
            num_unlabeled += torch.sum(pseudo_label_co * mask_l * maskcon, dim=0)
            batch_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f} | Loss_cls: {loss_cls:.4f} | Loss_con: {loss_con:.4f} | Loss_con2: {loss_con2:.4f}'.format(
                batch=batch_idx + 1,
                size=args.eval_step,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss_cls=losses_cls.avg,
                loss_con=losses_con.avg,
                loss_con2=losses_con2.avg,
            )
            bar.next()
        bar.finish()

        if epoch > args.est_epoch:
            py_unlabeled = args.alpha * py_unlabeled + (1 - args.alpha) * num_unlabeled / sum(num_unlabeled)
            py_all = args.beta * py_all + (1 - args.beta) * num_all / sum(num_all)
        print('\n')
        print(py_unlabeled)
        print(py_all)
        avg_time.append(batch_time.avg)

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        test_la = - compute_adjustment_by_py(1 / 2 * py_labeled + 1 / 2 * py_all, 1.0, args)
        if args.local_rank in [-1, 0]:

            test_loss, test_acc, test_top5_acc, test_acc_b, test_top5_acc_b = test(args, test_loader,
                                                                                                 test_model, epoch,
                                                                                                 test_la)
            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc_b, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc_b > best_acc_b

            best_acc = max(test_acc, best_acc)
            best_acc_b = max(test_acc_b, best_acc_b)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            if (epoch + 1) % 10 == 0 or (is_best and epoch > 250):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc': test_acc,
                    'best_acc': best_acc_b,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'py_unlabeled': py_unlabeled,
                    'py_all': py_all
                }, is_best, args.out, epoch_p=epoch + 1)

            test_accs.append(test_acc_b)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc_b))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

            args.logger.append([test_acc, test_top5_acc, best_acc, test_acc_b, test_top5_acc_b, best_acc_b])
    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch, la):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_b = AverageMeter()
    top5_b = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs_feat = model(inputs)
            outputs = model.classify(outputs_feat)
            outputs_b = model.classify1(outputs_feat)
            outputs_co = 1 / 2 * (outputs + la) + 1 / 2 * outputs_b
            loss = F.cross_entropy(outputs_b, targets)

            prec1_b, prec5_b = accuracy(outputs_b, targets, topk=(1, 5))
            prec1_co, prec5_co = accuracy(outputs_co, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1_co.item(), inputs.shape[0])
            top5.update(prec5_co.item(), inputs.shape[0])
            top1_b.update(prec1_b.item(), inputs.shape[0])
            top5_b.update(prec5_b.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info("top-1 acc: {:.2f}".format(top1_b.avg))
    logger.info("top-5 acc: {:.2f}".format(top5_b.avg))

    return losses.avg, top1.avg, top5.avg, top1_b.avg, top5_b.avg


if __name__ == '__main__':
    main()
