python train.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 150 --imb-ratio-unlabel 150 --out out/cifar-10/N500_M4000/consistent_150 --beta 0.95 --est-epoch 20 --gpu-id 0