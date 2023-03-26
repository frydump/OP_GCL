### OP-GCL Finetuning: ###
```
cd ./pretrain
python main.py --dataset NCI1 --epochs 200 --lr 0.001 --beta 0.3 --suffix 0
python main.py --dataset NCI1 --epochs 200 --lr 0.001 --beta 0.3 --suffix 1
python main.py --dataset NCI1 --epochs 200 --lr 0.001 --beta 0.3 --suffix 2
python main.py --dataset NCI1 --epochs 200 --lr 0.001 --beta 0.3 --suffix 3
python main.py --dataset NCI1 --epochs 200 --lr 0.001 --beta 0.3 --suffix 4
```

### OP-GCL Finetuning: ###

```
cd ./finetune
python main.py --dataset NCI1 --pretrain_epoch 200 --pretrain_lr 0.001 --suffix 0 --n_splits 10
python main.py --dataset NCI1 --pretrain_epoch 200 --pretrain_lr 0.001 --suffix 1 --n_splits 10
python main.py --dataset NCI1 --pretrain_epoch 200 --pretrain_lr 0.001 --suffix 2 --n_splits 10
python main.py --dataset NCI1 --pretrain_epoch 200 --pretrain_lr 0.001 --suffix 3 --n_splits 10
python main.py --dataset NCI1 --pretrain_epoch 200 --pretrain_lr 0.001 --suffix 4 --n_splits 10
```

Five suffixes stand for five runs (with mean & std reported)

```lr``` should be tuned from {0.01, 0.001, 0.0001}, ```beta``` from {0.05, 0.1, 0.3, 0.7} in pre-training, and ```pretrain_epoch``` in finetuning (this means the epoch checkpoint loaded from pre-trained model) from {20, 40, 60, 80, 100, 200}.



## Acknowledgements

The backbone implementation is reference to https://github.com/chentingpc/gfn.
