### OP-GCL Pre-Training: ###

```
cd ./bio
python pretrain.py --loss_num 3
cd ./chem
python pretrain.py --loss_num 3
```


### OP-GCL Finetuning: ###

```
cd ./bio
./finetune.sh ./weights/loss_lr_epoch.pth 1e-3 op-gcl_loss_0
cd ./chem
./finetune.py ./weights/loss_lr_epoch.pth 1e-3 op-gcl_loss_1
```

```loss_num``` is tuned from {1, 2, 3}. ```op-gcl_loss_0``` is the file name to store the results.




## Acknowledgements

The backbone implementation is reference to https://github.com/snap-stanford/pretrain-gnns.
