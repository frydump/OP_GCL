#!/bin/bash


for datasets in NCI1 DD PROTEINS MOLT-4H COLLAB github_stargazers IMDB-MULTI REDDIT-MULTI-5K COLORS-3
do
  for runseed in 0 1 2 3 4
  do
    python main.py --dataset $datasets --pretrain_epoch 200 --pretrain_lr 0.001  --suffix $runseed --n_splits 10
  done
done
