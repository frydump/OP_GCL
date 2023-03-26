#### GIN fine-tuning
split=scaffold

model_file=$1
lr=$2
resultFile_name=$3
for dataset in toxcast sider clintox hiv bace
do
for runseed in 0 1 2 3 4
do
  python finetune.py --input_model_file $model_file --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr $lr --epochs 100 --resultFile_name $resultFile_name
done
done

dataset=muv
for runseed in 0 1 2
do
  python finetune.py --input_model_file $model_file --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr $lr --epochs 100 --resultFile_name $resultFile_name
done