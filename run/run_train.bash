name=lxmert

output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/cosim_train.py \
    --train train --valid val --test test\
    --llayers 9 --xlayers 5 \
    --batchSize 2 --optim bert --lr 1e-5 --epochs 50 \
    --tqdm --output $output ${@:2}

