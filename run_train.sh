export CUDA_VISIBLE_DEVICES=0,1

data="/cs/natlang-expts/aditi/waitinfo/Wait-info/data_bin/iwslt14.tokenized.de-en"
modelfile="/cs/natlang-expts/aditi/Wait-info/checkpoints"

python train.py --ddp-backend=no_c10d ${data} --arch transformer \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --encoder-attention-heads 8 \
 --decoder-attention-heads 8 \
 --criterion label_smoothed_cross_entropy_waitinfo \
 --label-smoothing 0.1 \
 --left-pad-source False \
 --save-dir ${modelfile} \
 --max-tokens 8192 --update-freq 1 \
 --fp16 \
 --save-interval-updates 1000 \
 --keep-interval-updates 2 \
 --keep-last-epochs 5 \
 --log-interval 10 \
 --max-epoch 50 --tensorboard-logdir "/cs/natlang-expts/aditi/Wait-info/logs"