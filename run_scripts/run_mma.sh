
# ROOT="/local-scratch/nishant/simul/mma"
ROOT="/cs/natlang-expts/aditi/waitinfo"

DATA="/cs/natlang-expts/aditi/waitinfo/Wait-info/data_bin/vi_en/data-bin"


EXPT="${ROOT}/experiments/trial"
mkdir -p ${EXPT}

FAIRSEQ="${ROOT}/Wait-info"

USR="${FAIRSEQ}/examples/simultaneous_translation"




export CUDA_VISIBLE_DEVICES=0,1,2,3

# infinite lookback

mma_il(){
    CKPT="${EXPT}/checkpoints/infinite_latencyloss3"
    mkdir -p ${CKPT}

    fairseq-train \
    $DATA \
    --log-format simple --log-interval 100 \
    --source-lang de --target-lang en \
    --task translation \
    --simul-type infinite_lookback --tensorboard-logdir $CKPT/log \
    --user-dir $USR \
    --mass-preservation \
    --latency-weight-avg  0.1 \
    --criterion latency_augmented_label_smoothed_cross_entropy_pure \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en --save-dir $CKPT \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001\
    --dropout 0.3 \
    --label-smoothing 0.1\
    --max-tokens 8000 --update-freq 1 \
    --ddp-backend no_c10d --max-epoch 50 | tee -a "$CKPT/log/train.log"
}

# hard align
mma_h(){
     #"avg_0.1"
    latency_var=$1
    conf="var_$1"
    TGT_DICT="$DATA/dict.en.txt"
    name="hard${conf}"
    CKPT="${EXPT}/checkpoints/${name}"
    mkdir -p ${CKPT}

    export WANDB_NAME="${name}"

    fairseq-train \
    $DATA \
    --log-format simple --log-interval 100 \
    --source-lang vi --target-lang en \
    --task translation --tensorboard-logdir $CKPT/log \
    --simul-type hard_aligned \
    --user-dir $USR \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy_pure \
    --latency-weight-var ${latency_var} \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en --save-dir $CKPT \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001\
    --dropout 0.3 --patience 10 --keep-last-epochs 15 --best-checkpoint-metric ppl\
    --label-smoothing 0.1\
    --max-tokens 7000 \
    --ddp-backend no_c10d --max-epoch 50 --wandb-project MMAH
    #args for adaptive training 
    # --adaptive-training \
    # --dict-file ${TGT_DICT} \
    # --adaptive-method 'exp' \
    # --adaptive-T 1.75 \
    # --weight-drop 0.3 
}


# monotnic wait k
mma_wait_k(){
    CKPT="${EXPT}/checkpoints/waitk_latencyloss2"
    mkdir -p ${CKPT}

    fairseq-train \
    $DATA \
    --log-format simple --log-interval 100 \
    --source-lang de --target-lang en \
    --task translation --tensorboard-logdir $CKPT/log \
    --simul-type waitk \
    --waitk-lagging 3 \
    --user-dir $USR \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy_pure \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en --save-dir $CKPT \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001\
    --dropout 0.3 \
    --label-smoothing 0.1\
    --max-tokens 5000 --update-freq 2 \
    --ddp-backend no_c10d --max-epoch 50
}

#mma_il
#mma_h
mma_h 0.1
#mma_wait_k