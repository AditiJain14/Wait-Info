SRC_FILE="/cs/natlang-expts/aditi/waitinfo/data/iwslt14.tokenized.de-en/test.de"
TGT_FILE="/cs/natlang-expts/aditi/waitinfo/data/iwslt14.tokenized.de-en/test.en"
MODEL_PATH="/cs/natlang-expts/aditi/waitinfo/experiments/trial/checkpoints/waitk/checkpoint_best.pt"
RESULT_DIR="/cs/natlang-expts/aditi/waitinfo/experiments/trial/checkpoints/waitk"
data="/cs/natlang-expts/aditi/waitinfo/Wait-info/data_bin/iwslt14.tokenized.de-en"

python ./eval/evaluate.py \
    --local \
    --src-file $SRC_FILE \
    --tgt-file $TGT_FILE \
    --data-bin $data \
    --model-path $MODEL_PATH \
    --scores --output $RESULT_DIR