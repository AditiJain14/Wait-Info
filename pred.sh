export CUDA_VISIBLE_DEVICES=0,1
data="/cs/natlang-expts/aditi/waitinfo/Wait-info/data_bin/iwslt14.tokenized.de-en"
modelfile="/cs/natlang-expts/aditi/waitinfo/checkpoints"
testk=4
ref_dir="/cs/natlang-expts/aditi/waitinfo/Wait-info/outputs/${testk}"
# average last 5 checkpoints
# python scripts/average_checkpoints.py --inputs ${modelfile} --num-epoch-checkpoints 5 --output ${modelfile}/average-model.pt 

# # # generate translation
# # #python generate.py ${data} --path $modelfile/average-model.pt --batch-size 1 --beam 1 --left-pad-source False --fp16  --remove-bpe --test-wait-k ${testk} --sim-decoding > pred.out
# # # generate translation
mkdir -p ${ref_dir}

python generate.py ${data} --path $modelfile/average-model.pt --batch-size 8 --beam 1 --left-pad-source False --fp16  --remove-bpe --test-wait-k ${testk} > $ref_dir/pred.out
grep ^T- $ref_dir/pred.out | cut -f1,2- | cut -c3- | sort -k1n | cut -f2- > $ref_dir/pred.ref
grep ^H $ref_dir/pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > $ref_dir/pred.translation
perl multi-bleu.perl -lc ${ref_dir}/pred.ref < $ref_dir/pred.translation