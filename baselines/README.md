## Baselines

*** This README is being updated. ***

### Preparation 

```
python3 gen_delex.py

python3 gen_parlai_data.py

parlai train_model -t fromfile:parlaiformat --fromfile_datapath ./parlai --fromfile-datatype-extension true  -m transformer/generator --init-model zoo:tutorial_transformer_generator/model --dict-file zoo:tutorial_transformer_generator/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --skip-generation True --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True --model-file ./train_90M

parlai interactive -mf ./train_90M < lm.input.dev.cc.txt > lm.output.dev.cc.txt

parlai interactive -mf ./train_90M < lm.input.test.cc.txt > lm.output.test.cc.txt

python3 run_language_modeling.py --output_dir=output_gpt2_10epoch_1e-3_fp16 --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=lm.input.train.txt --do_eval  --eval_data_file=lm.input.dev.txt --per_device_train_batch_size 2 --gradient_accumulation_steps 18 --num_train_epochs 10 --learning_rate 1e-3 --fp16 --overwrite_output_dir

python3 run_generation.py --input lm.input.dev.eval.txt --output dev.inference.gpt2_10epoch_1e-3_fp16.json --model_name_or_path ./output_gpt2_10epoch_1e-3_fp16 --eos_token_id 50262

python3 run_generation.py --input lm.input.test.eval.txt --output test.inference.gpt2_10epoch_1e-3_fp16.json --model_name_or_path ./output_gpt2_10epoch_1e-3_fp16 --eos_token_id 50262

```

### SimpleTOD+

```
python3 run_language_modeling.py --output_dir=output_both_gpt2_10epoch_1e-3_fp16 --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=lm.input.train.both.txt --do_eval  --eval_data_file=lm.input.dev.both.txt --per_device_train_batch_size 2 --gradient_accumulation_steps 18 --num_train_epochs 10 --learning_rate 1e-3 --fp16 --overwrite_output_dir

python3 run_generation.py --input lm.input.dev.eval.txt --output dev.inference.both_gpt2_10epoch_1e-3_fp16.json --model_name_or_path ./output_both_gpt2_10epoch_1e-3_fp16 --eos_token_id 50262

python3 run_generation.py --input lm.input.test.eval.txt --output test.inference.both_gpt2_10epoch_1e-3_fp16.json --model_name_or_path ./output_both_gpt2_10epoch_1e-3_fp16 --eos_token_id 50262

```

### Arranger

```
python3 gen_arranger_input.py

python3 run_multiple_choice.py --model_type roberta --task_name acc --model_name_or_path roberta-base --do_train --do_eval --do_test --do_lower_case --data_dir . --learning_rate 2e-5 --num_train_epochs 3 --max_seq_length 512 --output_dir acc2_roberta_base_3 --per_gpu_eval_batch_size=16 --per_gpu_train_batch_size=1 --gradient_accumulation_steps 24 --overwrite_output --save_steps 10000

python3 gen_arranger_output.py
```

### Rewriter

```
python3 gen_rewriter_data.py

python3 run_language_modeling.py  --output_dir=output_ff_gpt2_10epoch_1e-3_fp16  --model_type=gpt2  --model_name_or_path=gpt2  --do_train  --train_data_file=lm.input.train.ff.txt  --do_eval   --eval_data_file=lm.input.dev.ff.txt  --per_device_train_batch_size 2  --gradient_accumulation_steps 18 --num_train_epochs 10 --learning_rate 1e-3 --fp16 --overwrite_output_dir

python3 run_generation.py --input lm.input.dev.eval.ff.txt --output dev.inference.ff_gpt2_10epoch_1e-3_fp16.json --model_name_or_path ./output_ff_gpt2_10epoch_1e-3_fp16 --eos_token_id 50262

python3 run_generation.py --input lm.input.test.eval.ff.txt --output test.inference.ff_gpt2_10epoch_1e-3_fp16.json --model_name_or_path ./output_ff_gpt2_10epoch_1e-3_fp16 --eos_token_id 50262

```

### Evaluation

Pass the output inference files (i.e., ```{dev,test}.inference*.json```) to ```gen_predict.py``` to obtain act-slot F1 and BLEU-4 scores. For example,
```
python3 gen_predict.py --inference test.inference.both_gpt2_10epoch_1e-3_fp16.json --split test
```

The above command will also generate a folder (named ```./prediction/``` by default), which can be used by the [official evaluation script of SGD](https://github.com/google-research/google-research/tree/master/schema_guided_dst) to obtain the joint goal accuracy and average accuracy. For example,
```
python3 -m schema_guided_dst.evaluate --dstc8_data_dir ./simpletod/ --prediction_dir ./prediction/test/ --eval_set test --output_metric_file simpletod+_test_result.json
``` 

