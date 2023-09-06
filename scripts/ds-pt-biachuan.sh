#!/bin/bash
pip install -r requirements.txt
# 1 - 模型参数文件，从s3拷贝到资源机
	# s5cmd等效于aws cli的aws s3 cp命令，速度更快
	# （大模型以外的场景不需要该操作）
    # 注意destination必须为/tmp的prefix（大机型自带的NVME存储）
    # In multi-node training, only copy on LocalRank 0
chmod +x ./s5cmd
./s5cmd sync s3://sagemaker-us-west-2-000274595152/baichuan13/* /tmp/large_model_pretrain/

# 2 - 训练数据从s3拷贝到资源机
	# （默认不需要操作，Sagemaker会自动拷贝到资源机的默认路径/opt/ml/input/data/）
	# e.g. /opt/ml/input/data/train123，这里train123跟Sagemaker Estimator.fit() 传入的channel中的key名称一致
    # *也可以使用1中的加速拷贝形式，或其他Steam传输形式
./s5cmd sync s3://sagemaker-us-west-2-000274595152/anno_data/* /opt/ml/input/data/

# 3 - 代码及脚本
	# （不需要操作，Sagemaker Estimator参数中指定的source_dir/dependency等，会自动上传到资源机的默认路径/opt/ml/code）

torchrun --num_gpus=8  /opt/ml/code/src/train_bash.py \
    --deepspeed /opt/ml/code/scripts/deepspeed.json \
    --stage pt \
    --model_name_or_path /tmp/large_model_pretrain \
    --do_train \
    --do_eval \
    --dataset_dir /opt/ml/input/data \
    --dataset announcement,report,translation \
    --template default \
    --finetuning_type lora \
    --lora_target W_pack \
    --output_dir /tmp/large_model_out \
    --overwrite_cache \
    --cache_dir /tmp \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4  \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --val_size 0.01 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --fp16

# ************************************
# 4 - 训练后的模型参数，从资源机拷贝到S3
	# （*注意LLM场景，务必不能用/opt/ml/model作为模型的输出路径，否则Sagemaker会执行 model file -> tar --s3 cp--> s3
	# 会有小时级别的时间消耗
./s5cmd sync /tmp/large_model_out s3://sagemaker-us-west-2-000274595152/anno_train_output/$(date +%Y-%m-%d-%H-%M-%S)/
