NUM_GPUS=2
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

# arguments that are very likely to be changed
# according to your own case
CONFIG=classonly_4c_var30s_numlabel
MODEL_ID=llava-interleave-qwen-7b                                   # model id; pick on by running `python supported_models.py`
TRAIN_DATA_PATH=data/image/train/${CONFIG}_train.json               # path to the training data json file
EVAL_DATA_PATH=data/image/train/${CONFIG}_test.json                 # path to the evaluation data json file (optional)
IMAGE_FOLDER=/imagenet21k                                           # path to the image root folder; if provided, the image paths in the json should be relative
VIDEO_FOLDER=None                                       # path to the video root folder; if provided, the video paths in the json should be relative
NUM_FRAMES=1                                            # how many frames are sampled from each video

TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=False                            # whether train the vision projector (only full finetuning is supported)

DS_STAGE=zero2                                          # deepspeed stage; < zero2 | zero3 >
LR=$1                                                   # learning rate     
PATCH=$2                                                # patch size
RUN_ID=${CONFIG}_${LR}_patch${PATCH}                    # a custom run id that determines the checkpoint folder and wandb run name
BZ=$3

ACC=$((16/${BZ}))


torchrun $DISTRIBUTED_ARGS src/finetune/train_img.py \
    --model_id $MODEL_ID \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --num_frames $NUM_FRAMES \
    --output_dir checkpoints/image/${RUN_ID} \
    --report_to wandb \
    --run_name $RUN_ID \
    --deepspeed src/finetune/ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BZ} \
    --per_device_eval_batch_size ${BZ} \
    --gradient_accumulation_steps ${ACC} \
    --eval_strategy "steps" \
    --save_strategy "best" \
    --metric_for_best_model loss \
    --eval_steps 200 \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --warmup_ratio 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 200 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora True \
    --q_lora False \
    --lora_r 64 \
    --lora_alpha 16 \
    --patch_size ${PATCH}
    