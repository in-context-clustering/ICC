NUM_GPUS=1
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

# arguments that are very likely to be changed
# according to your own case
CONFIG=t
MODEL_ID=llama-3.1-8b-instruct                              # model id; pick on by running `python supported_models.py`
TRAIN_DATA_PATH=data/numeric/train/${CONFIG}_train.json     # path to the training data json file
EVAL_DATA_PATH=data/numeric/train/${CONFIG}_val.json        # path to the evaluation data json file (optional)
IMAGE_FOLDER=None                                       # path to the image root folder; if provided, the image paths in the json should be relative
VIDEO_FOLDER=None                                       # path to the video root folder; if provided, the video paths in the json should be relative
NUM_FRAMES=1                                            # how many frames are sampled from each video

TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=False                            # whether train the vision projector (only full finetuning is supported)

DS_STAGE=zero2                                          # deepspeed stage; < zero2 | zero3 >
LR=$1                                                   # learning rate
RUN_ID=${CONFIG}_${MODEL_ID}_${LR}                      # a custom run id that determines the checkpoint folder and wandb run name

torchrun $DISTRIBUTED_ARGS src/finetune/train_num.py \
    --model_id $MODEL_ID \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --num_frames $NUM_FRAMES \
    --output_dir checkpoints/numeric/${RUN_ID} \
    --report_to wandb \
    --run_name $RUN_ID \
    --deepspeed src/finetune/ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs 1 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --warmup_ratio 0 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type "cosine" \
    --eval_steps 200 \
    --logging_steps 200 \
    --model_max_length 10000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora True \
    --q_lora False \
    --lora_r 64 \
    --lora_alpha 16
    