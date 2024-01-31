BASE_DIR="."
DATASET_DIR="$BASE_DIR/dataset/unseen_structure"

mkdir log

exp_name="selfattn_drop0.05_layerscale_lr0.001_ff"
echo $exp_name
CUDA_VISIBLE_DEVICES=0 python $BASE_DIR/main.py \
    --gpu 0 \
    --lr 0.001 \
    --epochs 50 \
    --batch_size 16 \
    --data_root "$DATASET_DIR/data" \
    --all_latency_file "${DATASET_DIR}/gt_stage.txt" \
    --norm_sf \
    --onnx_dir "${DATASET_DIR}" \
    --log "log/"${exp_name}".log" \
    --model_dir "output/"${exp_name}"/" \
    --ckpt_save_freq 1000 \
    --test_freq 1 \
    --print_freq 50 \
    --embed_type trans \
    --num_node_features 152 \
    --glt_norm LN \
    --warmup_rate 0.1 \
    --train_test_stage \
    --hidden_size 512 \
    --n_attned_gnn 2 \
    --exp_name $exp_name \
