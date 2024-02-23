BASE_DIR="."
DATASET_DIR="$BASE_DIR/dataset/unseen_structure"

exp_name="gnn_linearattn_nodegree"
echo $exp_name
CUDA_VISIBLE_DEVICES=0 python $BASE_DIR/analyze.py \
    --only_test \
    --gpu 0 \
    --batch_size 1024 \
    --data_root "$DATASET_DIR/data" \
    --all_latency_file "${DATASET_DIR}/gt_stage.txt" \
    --norm_sf \
    --onnx_dir "${DATASET_DIR}" \
    --log "log/test.log" \
    --pretrain "output/"${exp_name}"/ckpt_best.pth" \
    --ckpt_save_freq 1000 \
    --test_freq 1 \
    --print_freq 50 \
    --embed_type trans \
    --num_node_features 152 \
    --glt_norm LN \
    --train_test_stage \
    # --use_degree \