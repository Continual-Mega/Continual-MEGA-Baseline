

# evaluation continual tasks
for i in $(seq 0 2); do
    python eval_continual.py \
    --meta_file "meta_files/scenario2_30classes_tasks.json" \
    --base_meta_file "meta_files/scenario2_base.json" \
    --num_tasks 2 \
    --checkpoints "checkpoints/scenario2/30classes_tasks" \
    --checkpoint_base "checkpoints/scenario2/checkpoint_base.pth" \
    --save_path "results/scenario2_30classes" \
    --data_root "data" \
    --task_id $i
done

# calculate ACC, FM
python calculate_metrics.py \
--image_csv "results/scenario2_30classes/results_image.csv" \
--pixel_csv "results/scenario2_30classes/results_pixel.csv"
