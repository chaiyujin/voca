speaker=m001_trump

for seq in "vld-000" "vld-001"; do
  python3 run_voca.py \
    --tf_model_fname ./training/${speaker}/checkpoints/gstep_56400.model \
    --audio_fname ~/Documents/Project2021/stylized-sa/data/datasets/talk_video/celebtalk/data/${speaker}/${seq}/audio.wav \
    --out_path results/${speaker}/${seq} \
    --template_fname ./template/${speaker}.ply \
    --condition_idx 1 \
    --visualize=False \
  ;
done
