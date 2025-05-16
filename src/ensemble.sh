DEVICE=0

CUDA_VISIBLE_DEVICES=$DEVICE python test_ESCAPE.py \
  --maps_dir ../data/struct_maps \
  --test_file ../data/sequences/Test.csv \
  --checkpoint1 ../checkpoints/ESCAPE/OLD_ESCAPE_Fold1.pth \
  --checkpoint2 ../checkpoints/ESCAPE/OLD_ESCAPE_Fold2.pth \
  --batch_size 64 \
  --seq_max_len 200 \
  --img_size 224 \
  --seed 42