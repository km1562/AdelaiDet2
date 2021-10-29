CUDA_VISIBLE_DEVICES=0,3,4 OMP_NUM_THREADS=1 python tools/train_net.py \
  --config-file configs/BAText/Pretrain/add_global/CROP_Fasle_3bh_add_global_v2_attn_R_50.yaml \
  --num-gpus 3