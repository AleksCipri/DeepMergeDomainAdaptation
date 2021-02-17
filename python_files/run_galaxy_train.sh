
#!/bin/bash

ARGS="--gpu_id 0"
ARGS+=" --net DeepMerge"
ARGS+=" --dset galaxy"
ARGS+=" --dset_path arrays"
ARGS+=" --output_dir 'output_DeepMerge"
ARGS+=" --pristine_xfile Xdata_source.npy"
ARGS+=" --pristine_yfile ydata_source.npy"
ARGS+=" --noisy_xfile Xdata_target.npy"
ARGS+=" --noisy_yfile ydata_target.npy"
ARGS+=" --ly_type cosine"
ARGS+=" --loss_type mmd"
ARGS+=" --one_cycle yes"
ARGS+=" --lr 0.001"
ARGS+=" --epoch 200"
ARGS+=" --early_stop_patience 20"
ARGS+=" --weight_decay 0.001"
ARGS+=" --optim_choice Adam"
ARGS+=" --seed 1"
ARGS+=" --trade_off 1.0"
ARGS+=" --fisher_or_no Fisher"
ARGS+=" --em_loss_coef 0.05"
ARGS+=" --inter_loss_coef 1.0"
ARGS+=" --intra_loss_coef 0.01"
ARGS+=" --blobs yes"
ARGS+=" --grad_vis yes"
ARGS+=" --ckpt_path output_DeepMerge_SDSS"

cat << EOF
python train_MMD.py $ARGS
EOF
python train_MMD.py $ARGS
