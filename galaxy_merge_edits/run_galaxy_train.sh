
#!/bin/bash

ARGS="--gpu_id 0"
ARGS+=" --net ResNet18"
ARGS+=" --dset galaxy"
ARGS+=" --dset_path /Users/perdue/Dropbox/HVAI/DeepMergeDomainAdaptation/galaxy_merge_edits/small"
ARGS+=" --pristine_xfile Pristine_small.npy"
ARGS+=" --pristine_yfile Pristine_small_labels.npy"
ARGS+=" --noisy_xfile Noisy_small.npy"
ARGS+=" --noisy_yfile Noisy_small_labels.npy"
ARGS+=" --test_interval 500"
ARGS+=" --snapshot_interval 10000"
ARGS+=" --ly_type cosine"
ARGS+=" --loss_type mmd"
ARGS+=" --fisher_loss_type tr"
ARGS+=" --output_dir gnp_output"
ARGS+=" --em_loss_coef 0.1"
ARGS+=" --inter_loss_coef 1."
ARGS+=" --intra_loss_coef 1."
ARGS+=" --trade_off 1."
ARGS+=" --optim_choice 'SGD'"

cat << EOF
python galaxy_train.py $ARGS
EOF
python galaxy_train.py $ARGS
