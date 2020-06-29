#!/bin/bash

echo "started "`date`" "`date +%s`""

BASEPATH="/home/${USER}/DeepMergeDomainAdaptation/galaxy_merge_edits"

cp ${BASEPATH}/*.py ./

EXE="galaxy_train.py"

ARGS="--gpu_id 0"   # think we shouldn't pass this...
ARGS+=" --net ResNet18"
ARGS+=" --dset galaxy"
ARGS+=" --dset_path  /data/perdue/dmda/small/"
ARGS+=" --pristine_xfile Pristine_small.npy"
ARGS+=" --pristine_yfile Pristine_small_labels.npy"
ARGS+=" --noisy_xfile Noisy_small.npy"
ARGS+=" --noisy_yfile Noisy_small_labels.npy"
ARGS+=" --test_interval 500"
ARGS+=" --snapshot_interval 10000"
ARGS+=" --ly_type cosine"
ARGS+=" --loss_type mmd"
ARGS+=" --fisher_loss_type tr"
ARGS+=" --output_dir log_output"
ARGS+=" --em_loss_coef 0.1"
ARGS+=" --inter_loss_coef 1."
ARGS+=" --intra_loss_coef 1."
ARGS+=" --trade_off 1."
ARGS+=" --optim_choice 'SGD'"


SNGLRTY="/data/perdue/singularity/gnperdue-singularity_imgs-master-py3_dmda.simg"

cat << EOF
singularity exec --nv $SNGLRTY python3 $EXE $ARGS
EOF
singularity exec --nv $SNGLRTY python3 $EXE $ARGS
