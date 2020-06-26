#!/bin/bash

SCRIPTKEY=`date +%s`
mkdir job${SCRIPTKEY}

NGPU=1
NODES=gpu2
NODES=gpu3
NODES=gpu4

EXE="galaxy_train_exe.sh"
GROUP="accelai"

cat << EOF
sbatch --gres=gpu:${NGPU} --nodelist=${NODES} -A $GROUP -p gpu $EXE
EOF

# do the thing, etc.
cp ${EXE} job${SCRIPTKEY}
pushd job${SCRIPTKEY}
sbatch --gres=gpu:${NGPU} --nodelist=${NODES} -A $GROUP -p gpu $EXE
popd
