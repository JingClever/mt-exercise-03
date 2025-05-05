#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

models=$base/models
configs=$base/configs

mkdir -p $models

# 8 cores of CPU
num_threads=8
#device=0

# measure time

SECONDS=0

logs=$base/logs

model_name=deen_transformer_regular_post

mkdir -p $logs

mkdir -p $logs/$model_name

OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/$model_name.yaml > $logs/$model_name/out 2> $logs/$model_name/err

echo "time taken:"
echo "$SECONDS seconds"


# numpy 2.2.5

# tried mps but it does not work because of the following error:
# /AppleInternal/Library/BuildRoots/ce725a5f-c761-11ee-a4ec-b6ef2fd8d87b/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPSCore/Types/MPSNDArray.mm:869: failed assertion `[MPSNDArray, initWithBuffer:descriptor:] Error: buffer is not large enough. Must be 6348 bytes
