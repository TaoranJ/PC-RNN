# for debug
# CUDA_LAUNCH_BLOCKING=1

for ratio in 0.1 0.3 0.5 0.8
do
    python train.py --epochs 200 --embed-dim 4 --pencoder-hidden-dim 16 \
        --oencoder-hidden-dim 8 --decoder-hidden-dim 16 \
        --decoder-inner-dim 32 --batch-size 1024 --lr 0.01 --tune-lr \
        --eval-train --eval-test --use-cuda 0 --ob-ratio $ratio \
        --checkpoint-path 'checkpoint'
done
