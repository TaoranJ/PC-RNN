# for debug
# CUDA_LAUNCH_BLOCKING=1

for ratio in 0.1 0.3 0.5 0.8
do
    python train.py --epochs 400 --batch-size 256 --eval-train --eval-test \
        --ob-ratio $ratio --checkpoint-path 'checkpoint'
done
