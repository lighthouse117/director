python agents/director/train.py \
    --logdir logdir/$(date +%Y%m%d-%H%M%S) \
    --configs dmc_vision_acrobot \
    --imag_horizon 16 \
    --train_skill_duration 16

python agents/director/train.py \
    --logdir logdir/$(date +%Y%m%d-%H%M%S) \
    --configs dmc_vision_acrobot \
    --imag_horizon 16 \
    --train_skill_duration 8

python agents/director/train.py \
    --logdir logdir/$(date +%Y%m%d-%H%M%S) \
    --configs dmc_vision_acrobot \
    --imag_horizon 16 \
    --train_skill_duration 4

python agents/director/train.py \
    --logdir logdir/$(date +%Y%m%d-%H%M%S) \
    --configs dmc_vision_acrobot \
    --imag_horizon 16 \
    --train_skill_duration 2

python agents/director/train.py \
    --logdir logdir/$(date +%Y%m%d-%H%M%S) \
    --configs dmc_vision_acrobot \
    --imag_horizon 8 \
    --train_skill_duration 8

python agents/director/train.py \
    --logdir logdir/$(date +%Y%m%d-%H%M%S) \
    --configs dmc_vision_acrobot \
    --imag_horizon 8 \
    --train_skill_duration 4

python agents/director/train.py \
    --logdir logdir/$(date +%Y%m%d-%H%M%S) \
    --configs dmc_vision_acrobot \
    --imag_horizon 8 \
    --train_skill_duration 2

python agents/director/train.py \
    --logdir logdir/$(date +%Y%m%d-%H%M%S) \
    --configs dmc_vision_acrobot \
    --imag_horizon 4 \
    --train_skill_duration 4

