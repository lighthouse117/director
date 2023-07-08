# Script for training

python3 /embodied/agents/director/train.py \
  --logdir /embodied/logdir/$(date +%Y%m%d-%H%M%S) \
  --configs atari \
  --task atari_pong