# Billy
# VCE scores each video frame's reliability, outputs alpha in [0,1]. High means
# the face is clear and useful, low means blocked or blurry.
#
# TODO implement VCE and VCEWithTemporalSmoothing
#
# VCE is a 3-layer MLP: 512->256->64->1 with ReLU in between and sigmoid at
# the end. Each frame is scored independently, no time info needed here.
#
# VCEWithTemporalSmoothing inherits VCE and adds a causal Conv1d to smooth
# alpha over time so it doesn't jump around frame to frame. Causal means you
# only pad on the left (no future frames). Init conv weights to 1/kernel_size
# so it starts as a simple moving average. Clamp output to [0,1] after
# smoothing cause the conv can push it slightly out of range.
