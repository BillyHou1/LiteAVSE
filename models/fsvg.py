# Dominic + Zhenning + Billy (support)
# FSVG controls how much visual info gets injected at each frequency bin.
# The idea is lip movements help most with speech frequencies (300Hz-3kHz)
# but don't do much for high-freq noise, so the gate learns to be selective.
#
# TODO implement FSVG and FSVGWithPrior
#
# FSVG: concat audio_feat and visual_feat along channel dim to get
# [B, 128, T, F], run through 3 Conv2d layers (128->64->32->1) with ReLU
# in between and sigmoid at the end. kernel_size=3 padding=1 so T and F
# stay the same. Output gate is [B, 1, T, F] between 0 and 1.
#
# FSVGWithPrior inherits FSVG and adds a learnable frequency prior.
# Speech energy lives around 300Hz-3kHz which maps to encoded bins [4:38]
# after DenseEncoder downsampling. Init a nn.Parameter of shape
# [1, 1, 1, n_freq] with 0.8 for bins [4:38] and 0.3 for everything else.
# In forward just multiply the base gate with sigmoid(freq_prior) so the
# model starts biased toward speech freqs but can adjust during training.
