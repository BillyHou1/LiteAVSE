# Fan
# Convolves clean speech with a room impulse response to simulate reverb,
# makes the model more robust to echoey rooms.
#
# TODO implement RIRAugmentor class and apply_rir function
#
# RIRAugmentor takes rir_json, prob=0.3, target_sr=16000. Load all RIR wav
# files listed in rir_json at init, store in a list. If any has a different
# sample rate resample to target_sr. prob=0.3 means 30% of training samples
# get reverb applied.
#
# __call__ rolls a random number, if above prob return audio unchanged.
# Otherwise pick a random RIR, fftconvolve audio with rir mode='full', trim
# to original length, energy-normalize so output has same RMS as input cause
# reverb changes loudness and we don't want that.
#
# apply_rir(audio, rir) is a standalone version without the probability check,
# just convolve + trim + normalize, in case dataloader calls it directly.
