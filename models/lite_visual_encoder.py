# Zhenning
# Extracts visual features from video frames (lip region) for the audio model.
# Two versions: EncoderA uses pretrained MobileNetV3 so it works with less data,
# EncoderB is a custom 3D CNN that trains from scratch but needs more data.
#
# Both take video [B, 3, Tv, 96, 96] and output:
# visual_raw [B, 512, T_audio] goes to VCE for reliability scoring
# visual_feat [B, 1, T_audio, F_audio] goes to FSVG for fusion
#
# TODO implement LiteVisualEncoderA and LiteVisualEncoderB
#
# EncoderA: use torchvision mobilenet_v3_small pretrained, take backbone.features
# which outputs 576 channels (not 512). Process each frame independently through
# the 2D backbone then AdaptiveAvgPool2d(1) to get [B*Tv, 576]. Reshape to
# [B, 576, Tv] then Conv1d temporal conv 576->512 kernel_size=5 padding=2 with
# BatchNorm and ReLU to capture lip dynamics. Interpolate to T_audio cause video
# fps and audio frame rate don't match. Then Conv1d(512, 1, 1) to project down
# and unsqueeze + expand for the freq dim. If freeze_backbone=True set all
# backbone params requires_grad=False.
#
# EncoderB: 4 layers Conv3d (3->16->32->64->128) with BatchNorm3d and ReLU,
# spatial stride 2 on all layers, temporal stride 2 on the last two. Then
# AdaptiveAvgPool3d to kill spatial dims, then same temporal conv + interpolate
# + projection as EncoderA. About 0.8M params, all trainable.
