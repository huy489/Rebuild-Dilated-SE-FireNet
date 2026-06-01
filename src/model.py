import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from . import config as cfg

def conv_bn_lrelu(x, filters, kernel_size, strides=1, dilation_rate=1, name=None):
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(cfg.L2_RATE),
        name=name,
    )(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = layers.LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA, name=None if name is None else f"{name}_lrelu")(x)
    return x

def se_block(x, ratio=cfg.SE_RATIO, name="se"):
    channels = int(x.shape[-1])
    hidden = max(channels // ratio, 1)

    se = layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)
    se = layers.Dense(
        hidden,
        use_bias=False,
        kernel_regularizer=regularizers.l2(cfg.L2_RATE),
        name=f"{name}_fc1",
    )(se)
    se = layers.LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA, name=f"{name}_lrelu")(se)
    se = layers.Dense(
        channels,
        activation="sigmoid",
        use_bias=False,
        kernel_regularizer=regularizers.l2(cfg.L2_RATE),
        name=f"{name}_fc2_sigmoid",
    )(se)
    se = layers.Reshape((1, channels), name=f"{name}_reshape")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])

def dilated_fire_module(x, squeeze_filters, expand_filters, dilation_rate, block_name):
    shortcut = x

    # 1. Squeeze
    s = conv_bn_lrelu(
        x,
        filters=squeeze_filters,
        kernel_size=1,
        strides=1,
        name=f"fire_{block_name}_squeeze_1x1",
    )

    # 2. Expand branch 1x1
    e1 = layers.Conv1D(
        filters=expand_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(cfg.L2_RATE),
        name=f"fire_{block_name}_expand_1x1",
    )(s)
    e1 = layers.LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA, name=f"fire_{block_name}_expand_1x1_lrelu")(e1)

    # 3. Expand branch 3x3 dilated
    e3 = layers.Conv1D(
        filters=expand_filters,
        kernel_size=3,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(cfg.L2_RATE),
        name=f"fire_{block_name}_expand_3x3_dil{dilation_rate}",
    )(s)
    e3 = layers.LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA, name=f"fire_{block_name}_expand_3x3_lrelu")(e3)

    # 4. Concatenate e1 and e3
    out = layers.Concatenate(axis=-1, name=f"fire_{block_name}_concat")([e1, e3])
    out = layers.BatchNormalization(name=f"fire_{block_name}_concat_bn")(out)

    # 5. SE attention
    out = se_block(out, ratio=cfg.SE_RATIO, name=f"fire_{block_name}_se")

    # 6. Residual shortcut projection if channels mismatch
    out_channels = int(out.shape[-1])
    shortcut_channels = int(shortcut.shape[-1])

    if shortcut_channels != out_channels:
        shortcut = layers.Conv1D(
            filters=out_channels,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(cfg.L2_RATE),
            name=f"fire_{block_name}_shortcut_proj",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"fire_{block_name}_shortcut_bn")(shortcut)

    out = layers.Add(name=f"fire_{block_name}_add")([out, shortcut])
    out = layers.LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA, name=f"fire_{block_name}_out_lrelu")(out)
    out = layers.SpatialDropout1D(cfg.SPATIAL_DROPOUT, name=f"fire_{block_name}_spatial_dropout")(out)
    return out

def learnable_downsample(x, filters, block_name):
    return conv_bn_lrelu(
        x,
        filters=filters,
        kernel_size=3,
        strides=2,
        name=f"{block_name}_learnable_downsample",
    )

def build_dilated_se_firenet(input_shape=cfg.INPUT_SHAPE):
    inputs = layers.Input(shape=input_shape, name="Input_ECG")
    x = inputs

    # Stem block
    x = conv_bn_lrelu(
        x,
        filters=cfg.STEM_FILTERS,
        kernel_size=7,
        strides=2,
        name="Stem_Conv1D_k7_s2",
    )

    # Fire Blocks with learnable downsampling
    for i, (name, squeeze, expand, dilation, out_channels) in enumerate(cfg.FIRE_BLOCKS):
        x = dilated_fire_module(
            x,
            squeeze_filters=squeeze,
            expand_filters=expand,
            dilation_rate=dilation,
            block_name=name,
        )
        
        # Downsample except after last block
        if i < len(cfg.FIRE_BLOCKS) - 1:
            x = learnable_downsample(x, filters=out_channels, block_name=f"down_after_{name}")

    # Classifier
    x = layers.GlobalAveragePooling1D(name="Global_Average_Pooling")(x)
    x = layers.Dropout(cfg.DROPOUT_CLASSIFIER, name="Classifier_Dropout")(x)
    outputs = layers.Dense(
        cfg.NUM_CLASSES,
        activation="softmax",
        dtype="float32",
        name="Output_Softmax",
    )(x)

    return models.Model(inputs=inputs, outputs=outputs, name="Dilated_SE_FireNet")

if __name__ == "__main__":
    model = build_dilated_se_firenet()
    model.summary()
