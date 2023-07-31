import tensorflow as tf
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.layers import Layer, Conv2D, Activation, BatchNormalization, UpSampling2D

L2_WEIGHT_DECAY = 1e-4


class SegmentationHead(Layer):
    def __init__(self, name="seg_head", n_classes=3, kernel_size=1, activation_value="softmax", **kwargs):
        super(SegmentationHead, self).__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.activation_value = activation_value

    def build(self, *args):
        self.conv = Conv2D(
            filters=self.n_classes, kernel_size=self.kernel_size, padding="same",
            kernel_regularizer=regularizers.L2(L2_WEIGHT_DECAY), 
            kernel_initializer=initializers.LecunNormal())
        self.activation = Activation(self.activation_value)

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.activation(x)
        return x


class Conv2DReLu(Layer):
    def __init__(self, filters, kernel_size, padding="same", strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

    def build(self, input_shape):
        self.conv = Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
            padding=self.padding, use_bias=False, kernel_regularizer=regularizers.L2(L2_WEIGHT_DECAY), 
            kernel_initializer="lecun_normal")

        self.bn = BatchNormalization(momentum=0.9, epsilon=1e-5)

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x


class DecoderBlock(Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = Conv2DReLu(filters=self.filters, kernel_size=3)
        self.conv2 = Conv2DReLu(filters=self.filters, kernel_size=3)
        self.upsampling = UpSampling2D(
            size=2, interpolation="bilinear")

    def call(self, inputs, skip=None):
        x = self.upsampling(inputs)
        if skip is not None:
            x = tf.concat([x, skip], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(Layer):
    def __init__(self, decoder_channels, n_skip=3, **kwargs):
        super().__init__(**kwargs)
        self.decoder_channels = decoder_channels
        self.n_skip = n_skip

    def build(self, input_shape):
        self.conv_more = Conv2DReLu(filters=512, kernel_size=3)
        self.blocks = [DecoderBlock(filters=out_ch)
                       for out_ch in self.decoder_channels]

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, hidden_states, features):
        x = self.conv_more(hidden_states)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x
