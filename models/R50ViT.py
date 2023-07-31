import tensorflow as tf
from keras.layers import Layer
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, ZeroPadding2D
from tensorflow_addons.layers import GroupNormalization


def ws_reg(kernel):
    kernel_mean, kernel_std = tf.nn.moments(
        kernel, axes=[0, 1, 2], keepdims=True)
    kernel = (kernel - kernel_mean) / (kernel_std + 1e-5)


def conv3x3(cout, strides=1, groups=1, bias=False, name=""):
    return Conv2D(cout, kernel_size=3, strides=strides, padding="same", use_bias=bias, groups=groups, name=name,
                  kernel_regularizer=ws_reg)


def conv1x1(cout, strides=1, groups=1, bias=False, name=""):
    return Conv2D(cout, kernel_size=1, strides=strides, padding="same", use_bias=bias, groups=groups, name=name,
                  kernel_regularizer=ws_reg)


def bottleneck(x, cin, cout=None, cmid=None, strides=1, name='', step=1):
    cout = cout or cin
    cmid = cmid or cout // 4
    if strides != 1 or cin != cout:
        shortcut = conv1x1(cout, strides=strides, name=name + f"_{step}_conv0")(x)
        shortcut = GroupNormalization(cout, epsilon=1e-5, name=name + f"_{step}_gn0")(shortcut)
    else:
        shortcut = x

    x = conv1x1(cmid, name=name + f"_{step}_conv1")(x)
    x = GroupNormalization(epsilon=1e-6, name=name + f"_{step}_gn1")(x)
    x = Activation("relu", name=name + f"_{step}_relu1")(x)

    x = conv3x3(cmid, strides=strides, name=name + f'_{step}_conv2')(x)
    x = GroupNormalization(epsilon=1e-6, name=name + f'_{step}_gn2')(x)
    x = Activation("relu", name=name + f"_{step}_relu2")(x)

    x = conv1x1(cout, name=name + f"_{step}_conv3")(x)
    x = GroupNormalization(epsilon=1e-6, name=name + f"_{step}_gn3")(x)
    x = Add(name=name + f"_{step}_residual")([shortcut, x])
    x = Activation("relu", name=name + f"_{step}_relu3")(x)
    return x


def conv_block(x, filters, strides=1, name="conv_block", steps=1):
    if strides == 1:
        x = bottleneck(x, cin=filters, cout=filters * 4, cmid=filters, strides=strides, name=name, step=1)
        for step in range(2, steps + 1):
            x = bottleneck(x, cin=filters * 4, cout=filters * 4, cmid=filters, name=name, step=step)
        return x

    x = bottleneck(x, cin=filters, cout=filters * 2, cmid=filters >> 1, strides=strides, name=name, step=1)
    for step in range(2, steps + 1):
        x = bottleneck(x, cin=filters * 2, cout=filters * 2, cmid=filters >> 1, name=name, step=step)
    return x


class R50ViT(Layer):
    """This is the backbone model class used in the TransUnet paper.
    Paper : https://arxiv.org/pdf/2102.04306.pdf
    """
    def __init__(self, filters=64, num_classes=1000, activation="softmax", **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.activation = activation
        self.filters = filters
        self.skip = list()

    def call(self, _input, **kwargs):
        self.bn_axis = 3 if _input.shape[-1] == 3 else 1
        x = ZeroPadding2D((3, 3), name="conv1_pad1")(_input)
        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), name="conv1_conv1", kernel_regularizer=ws_reg)(x)
        x = GroupNormalization(32, epsilon=1e-6, name="conv1_gn1")(x)
        x = Activation("relu", name="conv1_relu1")(x)
        self.skip.append(x)

        x = ZeroPadding2D((1, 1), name="conv1_pad2")(x)
        x = MaxPooling2D((3, 3), strides=2, name="conv1_pool1")(x)

        x = conv_block(x, filters=self.filters, name="conv2", steps=3)
        self.skip.append(x)
        x = conv_block(x, filters=self.filters * 4, strides=2, name="conv3", steps=4)
        self.skip.append(x)
        x = conv_block(x, filters=self.filters * 8, strides=2, name="conv4", steps=9)
        return x, self.skip[::-1]
