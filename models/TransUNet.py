import math
from pathlib import Path

from tensorflow.keras.layers import Conv2D, Input, Reshape, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file

from models import R50ViT, AddPositionEmbs, TransformerBlock, DecoderCup, SegmentationHead
from models.utils import load_weights_numpy

"""
Paper:
    https://arxiv.org/pdf/2102.04306.pdf
Reference Source:
    https://github.com/Beckschen/TransUNet
    https://github.com/kenza-bouzid/TransUnet
"""
IMAGENET21K_URL = "https://storage.googleapis.com/vit_models/imagenet21k/"
PRETRAINED_NAME = "R50+ViT-B_16.npz"


def load_weights(model):
    pretrained_path = Path("pretrained_model")
    pretrained_filepath = (pretrained_path / PRETRAINED_NAME).absolute()
    pretrained_path.mkdir(exist_ok=True)
    get_file(pretrained_filepath, IMAGENET21K_URL + PRETRAINED_NAME, cache_subdir="weights")
    load_weights_numpy(model, pretrained_filepath)


class TransUNet:
    def __init__(self, input_size=(512, 512, 1), n_channels=64, n_classes=1, activation="sigmoid", layers=12,
                 patch_size=16, hidden_size=768, dropout_rate=0.1, mlp_dim=3072, n_heads=12,
                 decoder_channels=(256, 128, 64, 16), loss_value="binary_crossentropy", use_pretrained=False):
        """
        Args:
            input_size (tuple): Size of the input image.
            n_channels (int): Number of channels in the input image.
            n_classes (int): Number of classes in the output image.
            activation (str): Final activation function.
            layers (int): Number of encoder-decoder layers in the U-Net.
            patch_size (int): Input patch size.
            hidden_size (int): Hidden feature size D
            dropout_rate (float): Dropout rate
            mlp_dim (int): Size of the MLP
            n_heads (int): Number of heads
            decoder_channels (list): List of decoder channels
            loss_value (str): Loss value
            use_pretrained (bool): Use pretrained or not
        """
        self.input_shape = input_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.activation = activation
        self.layers = layers
        patch_grid = int(input_size[0] / patch_size)
        self.patch_size = input_size[0] // 16 // patch_grid
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.mlp_dim = mlp_dim
        self.n_heads = n_heads
        self.decoder_channels = decoder_channels
        self.loss_value = loss_value
        self.use_pretrained = use_pretrained

        self.encoder_features = list()
        self.model = self.build_model()

    def build_model(self):
        input_data = Input(shape=self.input_shape, name="input")
        encoder, features = self.encoder(input_data)
        transformer = self.transformer(encoder)
        decoder = self.decoder(transformer, features)
        model = Model(input_data, decoder, name="TransUnet")
        model.compile(optimizer=Adam(), loss=self.loss_value, metrics=['accuracy'])
        if self.use_pretrained:
            load_weights(model)
        return model

    def embedding(self, x=None):
        x = Conv2D(filters=self.hidden_size, kernel_size=self.patch_size, strides=self.patch_size, name="embedding")(x)
        x = Reshape((x.shape[1] * x.shape[2], self.hidden_size))(x)
        x = AddPositionEmbs(name="Transformer/posembed_input")(x)
        x = Dropout(rate=self.dropout_rate)(x)
        return x

    def encoder(self, x=None):
        x, features = R50ViT(filters=self.n_channels, num_classes=self.n_classes, activation=self.activation)(x)
        x = self.embedding(x)
        return x, features

    def transformer(self, x=None):
        """Number of image patches N = (H*W)/P**2 """
        for n in range(self.layers):
            x, _ = TransformerBlock(n_heads=self.n_heads, mlp_dim=self.mlp_dim, dropout=self.dropout_rate,
                                    name=f"Transformer/encoderblock_{n}")(x)
        x = LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(x)
        return x

    def decoder(self, x=None, features=None):
        n_patch_sqrt = int(math.sqrt(x.shape[1]))
        x = Reshape((n_patch_sqrt, n_patch_sqrt, self.hidden_size))(x)
        x = DecoderCup(decoder_channels=self.decoder_channels)(x, features)
        x = SegmentationHead(n_classes=self.n_classes, activation_value="sigmoid")(x)
        return x


if __name__ == "__main__":
    trans_unet = TransUNet(n_classes=1, activation="sigmoid", use_pretrained=True)
    trans_unet.model.summary()
