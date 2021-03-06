from tensorflow import keras
from tensorflow.keras import layers


def vgg_style(input_tensor):
    """
    The original feature extraction structure from CRNN paper.
    Related paper: https://ieeexplore.ieee.org/abstract/document/7801919
    """
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(512, 2, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def build_model(num_classes, image_width=None, image_height=None, channels=1):
    """build CNN-RNN model"""

    img_input = keras.Input(shape=(image_height, image_width, channels))
    backbone = keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_tensor=img_input
    )
    x = backbone(img_input)
    # x = vgg_style(img_input)
    x = layers.Reshape((-1, 2048))(x)
    x = layers.Bidirectional(layers.LSTM(units=512, return_sequences=True))(x)
    # x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)
    x = layers.Dense(units=num_classes)(x)
    return keras.Model(inputs=img_input, outputs=x, name='CRNN')
