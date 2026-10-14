"""Model architecture for image forgery detection.

This is the network that produced ``image_real_or_fake.keras``: a two-stream
fusion of a frozen InceptionV3 (the "GoogleNet" branch in the original
notebook) and a frozen VGG16, combined and passed through dense heads that
output a single forgery probability.

Import :func:`build_fusion_model` to construct an untrained network, or use
``train.py`` to run the full training pipeline.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3, VGG16

DEFAULT_INPUT_SHAPE = (224, 224, 3)
DEFAULT_DENSE_UNITS = 1024


def build_fusion_model(
    input_shape=DEFAULT_INPUT_SHAPE,
    dense_units=DEFAULT_DENSE_UNITS,
    learning_rate=1e-3,
    pretrained=True,
):
    """Build the InceptionV3 + VGG16 fusion classifier.

    Parameters
    ----------
    input_shape : tuple
        ``(height, width, channels)`` fed to both backbones.
    dense_units : int
        Width of the fusion dense layers.
    learning_rate : float
        Learning rate for the Adam optimiser.
    pretrained : bool
        When True, initialise the backbones with ImageNet weights.
    """
    weights = "imagenet" if pretrained else None

    inception = InceptionV3(weights=weights, include_top=False, input_shape=input_shape)
    vgg = VGG16(weights=weights, include_top=False, input_shape=input_shape)

    for layer in inception.layers:
        layer.trainable = False
    for layer in vgg.layers:
        layer.trainable = False

    input_layer = tf.keras.Input(shape=input_shape)

    inception_features = inception(input_layer)
    vgg_features = vgg(input_layer)

    inception_features = layers.GlobalAveragePooling2D()(inception_features)
    inception_features = layers.Dense(dense_units, activation="relu")(inception_features)
    # (batch, dense_units) -> (batch, 1, 1, dense_units) -> (batch, 7, 7, dense_units)
    inception_features = layers.Reshape((1, 1, dense_units))(inception_features)
    inception_features = layers.UpSampling2D(size=(7, 7))(inception_features)

    fusion = layers.Concatenate()([inception_features, vgg_features])
    fusion = layers.GlobalAveragePooling2D()(fusion)
    fusion = layers.Dense(dense_units, activation="relu")(fusion)
    fusion = layers.Dropout(0.3)(fusion)
    predictions = layers.Dense(1, activation="sigmoid", name="forgery")(fusion)

    model = models.Model(inputs=input_layer, outputs=predictions, name="ela_fusion")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


if __name__ == "__main__":
    # Build without ImageNet weights so this runs offline; training uses the default.
    build_fusion_model(pretrained=False).summary()
