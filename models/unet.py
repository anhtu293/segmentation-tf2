import tensorflow as tf
from backbones import build_backbone, get_feature_layers


def Conv2DBlock(filters):
    def layer(x):
        x = tf.keras.layers.Conv2D(filters, 
                                   (3, 3),
                                   strides=(1, 1),
                                   padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x
    return layer


def upsamling_decoder_block(name):
    name = 'decoder_' + name
    def upsample_block(x, skip, filters):
        x = tf.keras.layers.UpSampling2D((2, 2), name=name+'_upsample')(x)
        x = tf.keras.layers.Concatenate(axis=-1, name=name+'_concat')([skip, x])
        x = Conv2DBlock(filters)(x)
        x = Conv2DBlock(filters)(x)
        return x
    return upsample_block


def transpose_decoder_block(name):
    name = 'decoder_' + name
    def upsample_block(x, skip, filters):
        x = tf.keras.layers.Conv2DTranspose(filters,
                                            (4, 4),
                                            strides=(2, 2),
                                            padding='same',
                                            name=name+'_transpose')(x)
        x = tf.keras.layers.BatchNormalization(name=name+'_batchnorm')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Concatenate(axis=-1, name=name+'_concat')([skip, x])
        x = Conv2DBlock(filters)(x)
        return x
    return upsample_block


def Unet(backbone_name,
         decoder_block=None,
         skip_connection_layer=None,
         decoder_filters=(256, 128, 64, 32, 16),
         n_classes=1,
         multi_output=False,
         activation='sigmoid'):
    if skip_connection_layer is None:
        skip_connection_layer = get_feature_layers(backbone_name)
    n_upsample_blocks = len(skip_connection_layer)

    if decoder_block == 'upsampling':
        upsample_blocks = [upsamling_decoder_block(skip_connection_layer[i]) for i in range(n_upsample_blocks)]
    else:
        upsample_blocks = [transpose_decoder_block(skip_connection_layer[i]) for i in range(n_upsample_blocks)]

    # build unet
    backbone = build_backbone(backbone_name)
    inputs = backbone.inputs
    x = backbone.output
    skip_connection_features = [backbone.get_layer(skip_connection_layer[i]).output
                                    for i in range(len(skip_connection_layer))]
    
    for i in range(n_upsample_blocks):
        x = upsample_blocks[i](x, skip_connection_features[i], decoder_filters[i])

    # last upsampling
    if decoder_block == 'upsampling':
        x = tf.keras.layers.UpSampling2D((2, 2), name='decoder_last_upsampling')(x)
    else:
        x = tf.keras.layers.Conv2DTranspose(decoder_filters[-1],
                                            (4, 4),
                                            strides=(2, 2),
                                            padding='same',
                                            name='decoder_last_transpose')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2DBlock(decoder_filters[-1])(x)
    x = Conv2DBlock(decoder_filters[-1])(x)

    if not multi_output:
        outputs = tf.keras.layers.Conv2D(n_classes, (3, 3), activation=activation, padding='same')(x)
    else:
        outputs = [tf.keras.layers.Conv2D(1, (3, 3), activation=activation, padding='same')(x)
                    for _ in range(len(n_classes))]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
