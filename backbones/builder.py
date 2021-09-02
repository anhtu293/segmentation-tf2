import tensorflow as tf
from .models import get_model


SUPPORTED_MODELS = {
    'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),

    # ResNets
    'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet152': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),

    # ResNeXt
    'resnext50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnext101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),

    # Mobile Nets
    'mobilenet': ('conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'),
    'mobilenetv2': ('block_13_expand_relu', 'block_6_expand_relu', 'block_3_expand_relu',
                    'block_1_expand_relu'),

    # EfficientNets
    'efficientnetb0': ('block6a_expand_activation', 'block4a_expand_activation',
                        'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb1': ('block6a_expand_activation', 'block4a_expand_activation',
                        'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb2': ('block6a_expand_activation', 'block4a_expand_activation',
                        'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb3': ('block6a_expand_activation', 'block4a_expand_activation',
                        'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb4': ('block6a_expand_activation', 'block4a_expand_activation',
                        'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb5': ('block6a_expand_activation', 'block4a_expand_activation',
                        'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb6': ('block6a_expand_activation', 'block4a_expand_activation',
                        'block3a_expand_activation', 'block2a_expand_activation'),
    'efficientnetb7': ('block6a_expand_activation', 'block4a_expand_activation',
                        'block3a_expand_activation', 'block2a_expand_activation')
}


TF_MODEL_NAMES = {
    'vgg16': 'VGG16',
    'vgg19': 'VGG19',

    # ResNets
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
    'resnet152': 'ResNet152',

    # Mobile Nets
    'mobilenet': 'MobileNet',
    'mobilenetv2': 'MobileNetV2',

    # EfficientNets
    'efficientnetb0': 'EfficientNetB0',
    'efficientnetb1': 'EfficientNetB1',
    'efficientnetb2': 'EfficientNetB2',
    'efficientnetb3': 'EfficientNetB3',
    'efficientnetb4': 'EfficientNetB4',
    'efficientnetb5': 'EfficientNetB5',
    'efficientnetb6': 'EfficientNetB6',
    'efficientnetb7': 'EfficientNetB7'
}


def get_model(name, weights):
    if not hasattr(tf.keras.applications, name):
        raise ValueError('Not implemented')
    model = getattr(tf.keras.applications, name)(include_top=False,
                                                 weights=weights)
    return model


def build_backbone(name, weights='imagenet'):
    assert name in SUPPORTED_MODELS.keys(), 'Model {} is not supported'.format(name)
    tf_model = TF_MODEL_NAMES[name]
    model = get_model(tf_model, weights=weights)
    return model


def get_feature_layers(name):
    return SUPPORTED_MODELS[name]
