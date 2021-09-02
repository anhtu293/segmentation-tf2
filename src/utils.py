import tensorflow as tf
import numpy as np
from functools import partial


# IMG_SIZE = 320


def normalize(image, labels):
    return (tf.cast(image, tf.float32) / 127.5) - 1, labels


def resize(input_image, labels, img_size):
    return tf.image.resize(input_image, img_size), \
            tf.image.resize(labels, img_size)


class Parser():
    def __init__(self, img_size, num_class, augmentation, multiclass=False):
        self.image_shape = [*img_size, 3]
        self.labels_shape = [*img_size, num_class]
        self.augmentation = augmentation
        self.multiclass = multiclass
        print(self.image_shape, self.labels_shape)

    def __call__(self, example):
        tfrecord_format = (
            {
                "image": tf.io.FixedLenFeature([], tf.string),
                "labels": tf.io.FixedLenFeature([], tf.string),
            }
        )
        example = tf.io.parse_single_example(example, tfrecord_format)
        image = example["image"]
        labels = example["labels"]

        labels = tf.io.parse_tensor(labels, out_type=tf.uint8)#[:, :, 1:]
        labels.set_shape(self.labels_shape)
        labels = labels / 255

        image = tf.io.decode_image(image, channels=3, dtype=tf.dtypes.uint8)
        image.set_shape(self.image_shape)
        image = tf.cast(image, tf.float32)
        image, labels = resize(image, labels, (self.image_shape[0], self.image_shape[1]))

        outputs = labels
        return image, outputs


def load_dataset(filenames, img_size=(512, 512), num_class=29, augmentation=True, multiclass=False):
    ignore_order = tf.data.Options()
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    dataset = dataset.with_options(
        ignore_order
    )
    dataset = dataset.map(
        partial(Parser(img_size, num_class, augmentation, multiclass=multiclass)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def open_tfrecord(filenames, batch_size=8, img_size=(512, 512), num_class=29, preprocess_func=None, augmentation=False, multiclass=False):
    dataset = load_dataset(filenames, img_size=img_size, num_class=num_class, augmentation=augmentation, multiclass=multiclass)
    if preprocess_func != None:
        dataset = dataset.map(preprocess_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(batch_size * 2).batch(batch_size)
    print("________OK________")
    return dataset


def freeze_weights(model):
    for layer in model.layers:
        if 'decoder' not in layer.name:
            layer.trainable = False
    return model


def unfreeze_weights(model):
    for layer in model.layers:
        layer.trainable = True
    return model

