import tensorflow as tf
import numpy as np


SMOOTH = 1e-6


class FScore(tf.keras.metrics.Metric):
    def __init__(self,  num_classes, beta=1., threshold=0.5, cls_weights=None, **kwargs):
        super(FScore, self).__init__(**kwargs)
        self.beta = beta
        self.threshold = threshold
        self.num_classes = num_classes
        self.n_examples = self.add_weight(name='number examples', initializer='zero')
        self.fscore = self.add_weight(name='fscore', shape=(num_classes,), initializer='zero')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_cls_pred = tf.cast(y_pred >= self.threshold, y_true.dtype)
        tp = tf.math.reduce_sum(y_true * y_cls_pred, axis=[1, 2])
        fp = tf.math.reduce_sum(y_cls_pred, axis=[1, 2]) - tp
        fn = tf.math.reduce_sum(y_true, axis=[1, 2]) - tp
        result = ((1 + self.beta**2) * tp + SMOOTH)/ ((1 + self.beta**2) * tp \
                + self.beta**2 * fn + fp + SMOOTH)

        result = tf.cast(result, self.dtype)
        batch_size = tf.cast(tf.shape(y_pred)[0], self.dtype)
        self.fscore.assign_add(tf.reduce_sum(result, axis=0))
        self.n_examples.assign_add(batch_size)

    def result(self):
        return tf.math.reduce_mean(self.fscore / self.n_examples)

    def reset_state(self):
        self.n_examples.assign(0)
        self.fscore.assign(np.zeros(self.num_classes,))


class IoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, threshold=0.5, **kwargs):
        super(IoU, self).__init__(**kwargs)
        self.threshold = threshold
        self.num_classes = num_classes
        self.n_examples = self.add_weight(name='number of examples', initializer='zero')
        self.iou = self.add_weight(name='iou', shape=(num_classes,), initializer='zero')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_cls_pred = tf.cast(y_pred >= self.threshold, y_true.dtype)
        intersection = tf.math.reduce_sum(y_true * y_cls_pred, axis=(1, 2))
        union = tf.math.reduce_sum(y_true + y_cls_pred, axis=(1, 2)) - intersection
        result = (intersection + SMOOTH) / (union + SMOOTH)
        result = tf.cast(result, self.dtype)
        batch_size = tf.cast(tf.shape(y_pred)[0], self.dtype)

        self.n_examples.assign_add(batch_size)
        self.iou.assign_add(tf.reduce_sum(result, axis=0))

    def result(self):
        return tf.math.reduce_mean(self.iou / self.n_examples)

    def reset_state(self):
        self.n_examples.assign(0)
        self.iou.assign(np.zeros(self.num_classes,))
