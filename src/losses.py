import tensorflow as tf


SMOOTH = 1e-6


class BinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(BinaryCrossentropy, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, SMOOTH, 1 - SMOOTH)
        loss = - (y_true * tf.math.log(y_pred + SMOOTH) + (1-y_true) * tf.math.log(1 - y_pred + SMOOTH))
        loss = tf.math.reduce_mean(loss, axis=(1, 2, 3))
        return loss


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, beta=1, **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.beta = beta

    def call(self, y_true, y_pred):
        tp = tf.math.reduce_sum(y_true * y_pred, axis=(1, 2))
        fp = tf.math.reduce_sum(y_pred, axis=(1, 2)) - tp
        fn = tf.math.reduce_sum(y_true, axis=(1, 2)) - tp
        loss = 1 - (1 + self.beta**2) * tp / ((1 + self.beta**2) * tp + self.beta**2 * fn + fp)
        loss = tf.math.reduce_mean(loss, axis=-1)
        return loss


class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super(BinaryFocalLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, SMOOTH, 1 - SMOOTH)
        loss_0 = - self.alpha * y_true * tf.math.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred + SMOOTH)
        loss_1 = - (1 - self.alpha)  * (1 - y_true) * tf.math.pow(y_pred, self.gamma) * tf.math.log(1 - y_pred + SMOOTH)
        loss = loss_0 + loss_1
        loss = tf.math.reduce_mean(loss, axis=(1, 2, 3))
        return loss
