import tensorflow as tf
import os
import argparse
import glob

from models import Unet
# from utils.tfrecord import open_tfrecord
# from utils.freeze import freeze_weights, unfreeze_weights
from src.losses import BinaryCrossentropy, DiceLoss, BinaryFocalLoss
from src.metrics import FScore, IoU
from src.utils import open_tfrecord, freeze_weights, unfreeze_weights


IMG_SIZE = (512, 512)
LR = 5e-4
NUM_CLASSES = 23


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus',
                        help='gpu id for train')
    parser.add_argument('--output-dir',
                        help='output model directory')
    parser.add_argument('--backbone',
                        help='backbone model',
                        default='efficientnetb0')
    parser.add_argument('--batch-size',
                        help='mini batch',
                        type=int,
                        default=4)
    parser.add_argument('--freeze',
                        help='freeze',
                        action='store_true')
    args = parser.parse_args()
    return args


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, global_batch_size, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.focal = BinaryFocalLoss(reduction=tf.keras.losses.Reduction.NONE)
        self.dice = DiceLoss(reduction=tf.keras.losses.Reduction.NONE)
        self.bce = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.global_batch_size = global_batch_size

    def compute_distributed_loss(self, loss):
        loss = tf.math.reduce_sum(loss)
        loss = loss / self.global_batch_size
        return loss

    def call(self, y_true, y_pred):
        loss = self.focal(y_true, y_pred)  + self.dice(y_true, y_pred)
        loss = self.compute_distributed_loss(loss)
        return loss
        

def scheduler(epoch, lr):
    if epoch < 60:
        return lr
    elif epoch >= 60 and epoch < 100:
        return lr * 0.1
    else:
        return lr * 0.01


def main():
    class LRPrinter(tf.keras.callbacks.Callback):
        def __init__(self, **kwargs):
            super(LRPrinter, self).__init__(**kwargs)

        def on_epoch_begin(self, epoch, logs):
            print('\n Learning rate for epoch {} is {:.6f}'.format(epoch, model.optimizer.lr.numpy()))

    train_dataset_size = 7145
    validation_dataset_size = 376
    GLOBAL_BATCH_SIZE = args.batch_size * 2
    train_steps = train_dataset_size // GLOBAL_BATCH_SIZE

    # create output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # setup multi gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print(tf.config.get_visible_devices('GPU'))
    strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    
    # dataset
    files = glob.glob('./tfrecords/train*.tfrecords')
    print(files)
    train_dataset = open_tfrecord(files, GLOBAL_BATCH_SIZE, IMG_SIZE, NUM_CLASSES)
    train_dataset = train_dataset.repeat()
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    files = glob.glob('./tfrecords/val*.tfrecords')
    print(files)
    val_dataset = open_tfrecord(files, GLOBAL_BATCH_SIZE, IMG_SIZE, NUM_CLASSES)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)


    # define loss and metrics
    loss = CustomLoss(global_batch_size=args.batch_size)
    # loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    with strategy.scope():
        # define model
        model = Unet(args.backbone, decoder_block='upsampling', n_classes=NUM_CLASSES)
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[FScore(num_classes=NUM_CLASSES, name='f1'),
                               IoU(num_classes=NUM_CLASSES, name='iou')])
        model.summary()

    # freeze
    if args.freeze:
        model = freeze_weights(model)
        model.fit(train_dist_dataset, batch_size=GLOBAL_BATCH_SIZE, epochs=2, steps_per_epoch=train_steps)
        model = unfreeze_weights(model)
        with strategy.scope():
            model.compile(optimizer=optimizer,
                        loss=loss,
                        metrics=[FScore(num_classes=NUM_CLASSES, name='f1'),
                                IoU(num_classes=NUM_CLASSES, name='iou')])

    # checkpoint
    filepath = os.path.join(args.output_dir, 'saved-model-{epoch:02d}-{val_iou:.4f}-{val_f1:.4f}.hdf5')
    best_callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_f1',
                                                        save_best_only=True, mode='max',
                                                        save_freq='epoch', save_weights_only=False,
                                                        verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monirtor='val_f1',
                                                    save_best_only=False, period=5, verbose=1)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)
    lr_printer = LRPrinter()
                                                                
    # train
    history = model.fit(train_dataset, batch_size=GLOBAL_BATCH_SIZE, epochs=200,
                        callbacks=[best_callback, checkpoint, lr_schedule, lr_printer],
                        steps_per_epoch=train_steps, validation_data=val_dataset)


if __name__ == '__main__':
    args = parse_args()
    main()
