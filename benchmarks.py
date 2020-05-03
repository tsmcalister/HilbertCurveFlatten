import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten, \
    Activation
from custom_layers import HCFlatten

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 64
EPOCHS = 20
STEPS_PER_EPOCH = 250


train_ds = tfds.load('deep_weeds', as_supervised=True, split='train[:90%]')
val_ds = tfds.load('deep_weeds', as_supervised=True, split='train[-10%:]')


def pre_process(image, label):
    image = tf.cast(image, 'float32')/ (1./255)
    label = tf.one_hot(label, 9)
    return image, label


train_generator = train_ds.repeat().map(pre_process).shuffle(1024).batch(BATCH_SIZE)
val_generator = val_ds.repeat().map(pre_process).shuffle(1024).batch(BATCH_SIZE)


def get_2d_model():
    model = tf.keras.Sequential([
        Conv2D(32, (3,3), input_shape=IMAGE_SIZE + (3,)),
        MaxPooling2D((2,2)),
        Conv2D(32, (3,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3, 3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3, 3)),
        MaxPooling2D((2,2)),
        Conv2D(128, (3, 3)),
        MaxPooling2D((2,2)),
        Conv2D(128, (3, 3)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(9),
        Activation('softmax', dtype='float32')
    ])
    rmsprop = tf.optimizers.Nadam()
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def get_1d_model():
    model = tf.keras.Sequential([
        HCFlatten(input_shape=IMAGE_SIZE+(3,)),
        Conv1D(32, 9),
        MaxPooling1D(4),
        Conv1D(32, 9),
        MaxPooling1D(4),
        Conv1D(64, 9),
        MaxPooling1D(4),
        Conv1D(64, 9),
        MaxPooling1D(4),
        Conv1D(128, 9),
        MaxPooling1D(4),
        Conv1D(128, 9),
        MaxPooling1D(4),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(9),
        Activation('softmax', dtype='float32')
    ])
    rmsprop = tf.optimizers.Nadam()
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


model_2d = get_2d_model()
model_1d = get_1d_model()

print("training 2d model...")

history_2d = model_2d.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data = val_generator,
    validation_steps = STEPS_PER_EPOCH // 10
)

print("training 1d model...")

history_1d = model_1d.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_generator,
    validation_steps=STEPS_PER_EPOCH // 10
)
