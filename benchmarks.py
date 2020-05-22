import tensorflow_datasets as tfds
import tensorflow as tf
tf.random.set_seed(2718281)
import numpy as np
import matplotlib.pyplot as plt

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, BatchNormalization, Dense, Dropout, \
    Flatten, Lambda
from custom_layers import HCFlatten

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 64
EPOCHS = 20
SEED = 42


# train_ds = tfds.load('resisc45', as_supervised=True, split='train[:90%]')
# val_ds = tfds.load('resisc45', as_supervised=True, split='train[-10%:]')
train_ds = tfds.load('resisc45', as_supervised=True, split='train[:60%]')
val_ds = tfds.load('resisc45', as_supervised=True, split='train[-5%:]')

print("loading training data...")
train_X = []
train_y = []

for sample in train_ds:
    train_X.append(sample[0].numpy())
    train_y.append(tf.keras.utils.to_categorical(sample[1].numpy(), num_classes=45))

print(train_X[0].shape)
print(train_y[0].shape)

print("loading validation data")
val_X = []
val_y = []

for sample in val_ds:
    val_X.append(sample[0].numpy())
    val_y.append(tf.keras.utils.to_categorical(sample[1].numpy(), num_classes=45))

# print("converting to numpy arrays...")
train_X = np.array(train_X)
# train_y = np.array(train_y)
val_X = np.array(val_X)
# val_y = np.array(val_y)

print("defining image generators")
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

print("defining data generators")
train = train_datagen.flow(train_X, train_y, batch_size=BATCH_SIZE, seed=SEED)
val = val_datagen.flow(val_X, val_y, batch_size=BATCH_SIZE, seed=SEED)

def get_2d_model():
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), input_shape=IMAGE_SIZE + (3,)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(45, activation='softmax'),
    ])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
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
        Flatten(),
        Dropout(0.5),
        Dense(100, activation='relu'),
        Dense(45, activation='softmax'),
    ])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def get_1d_model_simple():
    def simple_flatten(x):
        shape = x.shape
        shape = [-1, shape[1]*shape[2], shape[3]]
        return tf.reshape(x, shape)
    model = tf.keras.Sequential([
        Lambda(simple_flatten, input_shape=IMAGE_SIZE+(3,)),
        Conv1D(32, 9),
        MaxPooling1D(4),
        Conv1D(32, 9),
        MaxPooling1D(4),
        Conv1D(64, 9),
        MaxPooling1D(4),
        Conv1D(64, 9),
        MaxPooling1D(4),
        Flatten(),
        Dropout(0.5),
        Dense(100, activation='relu'),
        Dense(45, activation='softmax'),
    ])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


model_2d = get_2d_model()
model_1d_hilbert = get_1d_model()
model_1d_simple = get_1d_model_simple()

steps_per_epoch = int( np.ceil(train_X.shape[0] / BATCH_SIZE) )
validation_steps = int(np.ceil(val_X.shape[0] / BATCH_SIZE))

print("training 2d model...")

history_2d = model_2d.fit(
    train,
    epochs=EPOCHS,
    validation_data=val,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

print("training 1d model with HCFlatten...")

history_1d = model_1d_hilbert.fit(
    train,
    epochs=EPOCHS,
    validation_data=val,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

print("training 1d model with simple Flatten...")

history_1d_simple = model_1d_simple.fit(
        train,
        epochs=EPOCHS,
        validation_data=val,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
)


def compare_training(history_2d, history_1d_simple, history_1d_hilbert):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Comparison of image classification architectures')

    axs[0, 0].plot(history_2d.history['accuracy'])
    axs[0, 0].plot(history_1d_simple.history['accuracy'])
    axs[0, 0].plot(history_1d_hilbert.history['accuracy'])
    axs[0, 0].set_xlabel('epoch')
    axs[0, 0].set_ylabel('accuracy')
    axs[0, 0].legend(['2d', '1d_simple', '1d_hilbert'], loc='upper left')

    axs[0, 1].plot(history_2d.history['val_accuracy'])
    axs[0, 1].plot(history_1d_simple.history['val_accuracy'])
    axs[0, 1].plot(history_1d_hilbert.history['val_accuracy'])
    axs[0, 1].set_xlabel('epoch')
    axs[0, 1].set_ylabel('validation accuracy')
    axs[0, 1].legend(['2d', '1d_simple', '1d_hilbert'], loc='upper left')

    axs[1, 0].plot(history_2d.history['loss'])
    axs[1, 0].plot(history_1d_simple.history['loss'])
    axs[1, 0].plot(history_1d_hilbert.history['loss'])
    axs[1, 0].set_xlabel('epoch')
    axs[1, 0].set_ylabel('loss')
    axs[1, 0].legend(['2d', '1d_simple', '1d_hilbert'], loc='upper left')

    axs[1, 1].plot(history_2d.history['val_loss'])
    axs[1, 1].plot(history_1d_simple.history['val_loss'])
    axs[1, 1].plot(history_1d_hilbert.history['val_loss'])
    axs[1, 1].set_xlabel('epoch')
    axs[1, 1].set_ylabel('validation loss')
    axs[1, 1].legend(['2d', '1d_simple', '1d_hilbert'], loc='upper left')

    plt.show()

compare_training(history_2d, history_1d_simple, history_1d)
