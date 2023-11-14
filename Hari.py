import os, shutil, random

import tensorflow as tf

from tensorflow.keras import utils

import pathlib

from Helper import *

filepath = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

root = GetOrigin(filepath=filepath)

[Getfiltrate(r) for r in root]

image_dir = "/tmp/images"

if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
else:
    os.mkdir(image_dir)

for folder in ["training", "testing"]:
    fone = os.path.join(image_dir, folder)
    os.mkdir(fone)
    for label in ["cat", "dog"]:
        flabel = os.path.join(fone, label)
        os.mkdir(flabel)

root_train = "/tmp/images/training"; root_testi = "/tmp/images/testing"

trc = os.path.join(root_train, "cat")

ttc = os.path.join(root_testi, "cat")

Traintest(root[0], trc, ttc)

trd = os.path.join(root_train, "dog")

ttd = os.path.join(root_testi, "dog")

Traintest(root[1], trd, ttd)

image_size = (150, 150); batch_size = 128

label_mode = "binary"; interpolation = "nearest"

train = utils.image_dataset_from_directory(
    root_train,
    image_size=image_size,
    batch_size=batch_size,
    label_mode=label_mode,
    interpolation=interpolation)

testi = utils.image_dataset_from_directory(
    root_testi,
    image_size=image_size,
    batch_size=batch_size,
    label_mode=label_mode,
    interpolation=interpolation)

def norm(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.math.divide(image, 255.0)
  return image, label

train = train.map(norm).shuffle(100).cache().prefetch(1)

testi = testi.map(norm)

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

input_shape = (150, 150, 3); node = 1; lr = 0.001

pre_model = Sequential([
    layers.RandomFlip(input_shape=input_shape),
    layers.RandomRotation(factor=0.4),
    layers.RandomTranslation(height_factor=0.2, width_factor=0.2, interpolation=interpolation),
    layers.RandomZoom(height_factor=0.2, interpolation=interpolation)
])


model = Sequential([
    pre_model,

    layers.Conv2D(16, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dense(node, activation="sigmoid")
])

optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, weight_decay=lr)
loss = tf.keras.losses.BinaryCrossentropy()
metrics = tf.keras.metrics.BinaryAccuracy()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

from tensorflow.keras.callbacks import EarlyStopping

epochs = 18

patience = 5; verbose = 1

model_hist = model.fit(train, validation_data=testi,
                       epochs=epochs, verbose=verbose,
                       callbacks=[EarlyStopping(monitor="val_binary_accuracy", patience=patience)])

model.evaluate(train, verbose=0)

model.evaluate(testi, verbose=0)

from Histplot import Histplot

image_hist = Histplot(model_hist, hist_acc="binary_accuracy")

image_hist.Trainplot()

image_hist.Testiplot()

model_path = "/tmp/model"

tf.saved_model.save(model, model_path)

optm = [tf.lite.Optimize.DEFAULT]

filemega = lambda filepath : os.path.getsize(filepath) / float(2**20)

tflite_model = Tensolite(model_path, optm=optm, filename="normal_model")

round(filemega("/content/normal_model.tflite"), 3)

"""```py
dataset = tf.data.Dataset

train_image, train_label = next(iter(train))

def mini_datagen():
  for t in dataset.from_tensor_slices(train_image).batch(1).take(100):
    yield [t]

tflite_model = lite.from_saved_model(model_path)

tflite_model.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model.representative_dataset = mini_datagen

tflite_model.target_spec.supported_ops

tflite_model.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model.target_spec.supported_ops

tflite_model.inference_input_type, tflite_model.inference_output_type

tflite_model.inference_input_type = tf.uint8

tflite_model.inference_output_type = tf.uint8

tflite_model.inference_input_type, tflite_model.inference_output_type

tflite_model_quant = tflite_model.convert()

tflite_model_file = pathlib.Path("quant_lite.tflite")

tflite_model_file.write_bytes(tflite_model_quant)

round(filemega("/content/quant_lite.tflite"), 3)
```
"""
