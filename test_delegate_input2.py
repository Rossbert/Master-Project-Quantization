import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import numpy.typing as npt 
import random
from typing import Tuple, List

def database_generator(database_size: int, shape: tuple) -> Tuple[npt.NDArray, List[int]]:
    """ Generator will detect if channel has values
    """
    coupled_data = []
    batch = database_size//3
    category = 0
    MAX = 20
    MIN = 15
    NOISE = 5
    for i in range(database_size):
        value = np.zeros(shape = shape)
        value[:,:,category] = np.random.randint(MIN, MAX, size = shape[:-1])
        coupled_data.append([value, category])
        if i > (category + 1)*batch:
            category = category + 1
    random.shuffle(coupled_data)
    database = [coupled_data[i][0] for i in range(len(coupled_data))]
    categories = [coupled_data[i][1] for i in range(len(coupled_data))]
    database = np.array(database) + np.random.randint(0, NOISE, size = (database_size, *shape))
    database = (database/np.max(database)).astype(np.float32)
    return database, categories

def category_builder(raw_category: int)->None:
    if raw_category == 0 or raw_category == 2:
        return 0
    elif raw_category == 1:
        return 1
    else:
        return 2

# Load paths
DATABASE_PATH = "./model/database2.npy"
TFLITE_OUT_PATH = "./model/add2_test.tflite"
DELEGATE_PATH = "./dependencies/custom_delegates.dll"
# print(sys.byteorder)

tensors_type = np.float32
IN_MAX = 1
IN_MIN = 0
CONV_MAX = 2.5 # Experimentally extracted by making tests with model, if weights change this value changes

input12_shape = (4, 5, 3)

kernel12_size = 3
kernel3_size = 2
kernel_filters = 4
dense_units = 3

conv12_shape = (kernel12_size, kernel12_size, input12_shape[-1], kernel_filters)

input_names = ["input1", "input2", "input3"]
input_1 = tf.keras.Input(shape = input12_shape, name = input_names[0], dtype = tensors_type)
input_2 = tf.keras.Input(shape = input12_shape, name = input_names[1], dtype = tensors_type)
conv1_layer = tf.keras.layers.Conv2D(filters = kernel_filters, 
                                     kernel_size = kernel12_size, 
                                     use_bias = True, 
                                     activation = 'relu',
                                     name = "conv1")(input_1)
conv2_layer = tf.keras.layers.Conv2D(filters = kernel_filters, 
                                     kernel_size = kernel12_size, 
                                     use_bias = True, 
                                     activation = 'relu',
                                     name = "conv2")(input_2)

add1_layer = tf.keras.layers.add([conv1_layer, conv2_layer])

conv3_shape = (kernel3_size, kernel3_size, add1_layer.shape[-1], kernel_filters)

conv3_layer = tf.keras.layers.Conv2D(filters = kernel_filters, 
                                     kernel_size = kernel3_size, 
                                     use_bias = True, 
                                     activation = 'relu',
                                     name = "conv3")(add1_layer)

# input3_shape = conv3_layer.shape[1:]
# input_3 = tf.keras.Input(shape = input3_shape, name = input_names[2], dtype = tensors_type)
# add2_layer = tf.keras.layers.add([conv3_layer, input_3])

flat_layer = tf.keras.layers.Flatten()(conv3_layer)

dense_shape = (flat_layer.shape[-1], dense_units)

dense_layer = tf.keras.layers.Dense(units = dense_units, 
                                    activation = 'softmax', 
                                    use_bias = True, 
                                    name = "dense")(flat_layer)

model = tf.keras.models.Model(inputs = [input_1, input_2], outputs = [dense_layer])
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Generate quantized model
qmodel = tfmot.quantization.keras.quantize_model(model)
qmodel.compile(optimizer = 'adam',
               loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])

N = 3000
x1, y1 = database_generator(N, input12_shape) 
x2, y2 = database_generator(N, input12_shape) 
y = np.array([category_builder(y1[i]|y2[i]) for i in range(N)])

train_log = qmodel.fit(x = [x1, x2], 
                       y = y,
                       batch_size = 200,
                       epochs = 20,
                       validation_split = 0.1,
                       verbose = 1)

# Conversion to TF Lite model
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(qmodel)
tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = tflite_converter.convert()

# Save database
with open(DATABASE_PATH, 'wb') as file:
    np.savez(file = file, 
             x1 = x1, 
             x2 = x2,
             y = y)

# Save TFLite model
with open(TFLITE_OUT_PATH, 'wb') as file:
    file.write(tflite_model)