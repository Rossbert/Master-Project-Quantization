import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

def evaluate_2input_model(interpreter: tf.lite.Interpreter, inputs: list) -> float:
    """ Evaluate TFLite Model:
    -
    Receives the interpreter and returns addition of inputs
    """
    idx_input1 = interpreter.get_input_details()[0]["index"]
    idx_input2 = interpreter.get_input_details()[1]["index"]
    idx_output = interpreter.get_output_details()[0]["index"]
    
    interpreter.set_tensor(idx_input1, inputs[0])
    interpreter.set_tensor(idx_input2, inputs[1])

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest probability.
    output = interpreter.tensor(idx_output)
    return output()

# Load paths
TFLITE_PATH = "./model/tflite_ep5_2023-07-02_16-50-58.tflite"
TFLITE_OUT_PATH = "./model/add_test.tflite"
DELEGATE_PATH = "./dependencies/custom_delegates.dll"
# print(sys.byteorder)

tensors_type = np.float32
IN_MAX = 1
IN_MIN = 0
CONV_MAX = 2.5 # Experimentally extracted by making tests with model, if weights change this value changes

input_shape = (4, 5, 3)
kernel1_size = 3
kernel2_size = 2
kernel_filters = 4
dense_units = 3

N = 1000
np.random.seed(10)
fake_dataset_x1 = np.random.randint(low = 0, high = 255, size = (N, *input_shape))/np.float32(255.0)
np.random.seed(20)
fake_dataset_x2 = np.random.randint(low = 0, high = 255, size = (N, *input_shape))/np.float32(255.0)
np.random.seed(30)
fake_dataset_y = np.random.randint(low = 0, high = 2 + 1, size = N)

conv1_shift = -40
conv2_shift = -20
conv3_shift = -20
dense_shift = -13
conv12_shape = (kernel1_size, kernel1_size, input_shape[-1], kernel_filters)

conv1_layer_kernel = np.arange(1 + conv1_shift, kernel1_size**2*input_shape[-1]*kernel_filters + 1 + conv1_shift, 1, dtype = tensors_type).reshape(conv12_shape)/100
conv1_layer_biases =  np.arange(1, kernel_filters + 1, 1, dtype = tensors_type)/50

conv2_layer_kernel = np.arange(kernel1_size**2*input_shape[-1]*kernel_filters + conv2_shift, 1 - 1 + conv2_shift, -1, dtype = tensors_type).reshape(conv12_shape)/100
conv2_layer_biases =  np.arange(kernel_filters, 1 - 1, -1, dtype = tensors_type)/50

input_1 = tf.keras.Input(shape = input_shape, name = "input1")
input_2 = tf.keras.Input(shape = input_shape, name = "input2")
conv1_layer = tf.keras.layers.Conv2D(filters = kernel_filters, 
                                     kernel_size = kernel1_size, 
                                     use_bias = True, 
                                     kernel_initializer = tf.constant_initializer(conv1_layer_kernel),
                                     bias_initializer = tf.constant_initializer(conv1_layer_biases),
                                     activation = 'relu',
                                     name = "conv1")(input_1)
conv2_layer = tf.keras.layers.Conv2D(filters = kernel_filters, 
                                     kernel_size = kernel1_size, 
                                     use_bias = True, 
                                     kernel_initializer = tf.constant_initializer(conv2_layer_kernel),
                                     bias_initializer = tf.constant_initializer(conv2_layer_biases),
                                     activation = 'relu',
                                     name = "conv2")(input_2)

add_layer = tf.keras.layers.add([conv1_layer, conv2_layer])

conv3_shape = (kernel2_size, kernel2_size, add_layer.shape[-1], kernel_filters)
conv3_layer_kernel = np.arange(kernel2_size**2*add_layer.shape[-1]*kernel_filters + conv3_shift, 1 - 1 + conv3_shift, -1, dtype = tensors_type).reshape(conv3_shape)/200
conv3_layer_biases =  np.arange(1, kernel_filters + 1, 1, dtype = tensors_type)/100

conv3_layer = tf.keras.layers.Conv2D(filters = kernel_filters, 
                                     kernel_size = kernel2_size, 
                                     use_bias = True, 
                                     kernel_initializer = tf.constant_initializer(conv3_layer_kernel),
                                     bias_initializer = tf.constant_initializer(conv3_layer_biases),
                                     activation = 'relu',
                                     name = "conv3")(add_layer)

# input_3 = tf.keras.Input(shape = input_shape, tensor = tf.constant(np.ones((1, *input_shape), dtype = tensors_type)), name = "input3")
# sub_layer = tf.keras.layers.subtract([conv3_layer, input_3])

flat_layer = tf.keras.layers.Flatten()(conv3_layer)

dense_shape = (flat_layer.shape[-1], dense_units)
dense_kernel = np.arange(1 + dense_shift, dense_units*flat_layer.shape[1] + 1 + dense_shift, dtype = tensors_type).reshape(dense_shape)
dense_biases = np.arange(dense_units, 1 - 1, -1, dtype = tensors_type)

dense_layer = tf.keras.layers.Dense(units = dense_units, 
                                    activation = 'softmax', 
                                    use_bias = True, 
                                    kernel_initializer = tf.constant_initializer(dense_kernel), 
                                    bias_initializer = tf.constant_initializer(dense_biases),
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

train_log = qmodel.fit(x = [fake_dataset_x1, fake_dataset_x2], y = fake_dataset_y,
                       batch_size = 500,
                       epochs = 5,
                       validation_split = 0.1,
                       verbose = 0)

# Conversion to TF Lite model
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(qmodel)
tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = tflite_converter.convert()

# Save TFLite model
with open(TFLITE_OUT_PATH, 'wb') as f:
    f.write(tflite_model)

# Interpreter creation
interpreter = tf.lite.Interpreter(model_content = tflite_model)

# Details of interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input 1 Name: ", input_details[0]['name'])
print("Input 1 Shape: ", input_details[0]['shape'])
print("Input 1 Dtype: ", input_details[0]['dtype'])
print("Input 2 Name: ", input_details[1]['name'])
print("Input 2 Shape: ", input_details[1]['shape'])
print("Input 2 Dtype: ", input_details[1]['dtype'])
# print("Input 3 Name: ", input_details[2]['name'])
# print("Input 3 Shape: ", input_details[2]['shape'])
# print("Input 3 Dtype: ", input_details[2]['dtype'])
print("Output Shape: ", output_details[0]['shape'])

interpreter.allocate_tensors()
inputs = np.ones((2, 1, *input_shape), dtype = tensors_type)
inputs[0] = 0.50*inputs[0]
inputs[1] = 0.25*inputs[1]
output = evaluate_2input_model(interpreter, inputs)
print(output)

# Using custom_delegate to add tensors
delegate = tf.lite.experimental.load_delegate(DELEGATE_PATH)
# new_interpreter = tf.lite.Interpreter(
#     model_path = TFLITE_PATH, experimental_delegates = [delegate])
new_interpreter = tf.lite.Interpreter(
    model_content = tflite_model, experimental_delegates = [delegate])

# Details of new_interpreter
input_details = new_interpreter.get_input_details()
output_details = new_interpreter.get_output_details()
print("Input 1 Shape: ", input_details[0]['shape'])
print("Input 1 Dtype: ", input_details[0]['dtype'])
print("Input 2 Shape: ", input_details[1]['shape'])
print("Input 2 Dtype: ", input_details[1]['dtype'])
# print("Input 3 Shape: ", input_details[2]['shape'])
# print("Input 3 Dtype: ", input_details[2]['dtype'])
print("Output Shape: ", output_details[0]['shape'])

new_interpreter.allocate_tensors()
output = evaluate_2input_model(new_interpreter, inputs)
print(output)

