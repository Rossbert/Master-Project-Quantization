import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

def get_functional_model(learning_rate : float) -> tf.keras.Model:
    """ Get Functional Model:
    -
    Returns a functional model with 3 convolutional layers and 1 densely connected layer
    """
    input_layer = tf.keras.layers.Input(shape = (28, 28, 1))
    conv_1 = tf.keras.layers.Conv2D(32, 5, use_bias = True, activation = 'relu')(input_layer)
    pool_1 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2)(conv_1)
    conv_2 = tf.keras.layers.Conv2D(64, 5, use_bias = True, activation = 'relu')(pool_1)
    pool_2 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2)(conv_2)
    conv_3 = tf.keras.layers.Conv2D(96, 3, use_bias = True, activation = 'relu')(pool_2)
    pool_3 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2)(conv_3)
    flat_1 = tf.keras.layers.Flatten()(pool_3)
    dense_out = tf.keras.layers.Dense(10, activation = 'softmax', name = "dense_last")(flat_1)
    
    model = tf.keras.models.Model(inputs = input_layer, outputs = dense_out)
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    
    model.compile(optimizer = opt, 
                    loss = 'sparse_categorical_crossentropy', 
                    metrics = ['accuracy'])
    return model

def evaluate_model(interpreter: tf.lite.Interpreter) -> float:
    """ Evaluate TFLite Model:
    -
    Receives the interpreter and returns accuracy.
    """
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for i, test_image in enumerate(test_images):
        if i % 1000 == 0:
            print('Evaluated on {n} results so far.'.format(n=i))
        # Pre-processing: add batch dimension and convert to float32 to match with the model's input data format.
        test_image = np.expand_dims(test_image, axis = 0).astype(np.float32)
        test_image = np.expand_dims(test_image, axis = 3).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()
    return accuracy

SAVE_PATH_MODEL =  "./model/" + "model_final_02"
SAVE_PATH_Q_AWARE = "./model/" + "model_q_aware_final_02"
TFLITE_MODEL_PATH = "./model/" + 'tflite_final_02.tflite'

# Load data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Train model
learning_rate = 4.0e-4 # best results .0004
epochs = 10
batch_size = 10 
model = get_functional_model(learning_rate)
model.summary()
train_log : tf.keras.callbacks.History
train_log = model.fit(train_images, train_labels,
                        batch_size = batch_size,
                        epochs = epochs,
                        validation_split = 0.1)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy : ', "{:0.2%}".format(test_acc))

# Quantize model
q_aware_model = tfmot.quantization.keras.quantize_model(model)
q_aware_model.compile(optimizer = 'adam', 
                        loss = 'sparse_categorical_crossentropy', 
                        metrics = ['accuracy'])
q_aware_model.summary()
train_log = q_aware_model.fit(train_images, train_labels,
                                batch_size = 128,
                                epochs = 1,
                                validation_split = 0.1)
q_aware_test_loss, q_aware_test_acc = q_aware_model.evaluate(test_images, test_labels)
print('Q Aware test accuracy : ', "{:0.2%}".format(q_aware_test_acc))

# Conversion to TF Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content = quantized_tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input Shape: ", input_details[0]['shape'])
print("Output Shape: ", output_details[0]['shape'])
tflite_accuracy = evaluate_model(interpreter)
print('TFLite test accuracy:', "{:0.2%}".format(tflite_accuracy))

# Save model
model.save(SAVE_PATH_MODEL)
# Save quantized model
q_aware_model.save(SAVE_PATH_Q_AWARE)
# Save TFLite model
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(quantized_tflite_model)