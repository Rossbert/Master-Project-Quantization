import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import Quantization
from keras.src.engine.functional import Functional
from typing import Tuple

def evaluate_model(interpreter: tf.lite.Interpreter) -> Tuple[float, float]:
    """ Evaluate TFLite Model:
    -
    Receives the interpreter and returns a tuple of loss and accuracy.
    """
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    print(interpreter.get_input_details())

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    predictions = []
    for i, test_image in enumerate(test_images):
        # Pre-processing: add batch dimension and convert to float32 to match with the model's input data format.
        test_image = np.expand_dims(test_image, axis = 0).astype(np.float32)
        test_image = np.expand_dims(test_image, axis = 3).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        predictions.append(np.copy(output()[0]))
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    predictions = np.array(predictions)
    scce = tf.keras.losses.SparseCategoricalCrossentropy()(test_labels, predictions)

    loss = scce.numpy()
    accuracy = (prediction_digits == test_labels).mean()
    return loss, accuracy

LOAD_PATH_Q_AWARE = "./model/model_q_aware_ep5_2023-07-02_16-50-58"
SAVE_TFLITE_PATH = "./model/tflite"+ LOAD_PATH_Q_AWARE[-24:] + ".tflite"

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load Q Aware model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

# Evaluate accuracy of both models in test set
q_aware_test_loss, q_aware_test_acc = q_aware_model.evaluate(test_images, test_labels)
print(f"Q Aware model test accuracy : {q_aware_test_acc:.2%}")
print(f"Q Aware model test loss: {q_aware_test_loss:.6f}")

q_model_info = Quantization.QuantizedModelInfo(q_aware_model)

# Conversion to TF Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content = quantized_tflite_model)
interpreter.allocate_tensors()
tflite_loss, tflite_accuracy = evaluate_model(interpreter)
print('TFLite model test accuracy:', "{:0.2%}".format(tflite_accuracy))
print('TFLite model test loss: ', tflite_loss)

# Save TFLite model
with open(SAVE_TFLITE_PATH, 'wb') as output_file:
    output_file.write(quantized_tflite_model)