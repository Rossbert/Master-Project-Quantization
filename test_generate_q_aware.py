import datetime
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.engine.functional import Functional
import Quantization

EPOCHS = 5
LOAD_PATH = "./model/model_final_01"
SAVE_PATH_Q_AWARE = f"./model/model_q_aware_ep{EPOCHS}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load model
model : Functional = tf.keras.models.load_model(LOAD_PATH)
# Quantize model
q_aware_model = tfmot.quantization.keras.quantize_model(model)
q_aware_model.compile(optimizer = 'adam', 
                        loss = 'sparse_categorical_crossentropy', 
                        metrics = ['accuracy'])
train_log = q_aware_model.fit(train_images, train_labels,
                                batch_size = 128,
                                epochs = EPOCHS,
                                validation_split = 0.1)
q_aware_test_loss, q_aware_test_acc = q_aware_model.evaluate(test_images, test_labels)
print(f"Q Aware test accuracy : {q_aware_test_acc:.2%}")
print(f"Q Aware model test loss: {q_aware_test_loss:.6f}")

q_model_info = Quantization.QuantizedModelInfo(q_aware_model)

# Save quantized model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model.save(SAVE_PATH_Q_AWARE)
Quantization.garbage_collection()
