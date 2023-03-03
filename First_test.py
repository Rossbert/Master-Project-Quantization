import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
# from datetime import datetime
quantize_model = tfmot.quantization.keras.quantize_model

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# print(train_labels.shape)
print(train_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 5, input_shape = (28, 28, 1)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

# log_directory = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%H%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_directory)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()
model.fit(train_images, train_labels, epochs = 5)

q_aware_model = quantize_model(model)

q_aware_model.compile(optimizer = 'adam', 
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
    metrics = ['accuracy'])

q_aware_model.summary()
q_aware_model.fit(train_images, train_labels, epochs = 5)

_, baseline_model_accuracy = model.evaluate(test_images, test_labels)

_, q_aware_model_accuracy = q_aware_model.evaluate(test_images, test_labels)

print("Baseline test accuracy : ", baseline_model_accuracy)
print("Q Aware test accuracy : ", q_aware_model_accuracy)


# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('Test accuracy : ', test_acc)
# prediction = model.predict(test_images)
# print(prediction[0], np.argmax(prediction[0]))
# print("Test code")