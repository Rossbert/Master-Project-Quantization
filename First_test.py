import numpy as np
import tensorflow as tf
from datetime import datetime

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
model.fit(train_images, train_labels,
    epochs = 5)
#     batch_size = 5, epochs = 3, 
#     callbacks = [tensorboard_callback])

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy : ', test_acc)

prediction = model.predict(test_images)

print(prediction[0], np.argmax(prediction[0]))