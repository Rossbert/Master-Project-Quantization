import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from collections import OrderedDict

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
train_images = train_images / 255.0
test_images = test_images / 255.0

def get_functional_model():
    input_layer = tf.keras.layers.Input(shape = (28, 28, 1))
    conv_1 = tf.keras.layers.Conv2D(32, 5, use_bias = False, activation = 'relu')(input_layer)
    pool_1 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2)(conv_1)
    conv_2 = tf.keras.layers.Conv2D(64, 5, use_bias = False, activation = 'relu')(pool_1)
    pool_2 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2)(conv_2)
    conv_3 = tf.keras.layers.Conv2D(96, 3, use_bias = False, activation = 'relu')(pool_2)
    pool_3 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2)(conv_3)
    flat_1 = tf.keras.layers.Flatten()(pool_3)
    dense_out = tf.keras.layers.Dense(10, activation = 'softmax', name = "dense_last")(flat_1)
    
    model = tf.keras.models.Model(inputs = input_layer, outputs = dense_out)
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    
    model.compile(optimizer = opt, 
        loss = 'sparse_categorical_crossentropy', 
        metrics = ['accuracy'])
    return model

# model = get_functional_model()
# model.summary()


# Load model_v3 Normal functional model
# save_dir = "./logs/"
# save_path = save_dir + "model_v3"
# model : tf.keras.models.Model = tf.keras.models.load_model(save_path)
# loss, acc = model.evaluate(test_images, test_labels)
# print('Test accuracy : ', "{:0.2%}".format(acc))

# Conversion and training into a Q Aware model
# q_aware_model = tfmot.quantization.keras.quantize_model(model)
# q_aware_model.compile(optimizer = 'adam', 
#     loss = 'sparse_categorical_crossentropy', 
#     metrics = ['accuracy'])
# q_aware_model.summary()

# Training of Q Aware model
# train_log = q_aware_model.fit(train_images, train_labels,
#     batch_size = 128,
#     epochs = 1,
#     validation_split = 0.1)

# Load Q Aware Model
save_dir = "./logs/"
save_path = save_dir + "model_q4_func"
q_aware_model : tf.keras.Model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model = tf.keras.models.load_model(save_path)

# Test Accuracy of loaded model
q_aware_test_loss, q_aware_test_acc = q_aware_model.evaluate(test_images, test_labels)
print('Test accuracy : ', "{:0.2%}".format(q_aware_test_acc))

bit_width = 8
quantized_and_dequantized = OrderedDict()
quantized = OrderedDict()
new_quantized_and_dequantized = OrderedDict()
new_quantized = OrderedDict()

idx = 0
for i, layer in enumerate(q_aware_model.layers):
    quantizer : tfmot.quantization.keras.quantizers.Quantizer
    weight : tf.Variable
    if hasattr(layer, '_weight_vars'):
        for weight, quantizer, quantizer_vars in layer._weight_vars:
            idx += 1
            min_var = quantizer_vars['min_var']
            max_var = quantizer_vars['max_var']

            quantized_and_dequantized[layer.name] = quantizer(inputs = weight, training = False, weights = quantizer_vars)
            quantized[layer.name] = np.round(quantized_and_dequantized[layer.name] / max_var * (2**(bit_width-1)-1))

            if "conv2d" in layer.name:
                new_quantized_and_dequantized[layer.name] = tf.quantization.fake_quant_with_min_max_vars_per_channel(weight, min_var, max_var, bit_width, narrow_range = True, name = "New_quantized_" + str(i))
                new_quantized[layer.name] = np.round(new_quantized_and_dequantized[layer.name] / max_var * (2**(bit_width-1)-1))
            elif "dense" in layer.name:
                new_quantized_and_dequantized[layer.name] = tf.quantization.fake_quant_with_min_max_vars(weight, min_var, max_var, bit_width, narrow_range = True, name = "New_quantized_" + str(i))
                new_quantized[layer.name] = np.round(new_quantized_and_dequantized[layer.name] / max_var * (2**(bit_width-1)-1))

for key in quantized:
    print("Fake Quantized")
    print(key)
    if "dense" not in key:
        print(quantized_and_dequantized[key][:,:,0,0])
        # print(quantized[key][:,:,0,0])
    else:
        print(quantized_and_dequantized[key][:,0])
        # print(quantized[key][:,0])

    print("New Fake Quantized")
    print(key)
    if "dense" not in key:
        print(new_quantized_and_dequantized[key][:,:,0,0])
        # print(new_quantized[key][:,:,0,0])
    else:
        print(new_quantized_and_dequantized[key][:,0])
        # print(new_quantized[key][:,0])


def quantize_function(input, min_var, max_var, bits, narrow_range = False):
    # Very important
    if not narrow_range:
        scale = (max_var - min_var) / (2**bits - 1)
    else:
        scale = (max_var - min_var) / (2**bits - 2)
    min_adj = scale * round(min_var / scale)
    max_adj = max_var + min_adj - min_var
    print("Scale : ", scale)
    return scale * np.round(input / scale)

self_quantized = quantize_function(q_aware_model.layers[11].get_weights()[0], min_var.numpy(), max_var.numpy(), bit_width, narrow_range = True)
print("Self quantized : ")
print(self_quantized[:5,:5])
new_quantized = np.round(new_quantized_and_dequantized / max_var * (2**(bit_width - 1) - 1))
print("New quantized")
print(new_quantized[:5,:5])

pos = (41,9)
print(quantized_and_dequantized[pos].numpy())
print(new_quantized_and_dequantized[pos].numpy())
print(self_quantized[pos])


print("Other variables stored in last layer")
print(q_aware_model.layers[11].variables[5:])

print("Quantized model biases")
print(q_aware_model.layers[11].get_weights()[1])

l = 11
print("Elements of layer ", l, "of q aware model")
# print([elem for elem in dir(q_aware_model.layers[l]) if not elem.startswith('_')])
for elem in dir(q_aware_model.layers[l]):
    if not elem.startswith('_'):
        print(elem)
# print([elem for elem in dir(q_aware_model.layers[l]) if elem.startswith('_')])

l = 11
print("Config :")
print(dir(q_aware_model.layers[l].quantize_config))
print(type(q_aware_model.layers[l].quantize_config))
conf : tfmot.quantization.keras.default_8bit = q_aware_model.layers[l].quantize_config
print("Type")
print(q_aware_model.layers[l].quantize_config.get_config())
print(q_aware_model.layers[l].quantize_config.activation_quantizer)
