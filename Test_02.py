import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load model_v2
save_dir = "./logs/"
save_path = save_dir + "model_v2"
model : tf.keras.models.Model = tf.keras.models.load_model(save_path)
loss, acc = model.evaluate(test_images, test_labels)
print('Test accuracy : ', "{:0.2%}".format(acc))

q_aware_model = tfmot.quantization.keras.quantize_model(model)

q_aware_model.compile(optimizer = 'adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics = ['accuracy'])
q_aware_model.summary()

train_log = q_aware_model.fit(train_images, train_labels,
    batch_size = 128,
    # epochs = 15,
    epochs = 1,
    validation_split = 0.1)

q_aware_test_loss, q_aware_test_acc = q_aware_model.evaluate(test_images, test_labels)
print('Test accuracy : ', "{:0.2%}".format(q_aware_test_acc))

for i, layer in enumerate(q_aware_model.layers):
    for elem in layer.__dict__:
        if elem == '_output_quantizer_vars':
            print(i, layer._output_quantizer_vars)
        if elem == '_weight_vars':
            print(i, layer._weight_vars)

bit_width = 8
quantizer :  tfmot.quantization.keras.quantizers.LastValueQuantizer
for i, layer in enumerate(q_aware_model.layers):
    if hasattr(layer, '_weight_vars'):
        for weight, quantizer, quantizer_vars in layer._weight_vars:
            quantized_and_dequantized = quantizer(weight, training = False, weights = quantizer_vars)
            min_var = quantizer_vars['min_var']
            max_var = quantizer_vars['max_var']
            new_quantized_and_dequantized = tf.quantization.fake_quant_with_min_max_vars(weight, min_var, max_var, bit_width, narrow_range = True, name = "New_quantized")
            quantized = np.round(quantized_and_dequantized / max_var * (2**(bit_width-1)-1))
print("Quantized model weights")
print(q_aware_model.layers[11].get_weights()[0][:5,:5])

print("Manual Quantized values")
print(quantized[:5,:5])

print("New fake quantized")
print(new_quantized_and_dequantized[:5,:5])


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