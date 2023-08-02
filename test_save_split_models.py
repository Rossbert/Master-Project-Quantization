import csv
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import Quantization
from keras.engine.functional import Functional

# Number of partitions for batch analysis
N_PARTITIONS = 2
# First index of q_aware_model
SPLIT_INDEX = 3
SEPARATION_MODE = Quantization.SeparationMode.first_quantized_weights

# LOAD_PATH_Q_AWARE = "./model/model_q_aware_final_01"
ID = "ep5_2023-07-02_16-50-58"
LOAD_PATH_Q_AWARE = "./model/model_q_aware_" + ID
SAVE_PATH_PART_2 = "./model/model_2" + ID[-9:]
SAVE_CONV_OUT_NPY = "./model/" + ID[-8:] + "_first_conv_out.npy"
SAVE_LABELS_NPY = "./model/" + ID[-8:] + "_labels.npy"

SAVE_KERNEL_NPY = "./model/" + ID[-8:] + "_kernel.npy"
SAVE_KERNEL_TXT = "./model/" + ID[-8:] + "_kernel.txt"
SAVE_KERNEL_TXT_RAW = "./model/" + ID[-8:] + "_kernel_raw.txt"

SAVE_BIASES_NPY = "./model/" + ID[-8:] + "_biases.npy"
SAVE_BIASES_SCALES_NPY = "./model/" + ID[-8:] + "_biases_scales.npy"
SAVE_OUTPUT_PARAMS_NPY = "./model/" + ID[-8:] + "_output_params.npy"

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load Q Aware model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

q_model_info = Quantization.QuantizedModelInfo(q_aware_model)
idx_conv = q_model_info.layers_indexes.index(SPLIT_INDEX - 1)
key_conv = q_model_info.keys[idx_conv]

# Preparing the quantized test set
quantized_test_images = np.round(test_images[:,:,:,np.newaxis]/q_model_info.output_scales[q_model_info.keys[0]]).astype(int)

# Generating the split models
model_1, model_2 = Quantization.split_model_mixed(q_aware_model, q_model_info, start_index = SPLIT_INDEX, first_part_mode = SEPARATION_MODE)

quantized_conv, test_loss, test_accuracy = Quantization.model_parts_predict_by_batches(
    data_input = quantized_test_images,
    test_labels = test_labels,
    n_partitions = N_PARTITIONS,
    q_model_info = q_model_info,
    model_1 = model_1,
    model_2 = model_2,
    evaluation_mode = Quantization.ModelEvaluationMode.manual_saturation,
    start_index = SPLIT_INDEX
    )

print(f"Model test accuracy: {test_accuracy:.2%}")
print(f"Model test loss: {test_loss:.6f}\n")

output_parameters = np.array([q_model_info.output_scales[key_conv], q_model_info.dequantized_output_max[key_conv], q_model_info.dequantized_output_min[key_conv]])
# Save quantized model
with tfmot.quantization.keras.quantize_scope():
    model_2.save(SAVE_PATH_PART_2)
np.save(SAVE_BIASES_NPY, q_model_info.quantized_biases[key_conv])
np.save(SAVE_BIASES_SCALES_NPY, q_model_info.bias_scales[key_conv])
np.save(SAVE_OUTPUT_PARAMS_NPY, output_parameters)

np.save(SAVE_CONV_OUT_NPY, quantized_conv)
np.save(SAVE_LABELS_NPY, test_labels)

np.save(SAVE_KERNEL_NPY, q_model_info.quantized_weights[key_conv])
with open(SAVE_KERNEL_TXT, 'w', newline = '') as main_file:
    main_file.write(str(q_model_info.quantized_weights[key_conv]))
    main_file.flush()
with open(SAVE_KERNEL_TXT_RAW, 'w', newline = '') as main_file:
    writer = csv.writer(main_file)
    for l in range(q_model_info.quantized_weights[key_conv].shape[3]):
        for k in range(q_model_info.quantized_weights[key_conv].shape[2]):
            for i in range(q_model_info.quantized_weights[key_conv].shape[0]):
                writer.writerow(q_model_info.quantized_weights[key_conv][i,:,k,l])
                main_file.flush()
