import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.engine.functional import Functional

""" Paths:
- VHDL_CONV
- VHDL_LABELS
Should contain the locations of the VHDL convolution results and labels.
Should be in the .npy format to be successfully acquired.
Manual saturation can be activated.
"""

manual_saturation : bool = False
ID = "16-50-58"
# Convolution results and labels location
# Replace with the path of the VHDL results
############################################
VHDL_CONV = "./model/" + ID + "_first_conv_out.npy"
VHDL_LABELS = "./model/" + ID + "_labels.npy"
############################################
# Models and important values location
LOAD_PATH_PART_2 = "./model/model_2_" + ID
LOAD_BIASES_NPY = "./model/" + ID + "_biases.npy"
LOAD_BIASES_SCALES_NPY = "./model/" + ID + "_biases_scales.npy"
LOAD_OUTPUT_PARAMS_NPY = "./model/" + ID + "_output_params.npy"

quant_conv : npt.NDArray[np.int32] = np.load(VHDL_CONV)
labels : npt.NDArray[np.uint8] = np.load(VHDL_LABELS)

if not labels.shape:
    labels = np.array([labels])

with tfmot.quantization.keras.quantize_scope():
    model_2 : Functional = tf.keras.models.load_model(LOAD_PATH_PART_2)

try:
    assert model_2.input_shape[1:] == quant_conv.shape[1:]
except AssertionError:
    if quant_conv.shape == model_2.input_shape[1:]:
        quant_conv = quant_conv[np.newaxis,:,:,:]
    else:
        print(f"Convolution shape should be {('Any number',)+ model_2.input_shape[1:]}... Exiting program")
        exit()

try:
    assert quant_conv.shape[0] == labels.shape[0] 
except AssertionError:
    print(f"Number of labels should coincide with convolution results... Exiting program")
    exit()

biases : npt.NDArray[np.int32] = np.load(LOAD_BIASES_NPY)
biases_scales : npt.NDArray[np.float64] = np.load(LOAD_BIASES_SCALES_NPY)
output_params : npt.NDArray[np.float64] = np.load(LOAD_OUTPUT_PARAMS_NPY)

quant_conv_bias : npt.NDArray[np.int32] = tf.nn.bias_add(quant_conv, biases).numpy().astype(int)
quant_postactiv : npt.NDArray[np.int32] = tf.nn.relu(quant_conv_bias).numpy()

dequant_postactiv : npt.NDArray[np.float32] = (biases_scales * quant_postactiv).astype(np.float32)
quant_rescaled_postactiv : npt.NDArray[np.int32] = np.round(dequant_postactiv/output_params[0]).astype(np.int32)
rescaled_postactiv : npt.NDArray[np.float32] = (output_params[0] * quant_rescaled_postactiv).astype(np.float32)

if manual_saturation:
    rescaled_postactiv[rescaled_postactiv >= output_params[1]] = output_params[1]
    rescaled_postactiv[rescaled_postactiv <= output_params[2]] = output_params[2]

loss, accuracy = model_2.evaluate(rescaled_postactiv, labels, verbose = 0)

print(f"Model test accuracy: {accuracy:.2%}")
print(f"Model test loss: {loss:.6f}\n")