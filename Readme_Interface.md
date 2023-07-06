# Interface functions Readme.md

Main file to test:
* test_interface_vhdl.py

Pre-trained models and values needed:
./model/model_2_16-50-58
./model/16-50-58_biases.npy
./model/16-50-58_biases_scales.npy
./model/16-50-58_output_params.npy

# Convolution results and labels location
# Locate your VHDL results as the following files
Names of inputs that should be generated in VHDL
./model/16-50-58_first_conv_out.npy
./model/16-50-58_labels.npy

Running the script will yield the accuracy and loss