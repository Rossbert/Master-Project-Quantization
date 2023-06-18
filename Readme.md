# Bienvenido to my first readme file

Main file to test:
* QSplit.py

Packages needed:
* Quantization.py

Models needed:
./model/model_q_aware_final_01

Test to affect convolution operation on first layer. 

* All the 10000 outputs on the test set will be affected
* The output channels are generated randomly
* The positions are also generated randomly
* The program iterates through all the bit positions for each 32-bit convolution result.

Parameters to be tuned:
- Number of simulations = repetitions.
- Limit of number of flips in total.
- Bit step that will be flipped in the 32 bit element.
- Operation mode:
    * 0 = 2nd part quantized: The second part will operate with an input-quantizing-layer with floating point weights.
    * 1 = 2nd part no input quantized: the second part will operate with floating point weights without an input-quantizing-layer.
    * 2 = 2nd part manual saturation: the second part will operate with floating point weights but their values are previously manually saturated.
    * 3 = 2nd part multichannel relu: applying an integer manual multichannel relu activation function.

# IMPORTANT
- The operation mode to test the new relu operation with integer saturation limit values is OPERATION_MODE = 3
- With OPERATION_MODE != 3, N_FLIPS_LIMIT = 4, BIT_STEPS_PROB = 1 each simulation takes around 7.5 minutes.
- With OPERATION_MODE = 3, N_FLIPS_LIMIT = 4, BIT_STEPS_PROB = 1 each simulation takes around 11 minutes.