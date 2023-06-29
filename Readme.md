# Bienvenido to my first readme file

Main file to test:
* QSplit.py

User-defined packages needed:
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
- Limit of number of flips in total (recommended 4).
- Bit step that will be flipped in the 32 bit element (if it is 1, the simulation will iterate through all bit positions of the 32 bit convolution result).
- Operation mode:
    * 1 = 2nd part no input saturation: the second part will operate with floating point weights without an input-quantizing-layer.
    * 2 = 2nd part manual saturation: the second part will operate with floating point weights but their values are previously manually saturated.
    * 3 = 2nd part multichannel relu: applying an integer manual multichannel relu activation function.

# IMPORTANT
- The operation mode to test the new relu operation with integer saturation limit values is OPERATION_MODE = 3
- With OPERATION_MODE != 3, N_FLIPS_LIMIT = 4, BIT_STEPS_PROB = 1 each simulation takes around 7.5 minutes.
- With OPERATION_MODE = 3, N_FLIPS_LIMIT = 4, BIT_STEPS_PROB = 1 each simulation takes around 11 minutes.

# Important functions descriptions
Quantization.split_model_mixed(
    q_aware_model : Functional | tf.keras.Model, 
    q_model_info : QuantizedModelInfo, 
    start_index : int = 3, 
    first_quantized : bool = False) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """ Divides the model in 2
        - First part model with either quantized or non-quantized weights
        - Second part model with dequantized weights (floating point)
        - Assumes selected start_index is always an index next to one of a convolutional layer index!
        """

Quantization.prediction_by_batches(data_input : npt.NDArray[np.int32], 
test_labels : npt.NDArray[np.uint8],
n_partitions : int, 
q_model_info : QuantizedModelInfo,
model_1 : tf.keras.Model | None = None,
model_2 : tf.keras.Model | None = None,
evaluation_mode : ModelEvaluationMode = ModelEvaluationMode.manual_saturation,
start_index : int = 3) -> Tuple[npt.NDArray[np.int32], float, float]:
    """ Evaluates both parts of the model and deletes memory unused
    - Different behaviour depending if you selected different evaluation_mode:
        * manual_saturation : Implements a manual floating-point saturation to output of model 1 before evaluating the second model
        * no_input_saturation : Feeds the output of model 1 directly to model 2
        * multi_relu : Applies an integer manual saturation to the output of model 1 before feeding it to model 2
        * m2_quantized : Not implemented 
    - Manual garbage collection functions are called multiple times
    """