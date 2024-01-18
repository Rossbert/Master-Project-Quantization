#pragma once
#include <iostream>
#include <string>
#include <random>

namespace tflite {
	// States of delegate enum class
	// With these states you can state the delegate effects:
	// - No effect
	// - Affect kernel weights
	// - Affect convolution operation
	enum class OperationMode {
		// Nothing is affected
		none,
		// The weights of the kernels are affected
		weights,
		// The multiplication operation during the convolution is affected
		convolution
	};

	// MyDelegateOptions
	// Stores the options to determine the behaviour of the delegate
	struct MyDelegateOptions
	{
		// Operation mode for the delegate options:
		// - Nothing is affected
		// - Weights are affected
		// - Convolution is affected
		OperationMode operation_mode = OperationMode::none;
		
		int node_index = -1; // useless?
		
		// The bit position to be affected
		// For operation mode = weights
		// * bit_positions should be [0, 7] because kernel weights are 8 bits long
		// For operation mode = convolution
		// * bit_positions should be [0, 31] because multiplication accumulator for convolution is 32 bits long
		int bit_position = -1;

		// Number of random flips that will affect the model 
		int number_flips = -1;

		// For MNIST Fashion's complete dataset it is 10000
		int dataset_size = 0;

		// The layer name to be affected
		// It should be precisely the same name that TfLite uses for the layer
		// This is the unique identifier used for filtering the layer
		std::string layer_name = "";
		
		//std::vector<int> errorWeightPositions;
		
		// Filled during MyDelegateKernel::Init
		std::vector<int> input_size;
		std::vector<int> kernel_size;
		std::vector<int> output_size;

		std::vector<int> indexes;

		// Filled during MyDelegateKernel::Init??
		std::vector<std::vector<std::pair<int, int>>> errorPositions;

		// Filled during MyDelegateKernel::Init??
		// They are needed to discriminate the channels for threads
		std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> realPositions;

		// Default constructor
		// Careful, the generator is not seeded by the default constructor
		// Must implement default constructor later for the resizing of errorpositions and realpositions
		MyDelegateOptions() = default;

		// Copy constructor
		MyDelegateOptions(const MyDelegateOptions& options);

		// Constructor with initializer options
		MyDelegateOptions(const OperationMode operation_mode,
			const int node_index, const int bit_position,
			const int number_flips, const int dataset_size, const std::string layer_name);

		// Constructor with char pointer options
		MyDelegateOptions(char** options_keys, char** options_values, size_t num_options);

		void convertPosition(int position, int max_size, const std::vector<int>& elements, std::vector<int>& values);

		// Adapt this function for the other code
		// Custom pair sorting
		bool pair_greater(const std::pair<int, int>& pair1, const std::pair<int, int>& pair2);

		// Adapt this function for the other code
		// Custom pair of vector positions sorting
		bool pair_vectors_greater(const std::pair<std::vector<int>, std::vector<int>>& pair1, const std::pair<std::vector<int>, std::vector<int>>& pair2, const std::vector<int>& out_elements);

		// Logger function of MyDelegateOptions
		void Log() const;
	};
}