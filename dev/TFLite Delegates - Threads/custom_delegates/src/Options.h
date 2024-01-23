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
		none,
		weights,
		convolution
	};

	// MyDelegateOptions
	// Stores the options to determine the behaviour of the delegate
	struct MyDelegateOptions
	{
		// Operation mode:
		//	- None: convolution runs normally
		//	- Kernel weights: kernel weights are affected
		//	- Convolution multiplication: convolution multiplication is affected
		OperationMode operation_mode = OperationMode::none;

		// Node index variable unused so far... delete later
		int node_index = -1; // useless?

		// Bit position to be flipped
		int bit_position = -1;

		// Number of flips per image in the dataset
		int number_flips = -1;
		
		// Size of the dataset
		int dataset_size = 0;

		// Number of threads for all processes
		int num_threads = 4;

		// Number of channels of kernel filter to distribute to make use of threads
		int channels = 0;

		// Size of the chunk of data of indexes to distribute to make use of threads
		int chunk_size = 0;

		// Name pattern of the layer to be affected
		std::string layer_name = "";
		
		// Position vector values of the input tensor
		// Filled during MyDelegateKernel::Init
		std::vector<int> input_size;

		// Position vector values of the kernel tensor
		// Filled during MyDelegateKernel::Init
		std::vector<int> kernel_size;

		// Position vector values of the output tensor
		// Filled during MyDelegateKernel::Init
		std::vector<int> output_size;

		// Holds the indexes sectioned according to the real positions
		std::vector<std::vector<std::vector<int>>> chunks_indexes;
		
		// Indexes for non-parallel solution
		std::vector<int> full_indexes;

		// Convert to vector to accept more than one node
		std::vector<std::vector<std::pair<int, int>>> errorPositions;

		// Convert to vector to accept more than one node
		// They are needed to discriminate the channels for threads
		std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> realPositions;

		// Default constructor
		// Careful, the generator is not seeded by the default constructor
		// Must implement default constructor later for the resizing of errorpositions and realpositions
		MyDelegateOptions() = default;

		/// <summary>
		/// Copy constructor
		/// </summary>
		/// <param name="options"></param>
		MyDelegateOptions(const MyDelegateOptions& options);

		/// <summary>
		/// Constructor with initializer options
		/// </summary>
		/// <param name="operation_mode"></param>
		/// <param name="node_index"></param>
		/// <param name="bit_position"></param>
		/// <param name="number_flips"></param>
		/// <param name="dataset_size"></param>
		/// <param name="channels"></param>
		/// <param name="chunk_size"></param>
		/// <param name="layer_name"></param>
		MyDelegateOptions(const OperationMode operation_mode,
			const int node_index, const int bit_position,
			const int number_flips, const int dataset_size, const int channels, const int chunk_size, const std::string layer_name);

		// Constructor with char pointer options
		MyDelegateOptions(char** options_keys, char** options_values, size_t num_options);

		// Convert integer position to vector position
		void convertPosition(int position, int max_size, const std::vector<int>& elements, std::vector<int>& values);

		// Custom pair sorting
		bool pair_greater(const std::pair<int, int>& pair1, const std::pair<int, int>& pair2);

		// Custom pair of vector positions sorting
		bool pair_vectors_greater(const std::pair<std::vector<int>, std::vector<int>>& pair1, const std::pair<std::vector<int>, std::vector<int>>& pair2, const std::vector<int>& out_elements);

		// Logger function of MyDelegateOptions
		void Log() const;
	};
}