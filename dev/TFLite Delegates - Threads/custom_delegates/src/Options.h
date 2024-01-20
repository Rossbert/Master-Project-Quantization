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
		OperationMode operation_mode = OperationMode::none;
		int node_index = -1; // useless?
		int bit_position = -1;
		int number_flips = -1;
		int dataset_size = 0;
		std::string layer_name = "";
		
		// Filled during MyDelegateKernel::Init
		std::vector<int> input_size;
		std::vector<int> kernel_size;
		std::vector<int> output_size;

		// Number of threads for all processes
		int num_threads = 4;

		// Number of channels of kernel filter to distribute to make use of threads
		int channels;

		// Size of the chunk of data of indexes to distribute to make use of threads
		int chunk_size;

		/// <summary>
		/// Holds the indexes sectioned according to the real positions
		/// </summary>
		std::vector<std::vector<std::vector<int>>> chunks_indexes;

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