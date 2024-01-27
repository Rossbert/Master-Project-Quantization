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
		// Overseer to check if same session has called 
		static bool new_call;

		// Maximum number of threads to avoid bottleneck
		// This number was obtained experimentally
		constexpr static int max_number_threads = 8;

		// Maximum number of operations to be performed by a thread
		// This number was obtained experimentally
		constexpr static int max_operations_per_thread = 100000;

		// Controls the index of the dataset image beig evaluated
		int dataset_index = 0;

		// Operation mode:
		//	- None: convolution runs normally
		//	- Kernel weights: kernel weights are affected
		//	- Convolution multiplication: convolution multiplication is affected
		OperationMode operation_mode = OperationMode::none;

		// Bit position to be flipped
		int bit_position = -1;

		// Number of flips per image in the dataset
		int number_flips = -1;
		
		// Size of the dataset
		int dataset_size = 0;

		// Convert into vector if accepting more than one node
		int node_index = -1;

		// Convert into vector if accepting more than one node
		int builtin_code = 0;

		// Convert to vector for more than one node
		// Number of channels of kernel filter to distribute to make use of threads
		int channels = 0;

		// Convert to vector for more than one node
		// Size of the chunk of data of indexes to distribute to make use of threads
		int chunk_size = 0;

		// Number of threads for all processes
		int num_threads = max_number_threads;

		// Threaded version necessary?
		bool is_threaded = false;

		// Convert to vector for more than one node
		// Name pattern of the layer to be affected
		// If accepting more than one node this logic need to be modified
		std::string layer_name = "";
		
		// Position vector values of the input tensor
		// Filled during MyDelegateKernel::Init
		std::vector<int> input_dimensions;

		// Position vector values of the kernel tensor
		// Filled during MyDelegateKernel::Init
		std::vector<int> kernel_dimensions;

		// Position vector values of the output tensor
		// Filled during MyDelegateKernel::Init
		std::vector<int> output_dimensions;

		// Holds the indexes sectioned according to the real positions
		std::vector<std::vector<std::vector<int>>> chunks_indexes;
		
		// Indexes for non-parallel solution
		std::vector<int> full_indexes;

		// Convert to vector to accept more than one node
		// Dimensions:
		//	- dataset_size x num_flips x pairs of int positions
		std::vector<std::vector<std::pair<int, int>>> error_flat_positions;

		// Convert to vector to accept more than one node.
		// They are needed to discriminate the channels for threads.
		// Dimensions:
		//	- dataset_size x num_flips x pairs of vector positions
		std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> error_vec_positions;

		// Default constructor
		// Careful, the generator is not seeded by the default constructor
		// Must implement default constructor later for the resizing of errorpositions and realpositions
		MyDelegateOptions() = default;

		/// <summary>
		/// Copy constructor<para/>
		/// 1st Called from the initialization list of MyDelegate constructor<para/>
		///	2nd Called from the initialization list of MyDelegateKernel constructor
		/// </summary>
		/// <param name="options">: Constant reference to an existing MyDelegateOptions instance</param>
		MyDelegateOptions(const MyDelegateOptions& options);

		/// <summary>
		/// Constructor with char pointers as input keys<para/>
		///	&#009; - It is the first constructor called from the entry point
		/// </summary>
		/// <param name="options_keys">: Field names</param>
		/// <param name="options_values">: Field values</param>
		/// <param name="num_options">: Total number of parameters</param>
		MyDelegateOptions(char** options_keys, char** options_values, size_t num_options);

		// Convert integer position to vector position
		void convertPositionInt2Vec(int position, int max_size, const std::vector<int>& tensor_dimensions, std::vector<int>& vec_position);

		// Custom pair sorting
		bool getPairIntGreater(const std::pair<int, int>& pair1, const std::pair<int, int>& pair2);

		// Convert vector position to integer position
		int convertPositionVec2Int(const std::vector<int>& output_dimensions, const std::vector<int>& vec_position);

		// Custom pair of vector positions sorting
		bool getPairVecGreater(const std::pair<std::vector<int>, std::vector<int>>& pair1, const std::pair<std::vector<int>, std::vector<int>>& pair2, const std::vector<int>& first_dimensions, const std::vector<int>& second_dimensions);

		// Logger function of MyDelegateOptions
		void Log() const;
	};
}