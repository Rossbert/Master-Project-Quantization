#pragma once
#include <iostream>
#include <string>

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
		int node_index = -1;
		int bit_position = -1;
		int number_flips = -1;
		std::string layer_name = "";

		// Default constructor
		MyDelegateOptions() = default;

		// Constructor with initializer options
		MyDelegateOptions(const OperationMode operation_mode,
			const int node_inde, const int bit_position,
			const int number_flips, const std::string layer_name);

		// Constructor with options
		MyDelegateOptions(char** options_keys, char** options_values, size_t num_options);

		// Logger function of MyDelegateOptions
		void Log() const;
	};
}