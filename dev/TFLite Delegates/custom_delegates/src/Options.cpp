#include "Options.h"

namespace tflite {

	// MyDelegateOptionsMethods

	MyDelegateOptions::MyDelegateOptions(const OperationMode operation_mode, const int node_index,
		const int bit_position, const int number_flips, const std::string layer_name)
		: operation_mode(operation_mode),
		node_index(node_index),
		bit_position(bit_position),
		number_flips(number_flips),
		layer_name(layer_name)
	{

	}
	MyDelegateOptions::MyDelegateOptions(char** options_keys, char** options_values, size_t num_options)
	{
		for (int i = 0; i < num_options; i++)
		{
			if (options_keys != nullptr && options_values != nullptr)
			{
				if (strcmp(*(options_keys + i), "operation_mode") == 0)
				{
					operation_mode = (OperationMode)std::stoi(*(options_values + i));
				}
				else if (strcmp(*(options_keys + i), "node_index") == 0)
				{
					node_index = std::stoi(*(options_values + i));
				}
				else if (strcmp(*(options_keys + i), "bit_position") == 0)
				{
					bit_position = std::stoi(*(options_values + i));
				}
				else if (strcmp(*(options_keys + i), "number_flips") == 0)
				{
					number_flips = std::stoi(*(options_values + i));
				}
				else if (strcmp(*(options_keys + i), "layer_name") == 0)
				{
					layer_name = std::string(*(options_values + i));
				}
				else
				{
					std::cout << "Warning: unmatched key : " << *(options_keys + i) << " = " << *(options_values + i) << std::endl;
				}
			}
		}
	}
	void MyDelegateOptions::Log() const
	{
		std::cout << "layer name = " << layer_name << std::endl;
		switch (operation_mode)
		{
		case tflite::OperationMode::none:
			std::cout << "operation mode = none" << std::endl;
			break;
		case tflite::OperationMode::weights:
			std::cout << "operation mode = weights" << std::endl;
			break;
		case tflite::OperationMode::convolution:
			std::cout << "operation mode = convolution" << std::endl;
			break;
		default:
			std::cout << "operation mode = unknown" << std::endl;
			break;
		}
		std::cout << "node index = " << node_index << std::endl;
		std::cout << "bit position = " << bit_position << std::endl;
	}

}