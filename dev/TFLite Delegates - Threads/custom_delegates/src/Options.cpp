#include "Options.h"

namespace tflite {

	// MyDelegateOptionsMethods

	MyDelegateOptions::MyDelegateOptions(const MyDelegateOptions& options)
		: operation_mode(options.operation_mode),
		node_index(options.node_index),
		bit_position(options.bit_position),
		number_flips(options.number_flips),
		dataset_size(options.dataset_size),
		layer_name(options.layer_name)
	{
		// Copy constructor
		errorPositions.resize(dataset_size);
		realPositions.resize(dataset_size);
		chunks_indexes.resize(dataset_size);
	}
	MyDelegateOptions::MyDelegateOptions(const OperationMode operation_mode, const int node_index,
		const int bit_position, const int number_flips, const int dataset_size, const std::string layer_name)
		: operation_mode(operation_mode),
		node_index(node_index),
		bit_position(bit_position),
		number_flips(number_flips),
		dataset_size(dataset_size),
		layer_name(layer_name)
	{
		// Creates new seed for the generator
		errorPositions.resize(dataset_size);
		realPositions.resize(dataset_size);
		chunks_indexes.resize(dataset_size);
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
				else if (strcmp(*(options_keys + i), "dataset_size") == 0)
				{
					dataset_size = std::stoi(*(options_values + i));
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

		errorPositions.resize(dataset_size);
		realPositions.resize(dataset_size);
		chunks_indexes.resize(dataset_size);
	}
	
	void MyDelegateOptions::convertPosition(int position, int max_size, const std::vector<int>& tensor_size, std::vector<int>& position_values)
	{
		int acc = position;
		int div = max_size;
		for (int dimension : tensor_size)
		{
			div /= dimension;
			position_values.emplace_back(acc / div);
			acc -= position_values.back() * div;
		}
	}

	bool MyDelegateOptions::pair_greater(const std::pair<int, int>& pair1, const std::pair<int, int>& pair2)
	{
		// Sort in decreasing order based on the first element of the pair
		return pair1.first > pair2.first;
	}

	bool MyDelegateOptions::pair_vectors_greater(const std::pair<std::vector<int>, std::vector<int>>& pair1, const std::pair<std::vector<int>, std::vector<int>>& pair2, const std::vector<int>& output_size)
	{
		// Sort in decreasing order based on the first element of the pair
		const std::vector<int>& first1 = pair1.first;
		const std::vector<int>& second1 = pair1.second;
		const std::vector<int>& first2 = pair2.first;
		const std::vector<int>& second2 = pair2.second;

		const int& batch1 = first1[0];
		const int& output_y1 = first1[1];
		const int& output_x1 = first1[2];
		const int& output_channel1 = first1[3];

		const int& batch2 = first2[0];
		const int& output_y2 = first2[1];
		const int& output_x2 = first2[2];
		const int& output_channel2 = first2[3];

		const int& batches = output_size[0];
		const int& outputRows = output_size[1];
		const int& outputCols = output_size[2];
		const int& outputChannels = output_size[3];

		const int pos1 = batch1 * outputRows * outputCols * outputChannels + output_y1 * outputCols * outputChannels + output_x1 * outputChannels + output_channel1;
		const int pos2 = batch2 * outputRows * outputCols * outputChannels + output_y2 * outputCols * outputChannels + output_x2 * outputChannels + output_channel2;

		return pos1 > pos2;
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