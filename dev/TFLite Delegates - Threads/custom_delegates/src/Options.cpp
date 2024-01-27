#include "Options.h"
#include "Logger.h"

namespace tflite {

	bool MyDelegateOptions::new_call = false;

	MyDelegateOptions::MyDelegateOptions(const MyDelegateOptions& options)
		: operation_mode(options.operation_mode),
		bit_position(options.bit_position),
		number_flips(options.number_flips),
		dataset_size(options.dataset_size),
		node_index(options.node_index),
		builtin_code(options.builtin_code),
		channels(options.channels),
		chunk_size(options.chunk_size),
		layer_name(options.layer_name)
	{
		// Copy constructor
		error_flat_positions.resize(dataset_size);
		error_vec_positions.resize(dataset_size);
		chunks_indexes.resize(dataset_size);
		// This constructor is called from the initialization list of the constructor of MyDelegateKernel
#if LOGGER
		//std::cout << "MyDelegateOptions copy constructor\n";
#endif // LOGGER
	}

	MyDelegateOptions::MyDelegateOptions(char** options_keys, char** options_values, size_t num_options)
	{
		// This constructor is called from the entry point
#if LOGGER
		//std::cout << "MyDelegateOptions constructor from keys\n";
#endif // LOGGER

		for (int i = 0; i < num_options; i++)
		{
			if (options_keys != nullptr && options_values != nullptr)
			{
				if (strcmp(*(options_keys + i), "operation_mode") == 0)
				{
					operation_mode = (OperationMode)std::stoi(*(options_values + i));
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

		error_flat_positions.resize(dataset_size);
		error_vec_positions.resize(dataset_size);
		chunks_indexes.resize(dataset_size);
	}
	
	void MyDelegateOptions::convertPositionInt2Vec(int position, int max_size, const std::vector<int>& tensor_dimensions, std::vector<int>& vec_position)
	{
		int acc = position;
		int div = max_size;
		for (int dimension : tensor_dimensions)
		{
			div /= dimension;
			vec_position.emplace_back(acc / div);
			acc -= vec_position.back() * div;
		}
	}

	bool MyDelegateOptions::getPairIntGreater(const std::pair<int, int>& pair1, const std::pair<int, int>& pair2)
	{
		// Sort in decreasing order based on the first element of the pair
		return pair1.first > pair2.first || pair1.first == pair2.first && pair1.second > pair2.second;
	}

	int MyDelegateOptions::convertPositionVec2Int(const std::vector<int>& tensor_dimensions, const std::vector<int>& vec_position)
	{
		int position = 0;
		for (int i = 0; i < vec_position.size(); i++)
		{
			int acc = 1;
			for (int j = i + 1; j < tensor_dimensions.size(); j++)
			{
				acc *= tensor_dimensions[j];
			}
			position += vec_position[i] * acc;
		}
		return position;
	}

	bool MyDelegateOptions::getPairVecGreater(const std::pair<std::vector<int>, std::vector<int>>& pair1, const std::pair<std::vector<int>, std::vector<int>>& pair2, const std::vector<int>& first_dimensions, const std::vector<int>& second_dimensions)
	{
		// Sort in decreasing order based on the first element of the pair
		const std::vector<int>& first1 = pair1.first;
		const std::vector<int>& second1 = pair1.second;
		const std::vector<int>& first2 = pair2.first;
		const std::vector<int>& second2 = pair2.second;

		const int pos_first1 = convertPositionVec2Int(first_dimensions, first1);
		const int pos_first2 = convertPositionVec2Int(first_dimensions, first2);

		const int pos_second1 = convertPositionVec2Int(first_dimensions, second1);
		const int pos_second2 = convertPositionVec2Int(first_dimensions, second2);

		return pos_first1 > pos_first2 || pos_first1 == pos_first2 && pos_second1 > pos_second2;
	}

	void MyDelegateOptions::Log() const
	{
		std::cout << "layer name = " << layer_name << "\n";
		switch (operation_mode)
		{
		case tflite::OperationMode::none:
			std::cout << "operation mode = none\n";
			break;
		case tflite::OperationMode::weights:
			std::cout << "operation mode = weights\n";
			break;
		case tflite::OperationMode::convolution:
			std::cout << "operation mode = convolution\n";
			break;
		default:
			std::cout << "operation mode = unknown\n";
			break;
		}
		std::cout << "bit position = " << bit_position << "\n";
		std::cout << "number flips = " << number_flips << "\n";
		std::cout << "dataset size = " << dataset_size << "\n";
		std::cout << "node index = " << node_index << "\n";
		std::cout << "builtin code = " << custom_logger::get_builtin_code(builtin_code) << "\n";
		std::cout << "channels = " << channels << "\n";
		std::cout << "chunk size = " << chunk_size << "\n";
		std::cout << "num threads = " << num_threads << "\n";
		std::cout << "is threaded: " << (is_threaded ? "true" : "false") << "\n";
	}

}