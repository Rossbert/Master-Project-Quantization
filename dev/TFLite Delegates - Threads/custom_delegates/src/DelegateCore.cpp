#include "DelegateCore.h"

namespace tflite {

	// MyDelegateKernel Methods

	MyDelegateKernel::MyDelegateKernel()
		: operation_data_conv_(nullptr), 
		conv_params_(new TfLiteConvParams),
		operation_data_fully_(nullptr),
		fully_params_(new TfLiteFullyConnectedParams)
	{

	}

	MyDelegateKernel::MyDelegateKernel(const MyDelegateOptions& options)
		: options_(options), 
		operation_data_conv_(nullptr), 
		conv_params_(new TfLiteConvParams),
		operation_data_fully_(nullptr),
		fully_params_(new TfLiteFullyConnectedParams)
	{
		// Constructor with initializer options
#if LOGGER
		//std::cout << "MyDelegateKernel constructor with options\n";
#endif // LOGGER
	}

	MyDelegateKernel::~MyDelegateKernel()
	{
		// Frees the memory created in Init of type OpData
		custom_ops::conv::Free(nullptr, operation_data_conv_);
		delete conv_params_;

		custom_ops::fully_connected::Free(nullptr, operation_data_fully_);
		delete fully_params_;
	}

	TfLiteStatus MyDelegateKernel::Init(TfLiteContext* context, const TfLiteDelegateParams* params)
	{
		// Stores the neccessary information in MyDelegateKernel instance
		// Only gets called ONCE!!!
#if LOGGER
		// TfLiteDelegateParams logging
		//std::cout << std::endl << "Variables in MyDelegateKernel::Init" << std::endl;
		//custom_logger::LogTfLiteDelegateParams(params);
		//auto temp = params->input_tensors->data[0];
		//params->input_tensors->data[0] = params->input_tensors->data[1];
		//params->input_tensors->data[1] = params->input_tensors->data[2];
		//params->input_tensors->data[2] = temp;
		//std::cout << "Huh????????" << std::endl;
		//custom_logger::LogTfLiteDelegateParams(params);
#endif // LOGGER

		// Save index to all nodes which are part of this delegate.
		// Inputs and outputs are vectors of vectors
		// input shape is number_nodes x number of inputs per node 
		//inputs_.resize(params->nodes_to_replace->size);
		// output shape is number_nodes x number of outputs per node 
		//outputs_.resize(params->nodes_to_replace->size);

		for (int i = 0; i < params->nodes_to_replace->size; ++i)
		{
			const int node_index = params->nodes_to_replace->data[i];
			// This is why options_ should be a vector of options in the future for every node
			options_.node_index = node_index;
			// Get this node information.
			TfLiteNode* delegated_node = nullptr;
			TfLiteRegistration* delegated_node_registration = nullptr;
			TF_LITE_ENSURE_EQ(
				context,
				context->GetNodeAndRegistration(context, node_index, &delegated_node,
					&delegated_node_registration), kTfLiteOk);

			int input_index, bias_index, filter_index;
			// Warning: ASSUMING THAT THERE IS ONLY ONE OUTPUT!!!
			int output_index = 0;
			for (int j = 0; j < delegated_node->inputs->size; j++)
			{
				//inputs_[i].push_back(delegated_node->inputs->data[j]);
				custom_ops::GetTensorIndexes(context, delegated_node, &bias_index, &filter_index, &input_index);
				
			}
			//for (int j = 0; j < delegated_node->outputs->size; j++)
			//{
			//	outputs_[i].push_back(delegated_node->outputs->data[j]);
			//}

			const auto& input_tensor = context->tensors[delegated_node->inputs->data[input_index]];
			const auto& filter_tensor = context->tensors[delegated_node->inputs->data[filter_index]];
			const auto& output_tensor = context->tensors[delegated_node->outputs->data[output_index]];
			
			for (int k = 0; k < input_tensor.dims->size; k++)
			{
				options_.input_dimensions.push_back(input_tensor.dims->data[k]);
			}
			for (int k = 0; k < filter_tensor.dims->size; k++)
			{
				options_.kernel_dimensions.push_back(filter_tensor.dims->data[k]);
			}
			for (int k = 0; k < output_tensor.dims->size; k++)
			{
				options_.output_dimensions.push_back(output_tensor.dims->data[k]);
			}

			options_.builtin_code = delegated_node_registration->builtin_code;

#if LOGGER
			//std::cout << "Input size\n";
			//for (const int& val : options_.input_dimensions)
			//{
			//	std::cout << val << " ";
			//}
			//std::cout << "\n";
			//std::cout << "Filter size\n";
			//for (const int& val : options_.kernel_dimensions)
			//{
			//	std::cout << val << " ";
			//}
			//std::cout << "\n";
			//std::cout << "Output size\n";
			//for (const int& val : options_.output_dimensions)
			//{
			//	std::cout << val << " ";
			//}
			//std::cout << "\n";

			//std::cout << "Registration type: " << custom_logger::get_builtin_code(options_.builtin_code) << "\n";
#endif // LOGGER

			// Stores the Convolution Operation Options
			// can add more options later
			// Heap allocated, should be freed in the destructor
			operation_data_conv_ = reinterpret_cast<custom_ops::conv::OpData*>(custom_ops::conv::Init(context, nullptr, 0));

			// Stores the Fully Connected Operation Options
			// can add more options later
			// Heap allocated, should be freed in the destructor
			operation_data_fully_ = reinterpret_cast<custom_ops::fully_connected::OpData*>(custom_ops::fully_connected::Init(context, nullptr, 0));

			// Modify this to accept more than 1 node
			// For the moment it only stores 1 node's information
			// This operation fails for dense layer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			switch (options_.builtin_code)
			{
			case kTfLiteBuiltinConv2d: {
				GetConvOperationData(*reinterpret_cast<custom_ops::conv::OpData*>(delegated_node->user_data));
				GetConvParams(*reinterpret_cast<TfLiteConvParams*>(delegated_node->builtin_data));

				//custom_logger::LogTfLiteConvParams(conv_params_);
				//custom_logger::conv::LogTfLiteOpData(operation_data_conv_);
			}
				break;
			case kTfLiteBuiltinFullyConnected: {
				GetFullyOperationData(*reinterpret_cast<custom_ops::fully_connected::OpData*>(delegated_node->user_data));
				GetFullyParams(*reinterpret_cast<TfLiteFullyConnectedParams*>(delegated_node->builtin_data));

				//custom_logger::LogTfLiteFullyConnectedParams(fully_params_);
				//custom_logger::fully_connected::LogTfLiteOpData(operation_data_fully_);
			}
				break;
			default:
				break;
			}
			
			int output_flat_size = custom_ops::getFlatSize(output_tensor.dims);
			int kernel_partial_flat_size = custom_ops::getFlatSize(filter_tensor.dims, 1);

			/// Random variables generation!
			std::random_device random_device;
			std::mt19937 mt_generator(random_device());
			std::uniform_int_distribution<int> output_dist(0, output_flat_size - 1);
			std::uniform_int_distribution<int> kernel_dist(0, kernel_partial_flat_size - 1);

			// Get partial sizes, first element of the kernel size is not needed
			std::vector<int> kernel_partial_dimensions(options_.kernel_dimensions.begin() + 1, options_.kernel_dimensions.end());
			
#if LOGGER
			//std::cout << "output size " << output_flat_size << "\n";
			//std::cout << "kernel partial size " << kernel_partial_flat_size << "\n";
			//std::cout << "Kernel dimensions: ";
			//for (const auto& val : kernel_partial_dimensions)
			//{
			//	std::cout << val << " ";
			//}
			//std::cout << "\n";
#endif // LOGGER

			int stride_height;
			int stride_width;
			int dilation_height_factor;
			int dilation_width_factor;
			int pad_height;
			int pad_width;
			int input_height;
			int input_width;
			if (options_.builtin_code == kTfLiteBuiltinConv2d)
			{
				// Gathering the variables of convolution
				stride_height = conv_params_->stride_height;
				stride_width = conv_params_->stride_width;
				dilation_height_factor = conv_params_->dilation_height_factor;
				dilation_width_factor = conv_params_->dilation_width_factor;

				pad_height = operation_data_conv_->padding.height;
				pad_width = operation_data_conv_->padding.width;

				input_height = options_.input_dimensions[1];
				input_width = options_.input_dimensions[2];
			}

			// Here organize the indexes of the chunks of the channels
			options_.full_indexes.resize(options_.number_flips);
			// Fill values with increasing order from 0 to size of indexes
			std::iota(options_.full_indexes.begin(), options_.full_indexes.end(), 0);

			// Constants for accelerating the threaded version
			// channels are always the position 0 of the kernel dimensions
			int number_operations = getNumberOperations(options_.output_dimensions, options_.kernel_dimensions);
			options_.num_threads = number_operations / options_.max_operations_per_thread;
			if (options_.num_threads == 0)
				options_.num_threads = 1;
			options_.num_threads = std::min(options_.num_threads, options_.max_number_threads); // Ensuring the number of threads doesn't exceed the number of channels
			options_.channels = options_.kernel_dimensions[0];
			options_.chunk_size = options_.channels / options_.num_threads;
			// Here determine if it will be threaded or not
			if (options_.num_threads != 1)
			{
				options_.is_threaded = true;
			}

#if LOGGER
			std::cout << "Is threaded?: " << (options_.is_threaded ? "true" : "false") << "\n";
			std::cout << "Number of threads " << options_.num_threads << "\n";
			//std::cout << "Number of operations " << number_operations << "\n";
			//std::cout << "Chunk size: " << options_.chunk_size << "\n";
#endif // LOGGER


			// Put everything that follows on a loop to generate the whole dataset random positions beforehand
			// For MNIST Fashion options_.dataset_size = 10000
			for (int j = 0; j < options_.dataset_size; j++)
			{
				// MUST BE RESERVE not RESIZE
				options_.error_flat_positions[j].reserve(options_.number_flips);
				options_.error_vec_positions[j].reserve(options_.number_flips);

				// Generating the output error positions
				for (int k = 0; k < options_.number_flips; ++k)
				{
					std::vector<int> output_error_vec_pos;
					std::vector<int> kernel_error_vec_pos;
					output_error_vec_pos.reserve(options_.output_dimensions.size());
					kernel_error_vec_pos.reserve(options_.kernel_dimensions.size());

					// Generating the output error position
					int output_error_flat_pos = output_dist(mt_generator);
					int kernel_partial_flat_pos;
					options_.convertPositionInt2Vec(output_error_flat_pos, output_flat_size, options_.output_dimensions, output_error_vec_pos);
					
					int input_y, input_x;
					int in_y_origin;
					int in_x_origin;
					if (options_.builtin_code == kTfLiteBuiltinConv2d)
					{
						in_y_origin = (output_error_vec_pos[1] * stride_height) - pad_height;
						in_x_origin = (output_error_vec_pos[2] * stride_width) - pad_width;
					}
					bool flag_valid_pos = false;
					do
					{
						// This has to be done in a do while loop, to make certain it is inside the input
						kernel_error_vec_pos.clear();
						
						// Push the last element of out position = channels
						// It is the first element of the kernel position
						kernel_error_vec_pos.push_back(output_error_vec_pos.back());

						// Generating the random number
						kernel_partial_flat_pos = kernel_dist(mt_generator);

						// Convert the number to position values
						options_.convertPositionInt2Vec(kernel_partial_flat_pos, kernel_partial_flat_size, kernel_partial_dimensions, kernel_error_vec_pos);

						if (options_.builtin_code == kTfLiteBuiltinConv2d)
						{
							// Verify that the values are in range!!! Only happens if there is padding present
							// output : batch   output_y    output_x    output_channel
							//          0       1           2           3
							// kernel:  output_channel  kernel_y    kernel_x    input_channel
							//          0               1           2           3

							input_y = in_y_origin + dilation_height_factor * kernel_error_vec_pos[1];
							input_x = in_x_origin + dilation_width_factor * kernel_error_vec_pos[2];

							//std::cout << "input_y: " << input_y << " input_x: " << input_x << "\n";
						}
						
						std::pair<int, int> candidate_position = { output_error_flat_pos, kernel_partial_flat_pos };
						auto it = std::find(options_.error_flat_positions[j].begin(), options_.error_flat_positions[j].end(), candidate_position);
						bool repeated_pos = false;
						if (it != options_.error_flat_positions[j].end()) 
						{
							repeated_pos = true;
#if LOGGER
							//std::cout << "Repeated pos: " << candidate_position.first << " - " << candidate_position.second << "\n";
							//std::cout << "Element found at index " << std::distance(options_.error_flat_positions[j].begin(), it) << "\n";
#endif // LOGGER
						}

						bool is_inside = input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width;

						if (not repeated_pos && (options_.builtin_code == kTfLiteBuiltinFullyConnected || is_inside))
							flag_valid_pos = true;
						// Check not repeated positions
					} while (!flag_valid_pos);

					// After the verification
					options_.error_flat_positions[j].emplace_back(output_error_flat_pos, kernel_partial_flat_pos);
					options_.error_vec_positions[j].emplace_back(output_error_vec_pos, kernel_error_vec_pos);

				}
				
				std::sort(options_.error_flat_positions[j].begin(), 
					options_.error_flat_positions[j].end(), 
					[this](const std::pair<int, int>& pair1, const std::pair<int, int>& pair2) 
					{ 
						return options_.getPairIntGreater(pair1, pair2); 
					});
				std::sort(options_.error_vec_positions[j].begin(), 
					options_.error_vec_positions[j].end(), 
					[this](const std::pair<std::vector<int>, std::vector<int>>& pair1, const std::pair<std::vector<int>, std::vector<int>>& pair2) 
					{ 
						return options_.getPairVecGreater(pair1, pair2, options_.output_dimensions, options_.kernel_dimensions); 
					});

				// For threaded computation
				// Separates the indexes by chunks
				std::vector<int> chunk_indexes;
				for (int k = 0; k < options_.num_threads; ++k)
				{
					const int start = k * options_.chunk_size;
					const int end = std::min(start + options_.chunk_size, options_.channels);
					getChunkIndexes(start, end, options_.error_vec_positions[j], chunk_indexes);

					//std::cout << "Start: " << start << " End: " << end << "\n";
					//std::cout << "Indexes in chunk " << k << ": ";
					//for (const auto& val : chunk_indexes)
					//{
					//	std::cout << val << " ";
					//}
					//std::cout << "\n";

					options_.chunks_indexes[j].emplace_back(chunk_indexes);
				}

#if LOGGER
				//std::cout << "Item " << j << "\n";
				//for (int k = 0; k < options_.chunks_indexes[j].size(); k++)
				//{
				//	std::cout << "Chunk " << k << "\n";
				//	for (const auto& val : options_.chunks_indexes[j][k])
				//	{
				//		std::cout << val << " ";
				//	}
				//	std::cout << "\n";
				//}
				//std::cout << "Error flat positions\n";
				//for (const auto& val : options_.error_flat_positions[j])
				//{
				//	std::cout << val.first << " - " << val.second << "\n";
				//}
				//std::cout << "Error vector positions\n";
				//for (const auto& val : options_.error_vec_positions[j])
				//{
				//	for (const int& element : val.first)
				//	{
				//		std::cout << element << " ";
				//	}
				//	std::cout << "- ";
				//	for (const int& element : val.second)
				//	{
				//		std::cout << element << " ";
				//	}
				//	std::cout << "\n";
				//}
#endif // LOGGER
			
			}

#if LOGGER
			//std::cout << "Indexes\n";
			//for (const auto& val : options_.full_indexes)
			//{
			//	std::cout << val << " ";
			//}
			//std::cout << "\n";

			//int j = 0;
			//std::cout << "Error flat positions\n";
			//for (const auto& val : options_.error_flat_positions[j])
			//{
			//	std::cout << val.first << " - " << val.second << "\n";
			//}
			//std::cout << "Error vector positions\n";
			//for (const auto& val : options_.error_vec_positions[j])
			//{
			//	for (const int& element : val.first)
			//	{
			//		std::cout << element << " ";
			//	}
			//	std::cout << "- ";
			//	for (const int& element : val.second)
			//	{
			//		std::cout << element << " ";
			//	}
			//	std::cout << "\n";
			//}
			//std::cout << "Special logging! To be delegated node index: " << node_index << std::endl;
			//std::cout << "Memory address of node: " << reinterpret_cast<void*>(delegated_node) << std::endl;
			//custom_logger::LogTfLiteRegistration(delegated_node_registration);
			//custom_logger::LogTfLiteContext(context);
			//custom_logger::LogTfLiteNode(delegated_node);
#endif // LOGGER
		}
		return kTfLiteOk;
	}
	
	TfLiteStatus MyDelegateKernel::Prepare(TfLiteContext* context, TfLiteNode* node)
	{
		// This method gets called once when creating the Interpreter and once again when allocating the tensors PER NODE????
		// This function forbbids the use of the function context->GetNodeAndRegistration when runnning from the interpreter
		// CAREFUL!
		// DON'T USE THE NODE RECEIVED HERE!
		// The node received here is different than the node at context
		// This node is a corrupted copy of the node registered for this operation
		// They do not share the same memory address
		// Reassigning the pointer of OpData makes the program to crash!!!!!!!!!!
		// Has absolutely insane values stored, changing it crashes the program
		// using new or malloc doesn't work

#if LOGGER
		// Logging
		//std::cout << std::endl << "MyDelegateKernel::Prepare function!" << std::endl;
		//if (prepared_)
		//{
		//	std::cout << "Already prepared! This convolution does not need allocation again." << std::endl;
		//}
#endif // LOGGER
	
		TfLiteStatus prepared_success;
		if (!prepared_)
		{
			// Calling the Custom Preparation
			// Careful the order of inputs in the receiving node is not the same as the standard order
			prepared_ = true;
			if (options_.builtin_code == kTfLiteBuiltinConv2d)
			{
				prepared_success = custom_ops::conv::Prepare<tflite::custom_ops::conv::kReference>(context, node, conv_params_, operation_data_conv_);
			}
			else
			{
				prepared_success = custom_ops::fully_connected::Prepare<tflite::custom_ops::fully_connected::kReference>(context, node, fully_params_, operation_data_fully_);
			}
		}
		else
		{
			prepared_success = kTfLiteOk;
		}

#if LOGGER
		//std::cout << "Special logging! " << std::endl;
		//custom_logger::LogTfLiteContext(context);
		//std::cout << "New node: " << std::endl;
		//std::cout << "Memory address of node: " << reinterpret_cast<void*>(node) << std::endl;
		//custom_logger::LogTfLiteNode(node);
		//custom_logger::LogTfLiteConvParams(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
		//std::cout<< std::endl;
		//std::cout << "Preparation result: " << custom_logger::get_TfLiteStatus(prepared_success) << std::endl;
#endif // LOGGER

		return prepared_success;
	}
	
	TfLiteStatus MyDelegateKernel::Eval(TfLiteContext* context, TfLiteNode* node)
	{
		// Evaluate the delegated graph.
		// Here we loop over all the delegated nodes.
		// The number of nodes equals ''inputs_.size()'' and inputs[i] is a list of
		// tensor indices for inputs to node ''i'', while outputs_[i] is the list of
		// outputs for node
		// ''i''. Note, that it is intentional we have simple implementation as this
		// is for demonstration.

#if LOGGER
		// Logging
		//std::cout << std::endl << "MyDelegateKernel::Eval function!" << std::endl;
		//std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
		//std::cout << "New node?? : " << std::endl;
		//std::cout << "Memory address of node: " << reinterpret_cast<void*>(node) << std::endl;
		//custom_logger::LogTfLiteNode(node);
		//options_.Log();
		//custom_logger::LogTfLiteContext(context);
#endif // LOGGER
		TfLiteStatus evalued_success;

		if (options_.builtin_code == kTfLiteBuiltinConv2d)
		{
			evalued_success = custom_ops::conv::Eval<custom_ops::conv::kReference>(context, node, conv_params_, operation_data_conv_, options_);
		}
		else
		{
			evalued_success = custom_ops::fully_connected::Eval<custom_ops::fully_connected::kReference>(context, node, fully_params_, operation_data_fully_, options_);
		}

#if LOGGER
		//std::cout << "Evaluation result: " << custom_logger::get_TfLiteStatus(evalued_success) << std::endl;
#endif // LOGGER

		return evalued_success;
	}

	void MyDelegateKernel::GetConvOperationData(const custom_ops::conv::OpData& operation_data)
	{
		operation_data_conv_->im2col_id = operation_data.im2col_id;
		operation_data_conv_->hwcn_weights_id = operation_data.hwcn_weights_id;
		operation_data_conv_->input_quantized_id = operation_data.input_quantized_id;
		operation_data_conv_->scaling_factors_id = operation_data.scaling_factors_id;
		operation_data_conv_->input_offset_id = operation_data.input_offset_id;
		operation_data_conv_->accum_scratch_id = operation_data.accum_scratch_id;
		
		operation_data_conv_->row_sums_id = operation_data.row_sums_id;
		
		operation_data_conv_->padding = operation_data.padding;
		
		operation_data_conv_->output_multiplier = operation_data.output_multiplier;
		operation_data_conv_->output_shift = operation_data.output_shift;
		
		operation_data_conv_->per_channel_output_multiplier = operation_data.per_channel_output_multiplier;
		operation_data_conv_->per_channel_output_shift = operation_data.per_channel_output_shift;
		
		operation_data_conv_->output_activation_min = operation_data.output_activation_min;
		operation_data_conv_->output_activation_max = operation_data.output_activation_max;

		operation_data_conv_->im2col_index = operation_data.im2col_index;
		operation_data_conv_->hwcn_weights_index = operation_data.hwcn_weights_index;
		operation_data_conv_->input_quantized_index = operation_data.input_quantized_index;
		operation_data_conv_->scaling_factors_index = operation_data.scaling_factors_index;
		operation_data_conv_->accum_scratch_index = operation_data.accum_scratch_index;
		operation_data_conv_->input_offset_index = operation_data.input_offset_index;
		operation_data_conv_->row_sums_index = operation_data.row_sums_index;
		
		operation_data_conv_->need_hwcn_weights = operation_data.need_hwcn_weights;
		operation_data_conv_->have_weights_been_transposed = operation_data.have_weights_been_transposed;
		operation_data_conv_->need_im2col = operation_data.need_im2col;
		operation_data_conv_->im2col_oversized = operation_data.im2col_oversized;
		
		operation_data_conv_->supports_multithreaded_kernel = operation_data.supports_multithreaded_kernel;
		operation_data_conv_->is_hybrid_per_channel = operation_data.is_hybrid_per_channel;
		operation_data_conv_->compute_hybrid_row_sums = operation_data.compute_hybrid_row_sums;
		
		operation_data_conv_->groups = operation_data.groups;
		operation_data_conv_->quantized_bias_type = operation_data.quantized_bias_type;
	}
	
	void MyDelegateKernel::GetConvParams(const TfLiteConvParams& params)
	{
		conv_params_->padding = params.padding;
		conv_params_->stride_width = params.stride_width;
		conv_params_->stride_height = params.stride_height;
		conv_params_->activation = params.activation;
		conv_params_->dilation_width_factor = params.dilation_width_factor;
		conv_params_->dilation_height_factor = params.dilation_height_factor;
		conv_params_->quantized_bias_type = params.quantized_bias_type;
	}

	void MyDelegateKernel::GetFullyOperationData(const custom_ops::fully_connected::OpData& operation_data)
	{
		operation_data_fully_->output_multiplier = operation_data.output_multiplier;
		operation_data_fully_->output_shift = operation_data.output_shift;

		operation_data_fully_->per_channel_output_multiplier = operation_data.per_channel_output_multiplier;
		operation_data_fully_->per_channel_output_shift = operation_data.per_channel_output_shift;

		operation_data_fully_->output_activation_min = operation_data.output_activation_min;
		operation_data_fully_->output_activation_max = operation_data.output_activation_max;

		operation_data_fully_->scratch_tensor_index = operation_data.scratch_tensor_index;
		operation_data_fully_->compute_row_sums = operation_data.compute_row_sums;

		operation_data_fully_->ledger_initialized = operation_data.ledger_initialized;

		operation_data_fully_->quantized_bias_type = operation_data.quantized_bias_type;
	}

	void MyDelegateKernel::GetFullyParams(const TfLiteFullyConnectedParams& params)
	{
		fully_params_->activation = params.activation;
		fully_params_->weights_format = params.weights_format;
		fully_params_->keep_num_dims = params.keep_num_dims;
		fully_params_->asymmetric_quantize_inputs = params.asymmetric_quantize_inputs;
		fully_params_->quantized_bias_type = params.quantized_bias_type;
	}

	void MyDelegateKernel::getChunkIndexes(int start, int end, const std::vector<std::pair<std::vector<int>, std::vector<int>>>& error_vec_positions, std::vector<int>& indexes)
	{
		indexes.clear();
		for (int i = 0; i < error_vec_positions.size(); i++)
		{
			// We are checking the channel of the channel output of the first position 
			const int& output_channel = error_vec_positions[i].first.back();
			// Range does not include end
			if (output_channel >= start && output_channel < end)
			{
				indexes.push_back(i);
			}
		}
	}

	int MyDelegateKernel::getNumberOperations(const std::vector<int>& output_dimensions, const std::vector<int>& kernel_dimensions)
	{
		// It is assumed the last dimension of the output coincides with the first of the kernel
		int acc = 1;
		for (const int& out : output_dimensions)
		{
			acc *= out;
		}
		for (int j = 1; j < kernel_dimensions.size(); j++)
		{
			acc *= kernel_dimensions[j];
		}
		return acc;
	}

	// MyDelegate Methods

	MyDelegate::MyDelegate()
	{
		// This calls the default constructor of options_ (MyDelegateOptions)
	}
	MyDelegate::MyDelegate(const MyDelegateOptions& options)
		: options_(options)
	{
		// Called from the entry point by creating an unique pointer there
		// MyDelegate is created before the MyDelegateKernel
		// The initialization list calls the copy constructor of options_ MyDelegateOptions
#if LOGGER
		//std::cout << "MyDelegate constructor with options\n";
#endif // LOGGER
	}
	MyDelegate::~MyDelegate()
	{

	}
	bool MyDelegate::IsNodeSupportedByDelegate(const TfLiteRegistration* registration, const TfLiteNode* node, TfLiteContext* context) const
	{
		// Checking the TfLiteRegistration
		// Only supports 2D convolution operations.
#if LOGGER
		//std::cout << "Registration type: " << custom_logger::get_builtin_code(registration->builtin_code) << "\n";
#endif // LOGGER
		if (registration->builtin_code != kTfLiteBuiltinConv2d && registration->builtin_code != kTfLiteBuiltinFullyConnected)
			return false;
		
		// Checking the TfLiteNode inputs type
		// If biases are included in the layer the input has 3 tensors
		// Otherwise it has 2
		// This delegate only supports convolution with the following details
		// input[0] => Input = int8
		// input[1] => Convolution kernel weights = int8
		// input[2] => Biases = int32
		for (int i = 0; i < 2; ++i)
		{
			auto& tensor = context->tensors[node->inputs->data[i]];
			if (tensor.type != kTfLiteInt8)
				return false;
		}
		if (node->inputs->size == 3)
		{
			// Position 2 is always the biases tensor
			auto& tensor = context->tensors[node->inputs->data[2]];
			if (tensor.type != kTfLiteInt32)
				return false;
		}
		
		// WARNING: The original node always follows the next order:
		// input index = 0
		// kernel index = 1
		// bias index = 2
		// Checking the TfLiteTensor and TfLiteNode name
		// For a normal node the kernel index is always 1, in the inputs
		// By this function the context and node are not disturbed... as in the function MyDelegateKernel::Prepare
		auto& kernel_tensor = context->tensors[node->inputs->data[1]];

		// Looking by name, if layer is not named, logic should be changed
#if LOGGER
		//std::cout << "Kernel tensor name: " << kernel_tensor.name << "\n";
#endif // LOGGER
		// By this criteria only one node will be accepted!
		if (strstr(kernel_tensor.name, options_.layer_name.c_str()) == nullptr)
			return false;
		
		// Checking if it affects the weights or the convolution
		if (options_.operation_mode == OperationMode::weights)
		{
			// Generate random number here to affect the weights of the kernel
			signed char* tensor_ptr = reinterpret_cast<signed char*>(kernel_tensor.data.data);
			int size = custom_ops::getFlatSize(kernel_tensor.dims);
			std::random_device random_device;
			std::mt19937 mt_generator(random_device());
			std::uniform_int_distribution<int> dist(0, size - 1);
			int random_position = 0;
			for (int i = 0; i < options_.number_flips; i++)
			{
				random_position = dist(mt_generator);
				*(tensor_ptr + random_position) = (signed char)(*(tensor_ptr + random_position) ^ (1 << options_.bit_position));
			}
#if LOGGER
			//std::cout << "Random random_position: " << random_position << std::endl;
			//std::cout << "Bit position: " << options_.bit_position << std::endl;
			//std::cout << "Tensor " << node->inputs->data[1] << " original value " << +*(tensor_ptr + random_position) << std::endl;
#endif // LOGGER
			
			// Should not delegate but it has modified the context
			return false;
		}

		if (options_.operation_mode != OperationMode::convolution)
		{
			return false;
		}
#if LOGGER
		std::cout << "Delegate type accepted!" << std::endl;
		//std::cout << std::endl << "Variables in MyDelegate::IsNodeSupportedByDelegate" << std::endl;
		//std::cout << "Masked value " << +*(tensor_ptr + random_position) << std::endl;
		//custom_logger::LogTfLiteRegistration(registration);
		//custom_logger::LogTfLiteNode(node);
#endif // LOGGER
		// Take care if not a single node is supported, the functions in MyDelegateKernel are not called
		return true;
	}
	TfLiteStatus MyDelegate::Initialize(TfLiteContext* context)
	{

#if LOGGER
		//std::cout << std::endl << "Variables in MyDelegate::Initialize" << std::endl;
		//custom_logger::LogTfLiteContext(context);
		//options_.Log();
#endif // LOGGER

		return kTfLiteOk;
	}
	const char* MyDelegate::Name() const
	{
		static constexpr char kName[] = "DelegateConvSET";
		return kName;
	}
	std::unique_ptr<SimpleDelegateKernelInterface> MyDelegate::CreateDelegateKernelInterface()
	{
		// Creates one unique pointer of MyDelegateKernel
		// This calls the constructor of MyDelegateKernel and passes options_ as a parameter
#if LOGGER
		//std::cout << "Created Simple Interface\n";
#endif // LOGGER
		return std::make_unique<MyDelegateKernel>(options_);
	}
	SimpleDelegateInterface::Options MyDelegate::DelegateOptions() const
	{
		// Default options
		return SimpleDelegateInterface::Options();
	}

}