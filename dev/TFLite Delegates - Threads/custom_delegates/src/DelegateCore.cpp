#include "DelegateCore.h"

namespace tflite {

	// MyDelegateKernel Methods

	MyDelegateKernel::MyDelegateKernel()
		: operation_data_(nullptr), conv_params_(new TfLiteConvParams)
	{

	}
	MyDelegateKernel::MyDelegateKernel(const MyDelegateOptions& options)
		: options_(options), operation_data_(nullptr), conv_params_(new TfLiteConvParams)
	{

	}
	MyDelegateKernel::~MyDelegateKernel()
	{
		// Frees the memory created in Init of type OpData
		custom_ops::conv::Free(nullptr, operation_data_);
		delete conv_params_;
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
		// output shape is number_nodes x number of outputs per node 
		inputs_.resize(params->nodes_to_replace->size);
		outputs_.resize(params->nodes_to_replace->size);
		builtin_code_.resize(params->nodes_to_replace->size);
		// Stores the Convolution Operation Options
		// can add more options later
		// Heap allocated, should be freed in the destructor
		operation_data_ = reinterpret_cast<custom_ops::conv::OpData*>(custom_ops::conv::Init(context, nullptr, 0));

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
					&delegated_node_registration),
				kTfLiteOk);

			int input_index, bias_index, filter_index;
			// Warning: ASSUMING THAT THERE IS ONLY ONE OUTPUT!!!
			int output_index = 0;
			for (int j = 0; j < delegated_node->inputs->size; j++)
			{
				inputs_[i].push_back(delegated_node->inputs->data[j]);
				custom_ops::conv::GetTensorIndexes(context, delegated_node, &bias_index, &filter_index, &input_index);
				
			}
			for (int j = 0; j < delegated_node->outputs->size; j++)
			{
				outputs_[i].push_back(delegated_node->outputs->data[j]);
			}

			const auto& input_tensor = context->tensors[delegated_node->inputs->data[input_index]];
			const auto& filter_tensor = context->tensors[delegated_node->inputs->data[filter_index]];
			const auto& output_tensor = context->tensors[delegated_node->outputs->data[output_index]];
			// In the future options_ will be a vector
			//std::cout << "Input size\n";
			for (int k = 0; k < input_tensor.dims->size; k++)
			{
				options_.input_size.push_back(input_tensor.dims->data[k]);
				//std::cout << options_.input_size.back() << " ";
			}
			//std::cout << "\n";
			//std::cout << "Filter size\n";
			for (int k = 0; k < filter_tensor.dims->size; k++)
			{
				options_.kernel_size.push_back(filter_tensor.dims->data[k]);
				//std::cout << options_.kernel_size.back() << " ";
			}
			//std::cout << "\n";
			//std::cout << "Output size\n";
			for (int k = 0; k < output_tensor.dims->size; k++)
			{
				options_.output_size.push_back(output_tensor.dims->data[k]);
				//std::cout << options_.output_size.back() << " ";
			}
			//std::cout << "\n";

			builtin_code_[i] = delegated_node_registration->builtin_code;

			// Modify this to accept more than 1 node
			// For the moment it only stores 1 node's information
			GetOperationData(*reinterpret_cast<custom_ops::conv::OpData*>(delegated_node->user_data));
			GetConvParams(*reinterpret_cast<TfLiteConvParams*>(delegated_node->builtin_data));

			//////////////////
			int output_size = custom_ops::size_extraction(output_tensor.dims);
			int kernel_partial_size = custom_ops::size_extraction(filter_tensor.dims, 1);

			//std::cout << "output size " << output_size << "\n";
			//std::cout << "kernel partial size " << kernel_partial_size << "\n";

			std::random_device random_device;
			std::mt19937 mt_generator(random_device());
			std::uniform_int_distribution<int> output_dist(0, output_size - 1);
			std::uniform_int_distribution<int> kernel_dist(0, kernel_partial_size - 1);

			// Gathering the variables of convolution
			const int& stride_height = conv_params_->stride_height;
			const int& stride_width = conv_params_->stride_width;
			const int& dilation_height_factor = conv_params_->dilation_height_factor;
			const int& dilation_width_factor = conv_params_->dilation_width_factor;

			const int& pad_height = operation_data_->padding.height;
			const int& pad_width = operation_data_->padding.width;

			const int& input_height = options_.input_size[1];
			const int& input_width = options_.input_size[2];

			// Get partial sizes
			std::vector<int> kernel_partial_size_values(options_.kernel_size.begin() + 1, options_.kernel_size.end());
			// Put everything that follows on a loop to generate the whole dataset random positions beforehand
			for (int j = 0; j < options_.dataset_size; j++)
			{
				// MUST BE RESERVE not RESIZE
				options_.errorPositions[j].reserve(options_.number_flips);
				options_.realPositions[j].reserve(options_.number_flips);

				// Generating the output error positions
				for (int k = 0; k < options_.number_flips; ++k)
				{
					std::vector<int> out_values;
					std::vector<int> kernel_values;
					out_values.reserve(4);
					kernel_values.reserve(4);

					// Generating the output error position
					int output_error_position = output_dist(mt_generator);
					int kernel_partial_position;
					options_.convertPosition(output_error_position, output_size, options_.output_size, out_values);
				
					const int in_y_origin = (out_values[1] * stride_height) - pad_height;
					const int in_x_origin = (out_values[2] * stride_width) - pad_width;
					int input_y, input_x;
					bool flag_inside_input = false;
					do
					{
						// This has to be done in a do while loop, to make certain it is inside the input
						kernel_values.clear();
						kernel_values.push_back(out_values.back());

						// Generating the random number
						kernel_partial_position = kernel_dist(mt_generator);

						// Convert the number to position values
						options_.convertPosition(kernel_partial_position, kernel_partial_size, kernel_partial_size_values, kernel_values);

						// Verify that the values are in range!!! Only happens if there is padding present
						// output : batch   output_y    output_x    output_channel
						//          0       1           2           3
						// kernel:  output_channel  kernel_y    kernel_x    input_channel
						//          0               1           2           3

						input_y = in_y_origin + dilation_height_factor * kernel_values[1];
						input_x = in_x_origin + dilation_width_factor * kernel_values[2];
						//std::cout << "input_y: " << input_y << " input_x: " << input_x << "\n";
						if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width)
							flag_inside_input = true;
					} while (!flag_inside_input);

					// After the verification
					options_.errorPositions[j].emplace_back(output_error_position, kernel_partial_position);
					options_.realPositions[j].emplace_back(out_values, kernel_values);

				}
				std::sort(options_.errorPositions[j].begin(), options_.errorPositions[j].end(), [this](const std::pair<int, int>& pair1, const std::pair<int, int>& pair2) { return options_.pair_greater(pair1, pair2); });
				std::sort(options_.realPositions[j].begin(), options_.realPositions[j].end(), [this](const std::pair<std::vector<int>, std::vector<int>>& pair1, const std::pair<std::vector<int>, std::vector<int>>& pair2) { return options_.pair_vectors_greater(pair1, pair2, options_.output_size); });
			
#if LOGGER
			//std::cout << "Error positions\n";
			//for (const auto& val : options_.errorPositions[j])
			//{
			//	std::cout << val.first << " - " << val.second << "\n";
			//}
			//std::cout << "\n";
			//std::cout << "Real positions\n";
			//for (const auto& val : options_.realPositions[j])
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
			//std::cout << "\n";

#endif // LOGGER
			
			}

			options_.indexes.resize(options_.number_flips);
			std::iota(options_.indexes.begin(), options_.indexes.end(), 0);

#if LOGGER
			//std::cout << "Special logging! To be delegated node index: " << node_index << std::endl;
			////custom_logger::LogTfLiteContext(context);
			//custom_logger::LogTfLiteRegistration(delegated_node_registration);
			//std::cout << "Memory address of node: " << reinterpret_cast<void*>(delegated_node) << std::endl;
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
			prepared_success = custom_ops::conv::Prepare<tflite::custom_ops::conv::kReference>(context, node, conv_params_, operation_data_);
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

		evalued_success = custom_ops::conv::Eval<custom_ops::conv::kReference>(context, node, conv_params_, operation_data_, options_);

#if LOGGER
		//std::cout << "Evaluation result: " << custom_logger::get_TfLiteStatus(evalued_success) << std::endl;
#endif // LOGGER

		return evalued_success;
	}
	void MyDelegateKernel::GetOperationData(const custom_ops::conv::OpData& operation_data)
	{
		operation_data_->im2col_id = operation_data.im2col_id;
		operation_data_->hwcn_weights_id = operation_data.hwcn_weights_id;
		operation_data_->input_quantized_id = operation_data.input_quantized_id;
		operation_data_->scaling_factors_id = operation_data.scaling_factors_id;
		operation_data_->input_offset_id = operation_data.input_offset_id;
		operation_data_->accum_scratch_id = operation_data.accum_scratch_id;
		
		operation_data_->row_sums_id = operation_data.row_sums_id;
		
		operation_data_->padding = operation_data.padding;
		
		operation_data_->output_multiplier = operation_data.output_multiplier;
		operation_data_->output_shift = operation_data.output_shift;
		
		operation_data_->per_channel_output_multiplier = operation_data.per_channel_output_multiplier;
		operation_data_->per_channel_output_shift = operation_data.per_channel_output_shift;
		
		operation_data_->output_activation_min = operation_data.output_activation_min;
		operation_data_->output_activation_max = operation_data.output_activation_max;

		operation_data_->im2col_index = operation_data.im2col_index;
		operation_data_->hwcn_weights_index = operation_data.hwcn_weights_index;
		operation_data_->input_quantized_index = operation_data.input_quantized_index;
		operation_data_->scaling_factors_index = operation_data.scaling_factors_index;
		operation_data_->accum_scratch_index = operation_data.accum_scratch_index;
		operation_data_->input_offset_index = operation_data.input_offset_index;
		operation_data_->row_sums_index = operation_data.row_sums_index;
		
		operation_data_->need_hwcn_weights = operation_data.need_hwcn_weights;
		operation_data_->have_weights_been_transposed = operation_data.have_weights_been_transposed;
		operation_data_->need_im2col = operation_data.need_im2col;
		operation_data_->im2col_oversized = operation_data.im2col_oversized;
		
		operation_data_->supports_multithreaded_kernel = operation_data.supports_multithreaded_kernel;
		operation_data_->is_hybrid_per_channel = operation_data.is_hybrid_per_channel;
		operation_data_->compute_hybrid_row_sums = operation_data.compute_hybrid_row_sums;
		
		operation_data_->groups = operation_data.groups;
		operation_data_->quantized_bias_type = operation_data.quantized_bias_type;
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
	
	// MyDelegate Methods

	MyDelegate::MyDelegate()
	{
		// This calls the default constructor off options_ (MyDelegateOptions)
	}
	MyDelegate::MyDelegate(const MyDelegateOptions& options)
		: options_(options)
	{
		// The initialization list calls the copy constructor of options_ MyDelegateOptions
	}
	MyDelegate::~MyDelegate()
	{

	}
	bool MyDelegate::IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
		const TfLiteNode* node, TfLiteContext* context) const
	{
		// Checking the TfLiteRegistration
		// Only supports 2D convolution operations.
		if (kTfLiteBuiltinConv2d != registration->builtin_code)
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
		auto& kernel_tensor = context->tensors[node->inputs->data[1]];
		// Looking by name, if layer is not named, logic should be changed
		if (strstr(kernel_tensor.name, options_.layer_name.c_str()) == nullptr)
			return false;
		
		// Checking if it affects the weights or the convolution
		if (options_.operation_mode == OperationMode::weights)
		{
			// Generate random number here to affect the weights of the kernel
			signed char* tensor_ptr = reinterpret_cast<signed char*>(kernel_tensor.data.data);
			int size = custom_ops::size_extraction(kernel_tensor.dims);
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
		return std::make_unique<MyDelegateKernel>(options_);
	}
	SimpleDelegateInterface::Options MyDelegate::DelegateOptions() const
	{
		// Default options
		return SimpleDelegateInterface::Options();
	}

}