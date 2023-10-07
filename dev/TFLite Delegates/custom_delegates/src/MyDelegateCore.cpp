#include "MyDelegateCore.h"

namespace tflite {
	
	// MyDelegateKernel Methods

	MyDelegateKernel::MyDelegateKernel()
	{

	}
	MyDelegateKernel::~MyDelegateKernel()
	{

	}
	TfLiteStatus MyDelegateKernel::Init(TfLiteContext* context, const TfLiteDelegateParams* params)
	{
		// TfLiteDelegateParams logging
		// Only gets called once
		std::cout << std::endl << "Variables in MyDelegateKernel::Init" << std::endl;
		custom_logger::LogTfLiteDelegateParams(params);

		// Save index to all nodes which are part of this delegate.
		// Inputs and outputs are vectors of vectors
		// input shape is number_nodes x number of inputs per node 
		// output shape is number_nodes x number of outputs per node 
		inputs_.resize(params->nodes_to_replace->size);
		outputs_.resize(params->nodes_to_replace->size);
		builtin_code_.resize(params->nodes_to_replace->size);
		for (int i = 0; i < params->nodes_to_replace->size; ++i) {
			const int node_index = params->nodes_to_replace->data[i];
			// Get this node information.
			TfLiteNode* delegated_node = nullptr;
			TfLiteRegistration* delegated_node_registration = nullptr;
			TF_LITE_ENSURE_EQ(
				context,
				context->GetNodeAndRegistration(context, node_index, &delegated_node,
					&delegated_node_registration),
				kTfLiteOk);

			std::cout << "Special logging! To be delegated node index: "<< node_index << std::endl;
			custom_logger::LogTfLiteNode(delegated_node);
			custom_logger::LogTfLiteRegistration(delegated_node_registration);

			for (int j = 0; j < delegated_node->inputs->size; j++)
				inputs_[i].push_back(delegated_node->inputs->data[j]);
			for (int j = 0; j < delegated_node->outputs->size; j++)
				outputs_[i].push_back(delegated_node->outputs->data[j]);

			builtin_code_[i] = delegated_node_registration->builtin_code;
		}
		return kTfLiteOk;
	}
	TfLiteStatus MyDelegateKernel::Prepare(TfLiteContext* context, TfLiteNode* node)
	{
		// Check activations.cc for inspiration
		// Inside tflite::ops::builtin::conv
		// Inside tflite::ops::builtin::activation
		// Logging
		std::cout << std::endl << "MyDelegateKernel::Prepare function!" << std::endl;
		std::cout << "Special logging" << std::endl;
		custom_logger::LogTfLiteNode(node);
		std::cout<< std::endl;
		return kTfLiteOk;
	}
	TfLiteStatus MyDelegateKernel::Eval(TfLiteContext* context, TfLiteNode* node)
	{
		// Evaluate the delegated graph.
		// Here we loop over all the delegated nodes.
		// We know that all the nodes are either ADD or SUB operations and the
		// number of nodes equals ''inputs_.size()'' and inputs[i] is a list of
		// tensor indices for inputs to node ''i'', while outputs_[i] is the list of
		// outputs for node
		// ''i''. Note, that it is intentional we have simple implementation as this
		// is for demonstration.

		// Logging
		std::cout << std::endl << "MyDelegateKernel::Eval function!";
		std::cout << std::endl << "Variables in MyDelegateKernel::Eval" << std::endl;
		std::cout << "##############################################################" << std::endl;

		for (int i = 0; i < inputs_.size(); ++i) {
			// Get the node input tensors.
			// Add/Sub operation accepts 2 inputs.
			auto& input_tensor_1 = context->tensors[inputs_[i][0]];
			auto& input_tensor_2 = context->tensors[inputs_[i][1]];
			auto& output_tensor = context->tensors[outputs_[i][0]];

			// Logging
			std::cout << "TfLiteBuiltinOperator : " << custom_logger::get_builtin_code(builtin_code_[i]) << std::endl;

			switch (input_tensor_1.type)
			{
			case kTfLiteFloat32:
				TF_LITE_ENSURE_EQ(
					context,
					ComputeAddResult<float>(context, builtin_code_[i], &input_tensor_1, &input_tensor_2, &output_tensor),
					kTfLiteOk);
				std::cout << "Delegated sum of floats performed!!!" << std::endl;
				break;
			case kTfLiteInt32:
				TF_LITE_ENSURE_EQ(
					context,
					ComputeAddResult<int>(context, builtin_code_[i], &input_tensor_1, &input_tensor_2, &output_tensor),
					kTfLiteOk);
				std::cout << "Delegated sum of ints performed!!!" << std::endl;
				break;
			case kTfLiteInt8:
				TF_LITE_ENSURE_EQ(
					context,
					ComputeAddResult<signed char>(context, builtin_code_[i], &input_tensor_1, &input_tensor_2, &output_tensor),
					kTfLiteOk);
				std::cout << "Delegated sum of signed chars performed!!!" << std::endl;
				break;
			default:
				std::cout << "Error: unrecognized type" << std::endl;
				break;
			}
		}

		return kTfLiteOk;
	}

	// MyDelegate Methods

	MyDelegate::MyDelegate()
	{

	}
	MyDelegate::~MyDelegate()
	{

	}
	bool MyDelegate::IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
		const TfLiteNode* node,
		TfLiteContext* context) const
	{
		std::cout << std::endl << "Variables in MyDelegate::IsNodeSupportedByDelegate" << std::endl;
		custom_logger::LogTfLiteRegistration(registration);
		custom_logger::LogTfLiteNode(node);
		
		// Only supports Add and Sub ops.
		// Modify this in the future to affect convolutional layers
		if (kTfLiteBuiltinAdd != registration->builtin_code &&
			kTfLiteBuiltinSub != registration->builtin_code)
			return false;

		// This delegate only supports float32 types.
		// Modify this to accept int32 types as well
		for (int i = 0; i < node->inputs->size; ++i)
		{
			auto& tensor = context->tensors[node->inputs->data[i]];
			if (tensor.type != kTfLiteFloat32 && 
				tensor.type != kTfLiteInt32 &&
				tensor.type != kTfLiteInt8)
				return false;
		}
		return true;
	}
	TfLiteStatus MyDelegate::Initialize(TfLiteContext* context)
	{
		std::cout << std::endl << "Variables in MyDelegate::Initialize" << std::endl;
		custom_logger::LogTfLiteContext(context);
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
		return std::make_unique<MyDelegateKernel>();
	}
	SimpleDelegateInterface::Options MyDelegate::DelegateOptions() const
	{
		// Default options
		return SimpleDelegateInterface::Options();
	}

}