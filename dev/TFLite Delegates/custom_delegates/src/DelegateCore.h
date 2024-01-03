#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <tensorflow/lite/delegates/utils/simple_delegate.h>
#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/kernels/kernel_util.h>
#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>

#include "Options.h"
#include "ConvOps.h"
#include "Logger.h"

namespace tflite {

	// MyDelegateKernel
	// Each instance represents a single part of the graph (subgraph).
	class MyDelegateKernel : public SimpleDelegateKernelInterface
	{
	public:
		// MyDelegateKernel constructor
		MyDelegateKernel();

		// MyDelegateKernel constructor
		MyDelegateKernel(const MyDelegateOptions& options);

		// MyDelegateKernel destructor
		~MyDelegateKernel();
		
		// Initializes a delegated subgraph.
		// The nodes in the subgraph are inside TfLiteDelegateParams->nodes_to_replace
		TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params) override;
		
		// Will be called by the framework. Should handle any needed preparation
		// for the subgraph e.g. allocating buffers, compiling model.
		// Returns status, and signalling any errors.
		TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override;
		
		// Actual subgraph inference should happen on this call.
		// Returns status, and signalling any errors.
		// NOTE: Tensor data pointers (tensor->data) can change every inference, so
		// the implementation of this method needs to take that into account.
		TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override;
	private:
		// Holds the indices of the input/output tensors.
		// inputs_[i] is list of all input tensors to node at index 'i'.
		// outputs_[i] is list of all output tensors to node at index 'i'.
		std::vector<std::vector<int>> inputs_, outputs_;
		
		// Holds the builtin code of the ops.
		// builtin_code_[i] is the type of node at index 'i'
		std::vector<int> builtin_code_;

		// MyDelegateOptions to determine the behaviour of the delegate
		MyDelegateOptions options_;

		// Operation Data, later explore the possibility to store it as a vector
		custom_ops::conv::OpData* operation_data_;

		// Convolution Parameters, convert it to a vector
		TfLiteConvParams* conv_params_;

		// Prepared flag
		bool prepared_ = false;

		// Steals the Operation data from the to-be-replaced node
		void GetOperationData(const custom_ops::conv::OpData&);

		// Steals the Convolution Parameters from the to-be-replaced node
		void GetConvParams(const TfLiteConvParams&);

	};

	// MyDelegate
	// It represents a delegate's capabilities and provides a factory for MyDelegateKernel.
	class MyDelegate : public SimpleDelegateInterface
	{
	public:
		// MyDelegate constructor
		MyDelegate();

		// MyDelegate constructor with MyDelegateOptions
		MyDelegate(const MyDelegateOptions&);

		// MyDelegate destructor
		~MyDelegate();

		// Returns true if 'node' is supported by the delegate. False otherwise.
		bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
			const TfLiteNode* node,
			TfLiteContext* context) const override;
		
		// Initialize the delegate before finding and replacing TfLite nodes with
		// delegate kernels, for example, retrieving some TFLite settings from
		// 'context'.
		TfLiteStatus Initialize(TfLiteContext* context) override;
		
		// Returns a name that identifies the delegate.
		// This name is used for debugging/logging/profiling.
		const char* Name() const override;
		
		// Returns instance of an object that implements the interface
		// SimpleDelegateKernelInterface.
		// An instance of SimpleDelegateKernelInterface represents one subgraph to
		// be delegated.
		// Caller takes ownership of the returned object.
		std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface() override;
		
		// Returns SimpleDelegateInterface::Options which has delegate properties
		// relevant for graph partitioning.
		SimpleDelegateInterface::Options DelegateOptions() const override;

	private:
		// MyDelegateOptions to determine the behaviour of MyDelegate and MyDelegateKernel
		MyDelegateOptions options_;
	};

}