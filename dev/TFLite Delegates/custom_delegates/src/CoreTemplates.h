#pragma once

#include "MyDelegateCore.h"

namespace tflite {

	template<typename T>
	TfLiteStatus MyDelegateKernel::ComputeAddResult(TfLiteContext* context, int builtin_code,
		const TfLiteTensor* input_tensor_1,
		const TfLiteTensor* input_tensor_2,
		TfLiteTensor* output_tensor)
	{
		if (NumElements(input_tensor_1) != NumElements(input_tensor_2) ||
			NumElements(input_tensor_1) != NumElements(output_tensor)) {
			return kTfLiteDelegateError;
		}

		// This code assumes no activation, and no broadcasting needed (both inputs
		// have the same size).
		// Assumes inputs and outputs are of the same type

		auto* input_1 = GetTensorData<T>(input_tensor_1);
		auto* input_2 = GetTensorData<T>(input_tensor_2);
		auto* output = GetTensorData<T>(output_tensor);

		for (int i = 0; i < NumElements(input_tensor_1); ++i) {
			if (builtin_code == kTfLiteBuiltinAdd)
				output[i] = input_1[i] + input_2[i];
			else
				output[i] = input_1[i] - input_2[i];
		}

		return kTfLiteOk;
	}
}