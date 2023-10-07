#pragma once

#define STRINGIFY(variable) (#variable)

#include <iostream>
#include <tensorflow/lite/core/c/common.h>
#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>
#include <tensorflow/lite/kernels/kernel_util.h>

namespace tflite {
	namespace custom_logger {
		// Extract the number of total elements inside a TfLiteIntArray
		int size_extraction(const TfLiteIntArray* dimensions);

		// Gets the string version of a TfLiteType
		const char* get_TfLiteType_string(const TfLiteType type);

		// Gets the string version of a builtin_code inside a TfLiteRegistration
		const char* get_builtin_code(const int builtin_code);

		// Logs a TfLiteContext
		void LogTfLiteContext(const TfLiteContext* const context);

		// Logs a TfLiteNode
		void LogTfLiteNode(const TfLiteNode* const node);

		// Logs a TfLiteRegistration
		void LogTfLiteRegistration(const TfLiteRegistration* const registration);

		// Logs a TfLiteDelegateParams
		void LogTfLiteDelegateParams(const TfLiteDelegateParams* const params);
	
		// Logs a TfLiteDelegate
		void LogTfLiteDelegate(const TfLiteDelegate* const delegate);
	
	}
}