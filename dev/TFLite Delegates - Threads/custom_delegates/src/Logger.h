#pragma once

#define STRINGIFY(variable) #variable

#include <iostream>
#include <string>
#include <vector>
#include <tensorflow/lite/core/c/common.h>
#include <tensorflow/lite/core/c/builtin_op_data.h>
#include <tensorflow/lite/builtin_ops.h>

#include "ConvOps.h"

namespace tflite {

	namespace custom_logger {

		// Namespace for copied convolutional instances for logging purposes
		namespace conv {

			// Gets the Kernel Type
			std::string get_TfLiteKernelType(const custom_ops::conv::KernelType);

			// Gets the string version of OpData
			void LogTfLiteOpData(const custom_ops::conv::OpData* const data);

		}

		// Gets the string version of TfLiteDelegateFlags
		std::string get_TfLiteDelegateFlags(const TfLiteDelegateFlags);

		// Gets the string version of TfLiteStatus
		std::string get_TfLiteStatus(const TfLiteStatus);

		// Gets the string version of a TfLiteType
		std::string get_TfLiteType(const TfLiteType type);

		// Gets the string version of a builtin_code inside a TfLiteRegistration
		std::string get_builtin_code(const int builtin_code);

		// Gets the string version of the TfLitePaddding
		std::string get_TfLitePadding(const TfLitePadding padding);

		// Gets the string version of the TfLiteFusedActivation
		std::string get_TfLiteFusedActivation(const TfLiteFusedActivation activation);

		// Gets the string version of the TfLiteAllocationType
		std::string get_TfLiteAllocationType(const TfLiteAllocationType allocation_type);

		// Gets the string version of the TfLiteQuantizationType
		std::string get_TfLiteQuantizationType(const TfLiteQuantizationType type);

		// Logs a TfLiteAffineQuantization
		void LogTfLiteAffineQuantization(const TfLiteAffineQuantization* const affine_quantization);

		// Logs a TfLiteQuantization
		void LogTfLiteQuantization(const TfLiteQuantization& quantization);

		// Logs a TfLiteQuantizationParams
		void LogTfLiteQuantizationParams(const TfLiteQuantizationParams& params);

		// Logs a TfLitePaddingValues
		void LogTfLitePaddingValues(const TfLitePaddingValues& padding);

		// Logs a TfLiteTensor
		void LogTfLiteTensor(const TfLiteTensor& tensor);

		// Logs a TfLiteContext
		void LogTfLiteContext(const TfLiteContext* const context);

		// Logs a TfLiteConvParams
		void LogTfLiteConvParams(const TfLiteConvParams* const params);

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