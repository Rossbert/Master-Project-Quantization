#pragma once

#include <iostream>
#include <vector>
#include <bitset>
#include <thread>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#include "Options.h"

namespace tflite {

	namespace custom_ops {

		namespace fully_connected {

            // This file has four implementations of FullyConnected
            enum KernelType {
                kReference,
                kGenericOptimized,
                kLegacyPie,  // Legacy path used by the PIE team and related clients.
            };

            struct OpData {
                // The scaling factor from input to output (aka the 'real multiplier') can
                // be represented as a fixed point multiplier plus a left shift.
                int32_t output_multiplier;
                int output_shift;
                // Per channel output multiplier and shift.
                std::vector<int32_t> per_channel_output_multiplier;
                std::vector<int> per_channel_output_shift;
                // The range of the fused activation layer. For example for kNone and
                // uint8_t these would be 0 and 255.
                int32_t output_activation_min;
                int32_t output_activation_max;
                // The index of the temporary tensor where the quantized inputs are cached.
                int scratch_tensor_index;
                bool compute_row_sums = false;
                // Only used for sparse hybrid fully connected kernels.
                bool ledger_initialized;
                // Used for 4bit hybrid
                //std::unique_ptr<optimized_4bit::OpData4Bit> op_data_4bit = nullptr;
                TfLiteType quantized_bias_type = kTfLiteNoType;
            };

            constexpr int kInputTensor = 0;
            constexpr int kWeightsTensor = 1;
            constexpr int kBiasTensor = 2;
            constexpr int kOutputTensor = 0;
            constexpr int kShuffledInputWorkspaceTensor = 1;

            // Begin temporary tensor ids created at init and initialized during prepare.
            constexpr int kQuantizedInputTensor = 0;
            constexpr int kScalingFactorsTensor = 1;
            constexpr int kAccumulatorTensor = 2;
            constexpr int kInputOffsetsTensor = 3;

            void* Init(TfLiteContext* context, const char* buffer, size_t length);

            void Free(TfLiteContext* context, void* buffer);

            TfLiteStatus PrepareImpl(TfLiteContext* context,
                TfLiteNode* node,
                KernelType kernel_type,
                TfLiteFullyConnectedParams* params,
                OpData* data);

            inline TfLiteStatus CheckTypes(TfLiteContext* context,
                const TfLiteTensor* input,
                const TfLiteTensor* filter,
                const TfLiteTensor* bias, TfLiteTensor* output,
                TfLiteFullyConnectedParams* params) 
            {
                const bool is_quantized =
                    ((filter->type == kTfLiteUInt8) || (filter->type == kTfLiteInt8) ||
                        (filter->type == kTfLiteInt4));
                const bool is_hybrid = is_quantized && (input->type == kTfLiteFloat32);
                const bool is_shuffled =
                    is_quantized && (params->weights_format ==
                        kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8);

                // optional bias tensor.
                const bool is_optional_bias_float = !bias || (bias->type == kTfLiteFloat32);
                const bool is_optional_bias_int =
                    !bias || (bias->type == kTfLiteInt32) || (bias->type == kTfLiteInt64);

                if (is_quantized) 
                {
                    if (is_shuffled) 
                    {
                        TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteUInt8);
                        TF_LITE_ENSURE_TYPES_EQ(context, filter->type, kTfLiteUInt8);
                        TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt16);
                        TF_LITE_ENSURE_EQ(context, is_optional_bias_int, true);
                    }
                    else if (is_hybrid) 
                    {
                        TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
                        TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
                        TF_LITE_ENSURE_EQ(context, is_optional_bias_float, true);
                    }
                    else 
                    {
                        TF_LITE_ENSURE(context, input->type == kTfLiteUInt8 ||
                            input->type == kTfLiteInt8 ||
                            input->type == kTfLiteInt16);
                        TF_LITE_ENSURE(context, output->type == kTfLiteUInt8 ||
                            output->type == kTfLiteInt8 ||
                            output->type == kTfLiteInt16);
                        TF_LITE_ENSURE_EQ(context, is_optional_bias_int, true);
                    }
                }
                else 
                {
                    // Only float32 is supported currently
                    TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
                    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
                    TF_LITE_ENSURE_TYPES_EQ(context, filter->type, kTfLiteFloat32);
                    TF_LITE_ENSURE_EQ(context, is_optional_bias_float, true);
                }

                return kTfLiteOk;
            }

		} // fully_connected

        // Gets the input, filter, and output indexes if the order of tensor inputs is mixed
        void GetTensorIndexes(TfLiteContext* context, TfLiteNode* node,
            int* bias_index, int* filter_index, int* input_index);

	} // namespace custom_ops

} // namespace tflite


#include "FullyConnectedTemplates.h"