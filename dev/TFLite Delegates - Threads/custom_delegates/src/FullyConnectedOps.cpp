#include "FullyConnectedOps.h"

namespace tflite {

    namespace custom_ops {

        namespace fully_connected {

            void* Init(TfLiteContext* context, const char* buffer, size_t length) 
            {
                // This is a builtin op, so we don't use the contents in 'buffer', if any.
                // Instead, we allocate a new object to carry information from Prepare() to
                // Eval().
                auto* op_data = new OpData();
                return op_data;
            }

            void Free(TfLiteContext* context, void* buffer) 
            {
                delete reinterpret_cast<OpData*>(buffer);
            }

            TfLiteStatus UpdateOutputSize(TfLiteContext* context,
                TfLiteFullyConnectedParams* params,
                const TfLiteTensor* input, TfLiteTensor* output,
                int batch_size, int num_units, int cols) 
            {
                TfLiteIntArray* output_size_array = nullptr;
                if (params->keep_num_dims) 
                {
                    TF_LITE_ENSURE_EQ(context, input->dims->data[input->dims->size - 1], cols);
                    output_size_array = TfLiteIntArrayCopy(input->dims);
                    output_size_array->data[output_size_array->size - 1] = num_units;
                }
                else 
                {
                    // Otherwise, the output is (potentially flattened to) a 2-D matrix.
                    output_size_array = TfLiteIntArrayCreate(2);
                    output_size_array->data[0] = batch_size;
                    output_size_array->data[1] = num_units;
                }
                return context->ResizeTensor(context, output, output_size_array);
            }

            TfLiteStatus PrepareImpl(TfLiteContext* context, 
                TfLiteNode* node,
                KernelType kernel_type,
                TfLiteFullyConnectedParams* params,
                OpData* data)
            {
                //auto* params = reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
                //OpData* data = reinterpret_cast<OpData*>(node->user_data);
                // Check we have all the inputs and outputs we need.
                TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
                // Shuffled formats need a workspace to store the shuffled input activations.
                const int expected_outputs_count =
                    params->weights_format == kTfLiteFullyConnectedWeightsFormatDefault ? 1 : 2;
                TF_LITE_ENSURE_EQ(context, node->outputs->size, expected_outputs_count);

                int bias_index = -1, filter_index = -1, input_index = -1;
                GetTensorIndexes(context, node, &bias_index, &filter_index, &input_index);

                const TfLiteTensor* input;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, input_index, &input));
                const TfLiteTensor* filter;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, filter_index, &filter));
                const TfLiteTensor* bias =
                    (node->inputs->size == 3)
                    ? GetOptionalInputTensor(context, node, bias_index)
                    : nullptr;
                TfLiteTensor* output;
                TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));

                // Check proper datatype match among all Input Tensors
                TF_LITE_ENSURE_STATUS(CheckTypes(context, input, filter, bias, output, params));

                // Check all the parameters of tensor match within themselves and match the
                // input configuration.
                int input_size = 1;
                for (int i = 0; i < input->dims->size; i++) 
                {
                    input_size *= input->dims->data[i];
                }

                TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 2);

                // When the second dimension size of the filter tensor is 0, we need to
                // generate the output shape early to avoid dividing by 0.
                if (filter->dims->data[1] == 0) 
                {
                    TfLiteIntArray* output_size_array;
                    if (params->keep_num_dims) 
                    {
                        output_size_array = TfLiteIntArrayCopy(input->dims);
                        output_size_array->data[output_size_array->size - 1] =
                            filter->dims->data[0];
                    }
                    else 
                    {
                        output_size_array = TfLiteIntArrayCreate(2);
                        // If `keep_num_dims` is false, we need to flatten the output tensor to
                        // have rank 2.
                        int batch_size = 1;
                        for (int i = 0; i < input->dims->size - 1; ++i)
                            batch_size *= input->dims->data[i];
                        output_size_array->data[0] = batch_size;
                        output_size_array->data[1] = filter->dims->data[0];
                    }
                    TF_LITE_ENSURE_OK(
                        context, context->ResizeTensor(context, output, output_size_array));
                    return kTfLiteOk;
                }

                const int batch_size = input_size / filter->dims->data[1];
                const int num_units = filter->dims->data[0];

                if (bias) 
                {
                    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 0));
                }

                // Note that quantized inference requires that all tensors have their
                // parameters set. This is usually done during quantized training.
                if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8 ||
                    input->type == kTfLiteInt16) 
                {
                    // Populate scalar quantization parameters.
                    double real_multiplier = 0.0;
                    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
                        context, input, filter, bias, output, &real_multiplier));
                    int exponent;
                    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
                    data->output_shift = exponent;

                    // Populate per-channel quantization parameters, if per-channel
                    // quantization.
                    TF_LITE_ENSURE_EQ(context, input->quantization.type, kTfLiteAffineQuantization);
                    TF_LITE_ENSURE_EQ(context, filter->quantization.type, kTfLiteAffineQuantization);
                    const auto* affine_quantization = 
                        reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
                    TF_LITE_ENSURE(context, affine_quantization);
                    TF_LITE_ENSURE(context, affine_quantization->scale);
                    const int per_channel_quantization_size = affine_quantization->scale->size;
                    const bool is_per_channel = per_channel_quantization_size > 1;
                    if (is_per_channel) {
                        //  Currently only Int8/Int16 is supported for per channel quantization.
                        TF_LITE_ENSURE(context,
                            input->type == kTfLiteInt8 || input->type == kTfLiteInt16);
                        TF_LITE_ENSURE(context, (filter->type == kTfLiteInt8));
                        TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                            per_channel_quantization_size);
                        TF_LITE_ENSURE_EQ(
                            context, per_channel_quantization_size,
                            filter->dims->data[affine_quantization->quantized_dimension]);
                        // Populate multiplier and shift using affine quantization.
                        const float input_scale = input->params.scale;
                        const float output_scale = output->params.scale;
                        const float* filter_scales = affine_quantization->scale->data;
                        data->per_channel_output_multiplier.resize(per_channel_quantization_size);
                        data->per_channel_output_shift.resize(per_channel_quantization_size);
                        int32_t* per_channel_multiplier =
                            data->per_channel_output_multiplier.data();
                        int32_t* per_channel_shift = data->per_channel_output_shift.data();
                        for (int i = 0; i < per_channel_quantization_size; ++i) 
                        {
                            const float scale = filter_scales[i];
                            const double filter_scale = static_cast<double>(scale);
                            const double effective_output_scale = static_cast<double>(input_scale) *
                                filter_scale /
                                static_cast<double>(output_scale);
                            int32_t significand;
                            int channel_shift;
                            QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
                            per_channel_multiplier[i] = significand;
                            per_channel_shift[i] = channel_shift;
                        }
                    }

                    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
                        context, params->activation, output, &data->output_activation_min,
                        &data->output_activation_max));
                }

                // Resize output.
                return UpdateOutputSize(context, params, input, output, batch_size, num_units,
                    filter->dims->data[1]);
            }

        }  // namespace fully_connected

    } // namespace custom_ops
} // namespace tflite
