#pragma once

namespace tflite {
    namespace custom_ops {
        namespace conv {

            template <KernelType kernel_type>
            TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node,
                TfLiteConvParams* params, OpData* data)
            {
                return Prepare(kernel_type, context, node, params, data);
            }

            template <KernelType kernel_type>
            void EvalQuantizedPerChannel(
                TfLiteContext* context, TfLiteNode* node,
                TfLiteConvParams* params, OpData* data,
                const TfLiteTensor* input,
                const TfLiteTensor* filter,
                const TfLiteTensor* bias, TfLiteTensor* output,
                TfLiteTensor* im2col, 
                const MyDelegateOptions& options)
            {
#if LOGGER
                //custom_logger::conv::LogTfLiteOpData(data);
                //custom_logger::LogTfLiteConvParams(params);
                //custom_logger::LogTfLiteTensor(*input);
                //custom_logger::LogTfLiteTensor(*output);
#endif // LOGGER

                ConvParams op_params;
                op_params.input_offset = -input->params.zero_point;
                op_params.output_offset = output->params.zero_point;
                op_params.stride_height = params->stride_height;
                op_params.stride_width = params->stride_width;
                op_params.dilation_height_factor = params->dilation_height_factor;
                op_params.dilation_width_factor = params->dilation_width_factor;
                op_params.padding_values.height = data->padding.height;
                op_params.padding_values.width = data->padding.width;
                op_params.quantized_activation_min = data->output_activation_min;
                op_params.quantized_activation_max = data->output_activation_max;

                KernelType effective_kernel_type = kernel_type;
                // We have to fallback to reference execution path when im2col is needed but
                // disabled because to-be-allocated temporary im2col tensor is too large.
                // See b/178743262 for the detailed motivation.
                if (data->im2col_oversized) 
                {
                    effective_kernel_type = kReference;
                }

                // Grouped convolution is right now only supported on reference kernel.
                if (data->groups != 1) 
                {
                    effective_kernel_type = kReference;
                }

                const int8_t* filter_data;

                // Only invalid for Int4
                filter_data = GetTensorData<int8>(filter);

                switch (effective_kernel_type) 
                {
                case kReference: {
                    switch (filter->type) 
                    {
                    case kTfLiteInt4:
                    case kTfLiteInt8: {
                        //reference_integer_ops::ConvPerChannel(
                        //    op_params, data->per_channel_output_multiplier.data(),
                        //    data->per_channel_output_shift.data(), GetTensorShape(input),
                        //    GetTensorData<int8>(input), GetTensorShape(filter), filter_data,
                        //    GetTensorShape(bias), GetTensorData<int32>(bias),
                        //    GetTensorShape(output), GetTensorData<int8>(output));
                        
                        //ConvPerChannel(
                        //    op_params, data->per_channel_output_multiplier.data(),
                        //    data->per_channel_output_shift.data(), GetTensorShape(input),
                        //    GetTensorData<int8>(input), GetTensorShape(filter), filter_data,
                        //    GetTensorShape(bias), GetTensorData<int32>(bias),
                        //    GetTensorShape(output), GetTensorData<int8>(output), options);

                        ConvPerChannelDisturbed(
                            op_params, 
                            data->per_channel_output_multiplier.data(),
                            data->per_channel_output_shift.data(), 
                            GetTensorShape(input), GetTensorData<int8>(input), 
                            GetTensorShape(filter), filter_data,
                            GetTensorShape(bias), GetTensorData<int32>(bias),
                            GetTensorShape(output), GetTensorData<int8>(output), 
                            options);

                        break;
                    }

                    default: {
                        TF_LITE_KERNEL_LOG(context,
                            "Weight type %s (%d) not supported for filter.",
                            TfLiteTypeGetName(filter->type), filter->type);
                        break;
                    }
                    }
                    break;
                }
                case kGenericOptimized:
                case kMultithreadOptimized:
                case kCblasOptimized:
                    switch (filter->type) 
                    {
                    case kTfLiteInt4:
                    case kTfLiteInt8: {
                        //optimized_integer_ops::ConvPerChannel(
                        //    op_params, data->per_channel_output_multiplier.data(),
                        //    data->per_channel_output_shift.data(), GetTensorShape(input),
                        //    GetTensorData<int8>(input), GetTensorShape(filter), filter_data,
                        //    GetTensorShape(bias), GetTensorData<int32>(bias),
                        //    GetTensorShape(output), GetTensorData<int8>(output),
                        //    GetTensorShape(im2col), GetTensorData<int8>(im2col),
                        //    CpuBackendContext::GetFromContext(context));
                        break;
                    }
                    default: {
                        TF_LITE_KERNEL_LOG(context,
                            "Weight type %s (%d) not supported for filter.",
                            TfLiteTypeGetName(filter->type), filter->type);
                        break;
                    }
                    }
                }
            }

            template <KernelType kernel_type, TfLiteType input_type>
            TfLiteStatus EvalImpl(
                TfLiteContext* context, TfLiteNode* node,
                TfLiteConvParams* params, OpData* data, 
                const MyDelegateOptions& options)
            {
                int bias_index = -1, filter_index = -1, input_index = -1;
                GetTensorIndexes(context, node, &bias_index, &filter_index, &input_index);

                TfLiteTensor* output;
                TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
                const TfLiteTensor* input;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, input_index, &input));
                const TfLiteTensor* filter;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, filter_index, &filter));
                bool has_bias = node->inputs->size == 3;
                const TfLiteTensor* bias = has_bias ? GetInput(context, node, bias_index) : nullptr;
                TfLiteTensor* im2col =
                    data->need_im2col
                    ? &context->tensors[node->temporaries->data[data->im2col_index]]
                    : nullptr;

                TFLITE_DCHECK_EQ(input_type, input->type);
                switch (input_type) 
                {  // Already know in/outtypes are same.
                case kTfLiteInt8:
                    EvalQuantizedPerChannel<kernel_type>(
                        context, node, params, data, input,
                        filter, bias, output, im2col, 
                        options);
                    break;
                default:
                    TF_LITE_KERNEL_LOG(context, "Type %s currently not supported.",
                        TfLiteTypeGetName(input->type));
                    return kTfLiteError;
                }
                return kTfLiteOk;
            }

            template <KernelType kernel_type>
            TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node, 
                TfLiteConvParams* params, OpData* data, 
                const MyDelegateOptions& options)
            {
                int bias_index = -1, filter_index = -1, input_index = -1;
                GetTensorIndexes(context, node, &bias_index, &filter_index, &input_index);

                const TfLiteTensor* input;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, input_index, &input));

                switch (input->type) 
                {
                case kTfLiteInt8:
                    return EvalImpl<kernel_type, kTfLiteInt8>(context, node, params, data, options);
                default:
                    TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                        TfLiteTypeGetName(input->type));
                    return kTfLiteError;
                }
            }

        }
    }
}
