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
            void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                TfLiteConvParams* params, OpData* data,
                const TfLiteTensor* input, const TfLiteTensor* filter,
                const TfLiteTensor* bias, TfLiteTensor* im2col,
                TfLiteTensor* output) {
                auto input_offset = -input->params.zero_point;
                auto filter_offset = -filter->params.zero_point;
                auto output_offset = output->params.zero_point;

                KernelType effective_kernel_type;
                if ((kernel_type == kMultithreadOptimized ||
                    kernel_type == kCblasOptimized) &&
                    (params->dilation_width_factor != 1 ||
                        params->dilation_height_factor != 1)) {
                    // kMultithreadOptimized and kCblasOptimized do not support dilation.
                    // Therefore, fallback to optimized.
                    effective_kernel_type = kGenericOptimized;
                }
                else {
                    effective_kernel_type = kernel_type;
                }

                // We have to fallback to reference execution path when im2col is needed but
                // disabled because to-be-allocated temporary im2col tensor is too large.
                // See b/178743262 for the detailed motivation.
                if (data->im2col_oversized) {
                    effective_kernel_type = kReference;
                }

                // Grouped convolution is right now only supported on reference kernel.
                if (data->groups != 1) {
                    effective_kernel_type = kReference;
                }

                ConvParams op_params;
                op_params.padding_type = PaddingType::kSame;
                op_params.padding_values.width = data->padding.width;
                op_params.padding_values.height = data->padding.height;
                op_params.dilation_width_factor = params->dilation_width_factor;
                op_params.dilation_height_factor = params->dilation_height_factor;
                op_params.stride_width = params->stride_width;
                op_params.stride_height = params->stride_height;
                op_params.input_offset = input_offset;
                op_params.weights_offset = filter_offset;
                op_params.output_offset = output_offset;
                op_params.output_multiplier = data->output_multiplier;
                op_params.output_shift = -data->output_shift;
                op_params.quantized_activation_min = data->output_activation_min;
                op_params.quantized_activation_max = data->output_activation_max;
                switch (effective_kernel_type) {
                case kReference: {
                    reference_ops::Conv(
                        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
                        GetTensorShape(filter), GetTensorData<uint8_t>(filter),
                        GetTensorShape(bias), GetTensorData<int32_t>(bias),
                        GetTensorShape(output), GetTensorData<uint8_t>(output),
                        GetTensorShape(im2col), GetTensorData<uint8_t>(im2col),
                        /* cpu_backend_context = */ nullptr);
                    break;
                }
                case kGenericOptimized:
                case kMultithreadOptimized:
                case kCblasOptimized: {
                    // There is only one optimized implementation for Quantized Conv.
                    //optimized_ops::Conv(
                    //    op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
                    //    GetTensorShape(filter), GetTensorData<uint8_t>(filter),
                    //    GetTensorShape(bias), GetTensorData<int32_t>(bias),
                    //    GetTensorShape(output), GetTensorData<uint8_t>(output),
                    //    GetTensorShape(im2col), GetTensorData<uint8_t>(im2col),
                    //    CpuBackendContext::GetFromContext(context));
                    break;
                }
                }
            }

            template <KernelType kernel_type>
            void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                TfLiteConvParams* params, OpData* data,
                const TfLiteTensor* input,
                const TfLiteTensor* filter,
                const TfLiteTensor* bias, TfLiteTensor* output,
                TfLiteTensor* im2col, const MyDelegateOptions& options)
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
                const size_t bytes_unpacked = filter->bytes * 2;
                auto unpacked_filter_data = std::make_unique<int8_t[]>(bytes_unpacked);

                if (filter->type == kTfLiteInt4) 
                {
                    tflite::tensor_utils::UnpackDenseInt4IntoInt8(
                        GetTensorData<int8_t>(filter), GetTensorShape(filter).FlatSize(),
                        unpacked_filter_data.get());
                    filter_data = unpacked_filter_data.get();
                }
                else
                {
                    filter_data = GetTensorData<int8>(filter);
                }

                switch (effective_kernel_type) 
                {
                case kReference: {
                    switch (filter->type) 
                    {
                    case kTfLiteInt4:
                    case kTfLiteInt8: {
                        //reference_integer_ops::ConvPerChannel(
                        ConvPerChannel(
                            op_params, data->per_channel_output_multiplier.data(),
                            data->per_channel_output_shift.data(), GetTensorShape(input),
                            GetTensorData<int8>(input), GetTensorShape(filter), filter_data,
                            GetTensorShape(bias), GetTensorData<int32>(bias),
                            //GetTensorShape(output), GetTensorData<int8>(output));
                            GetTensorShape(output), GetTensorData<int8>(output), options);
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
                        optimized_integer_ops::ConvPerChannel(
                            op_params, data->per_channel_output_multiplier.data(),
                            data->per_channel_output_shift.data(), GetTensorShape(input),
                            GetTensorData<int8>(input), GetTensorShape(filter), filter_data,
                            GetTensorShape(bias), GetTensorData<int32>(bias),
                            GetTensorShape(output), GetTensorData<int8>(output),
                            GetTensorShape(im2col), GetTensorData<int8>(im2col),
                            CpuBackendContext::GetFromContext(context));
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

            template <KernelType kernel_type>
            void EvalQuantizedPerChannel16x8(TfLiteContext* context, TfLiteNode* node,
                TfLiteConvParams* params, OpData* data,
                const TfLiteTensor* input,
                const TfLiteTensor* filter,
                const TfLiteTensor* bias, TfLiteTensor* output,
                TfLiteTensor* im2col) {
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
                if (data->im2col_oversized) {
                    effective_kernel_type = kReference;
                }

                // Grouped convolution is right now only supported on reference kernel.
                if (data->groups != 1) {
                    effective_kernel_type = kReference;
                }

                // To prevent 32bit accum overflow for 16x8 quantization, it enables the
                // optimized path only when zero_point is 0.
                bool has_non_zero_point = input->params.zero_point ||
                    filter->params.zero_point ||
                    output->params.zero_point;

                if (data->quantized_bias_type == kTfLiteInt32) {
                    if (effective_kernel_type == kReference || has_non_zero_point) {
                        reference_integer_ops::ConvPerChannel(
                            op_params, data->per_channel_output_multiplier.data(),
                            data->per_channel_output_shift.data(), GetTensorShape(input),
                            GetTensorData<int16>(input), GetTensorShape(filter),
                            GetTensorData<int8>(filter), GetTensorShape(bias),
                            GetTensorData<int32_t>(bias), GetTensorShape(output),
                            GetTensorData<int16>(output));
                    }
                    else {
                        //optimized_integer_ops::ConvPerChannel(
                        //    op_params, data->per_channel_output_multiplier.data(),
                        //    data->per_channel_output_shift.data(), GetTensorShape(input),
                        //    GetTensorData<int16_t>(input), GetTensorShape(filter),
                        //    GetTensorData<int8_t>(filter), GetTensorShape(bias),
                        //    GetTensorData<int32_t>(bias), GetTensorShape(output),
                        //    GetTensorData<int16_t>(output), GetTensorShape(im2col),
                        //    GetTensorData<int16_t>(im2col),
                        //    CpuBackendContext::GetFromContext(context));
                    }
                }
                else {
                    TFLITE_DCHECK(!has_non_zero_point);
                    // Fallback to reference kernel when bias_type is int64 as
                    // there is no optimized kernel for int64 bias yet.
                    reference_integer_ops::ConvPerChannel(
                        op_params, data->per_channel_output_multiplier.data(),
                        data->per_channel_output_shift.data(), GetTensorShape(input),
                        GetTensorData<int16>(input), GetTensorShape(filter),
                        GetTensorData<int8>(filter), GetTensorShape(bias),
                        GetTensorData<int64_t>(bias), GetTensorShape(output),
                        GetTensorData<int16>(output));
                }
            }

            template <KernelType kernel_type>
            void EvalFloat(TfLiteContext* context, TfLiteNode* node,
                TfLiteConvParams* params, OpData* data,
                const TfLiteTensor* input, const TfLiteTensor* filter,
                const TfLiteTensor* bias, TfLiteTensor* im2col,
                TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
                float output_activation_min, output_activation_max;
                CalculateActivationRange(params->activation, &output_activation_min,
                    &output_activation_max);
                KernelType effective_kernel_type = kernel_type;
                // Fall back to the optimized path if multi-threaded conv is unsupported.
                if ((kernel_type == kMultithreadOptimized) &&
                    !data->supports_multithreaded_kernel) {
                    effective_kernel_type = kGenericOptimized;
                }

                // When im2col is needed (which is implied when 'im2col_oversized' is true),
                // the GEMMM-based optimized path requires im2col data be allocated to ensure
                // the correctness. Therefore, when im2col is disabled because of the
                // oversized temporary im2col tensor, fallback to a non-optimized path is
                // needed.
                // See b/178743262 for the detailed motivation.
                if (data->im2col_oversized) {
                    effective_kernel_type = kReference;
                }

                // Grouped convolution is right now only supported on reference kernel.
                if (data->groups != 1) {
                    effective_kernel_type = kReference;
                }

                ConvParams op_params;
                op_params.padding_type = RuntimePaddingType(params->padding);
                op_params.padding_values.width = data->padding.width;
                op_params.padding_values.height = data->padding.height;
                op_params.stride_width = params->stride_width;
                op_params.stride_height = params->stride_height;
                op_params.dilation_width_factor = params->dilation_width_factor;
                op_params.dilation_height_factor = params->dilation_height_factor;
                op_params.float_activation_min = output_activation_min;
                op_params.float_activation_max = output_activation_max;
                switch (effective_kernel_type) {
                case kReference: {
                    reference_ops::Conv(op_params, GetTensorShape(input),
                        GetTensorData<float>(input), GetTensorShape(filter),
                        GetTensorData<float>(filter), GetTensorShape(bias),
                        GetTensorData<float>(bias), GetTensorShape(output),
                        GetTensorData<float>(output), GetTensorShape(im2col),
                        GetTensorData<float>(im2col));
                    break;
                }
                case kCblasOptimized:
                case kGenericOptimized: {
                    //optimized_ops::Conv(op_params, GetTensorShape(input),
                    //    GetTensorData<float>(input), GetTensorShape(filter),
                    //    GetTensorData<float>(filter), GetTensorShape(bias),
                    //    GetTensorData<float>(bias), GetTensorShape(output),
                    //    GetTensorData<float>(output), GetTensorShape(im2col),
                    //    GetTensorData<float>(im2col),
                    //    CpuBackendContext::GetFromContext(context));
                    break;
                }
                case kMultithreadOptimized: {
                    // See Register_CONV_2D: we should never be here when TFLITE_WITH_RUY
                    // was enabled. We #if out this code in order to get the corresponding
                    // binary size benefits.
                    TFLITE_DCHECK(false);
                }
                }
            }

            template <KernelType kernel_type>
            TfLiteStatus EvalHybridPerChannel(TfLiteContext* context, TfLiteNode* node,
                TfLiteConvParams* params, OpData* data,
                const TfLiteTensor* input,
                const TfLiteTensor* filter,
                const TfLiteTensor* bias,
                TfLiteTensor* im2col, TfLiteTensor* output) {
                float output_activation_min, output_activation_max;
                CalculateActivationRange(params->activation, &output_activation_min,
                    &output_activation_max);

                const int batch_size = SizeOfDimension(input, 0);
                TF_LITE_ENSURE(context, batch_size != 0);
                const int input_size = NumElements(input) / batch_size;
                TfLiteTensor* quantized_input_tensor;
                TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_quantized_index,
                        &quantized_input_tensor));
                int8_t* quantized_input_ptr_batch =
                    GetTensorData<int8_t>(quantized_input_tensor);
                TfLiteTensor* scaling_factors_tensor;
                TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->scaling_factors_index,
                        &scaling_factors_tensor));
                float* scaling_factors_ptr = GetTensorData<float>(scaling_factors_tensor);
                TfLiteTensor* input_offset_tensor;
                TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_offset_index,
                        &input_offset_tensor));
                int32_t* input_offset_ptr = GetTensorData<int32_t>(input_offset_tensor);

                for (int b = 0; b < batch_size; ++b) {
                    const int offset = b * input_size;
                    tensor_utils::AsymmetricQuantizeFloats(
                        GetTensorData<float>(input) + offset, input_size,
                        quantized_input_ptr_batch + offset, &scaling_factors_ptr[b],
                        &input_offset_ptr[b]);
                }

                int8_t* im2col_ptr = nullptr;
                int8_t* filter_ptr = nullptr;
                if (im2col != nullptr) {
                    im2col_ptr = im2col->data.int8;
                }
                filter_ptr = filter->data.int8;
                const auto* affine_quantization =
                    reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);

                KernelType effective_kernel_type = kernel_type;
                // We have to fallback to reference execution path when im2col is needed but
                // disabled because to-be-allocated temporary im2col tensor is too large.
                // See b/178743262 for the detailed motivation.
                if (data->im2col_oversized) {
                    effective_kernel_type = kReference;
                }

                // Grouped convolution is right now only supported on reference kernel.
                if (data->groups != 1) {
                    effective_kernel_type = kReference;
                }

                ConvParams op_params;
                op_params.padding_type = PaddingType::kSame;
                op_params.padding_values.width = data->padding.width;
                op_params.padding_values.height = data->padding.height;
                op_params.dilation_width_factor = params->dilation_width_factor;
                op_params.dilation_height_factor = params->dilation_height_factor;
                op_params.stride_width = params->stride_width;
                op_params.stride_height = params->stride_height;
                op_params.float_activation_min = output_activation_min;
                op_params.float_activation_max = output_activation_max;
                switch (effective_kernel_type) {
                case kReference:
                    reference_ops::HybridConvPerChannel(
                        op_params, scaling_factors_ptr, GetTensorShape(input),
                        quantized_input_ptr_batch, GetTensorShape(filter), filter_ptr,
                        GetTensorShape(bias), GetTensorData<float>(bias),
                        GetTensorShape(output), GetTensorData<float>(output),
                        GetTensorShape(im2col), im2col_ptr, affine_quantization->scale->data,
                        input_offset_ptr);
                    break;
                case kGenericOptimized:
                case kMultithreadOptimized:
                case kCblasOptimized: {
                    //TfLiteTensor* row_sums;
                    //TF_LITE_ENSURE_OK(
                    //    context,
                    //    GetTemporarySafe(context, node, data->row_sums_index, &row_sums));
                    //TfLiteTensor* scratch;
                    //TF_LITE_ENSURE_OK(
                    //    context,
                    //    GetTemporarySafe(context, node, data->accum_scratch_index, &scratch));
                    //optimized_ops::HybridConvPerChannel(
                    //    op_params, scaling_factors_ptr, GetTensorShape(input),
                    //    quantized_input_ptr_batch, GetTensorShape(filter), filter_ptr,
                    //    GetTensorShape(bias), GetTensorData<float>(bias),
                    //    GetTensorShape(output), GetTensorData<float>(output),
                    //    GetTensorShape(im2col), im2col_ptr, affine_quantization->scale->data,
                    //    input_offset_ptr, GetTensorShape(scratch),
                    //    GetTensorData<int32>(scratch), GetTensorData<int32_t>(row_sums),
                    //    &data->compute_hybrid_row_sums,
                    //    CpuBackendContext::GetFromContext(context));
                    //data->compute_hybrid_row_sums = false;
                    break;
                }
                }

                return kTfLiteOk;
            }

            template <KernelType kernel_type>
            TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                TfLiteConvParams* params, OpData* data,
                const TfLiteTensor* input, const TfLiteTensor* filter,
                const TfLiteTensor* bias, TfLiteTensor* im2col,
                TfLiteTensor* accum_scratch, TfLiteTensor* output) {
                float output_activation_min, output_activation_max;
                CalculateActivationRange(params->activation, &output_activation_min,
                    &output_activation_max);

                const int batch_size = SizeOfDimension(input, 0);
                TF_LITE_ENSURE(context, batch_size != 0);
                const int input_size = NumElements(input) / batch_size;

                const float* input_ptr = GetTensorData<float>(input);
                TfLiteTensor* quantized_input_tensor;
                TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_quantized_index,
                        &quantized_input_tensor));
                int8_t* quantized_input_ptr_batch =
                    GetTensorData<int8_t>(quantized_input_tensor);
                TfLiteTensor* scaling_factors_tensor;
                TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->scaling_factors_index,
                        &scaling_factors_tensor));
                float* scaling_factors_ptr = GetTensorData<float>(scaling_factors_tensor);

                // Per-batch input quantization for higher accuracy.
                {
                    // To run this line you need to include ruy/profiler/instrumentation.h
                    // Which is useless in non multithreaded
                    //ruy::profiler::ScopeLabel label("ConvHybridQuantizeInputs");
                    for (int b = 0; b < batch_size; ++b) {
                        float unused_min, unused_max;
                        const int offset = b * input_size;
                        tensor_utils::SymmetricQuantizeFloats(
                            input_ptr + offset, input_size, quantized_input_ptr_batch + offset,
                            &unused_min, &unused_max, &scaling_factors_ptr[b]);
                        scaling_factors_ptr[b] *= filter->params.scale;
                    }
                }

                switch (kernel_type) {
                case kReference:
                case kGenericOptimized:
                case kMultithreadOptimized:
                case kCblasOptimized: {
                    // There is only one implementation for hybrid kernel.
                    ConvParams op_params;
                    op_params.padding_type = PaddingType::kSame;
                    op_params.padding_values.width = data->padding.width;
                    op_params.padding_values.height = data->padding.height;
                    op_params.stride_width = params->stride_width;
                    op_params.stride_height = params->stride_height;
                    op_params.dilation_width_factor = params->dilation_width_factor;
                    op_params.dilation_height_factor = params->dilation_height_factor;
                    op_params.float_activation_min = output_activation_min;
                    op_params.float_activation_max = output_activation_max;
                    if (data->groups == 1) {
                        //optimized_ops::HybridConv(
                        //    op_params, scaling_factors_ptr, GetTensorShape(input),
                        //    quantized_input_ptr_batch, GetTensorShape(filter),
                        //    GetTensorData<int8_t>(filter), GetTensorShape(bias),
                        //    GetTensorData<float>(bias), GetTensorShape(accum_scratch),
                        //    GetTensorData<int32_t>(accum_scratch), GetTensorShape(output),
                        //    GetTensorData<float>(output), GetTensorShape(im2col),
                        //    GetTensorData<int8_t>(im2col),
                        //    CpuBackendContext::GetFromContext(context));
                    }
                    else {
                        // This case is handled by (fallbacked to) per channel hybrid group conv
                        // and shouldn't hit this branch.
                        TF_LITE_KERNEL_LOG(
                            context,
                            "Group convolution currently not supported for hybrid kernel.");
                        return kTfLiteError;
                    }
                    break;
                }
                }

                return kTfLiteOk;
            }

            template <KernelType kernel_type, TfLiteType input_type>
            TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                TfLiteConvParams* params, OpData* data, const MyDelegateOptions& options)
            {
                //auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
                //OpData* data = reinterpret_cast<OpData*>(node->user_data);

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
                TfLiteTensor* hwcn_weights =
                    data->need_hwcn_weights
                    ? &context->tensors[node->temporaries->data[data->hwcn_weights_index]]
                    : nullptr;

                if (data->need_hwcn_weights && !data->have_weights_been_transposed) 
                {
                    TransposeFloatTensor(filter, hwcn_weights);
                    data->have_weights_been_transposed = true;
                }

                TFLITE_DCHECK_EQ(input_type, input->type);
                switch (input_type) 
                {  // Already know in/outtypes are same.
                case kTfLiteFloat32:
                    if (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8) 
                    {
                        if (data->is_hybrid_per_channel ||
                            // TODO(b/162870360): Fallback to PerChannel implementation
                            // before we have grouped hybrid convolution.
                            data->groups != 1) 
                        {
                            TF_LITE_ENSURE_OK(context, EvalHybridPerChannel<kernel_type>(
                                context, node, params, data, input,
                                filter, bias, im2col, output));
                        }
                        else 
                        {
                            TfLiteTensor* accum_scratch =
                                &context->tensors[node->temporaries
                                ->data[data->accum_scratch_index]];
                            TF_LITE_ENSURE_OK(context,
                                EvalHybrid<kernel_type>(context, node, params, data,
                                    input, filter, bias, im2col,
                                    accum_scratch, output));
                        }
                    }
                    else 
                    {
                        EvalFloat<kernel_type>(context, node, params, data, input, filter, bias,
                            im2col, hwcn_weights, output);
                    }
                    break;
                case kTfLiteUInt8:
                    EvalQuantized<kernel_type>(context, node, params, data, input, filter,
                        bias, im2col, output);
                    break;
                case kTfLiteInt8:
                    EvalQuantizedPerChannel<kernel_type>(context, node, params, data, input,
                        filter, bias, output, im2col, options);
                    break;
                case kTfLiteInt16:
                    EvalQuantizedPerChannel16x8<kernel_type>(
                        context, node, params, data, input, filter, bias, output, im2col);
                    break;
                default:
                    TF_LITE_KERNEL_LOG(context, "Type %s currently not supported.",
                        TfLiteTypeGetName(input->type));
                    return kTfLiteError;
                }
                return kTfLiteOk;
            }

            template <KernelType kernel_type>
            TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node, TfLiteConvParams* params, OpData* data, const MyDelegateOptions& options)
            {
                int bias_index = -1, filter_index = -1, input_index = -1;
                GetTensorIndexes(context, node, &bias_index, &filter_index, &input_index);

                const TfLiteTensor* input;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, input_index, &input));

                switch (input->type) 
                {
                case kTfLiteFloat32:
                    return EvalImpl<kernel_type, kTfLiteFloat32>(context, node, params, data, options);
                case kTfLiteUInt8:
                    return EvalImpl<kernel_type, kTfLiteUInt8>(context, node, params, data, options);
                case kTfLiteInt8:
                    return EvalImpl<kernel_type, kTfLiteInt8>(context, node, params, data, options);
                case kTfLiteInt16:
                    return EvalImpl<kernel_type, kTfLiteInt16>(context, node, params, data, options);
                default:
                    TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                        TfLiteTypeGetName(input->type));
                    return kTfLiteError;
                }
            }

        }
    }
}
