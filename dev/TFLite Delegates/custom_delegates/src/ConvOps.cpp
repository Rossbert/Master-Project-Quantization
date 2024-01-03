#include "ConvOps.h"
#include "Logger.h"
namespace tflite {

    namespace custom_ops {

        namespace conv {

            void* Init(TfLiteContext* context, const char* buffer, size_t length) 
            {
                // This based on a builtin op, so data in buffer is not neeeded
                auto* data = new OpData;
                return data;
            }

            void Free(TfLiteContext* context, void* buffer) 
            {
                delete reinterpret_cast<OpData*>(buffer);
            }

            void TransposeFloatTensor(const TfLiteTensor* input, TfLiteTensor* output) 
            {
                const int rows = output->dims->data[1];
                const int cols = output->dims->data[0];
                const float* input_data = GetTensorData<float>(input);
                float* output_data = GetTensorData<float>(output);
                for (int i = 0; i < rows; ++i) 
                {
                    for (int j = 0; j < cols; ++j) 
                    {
                        const float in_value = input_data[i * cols + j];
                        output_data[j * rows + i] = in_value;
                    }
                }
            }

            void GetTensorIndexes(TfLiteContext* context, TfLiteNode* node,
                int* bias_index, int* filter_index, int* input_index)
            {
                for (int i = 0; i < node->inputs->size; i++)
                {
                    auto& tensor = context->tensors[node->inputs->data[i]];
                    // Input tensor has a reference to dimension signature
                    if (tensor.dims_signature != nullptr)
                        *input_index = i;
                    else
                    {
                        // Filter tensor must not have dimension = 1 (only when bias is present)
                        if (tensor.dims->size != 1)
                            *filter_index = i;
                        else
                            *bias_index = i;
                    }
                }
            }

            bool IsIm2ColRequired(const TfLiteTensor* input, TfLiteConvParams* params,
                const TfLiteTensor* filter, OpData* data, bool is_hybrid,
                KernelType kernel_type) 
            {
                // If HWCN weights are required, Im2Col not required
                if (data->need_hwcn_weights) return false;

                // segregate based on dilated conv & non-dialated conv
                const bool need_dilated_im2col =
                    params->dilation_width_factor != 1 || params->dilation_height_factor != 1;
                const bool need_non_dilated_im2col =
                    params->stride_width != 1 || params->stride_height != 1 ||
                    filter->dims->data[2] != 1 || filter->dims->data[1] != 1;

                const bool need_im2col = need_dilated_im2col || need_non_dilated_im2col;

                // Return early as basic requirement is not met
                if (!need_im2col) return false;

                switch (kernel_type) 
                {
                case kReference:
                    if (is_hybrid) 
                    {
                        return true;
                    }
                    else 
                    {
                        return false;
                    }
                case kGenericOptimized:
                case kCblasOptimized:
                    // `need_im2col` is always satisfied.
                    return true;
                case kMultithreadOptimized:
                    if (input->type == kTfLiteUInt8 ||  //
                        input->type == kTfLiteInt8 ||   //
                        input->type == kTfLiteInt16 ||  // quantized.
                        !data->supports_multithreaded_kernel) 
                    {
                        return true;
                    }
                    else 
                    {
                        return false;
                    }
                default:
                    return false;
                }
            }

            TfLiteStatus AllocateTemporaryTensorsIfRequired(
                TfLiteContext* context, TfLiteNode* node, bool is_hybrid,
                bool is_per_channel, KernelType kernel_type, size_t im2col_bytes,
                TfLiteConvParams* params, OpData* data)
            {
                // So far for Quantized models and kernel_type = kReference no memory is allocated
                //auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
                //OpData* data = reinterpret_cast<OpData*>(node->user_data);
                int bias_index = -1, filter_index = -1, input_index = -1;
                GetTensorIndexes(context, node, &bias_index, &filter_index, &input_index);

                TF_LITE_ENSURE(context, node->inputs->size >= 2);
                const TfLiteTensor* input;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, input_index, &input));
                const TfLiteTensor* filter;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, filter_index, &filter));

                // If we're using the optimized multithreaded EigenTensor implementation of
                // convolution, it expects the filter weights to be transposed compared to
                // the normal TF Lite buffer format. Typical TF Lite weights are
                // [filter_count, filter_height, filter_width, input_depth], but for the float
                // implementation we need them as [filter_height, filter_width, input_depth,
                // filter_count]. We get to that format by transposing, and create a temporary
                // buffer to store the results.
                // This path is only used for float processing, so only create the buffer if
                // we're running with that data type.
                data->need_hwcn_weights =
                    input->type == kTfLiteFloat32 && data->supports_multithreaded_kernel;

                // We don't always need to allocate im2col. It is only used in some versions
                // of the optimized Conv. This test just mimics something that happens inside
                // optimized_ops.h, in order to avoid a DCHECK(!im2col_data).
                data->need_im2col =
                    IsIm2ColRequired(input, params, filter, data, is_hybrid, kernel_type);

                // If im2col_oversized is found to be true, we have to fallback to an
                // execution path (like kReference in float/quantized cases) that doesn't
                // require im2col operation. Therefore, we have to skip checking the hybrid
                // case (but not the hybrid-per-channel one) where there's no such a fallback
                // execution path.
                // TODO(b/178743262): Consider making this check conditioned on the available
                // memory of the system, rather than coupling to the mobile platform check.
                //if (IsMobilePlatform() && !(is_hybrid && !is_per_channel) &&
                //    data->need_im2col && im2col_bytes >= kMaxIm2colBufferSizeMobile) {
                //    data->need_im2col = false;
                //    data->im2col_oversized = true;
                //}
                int temporaries_count = 0;
                if (data->need_im2col) 
                {
                    data->im2col_index = temporaries_count;
                    if (data->im2col_id == kTensorNotAllocated) 
                    {
                        context->AddTensors(context, 1, &data->im2col_id);
                    }
                    ++temporaries_count;
                }
                if (data->need_hwcn_weights) 
                {
                    data->hwcn_weights_index = temporaries_count;
                    if (data->hwcn_weights_id == kTensorNotAllocated) 
                    {
                        context->AddTensors(context, 1, &data->hwcn_weights_id);
                    }
                    ++temporaries_count;
                }

                if (is_hybrid) 
                {
                    // Allocate tensor to store the on-the-fly quantized inputs.
                    data->input_quantized_index = temporaries_count;
                    if (data->input_quantized_id == kTensorNotAllocated) 
                    {
                        TF_LITE_ENSURE_OK(
                            context, context->AddTensors(context, 1, &data->input_quantized_id));
                    }
                    ++temporaries_count;

                    // Allocate tensor to store the quantization params computed during
                    // on-the-fly input quantization.
                    data->scaling_factors_index = temporaries_count;
                    if (data->scaling_factors_id == kTensorNotAllocated) 
                    {
                        TF_LITE_ENSURE_OK(
                            context, context->AddTensors(context, 1, &data->scaling_factors_id));
                    }
                    ++temporaries_count;

                    // Allocate tensor to store the accumulators for the matrix multiply.
                    data->accum_scratch_index = temporaries_count;
                    if (data->accum_scratch_id == kTensorNotAllocated) 
                    {
                        TF_LITE_ENSURE_OK(
                            context, context->AddTensors(context, 1, &data->accum_scratch_id));
                    }
                    ++temporaries_count;
                    if (is_per_channel) 
                    {
                        data->input_offset_index = temporaries_count;
                        if (data->input_offset_id == kTensorNotAllocated) 
                        {
                            TF_LITE_ENSURE_OK(
                                context, context->AddTensors(context, 1, &data->input_offset_id));
                        }
                        ++temporaries_count;

                        data->row_sums_index = temporaries_count;
                        if (data->row_sums_id == kTensorNotAllocated) 
                        {
                            TF_LITE_ENSURE_OK(context,
                                context->AddTensors(context, 1, &data->row_sums_id));
                        }
                        ++temporaries_count;
                    }
                }

                TfLiteIntArrayFree(node->temporaries);
                node->temporaries = TfLiteIntArrayCreate(temporaries_count);

                return kTfLiteOk;
            }

            TfLiteStatus Prepare(KernelType kernel_type, 
                TfLiteContext* context, 
                TfLiteNode* node,
                TfLiteConvParams* params,
                OpData* data)
            {
                // This function has been modified to take into account the reordering of the inputs
                // Indexes are modified to reflect the changed order of inputs
                //auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
                //OpData* data = reinterpret_cast<OpData*>(node->user_data);
                bool has_bias = node->inputs->size == 3;

                // Check number of inputs/outputs
                TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
                TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

                // Added double check of shape of input tensors
                // Discriminate which are biases, filter and input
                int bias_index = -1, filter_index = -1, input_index = -1;
                GetTensorIndexes(context, node, &bias_index, &filter_index, &input_index);

                // Node is also needed to get size and tensor indexes
                TfLiteTensor* output;
                TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
                const TfLiteTensor* input;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, input_index, &input));
                const TfLiteTensor* filter;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, filter_index, &filter));

                // Check dimensionality of input, filter
                TF_LITE_ENSURE_EQ(context, input->dims->size, 4);
                TF_LITE_ENSURE_EQ(context, filter->dims->size, 4);
                // Check input channels matching filter
                // Filter input channel can be a factor of channels of input (grouped conv)
                // or equals (normal conv).
                auto input_channel = input->dims->data[3];
                auto filter_input_channel = filter->dims->data[3];
                TF_LITE_ENSURE(context, filter_input_channel > 0);
                TF_LITE_ENSURE_EQ(context, input_channel % filter_input_channel, 0);
                data->groups = input_channel / filter_input_channel;

                // Check types. (We assume that UINT8 refers to quantized tensors)
                TfLiteType input_type = input->type;
                TF_LITE_ENSURE(context,
                    input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
                    input_type == kTfLiteInt8 || input_type == kTfLiteInt16);
                TF_LITE_ENSURE_TYPES_EQ(context, output->type, input_type);

                // Filter must have zero zero-points in per-channel quantization.
                if (input_type == kTfLiteInt16 || input_type == kTfLiteInt8) 
                {
                    TF_LITE_ENSURE_EQ(context, filter->quantization.type, kTfLiteAffineQuantization);
                    const auto* affine_quantization =
                        reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
                    for (int i = 0; i < affine_quantization->zero_point->size; ++i) 
                    {
                        TF_LITE_ENSURE_EQ(context, affine_quantization->zero_point->data[i], 0);
                    }
                }

                const TfLiteTensor* bias = nullptr;
                
                // TODO(ahentz): At this point the optimized versions require 'bias'. We can
                // either change that or document that convolution requires it.
                TF_LITE_ENSURE(context, has_bias);

                if (has_bias) 
                {
                    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, bias_index, &bias));
                    if (input_type == kTfLiteUInt8 || input_type == kTfLiteInt8) {
                        TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt32);
                        TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
                    }
                    else if (input_type == kTfLiteInt16) {
                        TF_LITE_ENSURE(context, (bias->type == kTfLiteInt32) ||
                            (bias->type == kTfLiteInt64));
                        TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
                    }
                    else {
                        TF_LITE_ENSURE_TYPES_EQ(context, bias->type, input_type);
                    }
                    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 0));
                }

                if (input_type == kTfLiteInt16) 
                {
                    // Quantization should be symmetric.
                    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
                    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

                    // Check quantized_bias_type is either kTfLiteInt64 or kTfLiteInt32.
                    if (params->quantized_bias_type != kTfLiteFloat32) {
                        TF_LITE_ENSURE(context, params->quantized_bias_type == kTfLiteInt32 ||
                            params->quantized_bias_type == kTfLiteInt64);
                        TF_LITE_ENSURE(context, (bias == nullptr) ||
                            bias->type == params->quantized_bias_type);
                        data->quantized_bias_type = params->quantized_bias_type;
                    }
                }

                const bool is_hybrid =
                    (input->type == kTfLiteFloat32 &&
                        (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8));

                if (is_hybrid && filter->type == kTfLiteInt8 &&
                    filter->quantization.type == kTfLiteAffineQuantization &&
                    filter->quantization.params &&
                    reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params)->scale &&
                    reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params)->scale->size > 1) 
                {
                    const auto* affine_quantization =
                        reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
                    const float scale = affine_quantization->scale->data[0];
                    for (int i = 1; i < affine_quantization->scale->size; i++) 
                    {
                        if (affine_quantization->scale->data[i] != scale) 
                        {
                            data->is_hybrid_per_channel = true;
                            break;
                        }
                    }
                }

                // The multi-threaded kernel supports neither dilation nor hybrid kernels, and
                // is incompatible with mutable input filters that might change between evals.
                data->supports_multithreaded_kernel =
                    (kernel_type == kMultithreadOptimized) &&
                    (context->recommended_num_threads != 1) && !is_hybrid &&
                    (params->dilation_width_factor == 1) &&
                    (params->dilation_height_factor == 1) &&
                    (filter->allocation_type != kTfLiteArenaRw) && !IsDynamicTensor(filter);

                int channels_in = filter->dims->data[3];
                int channels_out = filter->dims->data[0];
                int width = input->dims->data[2];
                int height = input->dims->data[1];
                int filter_width = filter->dims->data[2];
                int filter_height = filter->dims->data[1];
                int batches = input->dims->data[0];

                // Matching GetWindowedOutputSize in TensorFlow.
                auto padding = params->padding;
                int out_width, out_height;
                data->padding = ComputePaddingHeightWidth(
                    params->stride_height, params->stride_width,
                    params->dilation_height_factor, params->dilation_width_factor, height,
                    width, filter_height, filter_width, padding, &out_height, &out_width);

                size_t im2col_type_size;
                TF_LITE_ENSURE_STATUS(GetSizeOfType(context, input->type, &im2col_type_size));
                // Note that we intentionally promote the first multiplicand (i.e. 'batches')
                // to 'size_t' to avoid integer overflow here.
                const size_t im2col_bytes = static_cast<size_t>(batches) * out_height *
                    out_width * channels_in * filter_height *
                    filter_width * im2col_type_size;
                TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequired(
                    context, node, is_hybrid, data->is_hybrid_per_channel, kernel_type,
                    im2col_bytes, params, data));

                TF_LITE_ENSURE(context, has_bias);

                // Note that full fixed-point inference requires that all tensors have their
                // parameters set. This is usually done during quantized training or
                // calibration.
                if (input_type != kTfLiteFloat32) 
                {
                    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                        kTfLiteAffineQuantization);
                    const auto* affine_quantization =
                        reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
                    TF_LITE_ENSURE(context, affine_quantization);
                    TF_LITE_ENSURE(context, affine_quantization->scale);
                    TF_LITE_ENSURE(context, (affine_quantization->scale->size == 1 ||
                        affine_quantization->scale->size == channels_out));

                    data->per_channel_output_multiplier.resize(channels_out);
                    data->per_channel_output_shift.resize(channels_out);

                    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
                        context, input, filter, bias, output, params->activation,
                        &data->output_multiplier, &data->output_shift,
                        &data->output_activation_min, &data->output_activation_max,
                        data->per_channel_output_multiplier.data(),
                        data->per_channel_output_shift.data(), channels_out));
                }

                TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
                output_size->data[0] = batches;
                output_size->data[1] = out_height;
                output_size->data[2] = out_width;
                output_size->data[3] = channels_out;
                auto output_status = context->ResizeTensor(context, output, output_size);

                if (output_status != kTfLiteOk) return output_status;

                // If kReference the following will always be false with input int8
                if (data->need_im2col) 
                {
                    node->temporaries->data[data->im2col_index] = data->im2col_id;

                    TfLiteIntArray* im2col_size = TfLiteIntArrayCreate(4);

                    auto filter_input_channel = filter->dims->data[3];
                    im2col_size->data[0] = output_size->data[0];
                    im2col_size->data[1] = output_size->data[1];
                    im2col_size->data[2] = output_size->data[2];
                    im2col_size->data[3] = filter_input_channel * filter_height * filter_width;

                    TfLiteTensor* im2col =
                        &context->tensors[node->temporaries->data[data->im2col_index]];
                    im2col->type = input->type;
                    if (is_hybrid) {
                        im2col->type = filter->type;
                    }
                    im2col->allocation_type = kTfLiteArenaRw;
                    auto im2col_status = context->ResizeTensor(context, im2col, im2col_size);
                    if (im2col_status != kTfLiteOk) return im2col_status;
                }

                // if filter is float32
                if (data->need_hwcn_weights) 
                {
                    node->temporaries->data[data->hwcn_weights_index] = data->hwcn_weights_id;
                    TfLiteIntArray* hwcn_weights_size = TfLiteIntArrayCreate(2);

                    // Because we're treating the filter weights as a matrix when we do the
                    // transpose, we allocate the buffer with a two-dimensional shape, where one
                    // dimension is the number of elements in each filter, and the second is the
                    // total number of filters.
                    auto filter_input_channel = filter->dims->data[3];
                    hwcn_weights_size->data[0] =
                        (filter_height * filter_width * filter_input_channel);
                    hwcn_weights_size->data[1] = channels_out;

                    TfLiteTensor* hwcn_weights =
                        &context->tensors[node->temporaries->data[data->hwcn_weights_index]];
                    hwcn_weights->type = input_type;
                    hwcn_weights->name = "Conv_hwcn_weights";
                    hwcn_weights->allocation_type = kTfLiteArenaRwPersistent;

                    auto hwcn_weights_status =
                        context->ResizeTensor(context, hwcn_weights, hwcn_weights_size);
                    if (hwcn_weights_status != kTfLiteOk) return hwcn_weights_status;

                    // TODO(petewarden): If Resize() is called when the size hasn't actually
                    // changed, this will do extra redundant work.
                    data->have_weights_been_transposed = false;
                }

                // If the input is float32 and filter is int8
                if (is_hybrid) 
                {
                    node->temporaries->data[data->input_quantized_index] =
                        data->input_quantized_id;
                    TfLiteTensor* input_quantized;
                    TF_LITE_ENSURE_OK(
                        context, GetTemporarySafe(context, node, data->input_quantized_index,
                            &input_quantized));
                    input_quantized->type = kTfLiteInt8;
                    input_quantized->allocation_type = kTfLiteArenaRw;
                    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) 
                    {
                        TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
                        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                            input_quantized_size));
                    }

                    node->temporaries->data[data->scaling_factors_index] =
                        data->scaling_factors_id;
                    TfLiteTensor* scaling_factors;
                    TF_LITE_ENSURE_OK(
                        context, GetTemporarySafe(context, node, data->scaling_factors_index,
                            &scaling_factors));
                    scaling_factors->type = kTfLiteFloat32;
                    scaling_factors->allocation_type = kTfLiteArenaRw;
                    // Only one scale factor per batch is typically necessary. See optimized
                    // implementation for why we need to allocate for the height of the inputs
                    // flattened to 2D.
                    TF_LITE_ENSURE(context, channels_in != 0);
                    const int height = NumElements(input) / channels_in;
                    int scaling_dims[1] = { height };
                    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) 
                    {
                        TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
                        scaling_factors_size->data[0] = height;
                        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                            scaling_factors_size));
                    }

                    node->temporaries->data[data->accum_scratch_index] = data->accum_scratch_id;
                    TfLiteTensor* accum_scratch;
                    TF_LITE_ENSURE_OK(context,
                        GetTemporarySafe(context, node, data->accum_scratch_index,
                            &accum_scratch));
                    accum_scratch->type = kTfLiteInt32;
                    accum_scratch->allocation_type = kTfLiteArenaRw;
                    const int scratch_width = batches * out_height * out_width;
                    int accum_scratch_dims[2] = { channels_out, scratch_width };
                    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                        accum_scratch_dims)) 
                    {
                        TfLiteIntArray* accum_scratch_size = TfLiteIntArrayCreate(2);
                        accum_scratch_size->data[0] = channels_out;
                        accum_scratch_size->data[1] = scratch_width;
                        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, accum_scratch,
                            accum_scratch_size));
                    }

                    if (data->is_hybrid_per_channel) 
                    {
                        const auto* affine_quantization =
                            reinterpret_cast<TfLiteAffineQuantization*>(
                                filter->quantization.params);
                        TF_LITE_ENSURE(context, affine_quantization);
                        TF_LITE_ENSURE(context, affine_quantization->scale);
                        TF_LITE_ENSURE_EQ(
                            context, affine_quantization->scale->size,
                            filter->dims->data[affine_quantization->quantized_dimension]);
                        node->temporaries->data[data->input_offset_index] = data->input_offset_id;
                        TfLiteTensor* input_offsets;
                        TF_LITE_ENSURE_OK(
                            context, GetTemporarySafe(context, node, data->input_offset_index,
                                &input_offsets));
                        input_offsets->type = kTfLiteInt32;
                        input_offsets->allocation_type = kTfLiteArenaRw;
                        // See above comment for the need to allocate for height of inputs.
                        TF_LITE_ENSURE(context, channels_in != 0);
                        const int height = NumElements(input) / channels_in;
                        const int input_offset_dims[1] = { height };
                        if (!TfLiteIntArrayEqualsArray(input_offsets->dims, 1,
                            input_offset_dims)) 
                        {
                            TfLiteIntArray* input_offsets_size = TfLiteIntArrayCreate(1);
                            input_offsets_size->data[0] = input_offset_dims[0];
                            TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_offsets,
                                input_offsets_size));
                        }
                        node->temporaries->data[data->row_sums_index] = data->row_sums_id;
                        TfLiteTensor* row_sums;
                        TF_LITE_ENSURE_OK(
                            context,
                            GetTemporarySafe(context, node, data->row_sums_index, &row_sums));
                        row_sums->type = kTfLiteInt32;
                        row_sums->name = "Conv_row_sums";
                        row_sums->allocation_type = kTfLiteArenaRwPersistent;
                        // See above comment for the need to allocate for height of inputs.
                        const int row_sums_dims[1] = { channels_out };
                        if (!TfLiteIntArrayEqualsArray(row_sums->dims, 1, row_sums_dims)) 
                        {
                            TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(1);
                            row_sums_size->data[0] = row_sums_dims[0];
                            TF_LITE_ENSURE_OK(
                                context, context->ResizeTensor(context, row_sums, row_sums_size));
                        }
                    }
                }
                
                //std::cout << std::endl << "\n\n ################ Checkpoint ################ \n\n" << std::endl;
                //custom_logger::LogTfLiteTensor(*input);
                //custom_logger::LogTfLiteTensor(*filter);
                //custom_logger::LogTfLiteTensor(*output);
                //custom_logger::LogTfLiteConvParams(params);
                //custom_logger::conv::LogTfLiteOpData(data);
                //std::cout << "Supports multithreaded kernel? " << (data->supports_multithreaded_kernel ? "true" : "false") << std::endl;
                //std::cout << "Kernel type " << custom_logger::conv::get_TfLiteKernelType(kernel_type) << std::endl;
                //std::cout << "im2col bytes " << im2col_bytes << std::endl;
                //std::cout << "is hybrid? " << (is_hybrid ? "true" : "false") << std::endl;
                //std::cout << "channels out " << channels_out << std::endl;
                //std::cout << "\n After \n" << std::endl;
                //custom_logger::conv::LogTfLiteOpData(data);
                return kTfLiteOk;
            }

        }

        int size_extraction(const TfLiteIntArray* const dimensions)
        {
            int acc = 1;
            for (int i = 0; i < dimensions->size; i++)
                acc = acc * dimensions->data[i];
            return acc;
        }
    }
}
