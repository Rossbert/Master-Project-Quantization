#pragma once

namespace tflite {
    namespace custom_ops {
        namespace fully_connected {

            template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
            void FullyConnectedPerChannel(
                const FullyConnectedParams& params, const int32_t* output_multiplier,
                const int* output_shift, const RuntimeShape& input_shape,
                const InputType* input_data, const RuntimeShape& filter_shape,
                const WeightType* filter_data, const RuntimeShape& bias_shape,
                const BiasType* bias_data, const RuntimeShape& output_shape,
                OutputType* output_data) 
            {
                const int32_t input_offset = params.input_offset;
                const int32_t output_offset = params.output_offset;
                const int32_t output_activation_min = params.quantized_activation_min;
                const int32_t output_activation_max = params.quantized_activation_max;
                TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
                TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

                TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
                const int filter_dim_count = filter_shape.DimensionsCount();
                const int batches = output_shape.Dims(0);
                const int output_depth = output_shape.Dims(1);
                TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
                const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
                for (int b = 0; b < batches; ++b) 
                {
                    for (int out_c = 0; out_c < output_depth; ++out_c) 
                    {
                        BiasType acc = 0;
                        for (int d = 0; d < accum_depth; ++d) 
                        {
                            int32_t input_val = input_data[b * accum_depth + d];
                            int32_t filter_val = filter_data[out_c * accum_depth + d];
                            acc += filter_val * (input_val + input_offset);
                        }
                        if (bias_data) 
                        {
                            acc += bias_data[out_c];
                        }
                        int32_t acc_scaled = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_c], output_shift[out_c]);
                        acc_scaled += output_offset;
                        acc_scaled = std::max(acc_scaled, output_activation_min);
                        acc_scaled = std::min(acc_scaled, output_activation_max);
                        output_data[out_c + output_depth * b] = static_cast<OutputType>(acc_scaled);
                    }
                }
            }

            template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
            void FullyConnected(const FullyConnectedParams& params,
                const RuntimeShape& input_shape,
                const InputType* input_data,
                const RuntimeShape& filter_shape,
                const WeightType* filter_data,
                const RuntimeShape& bias_shape, const BiasType* bias_data,
                const RuntimeShape& output_shape, OutputType* output_data) 
            {
                const int32_t input_offset = params.input_offset;
                const int32_t filter_offset = params.weights_offset;
                const int32_t output_offset = params.output_offset;
                const int32_t output_multiplier = params.output_multiplier;
                const int output_shift = params.output_shift;
                const int32_t output_activation_min = params.quantized_activation_min;
                const int32_t output_activation_max = params.quantized_activation_max;
                TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
                TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

                TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
                const int filter_dim_count = filter_shape.DimensionsCount();
                const int output_dim_count = output_shape.DimensionsCount();
                const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
                const int output_depth = output_shape.Dims(output_dim_count - 1);
                TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
                const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
                for (int b = 0; b < batches; ++b) 
                {
                    for (int out_c = 0; out_c < output_depth; ++out_c)
                    {
                        BiasType acc = 0;
                        for (int d = 0; d < accum_depth; ++d) 
                        {
                            int32_t input_val = input_data[b * accum_depth + d];
                            int32_t filter_val = filter_data[out_c * accum_depth + d];
                            acc += (filter_val + filter_offset) * (input_val + input_offset);
                        }
                        if (bias_data) 
                        {
                            acc += bias_data[out_c];
                        }
                        int32_t acc_scaled = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
                        acc_scaled += output_offset;
                        acc_scaled = std::max(acc_scaled, output_activation_min);
                        acc_scaled = std::min(acc_scaled, output_activation_max);
                        output_data[out_c + output_depth * b] = static_cast<OutputType>(acc_scaled);
                    }
                }
            }

            template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
            void DisturbedFullyConnectedOperation(
                const int dataset_index,
                const int32_t output_multiplier, const int32_t output_shift,
                const int batches, const int output_depth, const int accum_depth,
                const int input_offset, const int filter_offset, const int output_offset,
                const int output_activation_min, const int output_activation_max,
                const InputType* input_data,
                const WeightType* filter_data,
                const BiasType* bias_data,
                OutputType* output_data,
                const std::vector<int>& chunk_indexes,
                const MyDelegateOptions& options)
            {
                int idx_counter = chunk_indexes.size() - 1;
                for (int b = 0; b < batches; ++b)
                {
                    for (int out_c = 0; out_c < output_depth; ++out_c)
                    {
                        BiasType acc = 0;
                        int outputPosition = b * output_depth + out_c;
                        for (int d = 0; d < accum_depth; ++d)
                        {
                            int& kernelPartialPosition = d;
                            int32_t input_val = input_data[b * accum_depth + d];
                            int32_t filter_val = filter_data[out_c * accum_depth + d];

                            int32_t result = (filter_val + filter_offset) * (input_val + input_offset);

                            if (idx_counter >= 0 && options.error_flat_positions[dataset_index][chunk_indexes[idx_counter]].first == outputPosition && options.error_flat_positions[dataset_index][chunk_indexes[idx_counter]].second == kernelPartialPosition)
                            {
                                std::bitset<32> bits(result);
                                bits.flip(options.bit_position);
                                result = static_cast<int>(bits.to_ulong());
                                idx_counter--;
                            }

                            acc += result;
                        }
                        if (bias_data)
                        {
                            acc += bias_data[out_c];
                        }
                        int32_t acc_scaled = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
                        acc_scaled += output_offset;
                        acc_scaled = std::max(acc_scaled, output_activation_min);
                        acc_scaled = std::min(acc_scaled, output_activation_max);
                        output_data[outputPosition] = static_cast<OutputType>(acc_scaled);
                    }
                }
            }

            template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
            void DisturbedFullyConnectedOperationByChunks(
                const int dataset_index, const int start_chunk, const int end_chunk,
                const int32_t output_multiplier, const int32_t output_shift,
                const int batches, const int output_depth, const int accum_depth,
                const int input_offset, const int filter_offset, const int output_offset,
                const int output_activation_min, const int output_activation_max,
                const InputType* input_data,
                const WeightType* filter_data,
                const BiasType* bias_data,
                OutputType* output_data,
                const std::vector<int>& chunk_indexes,
                const MyDelegateOptions& options)
            {
                int idx_counter = chunk_indexes.size() - 1;
                for (int b = 0; b < batches; ++b)
                {
                    for (int out_c = start_chunk; out_c < end_chunk; ++out_c)
                    {
                        BiasType acc = 0;
                        int outputPosition = b * output_depth + out_c;
                        for (int d = 0; d < accum_depth; ++d)
                        {
                            int& kernelPartialPosition = d;
                            int32_t input_val = input_data[b * accum_depth + d];
                            int32_t filter_val = filter_data[out_c * accum_depth + d];

                            int32_t result = (filter_val + filter_offset) * (input_val + input_offset);

                            if (idx_counter >= 0 && options.error_flat_positions[dataset_index][chunk_indexes[idx_counter]].first == outputPosition && options.error_flat_positions[dataset_index][chunk_indexes[idx_counter]].second == kernelPartialPosition)
                            {
                                std::bitset<32> bits(result);
                                bits.flip(options.bit_position);
                                result = static_cast<int>(bits.to_ulong());
                                idx_counter--;
                            }

                            acc += result;
                        }
                        if (bias_data)
                        {
                            acc += bias_data[out_c];
                        }
                        int32_t acc_scaled = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
                        acc_scaled += output_offset;
                        acc_scaled = std::max(acc_scaled, output_activation_min);
                        acc_scaled = std::min(acc_scaled, output_activation_max);
                        output_data[outputPosition] = static_cast<OutputType>(acc_scaled);
                    }
                }
            }
            
            template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
            void ParallelDisturbedFullyConnected(
                const int dataset_index,
                const int32_t output_multiplier, const int32_t output_shift,
                const int batches, const int output_depth, const int accum_depth,
                const int input_offset, const int filter_offset, const int output_offset,
                const int output_activation_min, const int output_activation_max,
                const InputType* input_data,
                const WeightType* filter_data,
                const BiasType* bias_data,
                OutputType* output_data,
                const MyDelegateOptions& options)
            {
                std::vector<std::thread> threadPool;
                //std::mutex coutMutex;

                for (int i = 0; i < options.num_threads; ++i)
                {
                    const int start = i * options.chunk_size;
                    const int end = std::min(start + options.chunk_size, options.channels);

#if LOGGER
                    //std::cout << "Indexes size " << chunk_indexes.size() << "\n";
                    //std::cout << "Indexes capacity " << chunk_indexes.capacity() << "\n";
                    //std::cout << "Start: " << start << " End: " << end << "\n";
                    //std::cout << "Indexes in chunk " << i << ": ";
                    //for (const auto& val : chunk_indexes)
                    //{
                    //	std::cout << val << " ";
                    //}
                    //std::cout << "\n";
                    
                    //std::cout << "Real positions\n";
                    //for (const auto& val : options.error_vec_positions[dataset_index])
                    //{
                    //	for (const int& element : val.first)
                    //	{
                    //		std::cout << element << " ";
                    //	}
                    //	std::cout << "- ";
                    //	for (const int& element : val.second)
                    //	{
                    //		std::cout << element << " ";
                    //	}
                    //	std::cout << "\n";
                    //}
                    //std::cout << "\n";
#endif // LOGGER

                    threadPool.emplace_back(
                        DisturbedFullyConnectedOperationByChunks<InputType, WeightType, OutputType, BiasType>,
                        dataset_index, start, end,
                        output_multiplier, output_shift,
                        batches, output_depth, accum_depth,
                        input_offset, filter_offset, output_offset,
                        output_activation_min, output_activation_max,
                        input_data,
                        filter_data,
                        bias_data,
                        output_data,
                        std::cref(options.chunks_indexes[dataset_index][i]),
                        std::cref(options));
                }

                // Join all threads
                for (auto& thread : threadPool)
                {
                    thread.join();
                }
            }

            template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
            void FullyConnectedDisturbed(const FullyConnectedParams& params,
                const RuntimeShape& input_shape,
                const InputType* input_data,
                const RuntimeShape& filter_shape,
                const WeightType* filter_data,
                const RuntimeShape& bias_shape, const BiasType* bias_data,
                const RuntimeShape& output_shape, OutputType* output_data,
                const MyDelegateOptions& options)
            {
                const int32_t input_offset = params.input_offset;
                const int32_t filter_offset = params.weights_offset;
                const int32_t output_offset = params.output_offset;
                const int32_t output_multiplier = params.output_multiplier;
                const int output_shift = params.output_shift;
                const int32_t output_activation_min = params.quantized_activation_min;
                const int32_t output_activation_max = params.quantized_activation_max;
                TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
                TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

                TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
                const int filter_dim_count = filter_shape.DimensionsCount();
                const int output_dim_count = output_shape.DimensionsCount();
                const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
                const int output_depth = output_shape.Dims(output_dim_count - 1);
                TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
                const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

                // Because of threads this should be incorporated in the positions of the batches
                static int dataset_index = 0;

                if (options.is_threaded)
                {
                    // Parallel computing done here!
                    ParallelDisturbedFullyConnected(
                        dataset_index,
                        output_multiplier, output_shift,
                        batches, output_depth, accum_depth,
                        input_offset, filter_offset, output_offset,
                        output_activation_min, output_activation_max,
                        input_data,
                        filter_data,
                        bias_data,
                        output_data,
                        options
                    );
                }
                else
                {
                    DisturbedFullyConnectedOperation(
                        dataset_index,
                        output_multiplier, output_shift,
                        batches, output_depth, accum_depth,
                        input_offset, filter_offset, output_offset,
                        output_activation_min, output_activation_max,
                        input_data,
                        filter_data,
                        bias_data,
                        output_data,
                        options.full_indexes,
                        options
                    );
                }

                /*
                auto& chunk_indexes = options.full_indexes;
                int idx_counter = options.full_indexes.size() - 1;
                for (int b = 0; b < batches; ++b)
                {
                    for (int out_c = 0; out_c < output_depth; ++out_c)
                    {
                        BiasType acc = 0;
                        int outputPosition = b * output_depth + out_c;
                        for (int d = 0; d < accum_depth; ++d)
                        {
                            int& kernelPartialPosition = d;
                            int32_t input_val = input_data[b * accum_depth + d];
                            int32_t filter_val = filter_data[out_c * accum_depth + d];

                            int32_t result = (filter_val + filter_offset) * (input_val + input_offset);

                            if (idx_counter >= 0 && options.error_flat_positions[dataset_index][chunk_indexes[idx_counter]].first == outputPosition && options.error_flat_positions[dataset_index][chunk_indexes[idx_counter]].second == kernelPartialPosition)
                            {
                                std::bitset<32> bits(result);
                                bits.flip(options.bit_position);
                                result = static_cast<int>(bits.to_ulong());
                                idx_counter--;
                            }

                            acc += result;
                        }
                        if (bias_data)
                        {
                            acc += bias_data[out_c];
                        }
                        int32_t acc_scaled = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
                        acc_scaled += output_offset;
                        acc_scaled = std::max(acc_scaled, output_activation_min);
                        acc_scaled = std::min(acc_scaled, output_activation_max);
                        output_data[outputPosition] = static_cast<OutputType>(acc_scaled);
                    }
                }
                */

                dataset_index++;
            }

            namespace {
                template <KernelType kernel_type>
                void FullyConnectedInt8(const OpData* data, const TfLiteTensor* input,
                    const TfLiteTensor* filter, const TfLiteTensor* bias,
                    TfLiteTensor* output, const MyDelegateOptions& options)
                {
                    FullyConnectedParams op_params;
                    op_params.input_offset = -input->params.zero_point;
                    op_params.weights_offset = -filter->params.zero_point;
                    op_params.output_offset = output->params.zero_point;
                    op_params.output_multiplier = data->output_multiplier;
                    op_params.output_shift = data->output_shift;
                    op_params.quantized_activation_min = data->output_activation_min;
                    op_params.quantized_activation_max = data->output_activation_max;
                    op_params.lhs_cacheable = IsConstantTensor(filter);
                    op_params.rhs_cacheable = IsConstantTensor(input);

                    const int8_t* filter_data;

                    // Valid for different Int4
                    filter_data = GetTensorData<int8>(filter);
                    
                    if (kernel_type == kReference) 
                    {
                        //reference_integer_ops::FullyConnected(
                        //    op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
                        //    GetTensorShape(filter), filter_data, GetTensorShape(bias),
                        //    GetTensorData<int32_t>(bias), GetTensorShape(output),
                        //    GetTensorData<int8_t>(output));

                        //FullyConnected(
                        //    op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
                        //    GetTensorShape(filter), filter_data, GetTensorShape(bias),
                        //    GetTensorData<int32_t>(bias), GetTensorShape(output),
                        //    GetTensorData<int8_t>(output));

                        FullyConnectedDisturbed(
                            op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
                            GetTensorShape(filter), filter_data, GetTensorShape(bias),
                            GetTensorData<int32_t>(bias), GetTensorShape(output),
                            GetTensorData<int8_t>(output), options);
                    }
                }

                template <KernelType kernel_type>
                void FullyConnectedPerChannelInt8(const OpData* data, const TfLiteTensor* input,
                    const TfLiteTensor* filter,
                    const TfLiteTensor* bias,
                    TfLiteTensor* output, const MyDelegateOptions& options)
                {
                    // FullyConnectedPerChannel ops spec is that weights are symmetric.
                    // op_params.weights_offset is not set (filter.params.zero_point is not used),
                    // since it will be always assumed to be 0.
                    FullyConnectedParams op_params;
                    op_params.input_offset = -input->params.zero_point;
                    op_params.output_offset = output->params.zero_point;
                    op_params.quantized_activation_min = data->output_activation_min;
                    op_params.quantized_activation_max = data->output_activation_max;
                    op_params.lhs_cacheable = IsConstantTensor(filter);
                    op_params.rhs_cacheable = IsConstantTensor(input);
                    if (kernel_type == kReference) 
                    {
                        //reference_integer_ops::FullyConnectedPerChannel(
                        //    op_params, data->per_channel_output_multiplier.data(),
                        //    data->per_channel_output_shift.data(), GetTensorShape(input),
                        //    GetTensorData<int8_t>(input), GetTensorShape(filter),
                        //    GetTensorData<int8_t>(filter), GetTensorShape(bias),
                        //    GetTensorData<int32_t>(bias), GetTensorShape(output),
                        //    GetTensorData<int8_t>(output));

                        FullyConnectedPerChannel(
                            op_params, data->per_channel_output_multiplier.data(),
                            data->per_channel_output_shift.data(), GetTensorShape(input),
                            GetTensorData<int8_t>(input), GetTensorShape(filter),
                            GetTensorData<int8_t>(filter), GetTensorShape(bias),
                            GetTensorData<int32_t>(bias), GetTensorShape(output),
                            GetTensorData<int8_t>(output));
                    }
                }

            }  // namespace

            template <KernelType kernel_type>
            TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                TfLiteFullyConnectedParams* params, OpData* data,
                const TfLiteTensor* input,
                const TfLiteTensor* filter, const TfLiteTensor* bias,
                TfLiteTensor* output, const MyDelegateOptions& options)
            {
                const bool is_per_channel = data->per_channel_output_multiplier.size() > 1;
                int32_t input_offset = -input->params.zero_point;
                int32_t filter_offset = -filter->params.zero_point;
                int32_t output_offset = output->params.zero_point;
                // Only the Pie path supports quantized models and float inputs/outputs.
                if (input->type == kTfLiteFloat32)
                {
                }
                else
                {
                    FullyConnectedParams op_params;
                    op_params.input_offset = input_offset;
                    op_params.weights_offset = filter_offset;
                    op_params.output_offset = output_offset;
                    op_params.output_multiplier = data->output_multiplier;
                    op_params.output_shift = data->output_shift;
                    op_params.quantized_activation_min = data->output_activation_min;
                    op_params.quantized_activation_max = data->output_activation_max;
                    op_params.lhs_cacheable = IsConstantTensor(filter);
                    op_params.rhs_cacheable = IsConstantTensor(input);
                    switch (output->type)
                    {
                    case kTfLiteInt8:
                        if (filter->sparsity != nullptr)
                        {
                        }
                        else
                        {
                            is_per_channel ? FullyConnectedPerChannelInt8<kernel_type>(data, input, filter, bias, output, options)
                                : FullyConnectedInt8<kernel_type>(data, input, filter, bias, output, options);
                        }
                        break;
                    default:
                        TF_LITE_KERNEL_LOG(context,
                            "Quantized FullyConnected expects output data "
                            "type uint8, int8 or int16");
                        return kTfLiteError;
                    }
                }

                return kTfLiteOk;
            }

            template <KernelType kernel_type>
            TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node, TfLiteFullyConnectedParams* params, OpData* data)
            {
                // Check for supported activation types.
                //auto* params = reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);

                int bias_index = -1, filter_index = -1, input_index = -1;
                GetTensorIndexes(context, node, &bias_index, &filter_index, &input_index);

                const TfLiteTensor* filter;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, filter_index, &filter));
                const TfLiteTensor* input;
                TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, input_index, &input));
                const bool is_quantized =
                    ((filter->type == kTfLiteUInt8) || (filter->type == kTfLiteInt8) ||
                        (filter->type == kTfLiteInt4));
                const bool is_hybrid = is_quantized && (input->type == kTfLiteFloat32);
                const bool is_pie = kernel_type == kLegacyPie;

                // Pie and hybrid path supports all kinds of fused activations, otherwise only
                // clipping activations are supported.
                if (!is_pie && !is_hybrid)
                {
                    TF_LITE_ENSURE(context, params->activation == kTfLiteActNone ||
                        params->activation == kTfLiteActRelu ||
                        params->activation == kTfLiteActReluN1To1 ||
                        params->activation == kTfLiteActRelu6);
                }
                return PrepareImpl(context, node, kernel_type, params, data);
            }

            template <KernelType kernel_type>
            TfLiteStatus Eval(TfLiteContext* context, 
                TfLiteNode* node, TfLiteFullyConnectedParams* params, 
                OpData* data, const MyDelegateOptions& options)
            {
                //auto* params = reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
                //OpData* data = reinterpret_cast<OpData*>(node->user_data);

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
                // Do nothing if expected output is empty.
                if (NumElements(output) == 0) 
                {
                    return kTfLiteOk;
                }

                if (filter->dims->data[1] == 0) 
                {
                    memset(output->data.data, 0, output->bytes);
                    return kTfLiteOk;
                }

                switch (filter->type) 
                {
                case kTfLiteInt8:
                    if (params->weights_format == kTfLiteFullyConnectedWeightsFormatDefault) 
                    {
                        return EvalQuantized<kernel_type>(context, node, params, data, input,
                            filter, bias, output, options);
                    }
                    else 
                    {
                        TF_LITE_KERNEL_LOG(context, "Unhandled fully-connected weights format");
                        return kTfLiteError;
                    }
                default:
                    TF_LITE_KERNEL_LOG(context,
                        "Filter data type %s currently not supported.",
                        TfLiteTypeGetName(filter->type));
                    return kTfLiteError;
                }
                return kTfLiteOk;
            }

        } // fully_connected

    } // namespace custom_ops

} // namespace tflite
