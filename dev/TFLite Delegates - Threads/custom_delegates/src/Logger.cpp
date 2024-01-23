#include "Logger.h"
#include "DelegateCore.h"

namespace tflite {
	
	namespace custom_logger {

		namespace conv {

			std::string get_TfLiteKernelType(const custom_ops::conv::KernelType kernel_type)
			{
				switch (kernel_type)
				{
				case custom_ops::conv::kReference:
					return STRINGIFY(kReference);
				case custom_ops::conv::kGenericOptimized:
					return STRINGIFY(kGenericOptimized);
				case custom_ops::conv::kMultithreadOptimized:
					return STRINGIFY(kMultithreadOptimized);
				case custom_ops::conv::kCblasOptimized:
					return STRINGIFY(kCblasOptimized);
				default:
					return "unknown " + std::to_string((int)kernel_type);
				}
			}

			void LogTfLiteOpData(const custom_ops::conv::OpData* const data)
			{
				std::cout << "TfLiteConvOpData:" << std::endl;
				std::cout << "~~~~~~~~~~~~~~~~~" << std::endl;
				std::cout << "Im2Col ID(tensor index): " << data->im2col_id << std::endl;
				std::cout << "HWCN weights ID(tensor index): " << data->hwcn_weights_id << std::endl;
				std::cout << "Input quantized ID(tensor index): " << data->input_quantized_id << std::endl;
				std::cout << "Scaling factors ID(tensor index): " << data->scaling_factors_id << std::endl;
				std::cout << "Input offset ID(tensor index): " << data->input_offset_id << std::endl;
				std::cout << "Accum scratch ID(tensor index): " << data->accum_scratch_id << std::endl;
				std::cout << "Row sums ID(tensor index): " << data->row_sums_id << std::endl;
				LogTfLitePaddingValues(data->padding);
				std::cout << "Output multipier: " << data->output_multiplier << std::endl;
				std::cout << "Output shift: " << data->output_shift << std::endl;
				std::cout << "Per channel output multiplier (" << data->per_channel_output_multiplier.size() << "): ";
				for (auto& val : data->per_channel_output_multiplier)
				{
					std::cout << val << " ";
				}
				std::cout << std::endl;
				std::cout << "Per channel output shift (" << data->per_channel_output_shift.size() << "): ";
				if (data->per_channel_output_multiplier.size() == data->per_channel_output_shift.size())
				{
					for (auto& val : data->per_channel_output_shift)
					{
						std::cout << val << " ";
					}
				}
				std::cout << std::endl;
				std::cout << "Fuse output activation min: " << data->output_activation_min << std::endl;
				std::cout << "Fuse output activation max: " << data->output_activation_max << std::endl;
				std::cout << "Im2Col index: " << data->im2col_index << std::endl;
				std::cout << "HWCN weights index: " << data->hwcn_weights_index << std::endl;
				std::cout << "Input quantized index: " << data->input_quantized_index << std::endl;
				std::cout << "Scaling factors index: " << data->scaling_factors_index << std::endl;
				std::cout << "Accum scratch index: " << data->accum_scratch_index << std::endl;
				std::cout << "Input offset index: " << data->input_offset_index << std::endl;
				std::cout << "Row sums index: " << data->row_sums_index << std::endl;
				std::cout << "Need hwcn weights: " << ((data->need_hwcn_weights) ? "true" : "false") << std::endl;
				std::cout << "Have weights been transpoded: " << ((data->have_weights_been_transposed) ? "true" : "false") << std::endl;
				std::cout << "Need Im2Col: " << ((data->need_im2col) ? "true" : "false") << std::endl;
				std::cout << "Im2Col ovsersized: " << ((data->im2col_oversized) ? "true" : "false") << std::endl;
				std::cout << "Supports multithreaded kernel: " << ((data->supports_multithreaded_kernel) ? "true" : "false") << std::endl;
				std::cout << "Is hybrid per channel: " << ((data->is_hybrid_per_channel) ? "true" : "false") << std::endl;
				std::cout << "Compute hybrid row sums: " << ((data->compute_hybrid_row_sums) ? "true" : "false") << std::endl;
				std::cout << "Groups: " << data->groups << std::endl;
				std::cout << "Quantized bias type: " << get_TfLiteType(data->quantized_bias_type) << std::endl;
				std::cout << "~~~~~~~~~~~~~~~~~" << std::endl;
			}
		}

		std::string get_TfLiteDelegateFlags(const TfLiteDelegateFlags flag)
		{
			switch (flag)
			{
			case kTfLiteDelegateFlagsNone:
				return STRINGIFY(kTfLiteDelegateFlagsNone);
			case kTfLiteDelegateFlagsAllowDynamicTensors:
				return STRINGIFY(kTfLiteDelegateFlagsAllowDynamicTensors);
			case kTfLiteDelegateFlagsRequirePropagatedShapes:
				return STRINGIFY(kTfLiteDelegateFlagsRequirePropagatedShapes);
			case kTfLiteDelegateFlagsPerOperatorProfiling:
				return STRINGIFY(kTfLiteDelegateFlagsPerOperatorProfiling);
			case kTfLiteDelegateFlagsAllowDynamicTensors | kTfLiteDelegateFlagsRequirePropagatedShapes:
				return STRINGIFY(kTfLiteDelegateFlagsAllowDynamicTensors) + std::string(" + ") + STRINGIFY(kTfLiteDelegateFlagsRequirePropagatedShapes);
			case kTfLiteDelegateFlagsAllowDynamicTensors | kTfLiteDelegateFlagsPerOperatorProfiling:
				return STRINGIFY(kTfLiteDelegateFlagsAllowDynamicTensors) + std::string(" + ") + STRINGIFY(kTfLiteDelegateFlagsPerOperatorProfiling);
			case kTfLiteDelegateFlagsAllowDynamicTensors | kTfLiteDelegateFlagsRequirePropagatedShapes | kTfLiteDelegateFlagsPerOperatorProfiling:
				return STRINGIFY(kTfLiteDelegateFlagsAllowDynamicTensors) + std::string(" + ") + STRINGIFY(kTfLiteDelegateFlagsRequirePropagatedShapes) + std::string(" + ") + STRINGIFY(kTfLiteDelegateFlagsPerOperatorProfiling);
			default:
				return "unknown " + std::to_string((int)flag);
			}
		}

		std::string get_TfLiteStatus(const TfLiteStatus status)
		{
			switch (status)
			{
			case kTfLiteOk:
				return STRINGIFY(kTfLiteOk);
			case kTfLiteError:
				return STRINGIFY(kTfLiteError);
			case kTfLiteDelegateError:
				return STRINGIFY(kTfLiteDelegateError);
			case kTfLiteApplicationError:
				return STRINGIFY(kTfLiteApplicationError);
			case kTfLiteDelegateDataNotFound:
				return STRINGIFY(kTfLiteDelegateDataNotFound);
			case kTfLiteDelegateDataWriteError:
				return STRINGIFY(kTfLiteDelegateDataWriteError);
			case kTfLiteDelegateDataReadError:
				return STRINGIFY(kTfLiteDelegateDataReadError);
			case kTfLiteUnresolvedOps:
				return STRINGIFY(kTfLiteUnresolvedOps);
			case kTfLiteCancelled:
				return STRINGIFY(kTfLiteCancelled);
			default:
				return "unknown " + std::to_string((int)status);
			}
		}

		std::string get_TfLiteType(const TfLiteType type)
		{
			switch (type)
			{
			case kTfLiteNoType:
				return "no type";
			case kTfLiteFloat32:
				return "float (float32)";
			case kTfLiteInt32:
				return "int (int32)";
			case kTfLiteUInt8:
				return "unsigned char (uint8)";
			case kTfLiteInt64:
				return "long long (int64)";
			case kTfLiteString:
				return "char";
			case kTfLiteBool:
				return "bool";
			case kTfLiteInt16:
				return "short (int16)";
			case kTfLiteComplex64:
				return "{float float} (TfLiteComplex64)";
			case kTfLiteInt8:
				return "signed char (int8)";
			case kTfLiteFloat16:
				return "{unsigned short} (TfLiteFloat16)";
			case kTfLiteFloat64:
				return "double (float64)";
			case kTfLiteComplex128:
				return "{double double} (TfLiteComplex128)";
			case kTfLiteUInt64:
				return "unsigned long long (uint64)";
			case kTfLiteResource:
				return "kTfLiteResource";
			case kTfLiteVariant:
				return "kTfLiteVariant";
			case kTfLiteUInt32:
				return "unsigned int (uint32)";
			case kTfLiteUInt16:
				return "unsigned short (uint16)";
			case kTfLiteInt4:
				return "kTfLiteInt4";
			default:
				return "unknown " + std::to_string((int)type);
			}
		}

		std::string get_builtin_code(const int builtin_code)
		{
			switch (builtin_code)
			{
			case kTfLiteBuiltinAdd:
				return STRINGIFY(kTfLiteBuiltinAdd);
			case kTfLiteBuiltinAveragePool2d:
				return STRINGIFY(kTfLiteBuiltinAveragePool2d);
			case kTfLiteBuiltinConcatenation:
				return STRINGIFY(kTfLiteBuiltinConcatenation);
			case kTfLiteBuiltinConv2d:
				return STRINGIFY(kTfLiteBuiltinConv2d);
			case kTfLiteBuiltinDepthwiseConv2d:
				return STRINGIFY(kTfLiteBuiltinDepthwiseConv2d);
			case kTfLiteBuiltinDepthToSpace:
				return STRINGIFY(kTfLiteBuiltinDepthToSpace);
			case kTfLiteBuiltinDequantize:
				return STRINGIFY(kTfLiteBuiltinDequantize);
			case kTfLiteBuiltinEmbeddingLookup:
				return STRINGIFY(kTfLiteBuiltinEmbeddingLookup);
			case kTfLiteBuiltinFloor:
				return STRINGIFY(kTfLiteBuiltinFloor);
			case kTfLiteBuiltinFullyConnected:
				return STRINGIFY(kTfLiteBuiltinFullyConnected);
			case kTfLiteBuiltinHashtableLookup:
				return STRINGIFY(kTfLiteBuiltinHashtableLookup);
			case kTfLiteBuiltinL2Normalization:
				return STRINGIFY(kTfLiteBuiltinL2Normalization);
			case kTfLiteBuiltinL2Pool2d:
				return STRINGIFY(kTfLiteBuiltinL2Pool2d);
			case kTfLiteBuiltinLocalResponseNormalization:
				return STRINGIFY(kTfLiteBuiltinLocalResponseNormalization);
			case kTfLiteBuiltinLogistic:
				return STRINGIFY(kTfLiteBuiltinLogistic);
			case kTfLiteBuiltinLshProjection:
				return STRINGIFY(kTfLiteBuiltinLshProjection);
			case kTfLiteBuiltinLstm:
				return STRINGIFY(kTfLiteBuiltinLstm);
			case kTfLiteBuiltinMaxPool2d:
				return STRINGIFY(kTfLiteBuiltinMaxPool2d);
			case kTfLiteBuiltinMul:
				return STRINGIFY(kTfLiteBuiltinMul);
			case kTfLiteBuiltinRelu:
				return STRINGIFY(kTfLiteBuiltinRelu);
			case kTfLiteBuiltinReluN1To1:
				return STRINGIFY(kTfLiteBuiltinReluN1To1);
			case kTfLiteBuiltinRelu6:
				return STRINGIFY(kTfLiteBuiltinRelu6);
			case kTfLiteBuiltinReshape:
				return STRINGIFY(kTfLiteBuiltinReshape);
			case kTfLiteBuiltinResizeBilinear:
				return STRINGIFY(kTfLiteBuiltinResizeBilinear);
			case kTfLiteBuiltinRnn:
				return STRINGIFY(kTfLiteBuiltinRnn);
			case kTfLiteBuiltinSoftmax:
				return STRINGIFY(kTfLiteBuiltinSoftmax);
			case kTfLiteBuiltinSpaceToDepth:
				return STRINGIFY(kTfLiteBuiltinSpaceToDepth);
			case kTfLiteBuiltinSvdf:
				return STRINGIFY(kTfLiteBuiltinSvdf);
			case kTfLiteBuiltinTanh:
				return STRINGIFY(kTfLiteBuiltinTanh);
			case kTfLiteBuiltinConcatEmbeddings:
				return STRINGIFY(kTfLiteBuiltinConcatEmbeddings);
			case kTfLiteBuiltinSkipGram:
				return STRINGIFY(kTfLiteBuiltinSkipGram);
			case kTfLiteBuiltinCall:
				return STRINGIFY(kTfLiteBuiltinCall);
			case kTfLiteBuiltinCustom:
				return STRINGIFY(kTfLiteBuiltinCustom);
			case kTfLiteBuiltinEmbeddingLookupSparse:
				return STRINGIFY(kTfLiteBuiltinEmbeddingLookupSparse);
			case kTfLiteBuiltinPad:
				return STRINGIFY(kTfLiteBuiltinPad);
			case kTfLiteBuiltinUnidirectionalSequenceRnn:
				return STRINGIFY(kTfLiteBuiltinUnidirectionalSequenceRnn);
			case kTfLiteBuiltinGather:
				return STRINGIFY(kTfLiteBuiltinGather);
			case kTfLiteBuiltinBatchToSpaceNd:
				return STRINGIFY(kTfLiteBuiltinBatchToSpaceNd);
			case kTfLiteBuiltinSpaceToBatchNd:
				return STRINGIFY(kTfLiteBuiltinSpaceToBatchNd);
			case kTfLiteBuiltinTranspose:
				return STRINGIFY(kTfLiteBuiltinTranspose);
			case kTfLiteBuiltinMean:
				return STRINGIFY(kTfLiteBuiltinMean);
			case kTfLiteBuiltinSub:
				return STRINGIFY(kTfLiteBuiltinSub);
			case kTfLiteBuiltinDiv:
				return STRINGIFY(kTfLiteBuiltinDiv);
			case kTfLiteBuiltinSqueeze:
				return STRINGIFY(kTfLiteBuiltinSqueeze);
			case kTfLiteBuiltinUnidirectionalSequenceLstm:
				return STRINGIFY(kTfLiteBuiltinUnidirectionalSequenceLstm);
			case kTfLiteBuiltinStridedSlice:
				return STRINGIFY(kTfLiteBuiltinStridedSlice);
			case kTfLiteBuiltinBidirectionalSequenceRnn:
				return STRINGIFY(kTfLiteBuiltinBidirectionalSequenceRnn);
			case kTfLiteBuiltinExp:
				return STRINGIFY(kTfLiteBuiltinExp);
			case kTfLiteBuiltinTopkV2:
				return STRINGIFY(kTfLiteBuiltinTopkV2);
			case kTfLiteBuiltinSplit:
				return STRINGIFY(kTfLiteBuiltinSplit);
			case kTfLiteBuiltinLogSoftmax:
				return STRINGIFY(kTfLiteBuiltinLogSoftmax);
			case kTfLiteBuiltinDelegate:
				return STRINGIFY(kTfLiteBuiltinDelegate);
			case kTfLiteBuiltinBidirectionalSequenceLstm:
				return STRINGIFY(kTfLiteBuiltinBidirectionalSequenceLstm);
			case kTfLiteBuiltinCast:
				return STRINGIFY(kTfLiteBuiltinCast);
			case kTfLiteBuiltinPrelu:
				return STRINGIFY(kTfLiteBuiltinPrelu);
			case kTfLiteBuiltinMaximum:
				return STRINGIFY(kTfLiteBuiltinMaximum);
			case kTfLiteBuiltinArgMax:
				return STRINGIFY(kTfLiteBuiltinArgMax);
			case kTfLiteBuiltinMinimum:
				return STRINGIFY(kTfLiteBuiltinMinimum);
			case kTfLiteBuiltinLess:
				return STRINGIFY(kTfLiteBuiltinLess);
			case kTfLiteBuiltinNeg:
				return STRINGIFY(kTfLiteBuiltinNeg);
			case kTfLiteBuiltinPadv2:
				return STRINGIFY(kTfLiteBuiltinPadv2);
			case kTfLiteBuiltinGreater:
				return STRINGIFY(kTfLiteBuiltinGreater);
			case kTfLiteBuiltinGreaterEqual:
				return STRINGIFY(kTfLiteBuiltinGreaterEqual);
			case kTfLiteBuiltinLessEqual:
				return STRINGIFY(kTfLiteBuiltinLessEqual);
			case kTfLiteBuiltinSelect:
				return STRINGIFY(kTfLiteBuiltinSelect);
			case kTfLiteBuiltinSlice:
				return STRINGIFY(kTfLiteBuiltinSlice);
			case kTfLiteBuiltinSin:
				return STRINGIFY(kTfLiteBuiltinSin);
			case kTfLiteBuiltinTransposeConv:
				return STRINGIFY(kTfLiteBuiltinTransposeConv);
			case kTfLiteBuiltinSparseToDense:
				return STRINGIFY(kTfLiteBuiltinSparseToDense);
			case kTfLiteBuiltinTile:
				return STRINGIFY(kTfLiteBuiltinTile);
			case kTfLiteBuiltinExpandDims:
				return STRINGIFY(kTfLiteBuiltinExpandDims);
			case kTfLiteBuiltinEqual:
				return STRINGIFY(kTfLiteBuiltinEqual);
			case kTfLiteBuiltinNotEqual:
				return STRINGIFY(kTfLiteBuiltinNotEqual);
			case kTfLiteBuiltinLog:
				return STRINGIFY(kTfLiteBuiltinLog);
			case kTfLiteBuiltinSum:
				return STRINGIFY(kTfLiteBuiltinSum);
			case kTfLiteBuiltinSqrt:
				return STRINGIFY(kTfLiteBuiltinSqrt);
			case kTfLiteBuiltinRsqrt:
				return STRINGIFY(kTfLiteBuiltinRsqrt);
			case kTfLiteBuiltinShape:
				return STRINGIFY(kTfLiteBuiltinShape);
			case kTfLiteBuiltinPow:
				return STRINGIFY(kTfLiteBuiltinPow);
			case kTfLiteBuiltinArgMin:
				return STRINGIFY(kTfLiteBuiltinArgMin);
			case kTfLiteBuiltinFakeQuant:
				return STRINGIFY(kTfLiteBuiltinFakeQuant);
			case kTfLiteBuiltinReduceProd:
				return STRINGIFY(kTfLiteBuiltinReduceProd);
			case kTfLiteBuiltinReduceMax:
				return STRINGIFY(kTfLiteBuiltinReduceMax);
			case kTfLiteBuiltinPack:
				return STRINGIFY(kTfLiteBuiltinPack);
			case kTfLiteBuiltinLogicalOr:
				return STRINGIFY(kTfLiteBuiltinLogicalOr);
			case kTfLiteBuiltinOneHot:
				return STRINGIFY(kTfLiteBuiltinOneHot);
			case kTfLiteBuiltinLogicalAnd:
				return STRINGIFY(kTfLiteBuiltinLogicalAnd);
			case kTfLiteBuiltinLogicalNot:
				return STRINGIFY(kTfLiteBuiltinLogicalNot);
			case kTfLiteBuiltinUnpack:
				return STRINGIFY(kTfLiteBuiltinUnpack);
			case kTfLiteBuiltinReduceMin:
				return STRINGIFY(kTfLiteBuiltinReduceMin);
			case kTfLiteBuiltinFloorDiv:
				return STRINGIFY(kTfLiteBuiltinFloorDiv);
			case kTfLiteBuiltinReduceAny:
				return STRINGIFY(kTfLiteBuiltinReduceAny);
			case kTfLiteBuiltinSquare:
				return STRINGIFY(kTfLiteBuiltinSquare);
			case kTfLiteBuiltinZerosLike:
				return STRINGIFY(kTfLiteBuiltinZerosLike);
			case kTfLiteBuiltinFill:
				return STRINGIFY(kTfLiteBuiltinFill);
			case kTfLiteBuiltinFloorMod:
				return STRINGIFY(kTfLiteBuiltinFloorMod);
			case kTfLiteBuiltinRange:
				return STRINGIFY(kTfLiteBuiltinRange);
			case kTfLiteBuiltinResizeNearestNeighbor:
				return STRINGIFY(kTfLiteBuiltinResizeNearestNeighbor);
			case kTfLiteBuiltinLeakyRelu:
				return STRINGIFY(kTfLiteBuiltinLeakyRelu);
			case kTfLiteBuiltinSquaredDifference:
				return STRINGIFY(kTfLiteBuiltinSquaredDifference);
			case kTfLiteBuiltinMirrorPad:
				return STRINGIFY(kTfLiteBuiltinMirrorPad);
			case kTfLiteBuiltinAbs:
				return STRINGIFY(kTfLiteBuiltinAbs);
			case kTfLiteBuiltinSplitV:
				return STRINGIFY(kTfLiteBuiltinSplitV);
			case kTfLiteBuiltinUnique:
				return STRINGIFY(kTfLiteBuiltinUnique);
			case kTfLiteBuiltinCeil:
				return STRINGIFY(kTfLiteBuiltinCeil);
			case kTfLiteBuiltinReverseV2:
				return STRINGIFY(kTfLiteBuiltinReverseV2);
			case kTfLiteBuiltinAddN:
				return STRINGIFY(kTfLiteBuiltinAddN);
			case kTfLiteBuiltinGatherNd:
				return STRINGIFY(kTfLiteBuiltinGatherNd);
			case kTfLiteBuiltinCos:
				return STRINGIFY(kTfLiteBuiltinCos);
			case kTfLiteBuiltinWhere:
				return STRINGIFY(kTfLiteBuiltinWhere);
			case kTfLiteBuiltinRank:
				return STRINGIFY(kTfLiteBuiltinRank);
			case kTfLiteBuiltinElu:
				return STRINGIFY(kTfLiteBuiltinElu);
			case kTfLiteBuiltinReverseSequence:
				return STRINGIFY(kTfLiteBuiltinReverseSequence);
			case kTfLiteBuiltinMatrixDiag:
				return STRINGIFY(kTfLiteBuiltinMatrixDiag);
			case kTfLiteBuiltinQuantize:
				return STRINGIFY(kTfLiteBuiltinQuantize);
			case kTfLiteBuiltinMatrixSetDiag:
				return STRINGIFY(kTfLiteBuiltinMatrixSetDiag);
			case kTfLiteBuiltinRound:
				return STRINGIFY(kTfLiteBuiltinRound);
			case kTfLiteBuiltinHardSwish:
				return STRINGIFY(kTfLiteBuiltinHardSwish);
			case kTfLiteBuiltinIf:
				return STRINGIFY(kTfLiteBuiltinIf);
			case kTfLiteBuiltinWhile:
				return STRINGIFY(kTfLiteBuiltinWhile);
			case kTfLiteBuiltinNonMaxSuppressionV4:
				return STRINGIFY(kTfLiteBuiltinNonMaxSuppressionV4);
			case kTfLiteBuiltinNonMaxSuppressionV5:
				return STRINGIFY(kTfLiteBuiltinNonMaxSuppressionV5);
			case kTfLiteBuiltinScatterNd:
				return STRINGIFY(kTfLiteBuiltinScatterNd);
			case kTfLiteBuiltinSelectV2:
				return STRINGIFY(kTfLiteBuiltinSelectV2);
			case kTfLiteBuiltinDensify:
				return STRINGIFY(kTfLiteBuiltinDensify);
			case kTfLiteBuiltinSegmentSum:
				return STRINGIFY(kTfLiteBuiltinSegmentSum);
			case kTfLiteBuiltinBatchMatmul:
				return STRINGIFY(kTfLiteBuiltinBatchMatmul);
			case kTfLiteBuiltinPlaceholderForGreaterOpCodes:
				return STRINGIFY(kTfLiteBuiltinPlaceholderForGreaterOpCodes);
			case kTfLiteBuiltinCumsum:
				return STRINGIFY(kTfLiteBuiltinCumsum);
			case kTfLiteBuiltinCallOnce:
				return STRINGIFY(kTfLiteBuiltinCallOnce);
			case kTfLiteBuiltinBroadcastTo:
				return STRINGIFY(kTfLiteBuiltinBroadcastTo);
			case kTfLiteBuiltinRfft2d:
				return STRINGIFY(kTfLiteBuiltinRfft2d);
			case kTfLiteBuiltinConv3d:
				return STRINGIFY(kTfLiteBuiltinConv3d);
			case kTfLiteBuiltinImag:
				return STRINGIFY(kTfLiteBuiltinImag);
			case kTfLiteBuiltinReal:
				return STRINGIFY(kTfLiteBuiltinReal);
			case kTfLiteBuiltinComplexAbs:
				return STRINGIFY(kTfLiteBuiltinComplexAbs);
			case kTfLiteBuiltinHashtable:
				return STRINGIFY(kTfLiteBuiltinHashtable);
			case kTfLiteBuiltinHashtableFind:
				return STRINGIFY(kTfLiteBuiltinHashtableFind);
			case kTfLiteBuiltinHashtableImport:
				return STRINGIFY(kTfLiteBuiltinHashtableImport);
			case kTfLiteBuiltinHashtableSize:
				return STRINGIFY(kTfLiteBuiltinHashtableSize);
			case kTfLiteBuiltinReduceAll:
				return STRINGIFY(kTfLiteBuiltinReduceAll);
			case kTfLiteBuiltinConv3dTranspose:
				return STRINGIFY(kTfLiteBuiltinConv3dTranspose);
			case kTfLiteBuiltinVarHandle:
				return STRINGIFY(kTfLiteBuiltinVarHandle);
			case kTfLiteBuiltinReadVariable:
				return STRINGIFY(kTfLiteBuiltinReadVariable);
			case kTfLiteBuiltinAssignVariable:
				return STRINGIFY(kTfLiteBuiltinAssignVariable);
			case kTfLiteBuiltinBroadcastArgs:
				return STRINGIFY(kTfLiteBuiltinBroadcastArgs);
			case kTfLiteBuiltinRandomStandardNormal:
				return STRINGIFY(kTfLiteBuiltinRandomStandardNormal);
			case kTfLiteBuiltinBucketize:
				return STRINGIFY(kTfLiteBuiltinBucketize);
			case kTfLiteBuiltinRandomUniform:
				return STRINGIFY(kTfLiteBuiltinRandomUniform);
			case kTfLiteBuiltinMultinomial:
				return STRINGIFY(kTfLiteBuiltinMultinomial);
			case kTfLiteBuiltinGelu:
				return STRINGIFY(kTfLiteBuiltinGelu);
			case kTfLiteBuiltinDynamicUpdateSlice:
				return STRINGIFY(kTfLiteBuiltinDynamicUpdateSlice);
			case kTfLiteBuiltinRelu0To1:
				return STRINGIFY(kTfLiteBuiltinRelu0To1);
			case kTfLiteBuiltinUnsortedSegmentProd:
				return STRINGIFY(kTfLiteBuiltinUnsortedSegmentProd);
			case kTfLiteBuiltinUnsortedSegmentMax:
				return STRINGIFY(kTfLiteBuiltinUnsortedSegmentMax);
			case kTfLiteBuiltinUnsortedSegmentSum:
				return STRINGIFY(kTfLiteBuiltinUnsortedSegmentSum);
			case kTfLiteBuiltinAtan2:
				return STRINGIFY(kTfLiteBuiltinAtan2);
			case kTfLiteBuiltinUnsortedSegmentMin:
				return STRINGIFY(kTfLiteBuiltinUnsortedSegmentMin);
			case kTfLiteBuiltinSign:
				return STRINGIFY(kTfLiteBuiltinSign);
			case kTfLiteBuiltinBitcast:
				return STRINGIFY(kTfLiteBuiltinBitcast);
			case kTfLiteBuiltinBitwiseXor:
				return STRINGIFY(kTfLiteBuiltinBitwiseXor);
			case kTfLiteBuiltinRightShift:
				return STRINGIFY(kTfLiteBuiltinRightShift);
			case kTfLiteBuiltinStablehloLogistic:
				return STRINGIFY(kTfLiteBuiltinStablehloLogistic);
			case kTfLiteBuiltinStablehloAdd:
				return STRINGIFY(kTfLiteBuiltinStablehloAdd);
			case kTfLiteBuiltinStablehloDivide:
				return STRINGIFY(kTfLiteBuiltinStablehloDivide);
			case kTfLiteBuiltinStablehloMultiply:
				return STRINGIFY(kTfLiteBuiltinStablehloMultiply);
			case kTfLiteBuiltinStablehloMaximum:
				return STRINGIFY(kTfLiteBuiltinStablehloMaximum);
			case kTfLiteBuiltinStablehloReshape:
				return STRINGIFY(kTfLiteBuiltinStablehloReshape);
			case kTfLiteBuiltinStablehloClamp:
				return STRINGIFY(kTfLiteBuiltinStablehloClamp);
			case kTfLiteBuiltinStablehloConcatenate:
				return STRINGIFY(kTfLiteBuiltinStablehloConcatenate);
			case kTfLiteBuiltinStablehloBroadcastInDim:
				return STRINGIFY(kTfLiteBuiltinStablehloBroadcastInDim);
			case kTfLiteBuiltinStablehloConvolution:
				return STRINGIFY(kTfLiteBuiltinStablehloConvolution);
			case kTfLiteBuiltinStablehloSlice:
				return STRINGIFY(kTfLiteBuiltinStablehloSlice);
			case kTfLiteBuiltinStablehloCustomCall:
				return STRINGIFY(kTfLiteBuiltinStablehloCustomCall);
			case kTfLiteBuiltinStablehloReduce:
				return STRINGIFY(kTfLiteBuiltinStablehloReduce);
			case kTfLiteBuiltinStablehloAbs:
				return STRINGIFY(kTfLiteBuiltinStablehloAbs);
			case kTfLiteBuiltinStablehloAnd:
				return STRINGIFY(kTfLiteBuiltinStablehloAnd);
			case kTfLiteBuiltinStablehloCosine:
				return STRINGIFY(kTfLiteBuiltinStablehloCosine);
			case kTfLiteBuiltinStablehloExponential:
				return STRINGIFY(kTfLiteBuiltinStablehloExponential);
			case kTfLiteBuiltinStablehloFloor:
				return STRINGIFY(kTfLiteBuiltinStablehloFloor);
			case kTfLiteBuiltinStablehloLog:
				return STRINGIFY(kTfLiteBuiltinStablehloLog);
			case kTfLiteBuiltinStablehloMinimum:
				return STRINGIFY(kTfLiteBuiltinStablehloMinimum);
			case kTfLiteBuiltinStablehloNegate:
				return STRINGIFY(kTfLiteBuiltinStablehloNegate);
			case kTfLiteBuiltinStablehloOr:
				return STRINGIFY(kTfLiteBuiltinStablehloOr);
			case kTfLiteBuiltinStablehloPower:
				return STRINGIFY(kTfLiteBuiltinStablehloPower);
			case kTfLiteBuiltinStablehloRemainder:
				return STRINGIFY(kTfLiteBuiltinStablehloRemainder);
			case kTfLiteBuiltinStablehloRsqrt:
				return STRINGIFY(kTfLiteBuiltinStablehloRsqrt);
			case kTfLiteBuiltinStablehloSelect:
				return STRINGIFY(kTfLiteBuiltinStablehloSelect);
			case kTfLiteBuiltinStablehloSubtract:
				return STRINGIFY(kTfLiteBuiltinStablehloSubtract);
			case kTfLiteBuiltinStablehloTanh:
				return STRINGIFY(kTfLiteBuiltinStablehloTanh);
			case kTfLiteBuiltinStablehloScatter:
				return STRINGIFY(kTfLiteBuiltinStablehloScatter);
			case kTfLiteBuiltinStablehloCompare:
				return STRINGIFY(kTfLiteBuiltinStablehloCompare);
			case kTfLiteBuiltinStablehloConvert:
				return STRINGIFY(kTfLiteBuiltinStablehloConvert);
			case kTfLiteBuiltinStablehloDynamicSlice:
				return STRINGIFY(kTfLiteBuiltinStablehloDynamicSlice);
			case kTfLiteBuiltinStablehloDynamicUpdateSlice:
				return STRINGIFY(kTfLiteBuiltinStablehloDynamicUpdateSlice);
			case kTfLiteBuiltinStablehloPad:
				return STRINGIFY(kTfLiteBuiltinStablehloPad);
			case kTfLiteBuiltinStablehloIota:
				return STRINGIFY(kTfLiteBuiltinStablehloIota);
			case kTfLiteBuiltinStablehloDotGeneral:
				return STRINGIFY(kTfLiteBuiltinStablehloDotGeneral);
			case kTfLiteBuiltinStablehloReduceWindow:
				return STRINGIFY(kTfLiteBuiltinStablehloReduceWindow);
			case kTfLiteBuiltinStablehloSort:
				return STRINGIFY(kTfLiteBuiltinStablehloSort);
			case kTfLiteBuiltinStablehloWhile:
				return STRINGIFY(kTfLiteBuiltinStablehloWhile);
			case kTfLiteBuiltinStablehloGather:
				return STRINGIFY(kTfLiteBuiltinStablehloGather);
			case kTfLiteBuiltinStablehloTranspose:
				return STRINGIFY(kTfLiteBuiltinStablehloTranspose);
			case kTfLiteBuiltinDilate:
				return STRINGIFY(kTfLiteBuiltinDilate);
			case kTfLiteBuiltinStablehloRngBitGenerator:
				return STRINGIFY(kTfLiteBuiltinStablehloRngBitGenerator);
			default:
				return "unknown " + std::to_string((int)builtin_code);
			}
		}

		std::string get_TfLitePadding(const TfLitePadding padding)
		{
			switch (padding)
			{
			case kTfLitePaddingUnknown:
				return STRINGIFY(kTfLitePaddingUnknown);
			case kTfLitePaddingSame:
				return STRINGIFY(kTfLitePaddingSame);
			case kTfLitePaddingValid:
				return STRINGIFY(kTfLitePaddingValid);
			default:
				return "unknown " + std::to_string((int)padding);
			}
		}

		std::string get_TfLiteFusedActivation(const TfLiteFusedActivation activation)
		{
			switch (activation)
			{
			case kTfLiteActNone:
				return STRINGIFY(kTfLiteActNone);
			case kTfLiteActRelu:
				return STRINGIFY(kTfLiteActRelu);
			case kTfLiteActReluN1To1:
				return STRINGIFY(kTfLiteActReluN1To1 = min(max(-1,x),1));
			case kTfLiteActRelu6:
				return STRINGIFY(kTfLiteActRelu6 = min(max(0,x),6));
			case kTfLiteActTanh:
				return STRINGIFY(kTfLiteActTanh);
			case kTfLiteActSignBit:
				return STRINGIFY(kTfLiteActSignBit);
			case kTfLiteActSigmoid:
				return STRINGIFY(kTfLiteActSigmoid);
			default:
				return "unknown " + std::to_string((int)activation);
			}
		}
		
		std::string get_TfLiteAllocationType(const TfLiteAllocationType allocation_type)
		{
			switch (allocation_type)
			{
			case kTfLiteMemNone:
				return STRINGIFY(kTfLiteMemNone);
			case kTfLiteMmapRo:
				return STRINGIFY(kTfLiteMmapRo);
			case kTfLiteArenaRw:
				return STRINGIFY(kTfLiteArenaRw);
			case kTfLiteArenaRwPersistent:
				return STRINGIFY(kTfLiteArenaRwPersistent);
			case kTfLiteDynamic:
				return STRINGIFY(kTfLiteDynamic);
			case kTfLitePersistentRo:
				return STRINGIFY(kTfLitePersistentRo);
			case kTfLiteCustom:
				return STRINGIFY(kTfLiteCustom);
			case kTfLiteVariantObject:
				return STRINGIFY(kTfLiteVariantObject);
			default:
				return "unknown " + std::to_string((int)allocation_type);
			}
		}

		std::string get_TfLiteQuantizationType(const TfLiteQuantizationType type)
		{
			switch (type)
			{
			case kTfLiteNoQuantization:
				return STRINGIFY(kTfLiteNoQuantization);
			case kTfLiteAffineQuantization:
				return STRINGIFY(kTfLiteAffineQuantization);
			default:
				return "unknown " + std::to_string((int)type);
			}
		}

		void LogTfLiteAffineQuantization(const TfLiteAffineQuantization* const affine_quantization)
		{
			// quantized_dimension specifies which dimension the scales and zero_points
			// correspond to.
			// For a particular value in quantized_dimension, quantized values can be
			// converted back to float using:
			//     real_value = scale * (quantized_value - zero_point)
			if (affine_quantization != nullptr)
			{
				std::cout << "Affine Quantization -> tensor quantized dimension: " << (*affine_quantization).quantized_dimension << std::endl;
				if ((*affine_quantization).scale != nullptr)
				{
					std::cout << "scales: ";
					for (int i = 0; i < (*affine_quantization).scale->size; i++)
					{
						std::cout << (*affine_quantization).scale->data[i] << " ";
					}
				}
				if ((*affine_quantization).zero_point != nullptr)
				{
					std::cout << "zero points: ";
					for (int i = 0; i < (*affine_quantization).zero_point->size; i++)
					{
						std::cout << (*affine_quantization).zero_point->data[i] << " ";
					}
				}
				std::cout << std::endl;
			}
			else
			{
				std::cout << "No affine quantization parameters" << std::endl;
			}
		}

		void LogTfLiteQuantization(const TfLiteQuantization& quantization)
		{
			std::cout << "Quantization type: " << get_TfLiteQuantizationType(quantization.type) << std::endl;
			if (quantization.type != kTfLiteNoQuantization)
			{
				LogTfLiteAffineQuantization(reinterpret_cast<TfLiteAffineQuantization*>(quantization.params));
			}
		}

		void LogTfLiteQuantizationParams(const TfLiteQuantizationParams& params)
		{
			std::cout << "Quantization params -> scale: " << params.scale;
			std::cout << " zero point: " << params.zero_point << std::endl;
		}

		void LogTfLitePaddingValues(const TfLitePaddingValues& padding)
		{
			std::cout << "TfLitePaddingValues =";
			std::cout << " width: " << padding.width;
			std::cout << " height: " << padding.height;
			std::cout << " width offset: " << padding.width_offset;
			std::cout << " height offset: " << padding.height_offset << std::endl;
		}

		void LogTfLiteTensor(const TfLiteTensor& tensor)
		{
			// TfLiteType type
			std::cout << "Information -> type: " << get_TfLiteType(tensor.type) << std::endl;

			// const char* name
			if (tensor.name != nullptr)
			{
				std::cout << "Name: " << tensor.name << std::endl;
			}
			else
			{
				std::cout << "Name unavailable" << std::endl;
			}

			// TfLiteIntArray* dims and TfLitePtrUnion data
			if (tensor.dims != nullptr)
			{
				std::cout << "tensor dims size: " << tensor.dims->size;
				std::cout << " dimensions shape: ";
				for (int j = 0; j < tensor.dims->size; j++)
				{
					std::cout << tensor.dims->data[j] << " ";
				}
				std::cout << std::endl;

				if (tensor.data.data != nullptr)
				{
					std::cout << "data: ";
					for (int j = 0; j < custom_ops::size_extraction(tensor.dims); j++)
					{
						switch (tensor.type)
						{
						case kTfLiteFloat32:
							std::cout << *(reinterpret_cast<float*>(tensor.data.data) + j) << " ";
							break;
						case kTfLiteInt32:
							std::cout << *(reinterpret_cast<int*>(tensor.data.data) + j) << " ";
							break;
						case kTfLiteInt16:
							std::cout << *(reinterpret_cast<short*>(tensor.data.data) + j) << " ";
							break;
						case kTfLiteInt8:
							// Unary operator to print signed char with numerical value through std::cout
							//std::cout << +*(reinterpret_cast<signed char*>(tensor.data.data) + j) << " ";
							break;
						default:
							std::cout << "Error: unsupported type: " << get_TfLiteType(tensor.type) << std::endl;
							break;
						}
					}
					std::cout << std::endl;
				}
				else
				{
					std::cout << "Data unavailable" << std::endl;
				}
			}
			else
			{
				std::cout << "no tensor dimensions available" << std::endl;
			}
			
			// TfLiteQuantizationParams params
			LogTfLiteQuantizationParams(tensor.params);

			// TfLiteAllocationType allocation_type
			std::cout << "Allocation type: " << get_TfLiteAllocationType(tensor.allocation_type) << std::endl;

			// size_t(unsigned long long) bytes
			std::cout << "Bytes: " << tensor.bytes << std::endl;

			// TfLiteBufferHandle(int) buffer_handle
			std::cout << "Buffer handle: " << tensor.buffer_handle << std::endl;

			// bool is_variable
			std::cout << "Is tensor variable: " << (tensor.is_variable ? "true" : "false") << std::endl;

			// TfLiteQuantization
			LogTfLiteQuantization(tensor.quantization);

			// const TfLiteIntArray* dims_signature
			if (tensor.dims_signature != nullptr)
			{
				std::cout << "tensor dims signature size: " << tensor.dims_signature->size;
				std::cout << " signature dimensions: ";
				for (int j = 0; j < tensor.dims_signature->size; j++)
				{
					std::cout << tensor.dims_signature->data[j] << " ";
				}
				std::cout << std::endl;
			}
		
		}

		void LogTfLiteContext(const TfLiteContext* const context)
		{
			// TfLiteContext logging
			std::cout << "###############################################################" << std::endl;
			std::cout << "::::::::::::::::::::TfLiteContext variables::::::::::::::::::::" << std::endl;
			std::cout << "Number of tensors in TfLiteContext: " << context->tensors_size << std::endl;
			for (int i = 0; i < context->tensors_size; i++)
			{
				std::cout << "---------------------------------------------------------------" << std::endl;
				std::cout << "Tensor " << i << std::endl;
				LogTfLiteTensor(context->tensors[i]);
			}
			// To have access to the Subgraph class you should include
			// tensorflow/lite/core/subgraph.h
			// And add the include directory
			// C:\Users\rosal\tensorflow source\tflite_c_build\flatbuffers\include
			// context->impl_ is casted into a Subgraph
			if (context->impl_ != nullptr)
			{
				//std::cout << "..............................................................." << std::endl;
				//Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);
				//std::cout << "Subgraph name: " << subgraph->GetName() << std::endl;
				//std::cout << "tensors in subgraph: " << subgraph->tensors_size() << std::endl;
				//std::cout << "number of operations in subgraph: " << subgraph->nodes_size() << std::endl;
				//// The following are filled when calling them from the interpreter				
				//const auto& inputs = subgraph->inputs();
				//const std::vector<int>& outputs = subgraph->outputs();
				//const std::vector<int>& variables = subgraph->variables();
				//std::cout << "subgraph number of inputs: " << inputs.size() << std::endl;
				//std::cout << "subgraph number of outputs: " << outputs.size() << std::endl;
				//std::cout << "subgraph number of variables: " << variables.size() << std::endl;
			}
			std::cout << "---------------------------------------------------------------" << std::endl;
			std::cout << "Recommended number of threads " << context->recommended_num_threads << std::endl;
			std::cout << "Allow relax fp32 to fp16? " << (context->allow_fp32_relax_to_fp16 ? "true" : "false") << std::endl;
			std::cout << "###############################################################" << std::endl;
		}
	
		void LogTfLiteConvParams(const TfLiteConvParams* const params)
		{
			std::cout << "TfLiteConvParams:" << std::endl;
			std::cout << "-----------------" << std::endl;
			std::cout << "Padding: " << get_TfLitePadding(params->padding) << std::endl;
			std::cout << "Stride width: " << params->stride_width;
			std::cout << " height: " << params->stride_height << std::endl;
			std::cout << "Activation: " << get_TfLiteFusedActivation(params->activation) << std::endl;
			std::cout << "Dilation width: " << params->dilation_width_factor;
			std::cout << " height: " << params->dilation_height_factor << std::endl;
			std::cout << "Quantized bias type: " << get_TfLiteType(params->quantized_bias_type) << std::endl;
			std::cout << "-----------------" << std::endl;
		}

		void LogTfLiteNode(const TfLiteNode* const node)
		{
			// TfLiteNode logging
			std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
			std::cout << "::::::::::::::::::::::TfLiteNode variables:::::::::::::::::::::" << std::endl;
			std::cout << "number of inputs: " << node->inputs->size;
			std::cout << " tensor indexes: ";
			for (int i = 0; i < node->inputs->size; i++)
				std::cout << node->inputs->data[i] << " ";
			std::cout << std::endl;
			std::cout << "number of outputs: " << node->outputs->size;
			std::cout << " tensor indexes: ";
			for (int i = 0; i < node->outputs->size; i++)
				std::cout << node->outputs->data[i] << " ";
			std::cout << std::endl;
			std::cout << "number of intermediates: " << node->intermediates->size;
			std::cout << " tensor indexes: ";
			for (int i = 0; i < node->intermediates->size; i++)
				std::cout << node->intermediates->data[i] << " ";
			std::cout << std::endl;
			std::cout << "number of temporaries: " << node->temporaries->size;
			std::cout << " tensor indexes: ";
			for (int i = 0; i < node->temporaries->size; i++)
				std::cout << node->temporaries->data[i] << " ";
			std::cout << std::endl;

			// node->user_data can be casted to
			// SimpleDelegateKernelInterface*
			// MyDelegateKernel*
			// tflite::ops::builtin::conv::OpData*
			// tflite::ops::builtin::activations::OpData*
			// tflite::ops::builtin::add::OpData* // Which are structs that contain builtin data information for the builtin operations
			// TfLiteNode* if called from an opaque qualifier
			// tflite::xnnpack::::Subgraph*
			if (node->user_data != nullptr)
			{
				custom_ops::conv::OpData* data = reinterpret_cast<custom_ops::conv::OpData*>(node->user_data);
				conv::LogTfLiteOpData(data);
			}

			// node->builtin_data can be casted to
			// - TfLiteConvParams*
			// - TfLiteFullyConnectedParams*
			// - TfLiteAddParams*
			// And more, definitions are located in builtin_op_data.h
			if (node->builtin_data != nullptr)
			{
				TfLiteConvParams* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
				LogTfLiteConvParams(params);
			}

			// node->custom_initial_data can be casted to
			// uint32* or uint64*
			std::cout << "custom initial data size: " << node->custom_initial_data_size << std::endl;
			std::cout << "side effects: " << ((node->might_have_side_effect) ? "true" : "false") << std::endl;
		}

		void LogTfLiteRegistration(const TfLiteRegistration* const registration)
		{
			// TfLiteRegistration logging
			std::cout << "===============================================================" << std::endl;
			std::cout << "::::::::::::::::::TfLiteRegistration variables:::::::::::::::::" << std::endl;
			// To check what these functions do, check the
			// add.cc conv.cc activation.cc corresponding methods
			// init = Init
			// free = Free
			// prepare = Prepare
			// invoke = Eval

			//// tflite::ops::builtin::??::Init
			//if (registration->init != nullptr)
			//{
			//	std::cout << "init method is defined" << std::endl;
			//}
			//else
			//	std::cout << "init method is null" << std::endl;
			//// tflite::ops::builtin::??::Free
			//if (registration->free != nullptr)
			//{
			//	std::cout << "free method is defined" << std::endl;
			//}
			//else
			//	std::cout << "free method is null" << std::endl;
			//// tflite::ops::builtin::??::Prepare
			//if (registration->prepare != nullptr)
			//{
			//	std::cout << "prepare method is defined" << std::endl;
			//}
			//else
			//	std::cout << "prepare method is null" << std::endl;
			//// tflite::ops::builtin::??::Eval
			//if (registration->invoke != nullptr)
			//{
			//	std::cout << "invoke method is defined" << std::endl;
			//}
			//else
			//	std::cout << "invoke method is null" << std::endl;
			
			// This value is always a nullptr so far
			// profiling_string == nullptr

			std::cout << "Builtin code: " << get_builtin_code(registration->builtin_code) << std::endl;
			
			//if(registration->custom_name != nullptr)
			//	std::cout << "Custom name: " << *(registration->custom_name) << std::endl;
			//else
			//	std::cout << "No custom name" << std::endl;
			//std::cout << "Version: " << registration->version << std::endl;
			//// Looks that for C applications registration external is nullptr
			//if (registration->registration_external != nullptr)
			//{
			//	std::cout << "registration_external is defined" << std::endl;
			//}
			//else
			//	std::cout << "registration_external is null" << std::endl;
			// This value is always a nullptr so far
			// async_kernel == nullptr (method that returns a struct TfLiteAsyncKernel)

			// suposedly it is 4|8 = 12 for sum
			// for custom delegate it is uninitialized
			std::cout << "Inplace operator: " << registration->inplace_operator << std::endl;
		}

		void LogTfLiteDelegateParams(const TfLiteDelegateParams* const params)
		{
			// TfLiteDelegateParams logging
			std::cout << "***************************************************************" << std::endl;
			std::cout << ":::::::::::::::::TfLiteDelegateParams variables::::::::::::::::" << std::endl;
			std::cout << "Nodes to replace information" << std::endl;
			std::cout << "number of nodes to replace: " << params->nodes_to_replace->size;
			std::cout << " nodes indexes: ";
			for (int i = 0; i < params->nodes_to_replace->size; i++)
			{
				std::cout << params->nodes_to_replace->data[i] << " ";
			}
			std::cout << std::endl << "Input tensors" << std::endl;
			std::cout << "data size: " << params->input_tensors->size;
			std::cout << " tensor indexes: ";
			for (int i = 0; i < params->input_tensors->size; i++)
			{
				std::cout << params->input_tensors->data[i] << " ";
			}
			std::cout << std::endl << "Output tensors" << std::endl;
			std::cout << "data size: " << params->output_tensors->size;
			std::cout << " tensor indexes: ";
			for (int i = 0; i < params->output_tensors->size; i++)
			{
				std::cout << params->output_tensors->data[i] << " ";
			}
			std::cout << std::endl;
			LogTfLiteDelegate(params->delegate);
			std::cout << "***************************************************************" << std::endl;
		}

		void LogTfLiteDelegate(const TfLiteDelegate* const delegate)
		{
			// delegate->data_ can be casted to
			// ExternalDelegateWrapper*
			// SimpleDelegateInterface*
			// MyDelegate*
			std::cout << "TfLiteDelegate: " << std::endl;
			if (delegate->data_ != nullptr)
			{
				std::cout << "MyDelegate present!" << std::endl;
				auto data = reinterpret_cast<MyDelegate*>(delegate->data_);
				std::cout << "MyDelegate name: " << data->Name() << std::endl;
			}
			if (delegate->Prepare != nullptr)
			{
				std::cout << "Prepare function is declared" << std::endl;
			}
			if (delegate->CopyFromBufferHandle != nullptr)
			{
				std::cout << "CopyFromBufferHandle function is declared" << std::endl;
			}
			if (delegate->CopyToBufferHandle != nullptr)
			{
				std::cout << "CopyToBufferHandle function is declared" << std::endl;
			}
			if (delegate->FreeBufferHandle != nullptr)
			{
				std::cout << "FreeBufferHandle function is declared" << std::endl;
			}
			std::cout << "Flags: " << get_TfLiteDelegateFlags((TfLiteDelegateFlags)delegate->flags) << std::endl;
			if (delegate->opaque_delegate_builder)
			{
				std::cout << "Opaque delegate builder present" << std::endl;
			}

		}

	}
}