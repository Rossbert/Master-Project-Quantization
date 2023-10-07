# Steps for making things work on Windows

# Install latest VS Community version (2022)

# Error log
For some reason using a subtract-layer right after a sum-layer doesn't work on the delegate.
By calling the Interpreter.invoke the program exits without even raising an Exemption 



// The following are included in simple_delegate.cc and therefore not needed in MyDelegateCore.h
//#include "tensorflow/lite/delegates/utils.h"
//#include "tensorflow/lite/context_util.h"
//#include "tensorflow/lite/kernels/internal/compatibility.h"
//#include "tensorflow/lite/minimal_logging.h"

In TfLite the order of arrays is allocated in a transposed-like manner
e.g.
tensorflow shape = (4, 6)
tensorflow tensor
[
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24]
]

tflite shape = (4, 6)
[
    [1, 5, 9, 13, 17, 21,
    2, 6, 10, 14, 18, 22,
    3, 7, 11, 15, 19, 23,
    4, 8, 12, 16, 20, 24]
]

# Inside plugin delegate create
//std::unique_ptr<tflite::MyDelegate> my_delegate(new tflite::MyDelegate());
//return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(my_delegate));

// Alternative without creating the pointer in a variable
//return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(std::unique_ptr<tflite::MyDelegate>(new tflite::MyDelegate())));
        

# Extra things unneeded
//#include <tensorflow/lite/tools/delegates/delegate_provider.h>

//#include <tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.h>
//#include <tensorflow/lite/tools/command_line_flags.h>
//#include <tensorflow/lite/tools/logging.h>
//// Compiler requires following .cc added to the project:
//// dummy_delegate.cc
//// command_line_flags.cc
//// Linker requires:
//// absl_strings.lib
//// absl_strings_internal.lib
//// Don't forget to add the path for the headers
//namespace tflite {
//    namespace tools {
//
//        TfLiteDelegate* CreateDummyDelegateFromOptions(
//            const char* const* options_keys, const char* const* options_values,
//            size_t num_options) {
//            DummyDelegateOptions options = TfLiteDummyDelegateOptionsDefault();
//
//            // Parse key-values options to DummyDelegateOptions by mimicking them as
//            // command-line flags.
//            std::vector<const char*> argv;
//            argv.reserve(num_options + 1);
//            constexpr char kDummyDelegateParsing[] = "dummy_delegate_parsing";
//            argv.push_back(kDummyDelegateParsing);
//
//            std::vector<std::string> option_args;
//            option_args.reserve(num_options);
//            for (int i = 0; i < num_options; ++i) {
//                option_args.emplace_back("--");
//                option_args.rbegin()->append(options_keys[i]);
//                option_args.rbegin()->push_back('=');
//                option_args.rbegin()->append(options_values[i]);
//                argv.push_back(option_args.rbegin()->c_str());
//            }
//
//            constexpr char kAllowedBuiltinOp[] = "allowed_builtin_code";
//            constexpr char kReportErrorDuringInit[] = "error_during_init";
//            constexpr char kReportErrorDuringPrepare[] = "error_during_prepare";
//            constexpr char kReportErrorDuringInvoke[] = "error_during_invoke";
//
//            std::vector<tflite::Flag> flag_list = {
//                tflite::Flag::CreateFlag(kAllowedBuiltinOp, &options.allowed_builtin_code,
//                                         "Allowed builtin code."),
//                tflite::Flag::CreateFlag(kReportErrorDuringInit,
//                                         &options.error_during_init,
//                                         "Report error during init."),
//                tflite::Flag::CreateFlag(kReportErrorDuringPrepare,
//                                         &options.error_during_prepare,
//                                         "Report error during prepare."),
//                tflite::Flag::CreateFlag(kReportErrorDuringInvoke,
//                                         &options.error_during_invoke,
//                                         "Report error during invoke."),
//            };
//
//            int argc = num_options + 1;
//            if (!tflite::Flags::Parse(&argc, argv.data(), flag_list)) {
//                return nullptr;
//            }
//
//            TFLITE_LOG(INFO) << "Dummy delegate: allowed_builtin_code set to "
//                << options.allowed_builtin_code << ".";
//            TFLITE_LOG(INFO) << "Dummy delegate: error_during_init set to "
//                << options.error_during_init << ".";
//            TFLITE_LOG(INFO) << "Dummy delegate: error_during_prepare set to "
//                << options.error_during_prepare << ".";
//            TFLITE_LOG(INFO) << "Dummy delegate: error_during_invoke set to "
//                << options.error_during_invoke << ".";
//
//            return TfLiteDummyDelegateCreate(&options);
//        }
//
//    }  // namespace tools
//}  // namespace tflite


## To make use of the Subgraph class
- Include 
#include <tensorflow/lite/core/subgraph.h>
- Add the include directory
C:\Users\rosal\tensorflow source\tflite_c_build\flatbuffers\include
