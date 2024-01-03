#include <iostream>
#include "DelegateCore.h"

// Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

    // plugin to create a TfLiteDelegate from MyDelegate
    TFL_CAPI_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
        char** options_keys, char** options_values, size_t num_options,
        void (*report_error)(const char*)) 
    {
        // Constructor has to be called explicitly, use make_unique function for exception safety
        // Option keys and values are received here from python
        if (num_options > 0)
        {
            tflite::MyDelegateOptions options(options_keys, options_values, num_options);
            return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(std::make_unique<tflite::MyDelegate>(options)));
        }
        else
        {
            return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(std::make_unique<tflite::MyDelegate>()));
        }
    }

    // plugin to destroy a TfLiteDelegate from MyDelegate
    TFL_CAPI_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) 
    {
        tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
    }

#ifdef __cplusplus
}
#endif  // __cplusplus