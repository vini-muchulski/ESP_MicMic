#include <Arduino.h>
#include <pgmspace.h>
#include <cmath>
#include <climits>

#include "mnist_model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const uint8_t mnist_sample[28 * 28] PROGMEM = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  84, 185, 159, 151,  60,  36,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 222, 254, 254, 254, 254, 241, 198, 198, 198, 198,
    198, 198, 198, 198, 170,  52,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,  67, 114,  72, 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254,
    254, 140,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  17,  66,  14,  67,  67,  67,  59,  21, 236, 254, 106,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,  83, 253, 209,  18,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,  22, 233, 255,  83,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 129, 254, 238,
    44,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,  59, 249, 254,  62,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0, 133, 254, 187,   5,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   9,
    205, 248,  58,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 126, 254, 182,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,  75, 251, 240,  57,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,  19, 221, 254, 166,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3, 203, 254, 219,
    35,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,  38, 254, 254,  77,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  31, 224, 254, 115,   1,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 133,
    254, 254,  52,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  61, 242, 254, 254,  52,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 121, 254, 254, 219,  40,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0, 121, 254, 207,  18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};

namespace {
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input_tensor = nullptr;
    TfLiteTensor* output_tensor = nullptr;

    // Reduzido de 128KB para 80KB
    constexpr int kTensorArenaSize = 80 * 1024;
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
    Serial.begin(115200);
    delay(500);
    
    Serial.printf("Memoria livre no inicio: %d bytes\n", esp_get_free_heap_size());
    Serial.printf("Tamanho do tensor arena: %d bytes\n", kTensorArenaSize);

    model = tflite::GetModel(mnist_cnn_small_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Schema do modelo incompativel: v%d != v%d",
                               model->version(), TFLITE_SCHEMA_VERSION);
        while (1) delay(1000);
    }

    // Use MicroMutableOpResolver em vez de AllOpsResolver para economizar memória
    tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        error_reporter->Report("Falha ao alocar tensores. Arena necessaria: %lu bytes", 
                               (unsigned long)interpreter->arena_used_bytes());
        while (1) delay(1000);
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.printf("Arena usada: %lu de %d bytes (%.1f%%)\n", 
                  (unsigned long)interpreter->arena_used_bytes(), 
                  kTensorArenaSize,
                  100.0 * interpreter->arena_used_bytes() / kTensorArenaSize);
    Serial.printf("Memoria livre apos setup: %d bytes\n", esp_get_free_heap_size());
    Serial.println("Setup concluido com sucesso!");
}

void loop() {
    constexpr int kImageSize = 28 * 28;
    const bool is_quantized_input = (input_tensor->type == kTfLiteInt8 || input_tensor->type == kTfLiteUInt8);
    const float input_scale = input_tensor->params.scale;
    const int32_t input_zero_point = input_tensor->params.zero_point;

    for (int i = 0; i < kImageSize; ++i) {
        uint8_t pixel = pgm_read_byte(&mnist_sample[i]);
        if (is_quantized_input) {
            float normalized_pixel = pixel / 255.0f;
            int32_t quantized_value = lrintf(normalized_pixel / input_scale) + input_zero_point;
            if (input_tensor->type == kTfLiteUInt8) {
                input_tensor->data.uint8[i] = static_cast<uint8_t>(max(0, min(255, quantized_value)));
            } else {
                input_tensor->data.int8[i] = static_cast<int8_t>(max(-128, min(127, quantized_value)));
            }
        } else {
            input_tensor->data.f[i] = pixel / 255.0f;
        }
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        error_reporter->Report("Falha no Invoke()");
        delay(1000);
        return;
    }

    int best_index = 0;
    float best_score = -__FLT_MAX__;

    const bool is_quantized_output = (output_tensor->type == kTfLiteInt8 || output_tensor->type == kTfLiteUInt8);
    const float output_scale = output_tensor->params.scale;
    const int32_t output_zero_point = output_tensor->params.zero_point;
    const int output_size = output_tensor->dims->data[1];

    for (int i = 0; i < output_size; ++i) {
        float current_score;
        if (is_quantized_output) {
            int32_t raw_value = (output_tensor->type == kTfLiteUInt8)
                                    ? output_tensor->data.uint8[i]
                                    : output_tensor->data.int8[i];
            current_score = (raw_value - output_zero_point) * output_scale;
        } else {
            current_score = output_tensor->data.f[i];
        }

        if (current_score > best_score) {
            best_score = current_score;
            best_index = i;
        }
    }

    Serial.printf("Predicao MNIST: %d (score=%.3f)\n", best_index, best_score);
    delay(2000);
}