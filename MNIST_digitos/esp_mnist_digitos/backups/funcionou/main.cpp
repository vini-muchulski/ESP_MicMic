#include <Arduino.h>
#include <cmath>
#include <climits>

#include "mnist_model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Coloque a amostra MNIST na PROGMEM para economizar RAM
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
    198, 198, 198, 170,  52,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
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
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};

namespace {
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input_tensor = nullptr;
    TfLiteTensor* output_tensor = nullptr;

    // Aumentar arena - 80KB pode não ser suficiente para alguns modelos
    constexpr int kTensorArenaSize = 80 * 1024;
    alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
    
    // Variável global para controlar execução
    bool model_initialized = false;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.printf("Memoria livre no inicio: %d bytes\n", esp_get_free_heap_size());
    Serial.printf("Tamanho do tensor arena: %d bytes\n", kTensorArenaSize);
    
    // Verificar se há memória suficiente
    if (esp_get_free_heap_size() < kTensorArenaSize) {
        Serial.println("ERRO: Memoria insuficiente!");
        while (1) delay(1000);
    }
    
    // Carregar modelo com verificação de ponteiro nulo
    model = tflite::GetModel(mnist_cnn_small_int8_tflite);
    if (model == nullptr) {
        Serial.println("ERRO: Falha ao carregar modelo - ponteiro nulo!");
        while (1) delay(1000);
    }
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("ERRO: Schema do modelo incompativel: v%d != v%d\n",
                      model->version(), TFLITE_SCHEMA_VERSION);
        while (1) delay(1000);
    }
    Serial.println("Modelo carregado com sucesso");

    // Use MicroMutableOpResolver em vez de AllOpsResolver para economizar memória
    static tflite::MicroMutableOpResolver<8> resolver;
    
    // Adicionar operadores com verificação de erro
    TfLiteStatus status;
    status = resolver.AddConv2D();
    if (status != kTfLiteOk) Serial.println("Erro ao adicionar Conv2D");
    
    status = resolver.AddMaxPool2D();
    if (status != kTfLiteOk) Serial.println("Erro ao adicionar MaxPool2D");
    
    status = resolver.AddReshape();
    if (status != kTfLiteOk) Serial.println("Erro ao adicionar Reshape");
    
    status = resolver.AddFullyConnected();
    if (status != kTfLiteOk) Serial.println("Erro ao adicionar FullyConnected");
    
    status = resolver.AddSoftmax();
    if (status != kTfLiteOk) Serial.println("Erro ao adicionar Softmax");
    
    status = resolver.AddQuantize();
    if (status != kTfLiteOk) Serial.println("Erro ao adicionar Quantize");
    
    status = resolver.AddDequantize();
    if (status != kTfLiteOk) Serial.println("Erro ao adicionar Dequantize");

    status = resolver.AddMean();
    if (status != kTfLiteOk) Serial.println("Erro ao adicionar Mean");
    
    Serial.println("Operadores adicionados");
    
    // Criar interpretador
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Alocar tensores com verificação detalhada
    Serial.println("Alocando tensores...");
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    
    if (allocate_status != kTfLiteOk) {
        Serial.printf("ERRO: Falha ao alocar tensores. Status: %d\n", allocate_status);
        Serial.printf("Arena necessaria: %lu bytes\n", 
                      (unsigned long)interpreter->arena_used_bytes());
        Serial.printf("Arena disponivel: %d bytes\n", kTensorArenaSize);
        while (1) delay(1000);
    }

    // Obter tensores de entrada e saída com verificação
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    if (input_tensor == nullptr) {
        Serial.println("ERRO: Tensor de entrada é nulo!");
        while (1) delay(1000);
    }
    
    if (output_tensor == nullptr) {
        Serial.println("ERRO: Tensor de saída é nulo!");
        while (1) delay(1000);
    }
    
    // Verificar dimensões dos tensores
    Serial.printf("Tensor de entrada - tipo: %d, dims: %d\n", 
                  input_tensor->type, input_tensor->dims->size);
    if (input_tensor->dims->size >= 1) {
        Serial.printf("Tamanho esperado entrada: %d elementos\n", 
                      input_tensor->bytes / sizeof(float));
    }
    
    Serial.printf("Tensor de saída - tipo: %d, dims: %d\n", 
                  output_tensor->type, output_tensor->dims->size);

    Serial.printf("Arena usada: %lu de %d bytes (%.1f%%)\n", 
                  (unsigned long)interpreter->arena_used_bytes(), 
                  kTensorArenaSize,
                  100.0 * interpreter->arena_used_bytes() / kTensorArenaSize);
    Serial.printf("Memoria livre apos setup: %d bytes\n", esp_get_free_heap_size());
    
    model_initialized = true;
    Serial.println("Setup concluido com sucesso!");
}

void loop() {
    // Verificar se o modelo foi inicializado corretamente
    if (!model_initialized) {
        Serial.println("Modelo nao inicializado, pulando loop...");
        delay(2000);
        return;
    }
    
    // Verificar ponteiros antes de usar
    if (interpreter == nullptr || input_tensor == nullptr || output_tensor == nullptr) {
        Serial.println("ERRO: Ponteiros nulos detectados!");
        delay(2000);
        return;
    }
    
    Serial.println("Preparando dados de entrada...");
    
    constexpr int kImageSize = 28 * 28;
    const bool is_quantized_input = (input_tensor->type == kTfLiteInt8 || input_tensor->type == kTfLiteUInt8);
    const float input_scale = input_tensor->params.scale;
    const int32_t input_zero_point = input_tensor->params.zero_point;

    // Preencher tensor de entrada com verificação de bounds
    for (int i = 0; i < kImageSize; ++i) {
        // Ler da PROGMEM
        uint8_t pixel = pgm_read_byte(&mnist_sample[i]);
        
        if (is_quantized_input) {
            float normalized_pixel = pixel / 255.0f;
            int32_t quantized_value = lrintf(normalized_pixel / input_scale) + input_zero_point;
            
            if (input_tensor->type == kTfLiteUInt8) {
                quantized_value = max(0, min(255, quantized_value));
                input_tensor->data.uint8[i] = static_cast<uint8_t>(quantized_value);
            } else if (input_tensor->type == kTfLiteInt8) {
                quantized_value = max(-128, min(127, quantized_value));
                input_tensor->data.int8[i] = static_cast<int8_t>(quantized_value);
            }
        } else {
            input_tensor->data.f[i] = pixel / 255.0f;
        }
    }

    Serial.println("Executando inferencia...");
    
    // Executar inferência com verificação de erro
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.printf("ERRO: Falha no Invoke(), status: %d\n", invoke_status);
        delay(2000);
        return;
    }

    // Processar resultados
    int best_index = 0;
    float best_score = -INFINITY;

    const bool is_quantized_output = (output_tensor->type == kTfLiteInt8 || output_tensor->type == kTfLiteUInt8);
    const float output_scale = output_tensor->params.scale;
    const int32_t output_zero_point = output_tensor->params.zero_point;
    const int output_size = output_tensor->dims->data[output_tensor->dims->size - 1];

    Serial.printf("Analisando %d classes...\n", output_size);

    for (int i = 0; i < output_size; ++i) {
        float current_score;
        
        if (is_quantized_output) {
            int32_t raw_value;
            if (output_tensor->type == kTfLiteUInt8) {
                raw_value = output_tensor->data.uint8[i];
            } else {
                raw_value = output_tensor->data.int8[i];
            }
            current_score = (raw_value - output_zero_point) * output_scale;
        } else {
            current_score = output_tensor->data.f[i];
        }

        if (current_score > best_score) {
            best_score = current_score;
            best_index = i;
        }
    }

    Serial.printf("=== RESULTADO ===\n");
    Serial.printf("Predicao MNIST: %d\n", best_index);
    Serial.printf("Score: %.6f\n", best_score);
    Serial.printf("Memoria livre: %d bytes\n", esp_get_free_heap_size());
    Serial.println("================\n");
    
    delay(3000);
}