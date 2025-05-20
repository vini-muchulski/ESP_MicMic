// ============================================================================
// Inferência TFLite-Micro no ESP32-C3 (modelo de seno, float32)
// ============================================================================

#include <Arduino.h>

// --------- TensorFlow Lite-Micro ------------------------------------------------
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --------- Modelo convertido ----------------------------------------------------
#include "model_seno_data.h"     // modelo_seno_tflite / modelo_seno_tflite_len

// --------------------------- Configuração de memória ----------------------------
constexpr int   kTensorArenaSize = 12 * 1024;          // 12 kB
static uint8_t  tensor_arena[kTensorArenaSize];

// ---------------------------- Variáveis globais ---------------------------------
namespace {
  // Relator de erros
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter*     error_reporter = &micro_error_reporter;

  // Ponteiros principais
  const tflite::Model*     model       = nullptr;
  tflite::AllOpsResolver   resolver;                  // todas as ops
  tflite::MicroInterpreter* interpreter  = nullptr;

  // Tensores de entrada/saída
  TfLiteTensor* input  = nullptr;
  TfLiteTensor* output = nullptr;
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);
  delay(200);

  // 1) Carrega o modelo do array em flash/RAM
  model = tflite::GetModel(modelo_seno_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
        "Versão do modelo (%d) != schema (%d).",
        model->version(), TFLITE_SCHEMA_VERSION);
    while (true);   // trava
  }

  // 2) Cria o intérprete (passa arena + tamanho)
  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize,
      /* resource_variables */ nullptr,
      /* profiler           */ nullptr);
  interpreter = &static_interpreter;

  // 3) Aloca tensores
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() falhou.");
    while (true);
  }

  // 4) Ponteiros de entrada e saída
  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup concluído — modelo pronto para inferir!");
}

// ============================================================================
// LOOP
// ============================================================================
void loop() {
  // Preenche a entrada com um sweep 0‒2π
  const int len = input->bytes / sizeof(float);
  for (int i = 0; i < len; ++i) {
    float x = (2.0f * 3.14159265f) * i / (len - 1);
    input->data.f[i] = x;
  }

  // Executa a inferência
  if (interpreter->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke() falhou.");
    delay(500);
    return;
  }

  // Imprime resultados
  const int out_len = output->bytes / sizeof(float);
  for (int i = 0; i < out_len; ++i) {
    Serial.print("y[");
    Serial.print(i);
    Serial.print("] = ");
    Serial.println(output->data.f[i], 6);
  }
  Serial.println("----------------------------------");
  delay(1000);
}
