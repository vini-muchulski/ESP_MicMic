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




#define modelo_seno_tflite         modelo_seno_float32_tflite
#define modelo_seno_tflite_len     modelo_seno_float32_tflite_len

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
  // 1) Liste aqui os ângulos desejados (em radianos)
  constexpr float angles[] = {
    3.14159265f / 3.0f,   // π/3
    3.14159265f / 6.0f,   // π/6
    3.14159265f / 4.0f,   // π/4
    3.14159265f / 2.0f    // π/2
  };
  constexpr int num_angles = sizeof(angles) / sizeof(angles[0]);

  // 2) Para cada ângulo, faz a inferência e imprime
  for (int i = 0; i < num_angles; ++i) {
    float x = angles[i];
    input->data.f[0] = x;                  // ajusta o único valor de entrada

    if (interpreter->Invoke() != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke() falhou.");
      return;
    }

    float y = output->data.f[0];           // resultado para esse x
    Serial.print("sin(");
    Serial.print(x, 6);
    Serial.print(") = ");
    Serial.println(y, 6);
  }

  Serial.println("----------------------------");
  delay(1000);
}

