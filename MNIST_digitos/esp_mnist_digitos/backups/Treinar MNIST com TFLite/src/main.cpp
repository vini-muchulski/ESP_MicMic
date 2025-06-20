#include <Arduino.h>
#include <pgmspace.h>

#include "mnist_model_data.h"  // gerado com xxd -i e modificado para PROGMEM
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include <cfloat>

// Externas definidas em mnist_model_data.h (com PROGMEM)
// Externas definidas em mnist_model_data.h
extern unsigned char mnist_cnn_small_tflite[];      // símbolo correto do header
extern unsigned int  mnist_cnn_small_tflite_len;    // comprimento correto    // comprimento correto


// Exemplo de imagem MNIST em Flash (PROGMEM): substitua pelos 784 valores reais
static const uint8_t mnist_sample[28 * 28] PROGMEM =
{
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 185, 159, 151, 60, 36,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 222, 254, 254, 254, 254, 241, 198, 198, 198, 198,
  198, 198, 198, 198, 170, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 67, 114, 72, 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254,
  254, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 17, 66, 14, 67, 67, 67, 59, 21, 236, 254, 106, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 83, 253, 209, 18, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 22, 233, 255, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 129, 254, 238,
  44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 59, 249, 254, 62, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 133, 254, 187, 5, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,
  205, 248, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 254, 182, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 75, 251, 240, 57, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 19, 221, 254, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 203, 254, 219,
  35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 38, 254, 254, 77, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 31, 224, 254, 115, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133,
  254, 254, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 242, 254, 254, 52, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 121, 254, 254, 219, 40, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 121, 254, 207, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
























namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = nullptr;
  tflite::AllOpsResolver resolver;

  // Arena reduzida após medição em ~64 KB
  constexpr int kTensorArenaSize = 64 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
}

void setup() {
  Serial.begin(115200);
  delay(10);

  // Carrega o modelo em Flash
model = tflite::GetModel(mnist_cnn_small_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Schema %d != %d", model->version(), TFLITE_SCHEMA_VERSION
    );
    while (1);
  }

  // Cria o interpreter
  interpreter = new tflite::MicroInterpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );

  // Aloca tensores
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("Falha ao alocar tensores");
    while (1);
  }

  // Reporta uso de arena
  Serial.printf("Arena usada: %d bytes\n",
    interpreter->arena_used_bytes()
  );

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup concluído.");
}

void loop() {
  // 1) Lê e normaliza a imagem de PROGMEM
  const bool is_q_input = 
    (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8);
  const float in_scale  = input->params.scale;
  const int   in_zp     = input->params.zero_point;

  for (int i = 0; i < 28 * 28; ++i) {
    uint8_t pixel = pgm_read_byte(&mnist_sample[i]);
    float norm = pixel / 255.0f;
    if (is_q_input) {
      int32_t q = lrintf(norm / in_scale) + in_zp;
      if (input->type == kTfLiteUInt8) {
        input->data.uint8[i] = (uint8_t)max(0, min(255, q));
      } else {
        input->data.int8[i]  = (int8_t)max(-128, min(127, q));
      }
    } else {
      input->data.f[i] = norm;
    }
  }

  // 2) Executa inferência
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Falha no Invoke");
    delay(1000);
    return;
  }

  // 3) Processa saída e encontra argmax
  const bool is_q_output = 
    (output->type == kTfLiteInt8 || output->type == kTfLiteUInt8);
  const float out_scale  = output->params.scale;
  const int   out_zp     = output->params.zero_point;
  const int   out_len    = output->bytes /
    (is_q_output ? sizeof(int8_t) : sizeof(float));

  int   best_index = 0;
  float best_score = -FLT_MAX;

  for (int i = 0; i < out_len; ++i) {
    float score;
    if (is_q_output) {
      int32_t raw = 
        (output->type == kTfLiteUInt8)
          ? output->data.uint8[i]
          : output->data.int8[i];
      score = (raw - out_zp) * out_scale;
    } else {
      score = output->data.f[i];
    }
    if (score > best_score) {
      best_score = score;
      best_index = i;
    }
  }

  // 4) Mostra resultado
  Serial.printf(
    "Predicao MNIST: %d (score=%.3f)\n",
    best_index, best_score
  );

  delay(2000);
}
