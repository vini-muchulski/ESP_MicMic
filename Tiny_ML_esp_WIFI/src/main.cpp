/******************************************************************************
 * Inferência TensorFlow Lite Micro no ESP32-C3
 * Modelo de seno — compatível com FLOAT32 **ou** INT8/UINT8
 *
 * ▸ Basta alterar o #define “modelo_seno_tflite” para apontar ao modelo
 *   desejado e recompilar. O código detecta o tipo de tensor em tempo de
 *   execução e faz (de)quantização apenas quando necessário.
 ******************************************************************************/

 #include <WiFi.h>
#include <WebServer.h> 

#include <Arduino.h>
#include <math.h>                     // para M_PI


const char* ssid = "Starlink";
const char* password = "diversao";



// ───────── TensorFlow Lite Micro ──────────────────────────────────────────────
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ───────── Modelo convertido (escolha aqui) ──────────────────────────────────
//#include "model_seno_data.h"         // nomeado no Python como modelo_seno_int8[]
//#include "modelo_seno_float32.h"    // se ainda quiser testar o float32
#include "modelo_seno_float32.h"  

// Se quiser trocar de modelo sem mexer no restante do código,
// altere apenas estas duas linhas:
#define modelo_seno_tflite      modelo_seno_float32_tflite
#define modelo_seno_tflite_len  modelo_seno_float32_tflite_len

// ───────── Arena de memória do TFLM ──────────────────────────────────────────
constexpr int   kTensorArenaSize = 12 * 1024;   // 12 kB
static   uint8_t tensor_arena[kTensorArenaSize];

// ───────── Variáveis globais ─────────────────────────────────────────────────
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter*     error_reporter = &micro_error_reporter;

  const tflite::Model*       model          = nullptr;
  tflite::AllOpsResolver     resolver;                  // todas as ops
  tflite::MicroInterpreter*  interpreter     = nullptr;

  TfLiteTensor*              input          = nullptr;
  TfLiteTensor*              output         = nullptr;

  // Parâmetros de quantização (válidos só se o modelo for INT8/UINT8)
  float     in_scale   = 1.0f;
  int32_t   in_zp      = 0;
  float     out_scale  = 1.0f;
  int32_t   out_zp     = 0;
  bool      is_quant   = false;
}

// ═════════════════════════════════════════════════════════════════════════════
// SETUP
// ═════════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(200);

  // 1) Carrega o modelo da flash
  model = tflite::GetModel(modelo_seno_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
        "Modelo v%d ≠ Schema v%d", model->version(), TFLITE_SCHEMA_VERSION);
    while (true);
  }

  // 2) Intérprete + alocação de tensores
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() falhou.");
    while (true);
  }

  // 3) Ponteiros de entrada/saída
  input  = interpreter->input(0);
  output = interpreter->output(0);

  // 4) Checa se é quantizado
  is_quant = (input->type == kTfLiteInt8  || input->type == kTfLiteUInt8);

  if (is_quant) {
    in_scale   = input->params.scale;
    in_zp      = input->params.zero_point;
    out_scale  = output->params.scale;
    out_zp     = output->params.zero_point;
  }

  // 5) Log rápido
  Serial.printf("INPUT  type=%d  scale=%f  zp=%d\n",
                input->type,  in_scale,  in_zp);
  Serial.printf("OUTPUT type=%d  scale=%f  zp=%d\n",
                output->type, out_scale, out_zp);
  Serial.println("Setup concluído — modelo pronto para inferir!");



  // Conecta ao WiFi
  WiFi.begin(ssid, password);
  Serial.print("Conectando ao WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.printf("WiFi conectado! IP: %s\n", WiFi.localIP().toString().c_str());

}

// ═════════════════════════════════════════════════════════════════════════════
// LOOP
// ═════════════════════════════════════════════════════════════════════════════
void loop() {
  // Ângulos de teste (rad)
  constexpr float angles[] = { M_PI/3, M_PI/6, M_PI/4, M_PI/2 ,M_PI };
  constexpr int   num_angles = sizeof(angles) / sizeof(angles[0]);

  for (int i = 0; i < num_angles; ++i) {
    float x = angles[i];

    // ─── 1. Escreve no tensor de entrada ──────────────────────────────────
    if (is_quant) {
      int32_t q_in = static_cast<int32_t>(roundf(x / in_scale) + in_zp);

      // Clampa se exceder range do tipo
      if (input->type == kTfLiteInt8) {
        q_in = max(-128, min(127, q_in));
        input->data.int8[0] = static_cast<int8_t>(q_in);
      } else {                         // UINT8
        q_in = max(0, min(255, q_in));
        input->data.uint8[0] = static_cast<uint8_t>(q_in);
      }
    } else {
      input->data.f[0] = x;            // modelo float32 original
    }

    // ─── 2. Invoke ─────────────────────────────────────────────────────────
    if (interpreter->Invoke() != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke() falhou.");
      return;
    }

    // ─── 3. Lê a saída ─────────────────────────────────────────────────────
    float y;
    if (is_quant) {
      int32_t q_out = (output->type == kTfLiteInt8)
                        ? output->data.int8[0]
                        : output->data.uint8[0];
      y = (q_out - out_zp) * out_scale;
    } else {
      y = output->data.f[0];
    }

    // ─── 4. Print ──────────────────────────────────────────────────────────
    Serial.printf("sin(%f) = %f\n", x, y);
  }

  Serial.println("----------------------------");
  delay(1000);
}
