#include <Arduino.h>
#include "image_to_detect.h" // Inclui a nossa imagem de teste

// Inclui os cabeçalhos dos modelos de detecção de face da biblioteca da câmera
#include "human_face_detect_msr01.hpp"
#include "human_face_detect_mnp01.hpp"

// Define se usaremos o modelo de 2 estágios (mais preciso) ou apenas 1 (mais rápido)
#define TWO_STAGE 1

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Inicializando a detecção de face em uma imagem estática...");

  // Instancia os modelos. Os parâmetros são limiares de confiança e outras configurações.
  // s1 é o modelo rápido de primeiro estágio.
  //HumanFaceDetectMSR01 s1(0.05F, 0.5F, 10, 0.2F); // Diminuí para 0.05
  
  // s2 é o modelo de refinamento de segundo estágio.
  //HumanFaceDetectMNP01 s2(0.5F, 0.3F, 5);

  // HumanFaceDetectMSR01(score_threshold, nms_threshold, C, N)
HumanFaceDetectMSR01 s1(0.08F, 0.5F, 10, 0.2F);

// HumanFaceDetectMNP01(score_threshold, nms_threshold, N)
HumanFaceDetectMNP01 s2(0.5F, 0.3F, 5);

  Serial.printf("Imagem carregada: %d x %d pixels\n", IMAGE_WIDTH, IMAGE_HEIGHT);
  Serial.println("Iniciando a inferência (detecção)...");

  // Marca o tempo de início
  long startTime = micros();

  // A mágica acontece aqui!
  // A função 'infer' recebe os dados da imagem e suas dimensões.
  // Ela retorna uma lista de resultados.
  std::list<dl::detect::result_t> results;

  if (TWO_STAGE) {
    // Estágio 1: Encontra candidatos a rostos
    std::list<dl::detect::result_t> &candidates = s1.infer((uint16_t *)image_data, {(int)IMAGE_HEIGHT, (int)IMAGE_WIDTH, 3});
    // Estágio 2: Refina os candidatos para obter o resultado final
    results = s2.infer((uint16_t *)image_data, {(int)IMAGE_HEIGHT, (int)IMAGE_WIDTH, 3}, candidates);
  } else {
    // Apenas Estágio 1: Rápido, mas menos preciso
    results = s1.infer((uint16_t *)image_data, {(int)IMAGE_HEIGHT, (int)IMAGE_WIDTH, 3});
  }
  
  // Marca o tempo de fim
  long endTime = micros();

  // Processa e imprime os resultados
  if (results.size() > 0) {
    Serial.printf("\n%d rosto(s) detectado(s) em %lums!\n", results.size(), (endTime - startTime) / 1000);
    
    int i = 1;
    // Itera sobre a lista de rostos encontrados
    for (std::list<dl::detect::result_t>::iterator prediction = results.begin(); prediction != results.end(); prediction++, i++) {
      Serial.printf("--- Rosto #%d ---\n", i);
      Serial.printf("Confiança: %.2f%%\n", prediction->score * 100);
      
      // Imprime as coordenadas da caixa delimitadora (bounding box)
      int x = (int)prediction->box[0];
      int y = (int)prediction->box[1];
      int w = (int)prediction->box[2] - x;
      int h = (int)prediction->box[3] - y;
      Serial.printf("Caixa (x, y, w, h): %d, %d, %d, %d\n", x, y, w, h);
    }
  } else {
    Serial.println("\nNenhum rosto detectado.");
  }

  Serial.println("\nFim do teste.");
}

void loop() {
  // Nada a fazer aqui, o teste roda apenas uma vez no setup()
  delay(10000);
}
