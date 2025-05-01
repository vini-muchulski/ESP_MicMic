#include <Arduino.h>
#include <ESP32Servo.h>

const int  servoPin   = 18;   // pino de sinal do servo
const int  startAngle = 10;   // ângulo mínimo
const int  endAngle   = 170;  // ângulo máximo
const int  stepDeg    = 1;    // passo em graus
const int  sweepDelay = 15;   // atraso em ms

Servo servo;

void setup() {
  Serial.begin(115200);
  servo.setPeriodHertz(50);
  servo.attach(servoPin, 500, 2500);
}

void loop() {
  // imprime a faixa de varredura
  Serial.print("Faixa de ângulos: ");
  Serial.print(startAngle);
  Serial.print("° até ");
  Serial.print(endAngle);
  Serial.println("°");

  // vai de startAngle até endAngle
  for (int pos = startAngle; pos <= endAngle; pos += stepDeg) {
    servo.write(pos);
    Serial.print("Ângulo atual: ");
    Serial.print(pos);
    Serial.println("°");
    delay(sweepDelay);
  }

  // volta de endAngle até startAngle
  for (int pos = endAngle; pos >= startAngle; pos -= stepDeg) {
    servo.write(pos);
    Serial.print("Ângulo atual: ");
    Serial.print(pos);
    Serial.println("°");
    delay(sweepDelay);
  }
}
