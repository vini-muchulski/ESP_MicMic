#include <Arduino.h>
#include <ESP32Servo.h>

// pinos
const int servoPin   = 18;   // pino de sinal do servo
const int trigPin    = 27;   // pino Trig do HC-SR04
const int echoPin    = 26;   // pino Echo do HC-SR04

// sweep parameters
const int startAngle = 10;   // ângulo mínimo
const int endAngle   = 170;  // ângulo máximo
const int stepDeg    = 1;    // passo em graus
const int sweepDelay = 15;   // atraso em ms entre passos

Servo servo;
long duration;
float distance;

void setup() {
  Serial.begin(115200);
  // servo a 50 Hz (período de 20 ms)
  servo.setPeriodHertz(50);
  servo.attach(servoPin, 500, 2500);

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

void loop() {
  Serial.print("Varredura de ");
  Serial.print(startAngle);
  Serial.print("° até ");
  Serial.print(endAngle);
  Serial.println("°");

  // vai de startAngle até endAngle
  for (int pos = startAngle; pos <= endAngle; pos += stepDeg) {
    servo.write(pos);
    delay(sweepDelay);

    // disparo ultrassônico
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    duration = pulseIn(echoPin, HIGH);
    distance = duration * 0.034 / 2;  // em cm

    Serial.print("Ângulo: ");
    Serial.print(pos);
    Serial.print("°  Distância: ");
    Serial.print(distance);
    Serial.println(" cm");
  }

  // volta de endAngle até startAngle
  for (int pos = endAngle; pos >= startAngle; pos -= stepDeg) {
    servo.write(pos);
    delay(sweepDelay);

    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    duration = pulseIn(echoPin, HIGH);
    distance = duration * 0.034 / 2;

    Serial.print("Ângulo: ");
    Serial.print(pos);
    Serial.print("°  Distância: ");
    Serial.print(distance);
    Serial.println(" cm");
  }

  delay(1000);  // pausa antes de reiniciar o sweep
}
