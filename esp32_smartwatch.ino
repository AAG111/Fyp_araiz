/*
  FormCheck — ESP32 Smartwatch firmware
  ======================================
  Reads heart rate / SpO2 from a MAX30102 sensor, derives a simple stress
  score, and POSTs the reading to server.py every few seconds.

  Get DEVICE_TOKEN from the Streamlit app's sidebar → Devices → ESP32
  Smartwatch → "Generate device token". That token (not your username or
  password) is all the watch needs — it never sees your login credentials.

  Library required: SparkFun MAX3010x Pulse and Proximity Sensor Library
  (Arduino IDE → Library Manager → search "MAX3010x")
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"

// ── EDIT THESE ──────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL    = "http://192.168.1.50:5000";  // machine running server.py
const char* DEVICE_TOKEN  = "PASTE_TOKEN_FROM_STREAMLIT_SIDEBAR";
const unsigned long SEND_INTERVAL_MS = 3000;
// ─────────────────────────────────────────────────────────────

MAX30105 particleSensor;

uint32_t irBuffer[100];
uint32_t redBuffer[100];
int32_t bufferLength = 100;
int32_t spo2Value;
int8_t  spo2Valid;
int32_t heartRateValue;
int8_t  heartRateValid;

unsigned long lastSend = 0;

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println("\nConnected. IP: " + WiFi.localIP().toString());
}

void setup() {
  Serial.begin(115200);
  connectWiFi();

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found — check wiring.");
    while (1) delay(1000);
  }
  particleSensor.setup(); // default sensor config: 50mA LED, 100Hz sample rate
}

// Very simple stress proxy: heart-rate variability across the sample window.
// This is a placeholder — swap in a proper HRV/stress algorithm if you have one.
float estimateStress(uint32_t *ir, int32_t n) {
  float mean = 0;
  for (int32_t i = 0; i < n; i++) mean += ir[i];
  mean /= n;
  float variance = 0;
  for (int32_t i = 0; i < n; i++) variance += (ir[i] - mean) * (ir[i] - mean);
  variance /= n;
  float norm = constrain(sqrt(variance) / 500.0, 0, 1);
  return norm * 100.0;
}

void sendReading(int hr, int spo2, float stress) {
  if (WiFi.status() != WL_CONNECTED) { connectWiFi(); }

  HTTPClient http;
  http.begin(String(SERVER_URL) + "/api/devices/smartwatch/data");
  http.addHeader("Content-Type", "application/json");

  String body = String("{\"device_token\":\"") + DEVICE_TOKEN +
                "\",\"heart_rate\":" + hr +
                ",\"spo2\":" + spo2 +
                ",\"stress\":" + stress + "}";

  int code = http.POST(body);
  Serial.printf("POST /api/devices/smartwatch/data -> %d\n", code);
  http.end();
}

void loop() {
  // Collect a window of samples for the SpO2/HR algorithm
  for (int32_t i = 0; i < bufferLength; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    particleSensor.nextSample();
  }

  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, bufferLength, redBuffer,
    &spo2Value, &spo2Valid, &heartRateValue, &heartRateValid
  );

  if (millis() - lastSend > SEND_INTERVAL_MS) {
    lastSend = millis();
    if (heartRateValid && spo2Valid && heartRateValue > 30 && heartRateValue < 220) {
      float stress = estimateStress(irBuffer, bufferLength);
      sendReading(heartRateValue, spo2Value, stress);
    } else {
      Serial.println("No finger detected / invalid reading, skipping send.");
    }
  }
}
