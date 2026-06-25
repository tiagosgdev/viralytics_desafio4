#include <WiFi.h>
#include <HTTPClient.h>
#include <PubSubClient.h>
#include <HCSR04.h>

// ---------------------------------------------------------------------------
// Wireless config (ESP32-S3). Fill these in for your network/backend.
// ---------------------------------------------------------------------------
const char *WIFI_SSID = "iPhone de Filipe";
const char *WIFI_PASS = "Batatinha123";

// Backend base URL + door route. The event is sent as a POST to
// /api/robot/sensor/door?direction=entering|leaving (direction is a query param;
// the endpoint ignores the request body). Set API_BASE to the PC's IP on the
// SAME network as the ESP32 (on an iPhone hotspot this is usually 172.20.10.x).
const char *API_BASE = "http://172.20.10.3:8000";
const char *DOOR_PATH = "/api/robot/sensor/door";
const char *API_TOKEN = "";  // optional Bearer token; leave "" to disable

// MQTT broker (robot commands). Set ENABLE_MQTT to false to skip robot control.
const bool ENABLE_MQTT = true;
const char *MQTT_HOST = "192.168.1.80";
const int MQTT_PORT = 1883;
const char *MQTT_TOPIC = "cruzr/commands";
const char *MQTT_CLIENT_ID = "door_sensor_esp32";

// Texts spoken by the robot (mirrors the bridge defaults).
const char *WELCOME_TEXT = "Bem-vindo. Espero que tenha uma excelente visita.";
const char *GOODBYE_TEXT = "Obrigado pela visita. Ate breve.";
const char *ALERT_TEXT = "Alerta. Movimento suspeito detectado.";

// ---------------------------------------------------------------------------
// Sensor config
// ---------------------------------------------------------------------------
const byte TRIG_A = 6;
const byte ECHO_A = 7;
const byte TRIG_B = 4;
const byte ECHO_B = 5;

const float SENSOR_DISTANCE_CM = 15.0;
const float PERSON_THRESHOLD_CM = 1.0;
const float ALERT_SPEED_CM_S = 150.0;
const unsigned long EVENT_TIMEOUT_MS = 1500UL;
const unsigned long RESET_CLEAR_MS = 250UL;
const byte CALIBRATION_SAMPLES = 20;

// HCSR04 library handles the trigger pulse, echo timeout and cm conversion.
// dist() returns the distance in cm (0.0 when there is no echo / out of range).
HCSR04 sensorA(TRIG_A, ECHO_A);  // HCSR04(trig pin, echo pin)
HCSR04 sensorB(TRIG_B, ECHO_B);

float distChaoA = 0.0;
float distChaoB = 0.0;

WiFiClient httpWifiClient;
WiFiClient mqttWifiClient;
PubSubClient mqttClient(mqttWifiClient);

enum State {
  WAITING,
  A_FIRST,
  B_FIRST,
  WAIT_CLEAR
};

State state = WAITING;
unsigned long firstTriggerMs = 0;
float firstHeightCm = 0.0;
float maxHeightCm = 0.0;
unsigned long clearSinceMs = 0;

float readDistanceCm(HCSR04 &sensor) {
  return sensor.dist();
}

float readStableDistanceCm(HCSR04 &sensor) {
  float total = 0.0;
  byte count = 0;

  for (byte i = 0; i < CALIBRATION_SAMPLES; i++) {
    float distance = readDistanceCm(sensor);
    if (distance > 0.0) {
      total += distance;
      count++;
    }
    delay(50);
  }

  if (count == 0) {
    return 210.0;
  }

  return total / count;
}

// ---------------------------------------------------------------------------
// Connectivity helpers
// ---------------------------------------------------------------------------
void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.println("{\"status\":\"wifi_connecting\"}");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000UL) {
    delay(250);
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("{\"status\":\"wifi_connected\",\"ip\":\"");
    Serial.print(WiFi.localIP());
    Serial.println("\"}");
  } else {
    Serial.println("{\"status\":\"wifi_failed\"}");
  }
}

void ensureMqtt() {
  if (!ENABLE_MQTT || WiFi.status() != WL_CONNECTED) {
    return;
  }
  if (mqttClient.connected()) {
    return;
  }

  mqttClient.setServer(MQTT_HOST, MQTT_PORT);
  if (mqttClient.connect(MQTT_CLIENT_ID)) {
    Serial.println("{\"status\":\"mqtt_connected\"}");
  } else {
    Serial.print("{\"status\":\"mqtt_failed\",\"rc\":");
    Serial.print(mqttClient.state());
    Serial.println("}");
  }
}

// Notify the backend via POST /api/robot/sensor/door?direction=entering|leaving.
// The endpoint reads `direction` from the query string and ignores the body.
void postEvent(const char *tipo) {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  const char *direction = (strcmp(tipo, "entrada") == 0) ? "entering" : "leaving";
  String url = String(API_BASE) + DOOR_PATH + "?direction=" + direction;

  HTTPClient http;
  http.begin(httpWifiClient, url);
  if (strlen(API_TOKEN) > 0) {
    http.addHeader("Authorization", String("Bearer ") + API_TOKEN);
  }

  int code = http.POST("");  // no body; direction is in the query string
  Serial.print("{\"status\":\"posted\",\"direction\":\"");
  Serial.print(direction);
  Serial.print("\",\"code\":");
  Serial.print(code);
  Serial.println("}");
  http.end();
}

// Publish robot speak/sound commands over MQTT (replaces the bridge robot logic).
void publishRobotCommands(const char *tipo, bool alerta) {
  if (!ENABLE_MQTT || !mqttClient.connected()) {
    return;
  }

  if (alerta) {
    mqttClient.publish(MQTT_TOPIC, "{\"action\":\"sound\",\"name\":\"alert\"}");
    String msg = String("{\"action\":\"speak\",\"text\":\"") + ALERT_TEXT + "\"}";
    mqttClient.publish(MQTT_TOPIC, msg.c_str());
    return;
  }

  const char *text = (strcmp(tipo, "entrada") == 0) ? WELCOME_TEXT : GOODBYE_TEXT;
  String msg = String("{\"action\":\"speak\",\"text\":\"") + text + "\"}";
  mqttClient.publish(MQTT_TOPIC, msg.c_str());
}

void printStatus(const char *message) {
  Serial.print("{\"status\":\"");
  Serial.print(message);
  Serial.println("\"}");
}

void printDebugDistances(float distA, float distB, float heightA, float heightB) {
  Serial.print("{\"debug\":\"distancias\",\"distA\":");
  Serial.print(distA, 1);
  Serial.print(",\"distB\":");
  Serial.print(distB, 1);
  Serial.print(",\"alturaA\":");
  Serial.print(heightA, 1);
  Serial.print(",\"alturaB\":");
  Serial.print(heightB, 1);
  Serial.println("}");
}

// Build the event JSON, log it to Serial, POST it, and command the robot.
void reportEvent(const char *tipo, float altura, float velocidade, bool alerta) {
  const char *robotFala = (strcmp(tipo, "entrada") == 0) ? WELCOME_TEXT : GOODBYE_TEXT;

  String json = "{\"tipo\":\"";
  json += tipo;
  json += "\",\"altura\":";
  json += String(altura, 1);
  json += ",\"velocidade\":";
  json += String(velocidade, 1);
  json += ",\"alerta\":";
  json += (alerta ? "true" : "false");
  json += ",\"ts\":";
  json += String(millis());
  json += ",\"robot_fala\":\"";
  json += robotFala;
  json += "\"";
  if (alerta) {
    json += ",\"robot_alerta_som\":true";
  }
  json += "}";

  Serial.println(json);
  postEvent(tipo);
  publishRobotCommands(tipo, alerta);
}

bool personDetected(float floorDistance, float currentDistance, float &heightCm) {
  if (currentDistance <= 0.0) {
    heightCm = 0.0;
    return false;
  }

  heightCm = floorDistance - currentDistance;
  return heightCm > PERSON_THRESHOLD_CM;
}

void setup() {
  Serial.begin(9600);
  delay(1000);

  ensureWifi();
  ensureMqtt();

  printStatus("calibrar_sem_pessoas");
  distChaoA = readStableDistanceCm(sensorA);
  distChaoB = readStableDistanceCm(sensorB);

  Serial.print("{\"status\":\"calibrado\",\"distChaoA\":");
  Serial.print(distChaoA, 1);
  Serial.print(",\"distChaoB\":");
  Serial.print(distChaoB, 1);
  Serial.println("}");
}

void loop() {
  ensureWifi();
  ensureMqtt();
  if (ENABLE_MQTT) {
    mqttClient.loop();
  }

  float distA = readDistanceCm(sensorA);
  delay(35);
  float distB = readDistanceCm(sensorB);

  float heightA = 0.0;
  float heightB = 0.0;
  bool detectedA = personDetected(distChaoA, distA, heightA);
  bool detectedB = personDetected(distChaoB, distB, heightB);
  unsigned long now = millis();

  switch (state) {
    case WAITING:
      if (detectedA && !detectedB) {
        state = A_FIRST;
        firstTriggerMs = now;
        firstHeightCm = heightA;
        maxHeightCm = heightA;
      } else if (detectedB && !detectedA) {
        state = B_FIRST;
        firstTriggerMs = now;
        firstHeightCm = heightB;
        maxHeightCm = heightB;
      }
      break;

    case A_FIRST:
      if (detectedA && heightA > maxHeightCm) {
        maxHeightCm = heightA;
      }
      if (detectedB) {
        float elapsedSeconds = max((now - firstTriggerMs) / 1000.0, 0.05);
        float speed = SENSOR_DISTANCE_CM / elapsedSeconds;
        float height = max(maxHeightCm, max(firstHeightCm, heightB));
        reportEvent("entrada", height, speed, speed > ALERT_SPEED_CM_S);
        state = WAIT_CLEAR;
        clearSinceMs = 0;
      } else if (now - firstTriggerMs > EVENT_TIMEOUT_MS) {
        state = WAIT_CLEAR;
        clearSinceMs = 0;
      }
      break;

    case B_FIRST:
      if (detectedB && heightB > maxHeightCm) {
        maxHeightCm = heightB;
      }
      if (detectedA) {
        float elapsedSeconds = max((now - firstTriggerMs) / 1000.0, 0.05);
        float speed = SENSOR_DISTANCE_CM / elapsedSeconds;
        float height = max(maxHeightCm, max(firstHeightCm, heightA));
        reportEvent("saida", height, speed, speed > ALERT_SPEED_CM_S);
        state = WAIT_CLEAR;
        clearSinceMs = 0;
      } else if (now - firstTriggerMs > EVENT_TIMEOUT_MS) {
        state = WAIT_CLEAR;
        clearSinceMs = 0;
      }
      break;

    case WAIT_CLEAR:
      if (!detectedA && !detectedB) {
        if (clearSinceMs == 0) {
          clearSinceMs = now;
        } else if (now - clearSinceMs > RESET_CLEAR_MS) {
          state = WAITING;
        }
      } else {
        clearSinceMs = 0;
      }
      break;
  }

  delay(35);
}
