#include <WiFi.h>
#include <WiFiClientSecure.h>  // HTTPS (the ngrok tunnel is TLS)
#include <HTTPClient.h>
#include <esp_system.h>        // esp_reset_reason()
#include <WiFiManager.h>   // tzapu/WiFiManager  (install via Library Manager)
#include <HCSR04.h>

// ---------------------------------------------------------------------------
// Networking (ESP32-S3, runs off a power bank).
//
// WiFi: no credentials are hard-coded. On first boot (or when GPIO0/BOOT is held
// at power-up) the sensor opens a captive-portal AP named below: join it with a
// phone and pick the venue WiFi. Saved to flash and reused on every power cycle,
// so re-flashing is never needed to change network.
//
// API: reached through an ngrok tunnel, so the URL is FIXED and independent of
// which laptop/IP runs the backend. Whatever machine runs the API + the tunnel
// (`ngrok http --domain=<your-domain> 8000`) is reachable at the same URL. The
// sensor and laptop only each need internet — they need NOT be on the same WiFi.
//
// All the sensor does is POST the door event to the API; the SERVER decides what
// the robot does:
//   direction=entering -> navigate to entrance + greet
//   direction=leaving  -> ignored by the server (still sent)
// ---------------------------------------------------------------------------
const char *AP_NAME = "DoorSensorSetup";          // captive-portal SSID

// Reserved ngrok static domain (free tier, permanent until deleted in the dashboard).
// Run on the laptop:  ngrok http --url=https://unaltering-unabjectly-micha.ngrok-free.dev 8000
const char *API_HOST = "unaltering-unabjectly-micha.ngrok-free.dev";
const char *DOOR_PATH = "/api/robot/sensor/door";
const char *API_TOKEN = "";                        // optional Bearer token; "" disables

const byte PORTAL_PIN = 0;                          // GPIO0/BOOT: hold LOW at power-up to force portal
const unsigned long PORTAL_TIMEOUT_S = 180UL;      // don't sit in AP forever in the field

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

// A fresh WiFiClientSecure is created per POST (heap is plentiful, ~200 KB), which
// is more reliable than reusing one TLS context across requests.

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

// First-boot / forced-portal WiFi setup. Auto-connects to the last saved network;
// if none is saved (or BOOT is held), opens the captive portal so a phone can pick
// the venue WiFi. Credentials persist in flash across power cycles. The API URL is
// fixed in code (ngrok), so there is nothing else to configure here.
void setupWifi() {
  WiFiManager wm;
  wm.setConfigPortalTimeout(PORTAL_TIMEOUT_S);

  pinMode(PORTAL_PIN, INPUT_PULLUP);
  bool forcePortal = (digitalRead(PORTAL_PIN) == LOW);

  bool connected;
  if (forcePortal) {
    Serial.println("{\"status\":\"portal_forced\"}");
    connected = wm.startConfigPortal(AP_NAME);
  } else {
    connected = wm.autoConnect(AP_NAME);  // tries saved creds, else opens portal
  }

  if (connected) {
    // Keep the radio fully on. Modem sleep (the default) powers the radio down
    // between beacons and intermittently breaks the multi-round-trip TLS handshake
    // — the classic "POST works only sometimes" symptom.
    WiFi.setSleep(false);
    Serial.print("{\"status\":\"wifi_connected\",\"ip\":\"");
    Serial.print(WiFi.localIP());
    Serial.print("\",\"api_host\":\"");
    Serial.print(API_HOST);
    Serial.println("\"}");
  } else {
    Serial.println("{\"status\":\"wifi_failed\"}");
  }
}

// Lightweight reconnect for the main loop. Credentials saved by WiFiManager live
// in flash, so WiFi.reconnect() re-uses them after a drop (e.g. power-bank blip).
void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.println("{\"status\":\"wifi_reconnecting\"}");
  WiFi.reconnect();

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 10000UL) {
    delay(250);
  }
}

// Notify the backend via POST /api/robot/sensor/door?direction=entering|leaving.
// The endpoint reads `direction` from the query string and ignores the body, then
// drives the robot itself (navigate + greet on entering; nothing on leaving).
void postEvent(const char *tipo) {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  const char *direction = (strcmp(tipo, "entrada") == 0) ? "entering" : "leaving";
  String url = String("https://") + API_HOST + DOOR_PATH + "?direction=" + direction;

  // Retry only on connection-level failures (code <= 0, i.e. no HTTP response).
  // Any real HTTP status (200/404/503) means the request was delivered — stop, so
  // we never duplicate a delivered event.
  int code = 0;
  for (int attempt = 1; attempt <= 3; attempt++) {
    WiFiClientSecure client;         // fresh TLS context each attempt
    client.setInsecure();            // ngrok is TLS; skip cert validation
    client.setHandshakeTimeout(15);  // seconds

    HTTPClient http;
    http.begin(client, url);
    http.setConnectTimeout(15000);   // ms
    http.setTimeout(15000);          // ms
    // ngrok's free tier serves a browser-warning interstitial; this header skips it.
    http.addHeader("ngrok-skip-browser-warning", "true");
    if (strlen(API_TOKEN) > 0) {
      http.addHeader("Authorization", String("Bearer ") + API_TOKEN);
    }

    code = http.POST("");  // no body; direction is in the query string
    http.end();
    client.stop();

    Serial.print("{\"status\":\"posted\",\"direction\":\"");
    Serial.print(direction);
    Serial.print("\",\"attempt\":");
    Serial.print(attempt);
    Serial.print(",\"code\":");
    Serial.print(code);
    Serial.println("}");

    if (code > 0) {
      break;  // got an HTTP response (delivered) — done
    }
    delay(400);
  }
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

// Log the event JSON to Serial (diagnostics) and POST it to the door endpoint.
// Both enter and leave are always sent. The "run" speed (alerta) is still
// computed and logged for later use, but it no longer suppresses a real crossing.
void reportEvent(const char *tipo, float altura, float velocidade, bool alerta) {
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
  json += "}";

  Serial.println(json);
  postEvent(tipo);
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

  // Reset reason tells brownout (power) vs panic (crash) vs poweron apart.
  // 1=POWERON 3=SW 4=PANIC 7=TASK_WDT 8=INT_WDT 9=BROWNOUT (esp_reset_reason_t)
  Serial.print("{\"status\":\"boot\",\"reset_reason\":");
  Serial.print((int)esp_reset_reason());
  Serial.print(",\"free_heap\":");
  Serial.print(ESP.getFreeHeap());
  Serial.println("}");

  setupWifi();

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
