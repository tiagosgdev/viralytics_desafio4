const byte TRIG_A = 2;
const byte ECHO_A = 3;
const byte TRIG_B = 4;
const byte ECHO_B = 5;

const float SENSOR_DISTANCE_CM = 40.0;
const float PERSON_THRESHOLD_CM = 30.0;
const float ALERT_SPEED_CM_S = 150.0;
const unsigned long MAX_ECHO_US = 30000UL;
const unsigned long EVENT_TIMEOUT_MS = 1500UL;
const unsigned long RESET_CLEAR_MS = 250UL;
const byte CALIBRATION_SAMPLES = 20;

float distChaoA = 0.0;
float distChaoB = 0.0;

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

float readDistanceCm(byte trigPin, byte echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  unsigned long duration = pulseIn(echoPin, HIGH, MAX_ECHO_US);
  if (duration == 0) {
    return -1.0;
  }

  return duration / 58.0;
}

float readStableDistanceCm(byte trigPin, byte echoPin) {
  float total = 0.0;
  byte count = 0;

  for (byte i = 0; i < CALIBRATION_SAMPLES; i++) {
    float distance = readDistanceCm(trigPin, echoPin);
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

void printEvent(const char *tipo, float altura, float velocidade, bool alerta) {
  Serial.print("{\"tipo\":\"");
  Serial.print(tipo);
  Serial.print("\",\"altura\":");
  Serial.print(altura, 1);
  Serial.print(",\"velocidade\":");
  Serial.print(velocidade, 1);
  Serial.print(",\"alerta\":");
  Serial.print(alerta ? "true" : "false");
  Serial.print(",\"ts\":");
  Serial.print(millis());
  Serial.println("}");
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
  pinMode(TRIG_A, OUTPUT);
  pinMode(ECHO_A, INPUT);
  pinMode(TRIG_B, OUTPUT);
  pinMode(ECHO_B, INPUT);

  Serial.begin(9600);
  delay(1000);

  printStatus("calibrar_sem_pessoas");
  distChaoA = readStableDistanceCm(TRIG_A, ECHO_A);
  distChaoB = readStableDistanceCm(TRIG_B, ECHO_B);

  Serial.print("{\"status\":\"calibrado\",\"distChaoA\":");
  Serial.print(distChaoA, 1);
  Serial.print(",\"distChaoB\":");
  Serial.print(distChaoB, 1);
  Serial.println("}");
}

void loop() {
  float distA = readDistanceCm(TRIG_A, ECHO_A);
  delay(35);
  float distB = readDistanceCm(TRIG_B, ECHO_B);

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
        printEvent("entrada", height, speed, speed > ALERT_SPEED_CM_S);
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
        printEvent("saida", height, speed, speed > ALERT_SPEED_CM_S);
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
