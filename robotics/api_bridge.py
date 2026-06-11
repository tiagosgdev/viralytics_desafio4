import paho.mqtt.client as mqtt
import json

# 1. Configuração do Broker
BROKER_ADDRESS = "test.mosquitto.org" 
PORT = 1883
TOPIC = "cruzr/commands"

# 2. Inicialização do Cliente (Compatível com paho-mqtt v2.0+)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "ROS2_Bridge_Node")

try:
    client.connect(BROKER_ADDRESS, PORT)
    print(f"Connected to broker at {BROKER_ADDRESS}:{PORT}")
    
    # 3. NOVO PAYLOAD: Alterado de movimento para fala (TTS)
    payload = {
        "action": "speak",
    "text": "Hello, I am Fashion-Clanker! How can I assist you today? Would you like help with getting new clothes"
    }
    
    json_string = json.dumps(payload)
    client.publish(TOPIC, json_string)
    print(f"Published to topic '{TOPIC}': {json_string}")

except Exception as e:
    print(f"Failed to connect or publish: {e}")

finally:
    client.disconnect()