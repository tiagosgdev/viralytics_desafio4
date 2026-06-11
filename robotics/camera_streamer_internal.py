import cv2
import numpy as np
import base64
from flask import Flask, Response
import roslibpy
import threading

app = Flask(__name__)

# --- Configuration ---
# The IP of the Cruzr robot and the default ROS bridge port
ROBOT_IP = '192.168.1.100'
ROSBRIDGE_PORT = 9090 
# You will need to find the exact topic name next week using `rostopic list`
CAMERA_TOPIC = '/camera/rgb/image_raw/compressed'

current_frame = None

def receive_image(message):
    """Callback function that runs every time the robot publishes a new frame."""
    global current_frame
    try:
        # Decode the base64 compressed image coming from ROS
        img_bytes = base64.b64decode(message['data'])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding frame: {e}")

def connect_ros():
    """Connects to the ROS 1 websocket server on the robot."""
    client = roslibpy.Ros(host=ROBOT_IP, port=ROSBRIDGE_PORT)
    
    # We subscribe to the compressed image topic to save Wi-Fi bandwidth
    listener = roslibpy.Topic(client, CAMERA_TOPIC, 'sensor_msgs/CompressedImage')
    listener.subscribe(receive_image)
    
    print(f"Connecting to ROS 1 Bridge at {ROBOT_IP}...")
    client.run() # This runs in its own thread

def generate_frames():
    """Yields the latest frame for the Flask web server."""
    global current_frame
    while True:
        if current_frame is not None:
            # Re-encode the frame for the web stream
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the ROS connection in the background
    ros_thread = threading.Thread(target=connect_ros)
    ros_thread.daemon = True
    ros_thread.start()
    
    print("Internal Vision Streamer running on port 5001...")
    app.run(host='0.0.0.0', port=5001)