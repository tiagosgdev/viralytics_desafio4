import cv2
from flask import Flask, Response

app = Flask(__name__)

# Initialize the external webcam (0 is usually the default USB camera)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in byte format for the HTTP stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """
    The AI agents will subscribe to this endpoint to receive continuous video frames.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Vision Streamer running on port 5001...")
    print("AI Agents can connect to: http://localhost:5001/video_feed")
    app.run(host='0.0.0.0', port=5001)