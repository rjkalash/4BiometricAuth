from flask import Flask, render_template, Response, jsonify
from app.camera import VideoCamera
import threading

app = Flask(__name__)

# Global camera instance to prevent multiple access issues
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera()
    return camera

def gen(camera_obj):
    while True:
        frame = camera_obj.get_frame()
        if frame is None:
            # If camera fails or ends, send a placeholder or stop
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(get_camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "biometric-auth"})

@app.route('/stats')
def stats():
    cam = get_camera()
    return jsonify(cam.stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
