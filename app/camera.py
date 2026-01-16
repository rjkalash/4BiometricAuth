import cv2
import time
from .face_recognition import FaceAuthSystem

class VideoCamera:
    def __init__(self, source=0):
        # Try to open camera
        self.video = cv2.VideoCapture(source, cv2.CAP_DSHOW) # CAP_DSHOW for windows speedup, might fail on linux
        if not self.video.isOpened():
             self.video = cv2.VideoCapture(source)
             
        self.auth_system = FaceAuthSystem()
        self.is_running = self.video.isOpened()
        self.stats = {"latency": 0, "accuracy": 0}

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        if not self.is_running:
            return None
            
        success, image = self.video.read()
        if not success:
            return None

        # Process frame
        start_time = time.time()
        results = self.auth_system.process_frame(image)
        latency = (time.time() - start_time) * 1000 # in ms
        
        # Calculate stats for the frame
        # We take the max confidence of ANY face found, even if Unknown
        confidences = [r['confidence'] for r in results]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Ensure latency is at least displayed as >0 if it's super fast, but typically it should be >10ms
        self.stats = {
            "latency": latency,
            "accuracy": avg_conf
        }
        print(f"DEBUG: Latency={latency:.2f}ms, Conf={avg_conf:.2f}%") # remove later
        
        # Draw results
        for res in results:
            (x, y, w, h) = res['rect']
            name = res['name']
            conf = res['confidence']
            
            # Green for identified, Red for Unknown
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            
            text = f"{name}"
            if conf > 0:
                text += f" {conf:.1f}%"
                
            cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Encode
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
