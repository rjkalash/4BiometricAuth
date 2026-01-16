import cv2
import joblib
import os
import numpy as np
from skimage.feature import hog

class FaceAuthSystem:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default to absolute path parallel to app/
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'model', 'svm_model.pkl')
            
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model_path = model_path
        self.clf = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.clf = joblib.load(self.model_path)
                print(f"SVM Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print(f"No SVM model found at {self.model_path}. System in detection-only mode.")

    def preprocess_face(self, face_img):
        """
        Resize and extract HOG features.
        Must match the training preprocessing.
        """
        # Resize to fixed size (e.g., 64x64)
        face_img = cv2.resize(face_img, (64, 64))
        
        # Extract HOG features
        # orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2) is standard for human detection
        # We use a lighter config for speed/simplicity
        fd = hog(face_img, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False)
        return fd

    def process_frame(self, frame):
        """
        Detect faces and predict identity.
        Returns flattened frame with bounding boxes drawn (optional) or just metadata.
        Here we return metadata and let the caller draw.
        """
        if frame is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            name = "Unknown"
            confidence = 0.0
            
            if self.clf:
                try:
                    features = self.preprocess_face(face_roi)
                    features = features.reshape(1, -1)
                    
                    # Predict
                    name = self.clf.predict(features)[0]
                    
                    # Get confidence if available
                    if hasattr(self.clf, "predict_proba"):
                        probs = self.clf.predict_proba(features)
                        confidence = np.max(probs) * 100
                except Exception as e:
                    print(f"Inference error: {e}")
            
            results.append({
                'rect': (x, y, w, h),
                'name': name,
                'confidence': confidence
            })
            
        return results
