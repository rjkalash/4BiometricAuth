import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

MODEL_PATH = 'model/svm_model.pkl'
DATASET_PATH = 'dataset'

def preprocess_face(face_img):
    # Must match app/face_recognition.py
    face_img = cv2.resize(face_img, (64, 64))
    fd = hog(face_img, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), visualize=False)
    return fd

def load_dataset():
    X = []
    y = []
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path '{DATASET_PATH}' not found.")
        print("Please create a 'dataset' folder with subfolders for each person.")
        return [], []

    print("Loading dataset...")
    for person_name in os.listdir(DATASET_PATH):
        person_dir = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect face in training image
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                print(f"No face detected in {img_path}, skipping.")
                continue
                
            # Take the largest face if multiple
            (fx, fy, fw, fh) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_roi = gray[fy:fy+fh, fx:fx+fw]
            
            try:
                features = preprocess_face(face_roi)
                X.append(features)
                y.append(person_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return X, y

def train():
    X, y = load_dataset()
    
    print(f"Total samples: {len(X)}")
    unique_classes = list(set(y))
    print(f"Classes found: {unique_classes}")

    if len(unique_classes) == 0:
         print("No data found. Generating dummy model...")
         X = np.random.rand(20, 1764)
         y = ['PersonA'] * 10 + ['PersonB'] * 10
    elif len(unique_classes) == 1:
        print("Only 1 class found. Generating dummy 'Unknown' samples for contrast...")
        # Generate random noise as "Unknown" class
        # HOG feature size is 1764 for 64x64 input with current params
        feature_size = len(X[0])
        n_negatives = max(len(X), 10)
        
        X_neg = np.random.rand(n_negatives, feature_size)
        y_neg = ['Unknown'] * n_negatives
        
        X = np.concatenate([X, X_neg])
        y = y + y_neg
    
    print(f"Training on {len(X)} samples with classes: {set(y)}...")
    
    # Stratify only if we have enough samples per class
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = SVC(probability=True)
        clf.fit(X_train, y_train)
        
        if len(y_test) > 0:
            y_pred = clf.predict(X_test)
            print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        
        joblib.dump(clf, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    train()
