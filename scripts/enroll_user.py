import cv2
import os
import time

def capture_faces():
    print("=== Biometric Enrollment ===")
    name = input("Enter the name of the person to enroll: ").strip()
    if not name:
        print("Invalid name.")
        return

    save_dir = os.path.join("dataset", name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Directory created: {save_dir}")
    print("Instructions:")
    print(" - Press 'SPACE' to capture a photo")
    print(" - Capture at least 10-15 photos with different angles/expressions")
    print(" - Press 'ESC' to finish and train")
    print("Starting camera...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
            
        # Draw a guide box
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        box_size = 200
        cv2.rectangle(frame, (center_x - box_size//2, center_y - box_size//2),
                      (center_x + box_size//2, center_y + box_size//2), (0, 255, 0), 1)
        
        cv2.putText(frame, f"Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Enrollment - Press SPACE to Capture", frame)
        
        key = cv2.waitKey(1)
        if key % 256 == 27: # ESC
            break
        elif key % 256 == 32: # SPACE
            img_name = os.path.join(save_dir, f"face_{int(time.time()*1000)}.jpg")
            cv2.imwrite(img_name, frame)
            count += 1
            print(f"Saved {img_name}")
            # Visual feedback
            cv2.rectangle(frame, (0,0), (w,h), (0, 255, 0), 10)
            cv2.imshow("Enrollment - Press SPACE to Capture", frame)
            cv2.waitKey(200)

    cap.release()
    cv2.destroyAllWindows()
    
    if count > 0:
        print(f"\nCaptured {count} images for {name}.")
        print("Running training script...")
        os.system("python model/train_model.py")
        print("Done! You can now run the app.")
    else:
        print("No images captured.")

if __name__ == "__main__":
    capture_faces()
