from ultralytics import YOLO
import cv2

def main():
    # Load the YOLO model
    model = YOLO("yolo11n.pt") 

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Press 'q' to quit.")
    print("Detecting only: COWS")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # 1. Run YOLO inference
        # classes=[19] filters for 'cow' only
        results = model(frame, classes=[19])
        
        # 2. Visualize the results on the frame
        annotated_frame = results[0].plot()

        # 3. Count cows
        # results[0].boxes contains the detection boxes
        cow_count = len(results[0].boxes)

        # Display the count
        text = f"Cows detected: {cow_count}"
        cv2.putText(annotated_frame, text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow("YOLO Cow Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
