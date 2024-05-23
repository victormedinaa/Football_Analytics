import cv2

def process_video(video_path, yolo_model):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.detect_objects(frame)
        frame = draw_boxes(frame, results)

        cv2.imshow("Video Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def draw_boxes(image, results):
    for result in results:
        x, y, w, h = result['box']
        confidence = result['confidence']
        class_name = result['class']
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{class_name}: {confidence:.4f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image
