import cv2
import numpy as np
class YOLO:
    def __init__(self, model_path, config_path, classes_path, confidence_threshold=0.5, nms_threshold=0.4):
        self.net = cv2.dnn.readNet(model_path, config_path)
        self.classes = open(classes_path).read().strip().split('\n')
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect_objects(self, image):
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        unconnected_layers = self.net.getUnconnectedOutLayers()

        # Print the structure and type of unconnected_layers for debugging
        print("Unconnected layers:", unconnected_layers)
        print("Type of unconnected_layers:", type(unconnected_layers))
        for idx, item in enumerate(unconnected_layers):
            print(f"Type of unconnected_layers[{idx}]:", type(item))
            print(f"Content of unconnected_layers[{idx}]:", item)

        # Handling the structure of unconnected_layers
        output_layers = []
        if isinstance(unconnected_layers, np.ndarray):
            if len(unconnected_layers.shape) == 2 and unconnected_layers.shape[1] == 1:
                output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
            elif len(unconnected_layers.shape) == 1:
                output_layers = [layer_names[i - 1] for i in unconnected_layers]
        elif isinstance(unconnected_layers, (list, tuple)):
            if isinstance(unconnected_layers[0], (list, tuple, np.ndarray)):
                output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
            else:
                output_layers = [layer_names[i - 1] for i in unconnected_layers]

        print("Output layers:", output_layers)

        outputs = self.net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                result = {
                    "box": boxes[i],
                    "confidence": confidences[i],
                    "class": self.classes[class_ids[i]]
                }
                results.append(result)

        return results
