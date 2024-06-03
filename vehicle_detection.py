import cv2
import os
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

inputPath = os.getcwd() + "/test_images/"
outputPath = os.getcwd() + "/output_images/"

def detectVehicles(filename):
    global net, inputPath, outputPath
    img = cv2.imread(inputPath + filename, cv2.IMREAD_COLOR)

    height, width, _ = img.shape

    # Normalize and preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass to get detections
    detections = net.forward(layer_names)

    # Initialize lists to store filtered detections
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if class_id is within the valid range
            if 0 <= class_id < len(classes) and confidence > 0.5 and classes[class_id] in ["car", "bus", "bike", "truck", "rickshaw"]:
                center_x, center_y, box_width, box_height = (obj[0] * width, obj[1] * height, obj[2] * width, obj[3] * height)

                # Calculate top-left corner coordinates
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))

                # Convert box dimensions to integer
                box_width = int(box_width)
                box_height = int(box_height)

                # Append to lists
                boxes.append([x, y, box_width, box_height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw rectangles for the remaining detections
    for i in indices:
        i = i.item()  # Convert numpy scalar to Python scalar
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    outputFilename = outputPath + "output_" + filename
    cv2.imwrite(outputFilename, img)
    print('Output image stored at:', outputFilename)

for filename in os.listdir(inputPath):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        detectVehicles(filename)

print("Done!")
