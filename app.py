import cv2
import numpy as np

# Load the YOLO model
yolo_model_path = "best (1).onnx"
net = cv2.dnn.readNetFromONNX(yolo_model_path)

# Load the image
image = cv2.imread('path_to_image')

# Prepare the image for detection
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the model
net.setInput(blob)

# Perform inference
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)

# Process the detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # License plate detection logic here
            pass

# Additional code to visualize results or handle detections
