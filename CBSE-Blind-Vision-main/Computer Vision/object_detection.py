import cv2
import pyttsx3
import numpy as np

# Load the class names
classNames = []
classFile = "Weights/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Set up model configuration and weights paths
configPath = "Weights/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "Weights/frozen_inference_graph.pb"

# Initialize the neural network for object detection
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Dictionary to track the counts for each class
class_counts = {}

# Set to store detected classes
detected_classes = set()

# Video capture setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Function to detect objects
def getObjects(img, thres, nms, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)

    if len(objects) == 0:
        objects = classNames

    objectInfo = []

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])

                # Special handling for 'car' class
                if className == 'car':
                    if 'car' in class_counts and class_counts['car'] < 4:
                        engine.say(f"{className} detected")
                        engine.runAndWait()
                        # Increment the count for the 'car' class
                        class_counts['car'] += 1
                    elif 'car' not in class_counts:
                        class_counts['car'] = 1
                    elif class_counts['car'] == 4:
                        engine.say("A lot of traffic ahead! Multiple cars detected.")
                        engine.runAndWait()
                        # Increment the count for the 'car' class to prevent further speech
                        class_counts['car'] += 1
                else:
                    # For other classes, speak only if the class is not seen before
                    if className not in detected_classes:
                        engine.say(f"{className} detected")
                        engine.runAndWait()
                        detected_classes.add(className)

                # Draw a box and label on the image
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo

while True:
    success, img = cap.read()
    result, objectInfo = getObjects(img, 0.60, 0.2)
    cv2.imshow("Output", img)

    # Check for 'q' key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
