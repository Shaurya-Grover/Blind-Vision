import numpy as np
import cv2
from tensorflow.lite.python.interpreter import Interpreter
import pyttsx3

model_path = 'Weights/money.tflite'
label_path = 'Weights/money.txt'
min_confidence = 0.7

cap = cv2.VideoCapture(0)

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)
input_mean, input_std = 127.5, 127.5

with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Dictionary to track the counts for each class
class_counts = {}

while True:
    ret, frame = cap.read()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    detections = []

    for i in range(len(scores)):
        if min_confidence < scores[i] <= 1.0:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)

            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Speak out only if the count is less than or equal to two
            if object_name not in class_counts or class_counts[object_name] < 2:
                engine.say(f'{object_name} detected')
                engine.runAndWait()

                # Increment the count for the class
                if object_name in class_counts:
                    class_counts[object_name] += 1
                else:
                    class_counts[object_name] = 1

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
