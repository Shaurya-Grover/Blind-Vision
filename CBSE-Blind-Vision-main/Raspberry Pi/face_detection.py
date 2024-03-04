import numpy as np
import cv2
from tensorflow.lite.python.interpreter import Interpreter
import pyttsx3
import RPi.GPIO as GPIO
import time

modelpath = 'Weights/face.tflite'
lblpath = 'Weights/face.txt'
min_conf = 0.75
cap = cv2.VideoCapture(0)

interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

with open(lblpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# GPIO setup
button_pin = 4  # Replace with the GPIO pin connected to your button
GPIO.setmode(GPIO.BCM)
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

while True:
    ret, frame = cap.read()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    # Check if the button is pressed
    if GPIO.input(button_pin) == GPIO.LOW:
        for i in range(len(scores)):
            if min_conf <= scores[i] <= 1.0:
                object_name = labels[int(classes[i])]

                # Speak out the detected object name
                engine.say(f"{object_name} detected")
                engine.runAndWait()

    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
