from flask import Flask, render_template, Response
import cv2
import urllib.request
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from flask import jsonify

app = Flask(__name__)
# Object detection setup (same as before)
classNames = []
speak = False
speakface=False
speakmoney = False
max_objects_to_detect = 3
detected_objects = 0
classFile = "Weights/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "Weights/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "Weights/frozen_inference_graph.pb"

detected_faces = set()
detected_classes = set()
detected_money = set()
# class_counts = {}

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Face detection setup (same as before)
face_model_path = 'Weights/facev3.tflite'
face_label_path = 'Weights/face.txt'
face_min_confidence = 0.7

face_interpreter = Interpreter(model_path=face_model_path)
face_interpreter.allocate_tensors()
face_input_details = face_interpreter.get_input_details()
face_output_details = face_interpreter.get_output_details()
face_height, face_width = face_input_details[0]['shape'][1], face_input_details[0]['shape'][2]

face_float_input = (face_input_details[0]['dtype'] == np.float32)
face_input_mean, face_input_std = 127.5, 127.5

with open(face_label_path, 'r') as f:
    face_labels = [line.strip() for line in f.readlines()]

# Face recognition setup
face_recognition_model_path = 'Weights/money.tflite'
face_recognition_label_path = 'Weights/money.txt'
face_recognition_min_confidence = 0.7

face_recognition_interpreter = Interpreter(model_path=face_recognition_model_path)
face_recognition_interpreter.allocate_tensors()
face_recognition_input_details = face_recognition_interpreter.get_input_details()
face_recognition_output_details = face_recognition_interpreter.get_output_details()
face_recognition_height, face_recognition_width = face_recognition_input_details[0]['shape'][1], face_recognition_input_details[0]['shape'][2]

face_recognition_float_input = (face_recognition_input_details[0]['dtype'] == np.float32)
face_recognition_input_mean, face_recognition_input_std = 127.5, 127.5

with open(face_recognition_label_path, 'r') as f:
    face_recognition_labels = [line.strip() for line in f.readlines()]

# Common camera URL for all three features
camera_url = 'http://192.168.1.4/cam-hi.jpg'  # Update with your camera URL

def generate(url, detection_type):
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        if detection_type == 'object':
            result, _ = getObjects(frame, 0.60, 0.2)
        elif detection_type == 'face':
            result = detectFaces(frame)
        elif detection_type == 'face_recognition':
            result = recognizeFaces(frame)

        _, jpeg = cv2.imencode('.jpg', result)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def getObjects(img, thres, nms, draw=True, objects=[]):
    global speak, detected_classes, detected_objects
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    
    if len(objects) == 0:
        objects = classNames
    
    objectInfo = []
    
    if len(classIds) != 0 and detected_objects < max_objects_to_detect:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            
            if className in objects and className not in detected_classes:
                objectInfo.append([box, className])
                
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                
                speak = True
                detected_classes.add(className)
                detected_objects += 1

        # If the maximum number of objects has been detected, reset the counter and set speak to False
        if detected_objects >= max_objects_to_detect:
            detected_objects = 0
            speak = False

    return img, objectInfo

def detectFaces(frame):
    global speakface,detected_faces
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (face_width, face_height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model
    if face_float_input:
        input_data = (np.float32(input_data) - face_input_mean) / face_input_std

    face_interpreter.set_tensor(face_input_details[0]['index'], input_data)
    face_interpreter.invoke()

    boxes = face_interpreter.get_tensor(face_output_details[1]['index'])[0]
    classes = face_interpreter.get_tensor(face_output_details[3]['index'])[0]
    scores = face_interpreter.get_tensor(face_output_details[0]['index'])[0]

    for i in range(len(scores)):
        if face_min_confidence < scores[i] <= 1.0:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = face_labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)

            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            if object_name not in detected_faces:
                speakface = True
                detected_faces.add(object_name)

    return frame


def recognizeFaces(frame):
    global speakmoney,detected_money
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (face_recognition_width, face_recognition_height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model
    if face_recognition_float_input:
        input_data = (np.float32(input_data) - face_recognition_input_mean) / face_recognition_input_std

    face_recognition_interpreter.set_tensor(face_recognition_input_details[0]['index'], input_data)
    face_recognition_interpreter.invoke()

    boxes = face_recognition_interpreter.get_tensor(face_recognition_output_details[1]['index'])[0]
    classes = face_recognition_interpreter.get_tensor(face_recognition_output_details[3]['index'])[0]
    scores = face_recognition_interpreter.get_tensor(face_recognition_output_details[0]['index'])[0]

    for i in range(len(scores)):
        if face_recognition_min_confidence < scores[i] <= 1.0:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = face_recognition_labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)

            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            if object_name not in detected_money:
                speakmoney = True
                detected_money.add(object_name)

    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/object_detection_video')
def object_detection_video():
    return Response(generate(camera_url, 'object'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_detection_video')
def face_detection_video():
    return Response(generate(camera_url, 'face'), mimetype='multipart/x-mixed-replace; boundary=frame')

MAX_DETECTED_CLASSES = 3
@app.route('/get_objvar')
def speakobj():
    global detected_classes
    variable_to_send = list(detected_classes)
    # Limit the number of detected classes
    variable_to_send = variable_to_send[:MAX_DETECTED_CLASSES]
    detected_classes = set()  # Clear the set after sending the classes
    return jsonify({'variable': variable_to_send})

@app.route('/get_facevar')
def speakface():
    global speakface, detected_faces
    variable_to_send = list(detected_faces)
    detected_faces = set()  # Clear the set after sending the classes
    var2 = []
    if speakface:
        speakface = False  # Reset speakface to False
        return jsonify({'variable': variable_to_send})
    elif not speakface:
        return jsonify({'variable': var2})

@app.route('/getmoney_var')
def speakmoney():
    global speakmoney, detected_money
    variable_to_send = list(detected_money)
    detected_money = set()
    var2=[]
    if speakmoney:
        speakmoney=False
        return jsonify({'variable': variable_to_send})
    elif not speakface:
        return jsonify({'variable': var2})

@app.route('/face_recognition_video')
def face_recognition_video():
    return Response(generate(camera_url, 'face_recognition'), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/testface')
# def sample():
#     return "<h1>BITCH</h1>"

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)
