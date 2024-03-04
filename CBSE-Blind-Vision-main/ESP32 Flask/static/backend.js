var objectDetectionStarted = false;
var baseCameraStarted = false;
var faceRecogStarted = false;

function speak(text) {
    var msg = new SpeechSynthesisUtterance();
    msg.text = text;
    window.speechSynthesis.speak(msg);
}

function toggleObjectDetection() {
    var btn = document.getElementById("startObjectDetectionBtn");
    if (objectDetectionStarted) {
        document.getElementById("video-feed").src = "";
        objectDetectionStarted = false;
        btn.innerText = "Start Object Detection";
        speak("Object detection stopped");
        fetch('/get_objvar')
            .then(response => response.json())
            .then(data => {
                const outputElement = document.getElementById('output');

                // Check if the data variable is not an empty array before using it
                if (data.variable && data.variable.length > 0) {
                    outputElement.innerText = data.variable.join(', ');  // Join the array elements into a string
                    speak(data.variable.join(', ') + " detected");  // Speak out the variable
                } else {
                    outputElement.innerText = "No objects detected";
                }
            });
    } else {
        document.getElementById("video-feed").src = "/object_detection_video";  // Update the URL
        objectDetectionStarted = true;
        btn.innerText = "Stop Object Detection";
        speak("Object detection running");
    }
}


function toggleBaseCamera() {
    var btn = document.getElementById("startBaseCameraBtn");
    if (baseCameraStarted) {
        document.getElementById("video-feed").src = "";
        baseCameraStarted = false;
        btn.innerText = "Start Face Detection";
        speak("Face Detection Stopped");
        fetch('/get_facevar')
                .then(response => response.json())
                .then(data => {
                    const outputElement = document.getElementById('output2');
                    outputElement.innerText = data.variable;
                    
                    // Speak out the variable
                    speak(data.variable+"detected");
                });
    } else {
        document.getElementById("video-feed").src = "/face_detection_video";  // Update the URL
        baseCameraStarted = true;
        btn.innerText = "Stop Face Detection";
        speak("Face detection running");
    }
}


function toggleFaceRecog() {
    var btn = document.getElementById("startFaceRecog");
    if (faceRecogStarted) {
        document.getElementById("video-feed").src = "";
        faceRecogStarted = false;
        btn.innerText = "Run Currency Detection";
        speak("Currency detection stopped");
        fetch('/getmoney_var')
                .then(response => response.json())
                .then(data => {
                    const outputElement = document.getElementById('output3');
                    outputElement.innerText = data.variable;
                    
                    // Speak out the variable
                    speak(data.variable+"detected");
                });
    } else {
        document.getElementById("video-feed").src = "/face_recognition_video";  // Update the URL
        faceRecogStarted = true;
        btn.innerText = "Stop Currency Detection";
        speak("Currency detection running");
    }
}

document.getElementById("startObjectDetectionBtn").addEventListener("click", toggleObjectDetection);
document.getElementById("startBaseCameraBtn").addEventListener("click", toggleBaseCamera);
document.getElementById("startFaceRecog").addEventListener("click", toggleFaceRecog);

function loadResultPage() {
    var newUrl = '/testface';
    window.location.href = newUrl;
}
