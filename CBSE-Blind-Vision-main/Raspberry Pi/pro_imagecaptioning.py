import cv2
import google.generativeai as genai
from pathlib import Path
import threading
import os
import atexit
import pyttsx3
import RPi.GPIO as GPIO
from googletrans import Translator


engine = pyttsx3.init()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

genai.configure(api_key='AIzaSyB-IbJ3LvK2x10tWXxcDcQv_2CRoYBLPQI')

# Initialize counter
counter = 0
image_captured = False  # Flag to track whether an image has been captured

generation_config = {
    "temperature": 0.5,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)


def input_image_setup(file_loc):
    if not (img := Path(file_loc)).exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": Path(file_loc).read_bytes()
        }
    ]
    return image_parts


def generate_gemini_response_async(input_prompt, image_loc, question_prompt):
    response = generate_gemini_response(input_prompt, image_loc, question_prompt)
    translator = Translator()
    response_temp = response
    out = translator.translate(response_temp,dest="hi").text
    print(response)
    engine.say(response)
    engine.runAndWait()


def generate_gemini_response(input_prompt, image_loc, question_prompt):
    image_prompt = input_image_setup(image_loc)
    prompt_parts = [input_prompt, image_prompt[0], question_prompt]
    response = model.generate_content(prompt_parts)
    return response.text


def delete_saved_images():
    for i in range(1, counter + 1):
        filename = f'Videos/image_{i}.png'
        if os.path.exists(filename):
            os.remove(filename)


input_prompt = """"
               You are an expert in understanding scenarios and reading texts and identifying objects.
               You will receive input images as scenarios &
               you will have to describe scenarios and identify objects and read texts based on the input image
               """

# Register an exit handler to delete the saved images before exiting
atexit.register(delete_saved_images)

def delete_images_in_folder(folder_path):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        # Filter only image files (you can modify the condition as needed)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        # Delete each image file
        for image_file in image_files:
            file_path = os.path.join(folder_path, image_file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

        print("All images deleted successfully.")
    except Exception as e:
        print(f"Error deleting images: {e}")

# GPIO setup
button_pin = 14  # Replace with the GPIO pin connected to your button
GPIO.setmode(GPIO.BCM)
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

button_pin2 = 18  # Replace with the GPIO pin connected to your button
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# Replace 'your_folder_path' with the actual path of the folder containing images
folder_path = 'Videos'

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Check for GPIO pin press
    if GPIO.input(button_pin) == GPIO.LOW and not image_captured:
        # Increment counter
        counter += 1

        # Save the image with counter appended to the filename in the Videos folder
        filename = f'Videos/image_{counter}.png'
        cv2.imwrite(filename, frame)
        image_loc = f"Videos/image_{counter}.png"
        question_prompt = "What is this image? describe precisely"

        # Use a separate thread to generate Gemini response asynchronously
        threading.Thread(target=generate_gemini_response_async, args=(input_prompt, image_loc, question_prompt)).start()

        print(f"Image {counter} saved as {filename}")
        
        image_captured = True  # Set the flag to True

    # Reset the flag if the button is released
    if GPIO.input(button_pin) == GPIO.HIGH:
        image_captured = False

    # Check for the 'q' key press to exit the loop
    if GPIO.input(button_pin2) == GPIO.LOW:
        delete_images_in_folder(folder_path)
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
