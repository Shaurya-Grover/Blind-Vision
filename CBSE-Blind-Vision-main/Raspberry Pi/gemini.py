import google.generativeai as genai
from pathlib import Path
# import speech_recognition as sr
import pyttsx3

def say(text):
    engine = pyttsx3.init()
    engine.setProperty('sapi',10)
    engine.say(text)
    engine.runAndWait()

genai.configure(api_key='AIzaSyB-IbJ3LvK2x10tWXxcDcQv_2CRoYBLPQI')

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

model = genai.GenerativeModel(model_name = "gemini-pro-vision",
                              generation_config = generation_config,
                              safety_settings = safety_settings)


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

def generate_gemini_response(input_prompt, image_loc, question_prompt):

    image_prompt = input_image_setup(image_loc)
    prompt_parts = [input_prompt, image_prompt[0], question_prompt]
    response = model.generate_content(prompt_parts)
    return response.text

input_prompt = """"
               You are an expert in understanding scenarios and reading texts and identifying objects.
               You will receive input images as scenarios &
               you will have to describe scenarios and identify objects and read texts based on the input image
               """

image_loc = "Videos/shop.jpg"
question_prompt = "What is this image? describe precisely"
response = generate_gemini_response(input_prompt, image_loc, question_prompt)
print(response)
