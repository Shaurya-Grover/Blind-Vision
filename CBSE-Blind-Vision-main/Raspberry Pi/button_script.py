import sys
import subprocess
from gpiozero import Button
from signal import pause
import time

# Dictionary to map GPIO pin numbers to script names button PINS
script_mapping = {
    3: 'face_detection.py',
    4: 'currency_detection.py',
    17: 'object_detection.py',
    27: 'image_captioning.py'
}

# Dictionary to track the state of each script
script_states = {pin: False for pin in script_mapping}

# Function to toggle the script state
def toggle_script(pin):
    global script_states

    script_name = script_mapping[pin]

    if script_states[pin]:
        # If the script is running, terminate it
        subprocess.run(["pkill", "-f", script_name])
        print(f"Script {script_name} terminated.")
    else:
        # If the script is not running, start it
        subprocess.Popen([sys.executable, script_name])
        print(f"Script {script_name} started.")

    # Toggle the script state
    script_states[pin] = not script_states[pin]

# Function to terminate the program
def terminate_program():
    print("Exiting the program.")
    # You can add any cleanup or finalization code here if needed
    sys.exit(0)

# Create buttons for each script
buttons = [Button(pin, hold_time=4) for pin in script_mapping]

# Assign toggle_script function to each button
for button in buttons:
    button.when_pressed = lambda b=button: toggle_script(b.pin.number)

# Assign terminate_program function to the last button
buttons[-1].when_held = terminate_program

# Keep the program running
pause()
