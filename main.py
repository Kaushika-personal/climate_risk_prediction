import os
import eel
from engine.features import *
from engine.command import *
from engine.auth import recoganize

def start():
    # Initialize the eel app with the web directory
    eel.init("www")

    # Play an initial sound or action
    playAssistantSound()

    # Define the initialization function to handle face authentication
    @eel.expose
    def init():
        eel.hideLoader()  # Hide any loading indicators
        speak("Ready for Face Authentication")  # Prompt for face authentication
        flag = recoganize.AuthenticateFace()  # Call the face recognition function

        if flag == 1:
            eel.hideFaceAuth()  # Hide face authentication UI
            speak("Face Authentication Successful")  # Notify success
            eel.hideFaceAuthSuccess()  # Hide success message on UI
            speak("Hello, Welcome Kaushika, How can I help you?")  # Greet the user
            eel.hideStart()  # Hide the start UI

            # Play a sound to indicate readiness for commands
            playAssistantSound()

            # Start the loop for repeated command recognition
            while True:
                text = takecommand()  # Get the user's command
                if text == "exit" or text == "quit":
                    speak("Goodbye!")  # Exit command, say goodbye
                    break  # Exit the loop and stop processing commands

                speak(f"You said: {text}")  # Speak the command back to the user
                eel.senderText(text)  # Optionally send the recognized text to the frontend

                # Now you can process the command using allCommands or other functions
                allCommands(text)  # Process the recognized command
                speak("How can I assist you further?")  # Ask for the next command

        else:
            # Handle failed face authentication
            speak("Face Authentication Failed")  # Notify the failure
            eel.showStart()  # Optionally show a retry or start screen

    # Open the web interface in a browser (Edge in this case)
    os.system('start msedge.exe --app="http://localhost:8000/index.html"')

    # Start the eel server and load the 'index.html' page
    eel.start('index.html', mode=None, host='localhost', block=True)
