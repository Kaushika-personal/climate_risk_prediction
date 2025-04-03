import os
from pipes import quote
import re
import sqlite3
import struct
import subprocess
import time
import webbrowser
import eel
import pyaudio
import pyautogui
from engine.command import speak
from engine.config import ASSISTANT_NAME
# Playing assistant sound function
import pywhatkit as kit
import pvporcupine

from engine.helper import extract_yt_term, remove_words
from hugchat import hugchat

# Initialize the database connection
con = sqlite3.connect("jarvis.db")
cursor = con.cursor()

# Install and import pygame for sound playback
import pygame

# Function to play assistant sound using pygame
@eel.expose
def playAssistantSound():
    music_dir = os.path.abspath("www/assets/audio/start_sound.mp3")
    
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Load the sound
    pygame.mixer.music.load(music_dir)
    
    # Play the sound
    pygame.mixer.music.play()

    # Wait until the sound finishes
    while pygame.mixer.music.get_busy():  # Check if sound is still playing
        pygame.time.Clock().tick(10)  # Wait for the sound to finish

# Function to open commands
def openCommand(query):
    query = query.replace(ASSISTANT_NAME, "")
    query = query.replace("open", "")
    query.lower()

    app_name = query.strip()

    if app_name != "":

        try:
            cursor.execute(
                'SELECT path FROM sys_command WHERE name IN (?)', (app_name,))
            results = cursor.fetchall()

            if len(results) != 0:
                speak("Opening "+query)
                os.startfile(results[0][0])

            elif len(results) == 0:
                cursor.execute(
                    'SELECT url FROM web_command WHERE name IN (?)', (app_name,))
                results = cursor.fetchall()

                if len(results) != 0:
                    speak("Opening "+query)
                    webbrowser.open(results[0][0])

                else:
                    speak("Opening "+query)
                    try:
                        os.system('start '+query)
                    except:
                        speak("not found")
        except:
            speak("something went wrong")

# Function to play a YouTube video based on query
def PlayYoutube(query):
    search_term = extract_yt_term(query)
    speak("Playing "+search_term+" on YouTube")
    kit.playonyt(search_term)

# Hotword detection using Porcupine
def hotword():
    porcupine = None
    paud = None
    audio_stream = None
    try:
        # Pre-trained keywords    
        porcupine = pvporcupine.create(keywords=["jarvis", "alexa"]) 
        paud = pyaudio.PyAudio()
        audio_stream = paud.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=porcupine.frame_length)

        # Loop for streaming
        while True:
            keyword = audio_stream.read(porcupine.frame_length)
            keyword = struct.unpack_from("h"*porcupine.frame_length, keyword)

            # Processing keyword from mic 
            keyword_index = porcupine.process(keyword)

            # Checking if the first keyword is detected
            if keyword_index >= 0:
                print("Hotword detected")

                # Pressing shortcut key win+j
                import pyautogui as autogui
                autogui.keyDown("win")
                autogui.press("j")
                time.sleep(2)
                autogui.keyUp("win")
                
    except:
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if paud is not None:
            paud.terminate()

# Function to find contacts
def findContact(query):
    words_to_remove = [ASSISTANT_NAME, 'make', 'a', 'to', 'phone', 'call', 'send', 'message', 'whatsapp', 'video']
    query = remove_words(query, words_to_remove)

    try:
        query = query.strip().lower()
        cursor.execute("SELECT mobile_no FROM contacts WHERE LOWER(name) LIKE ? OR LOWER(name) LIKE ?", ('%' + query + '%', query + '%'))
        results = cursor.fetchall()
        print(results[0][0])
        mobile_number_str = str(results[0][0])

        if not mobile_number_str.startswith('+91'):
            mobile_number_str = '+91' + mobile_number_str

        return mobile_number_str, query
    except:
        speak('Not found in contacts')
        return 0, 0

# Function to send WhatsApp messages
def whatsApp(mobile_no, message, flag, name):
    if flag == 'message':
        target_tab = 12
        jarvis_message = "Message sent successfully to "+name
    elif flag == 'call':
        target_tab = 7
        message = ''
        jarvis_message = "Calling to "+name
    else:
        target_tab = 6
        message = ''
        jarvis_message = "Starting video call with "+name

    # Encode the message for URL
    encoded_message = quote(message)
    print(encoded_message)
    # Construct the URL
    whatsapp_url = f"whatsapp://send?phone={mobile_no}&text={encoded_message}"

    # Construct the full command
    full_command = f'start "" "{whatsapp_url}"'

    # Open WhatsApp with the constructed URL using cmd.exe
    subprocess.run(full_command, shell=True)
    time.sleep(5)
    subprocess.run(full_command, shell=True)
    
    pyautogui.hotkey('ctrl', 'f')

    for i in range(1, target_tab):
        pyautogui.hotkey('tab')

    pyautogui.hotkey('enter')
    speak(jarvis_message)

# Chatbot functionality
def chatBot(query):
    #user_input = query.lower()
    #chatbot = hugchat.ChatBot(cookie_path="engine\cookies.json")
    #id = chatbot.new_conversation()
    #chatbot.change_conversation(id)
    response = chatBot("Hello, how are you?")
    print(response)
    speak(response)
    return response



# Android automation: make a call
def makeCall(name, mobileNo):
    mobileNo = mobileNo.replace(" ", "")
    speak("Calling "+name)
    command = 'adb shell am start -a android.intent.action.CALL -d tel:'+mobileNo
    os.system(command)

# Send SMS functionality
def sendMessage(message, mobileNo, name):
    from engine.helper import replace_spaces_with_percent_s, goback, keyEvent, tapEvents, adbInput
    message = replace_spaces_with_percent_s(message)
    mobileNo = replace_spaces_with_percent_s(mobileNo)
    speak("Sending message")
    goback(4)
    time.sleep(1)
    keyEvent(3)
    # Open SMS app
    tapEvents(136, 2220)
    # Start chat
    tapEvents(819, 2192)
    # Search mobile no
    adbInput(mobileNo)
    # Tap on name
    tapEvents(601, 574)
    # Tap on input field
    tapEvents(390, 2270)
    # Enter message
    adbInput(message)
    # Send message
    tapEvents(957, 1397)
    speak("Message sent successfully to "+name)
