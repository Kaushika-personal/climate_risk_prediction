import requests
import pyttsx3
import speech_recognition as sr
import eel
import time
import os
import json
import cv2
from hugchat import hugchat

# Function to speak the text (unchanged)
def speak(text):
    text = str(text)
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices') 
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 174)
    engine.say(text)
    engine.runAndWait()

# Function to fetch weather information (new function)
def get_weather(city="Gandhinagar"):
    # Replace with your OpenWeatherMap API Key
    api_key = "b378409c5939d56d924d82438ea6f273"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    
    # Complete url
    url = f"{base_url}q={city}&appid={api_key}&units=metric"  # 'units=metric' for Celsius
    
    # Get weather data
    response = requests.get(url)
    
    # If response is OK, parse it
    if response.status_code == 200:
        data = response.json()
        
        # Extract required data
        main = data["main"]
        weather = data["weather"][0]
        
        # Prepare response message
        temperature = main["temp"]
        humidity = main["humidity"]
        description = weather["description"]
        
        weather_info = f"The weather in {city} is currently {description}. The temperature is {temperature}°C with a humidity of {humidity}%."
        return weather_info
    else:
        return "Sorry, I couldn't fetch the weather information right now."

# Function to listen to user's voice command
@eel.expose
def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('listening....')
        eel.DisplayMessage('listening....')
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        
        audio = r.listen(source, 10, 6)

    try:
        print('recognizing')
        eel.DisplayMessage('recognizing....')
        query = r.recognize_google(audio, language='en-in')
        print(f"user said: {query}")
        eel.DisplayMessage(query)
       
    except Exception as e:
        return ""
    
    return query.lower()

# Function to handle different commands including weather queries
@eel.expose
def allCommands(message=1):
    if message == 1:
        query = takecommand()
        print(query)
        eel.senderText(query)
    else:
        query = message
        eel.senderText(query)
    
    try:
        # Check if the user is asking about weather in Gandhinagar
        if "weather" in query and "gandhinagar" in query:
            weather_info = get_weather("Gandhinagar")
            speak(weather_info)  # Speak the weather information
            eel.senderText(weather_info)  # Send the response to the UI
        elif "open" in query:
            from engine.features import openCommand
            openCommand(query)
        elif "on youtube" in query:
            from engine.features import PlayYoutube
            PlayYoutube(query)
        elif "send message" in query or "phone call" in query or "video call" in query:
            from engine.features import findContact, whatsApp, makeCall, sendMessage
            contact_no, name = findContact(query)
            if(contact_no != 0):
                speak("Which mode you want to use whatsapp or mobile")
                preferance = takecommand()
                print(preferance)

                if "mobile" in preferance:
                    if "send message" in query or "send sms" in query: 
                        speak("what message to send")
                        message = takecommand()
                        sendMessage(message, contact_no, name)
                    elif "phone call" in query:
                        makeCall(name, contact_no)
                    else:
                        speak("please try again")
                elif "whatsapp" in preferance:
                    message = ""
                    if "send message" in query:
                        message = 'message'
                        speak("what message to send")
                        query = takecommand()
                    elif "phone call" in query:
                        message = 'call'
                    else:
                        message = 'video call'
                    whatsApp(contact_no, query, message, name)

        else:
            from .features import chatBot
            chatBot(query)
    except Exception as e:
        print("Error:", e)
    
    eel.ShowHood()
