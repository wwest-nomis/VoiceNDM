# Copyright (c) 2025 Nomis Solutions, Inc
# Licensed under the MIT License - see LICENSE file for details

# Imports
import os
import whisper
import sounddevice as sd
import numpy as np
from openai import OpenAI
import pyttsx3
import time
import random
import re
import json
from dotenv import load_dotenv
from datetime import datetime
from tabulate import tabulate
import queue
import threading

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# SSL Certificate workaround for macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Whisper models with custom download location
print("Loading Whisper models (this may take a moment on first run)...")
print("Downloading/loading tiny model...")
whisper_tiny = whisper.load_model("tiny", download_root="./whisper_model")
print("Downloading/loading base model...")
whisper_base = whisper.load_model("base", download_root="./whisper_model")
print("Downloading/loading small model...")
whisper_small = whisper.load_model("small", download_root="./whisper_model")
print("All Whisper models loaded successfully!")

# Model info for display
model_info = {
    "small": {"model": whisper_small, "size": "244M params"},
    "base": {"model": whisper_base, "size": "74M params"},
    "tiny": {"model": whisper_tiny, "size": "39M params"}
}

# Audio recording parameters
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1
BLOCKSIZE = 1024
SILENCE_THRESHOLD = 0.01  # Adjust based on your environment
SILENCE_DURATION = 2.0  # Seconds of silence before stopping

#%% Text-to-Speech Function

def speak(text):
    """
    Convert text to speech and play it
    """
    engine = pyttsx3.init()
    engine.say(text)
    print(f"Computer: {text}")
    engine.runAndWait()

#%% Voice Recognition with Whisper

def listen_for_response(max_attempts=3, max_duration=15):
    """
    Listen for user's speech using sounddevice and convert to text with Whisper
    Uses voice activity detection to automatically stop recording
    """
    
    for attempt in range(max_attempts):
        print(f"\n🎤 Listening... (speak now) - Attempt {attempt+1}/{max_attempts}")
        
        try:
            # Queue for audio data
            audio_queue = queue.Queue()
            
            # Variables for voice activity detection
            is_speaking = False
            silence_start = None
            audio_data = []
            
            def callback(indata, frames, time, status):
                """Callback for sounddevice stream"""
                if status:
                    print(f"Audio callback status: {status}")
                
                # Add audio to queue
                audio_queue.put(indata.copy())
                
            # Start the audio stream
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=callback,
                blocksize=BLOCKSIZE,
                dtype='float32'
            )
            
            with stream:
                start_time = time.time()
                
                while True:
                    try:
                        # Get audio chunk from queue
                        chunk = audio_queue.get(timeout=0.1)
                        audio_data.append(chunk)
                        
                        # Calculate volume
                        volume = np.sqrt(np.mean(chunk**2))
                        
                        # Voice activity detection
                        if volume > SILENCE_THRESHOLD:
                            is_speaking = True
                            silence_start = None
                            print(".", end="", flush=True)  # Visual feedback
                        else:
                            if is_speaking and silence_start is None:
                                silence_start = time.time()
                            elif is_speaking and silence_start and (time.time() - silence_start > SILENCE_DURATION):
                                # Stop recording after silence
                                print("\n✓ Recording complete")
                                break
                        
                        # Timeout check
                        if time.time() - start_time > max_duration:
                            print("\n⏱️ Maximum recording time reached")
                            break
                            
                    except queue.Empty:
                        continue
            
            # Convert audio data to numpy array
            if audio_data:
                audio_array = np.concatenate(audio_data).flatten()
                
                # Ensure audio is in the correct format for Whisper
                audio_array = audio_array.astype(np.float32)
                
                # Transcribe with all three models (slowest to fastest)
                print("\n" + "-" * 60)
                print("Starting transcription comparison...")
                
                transcription_results = []
                base_transcription = None
                
                for model_name in ["small", "base", "tiny"]:
                    model_data = model_info[model_name]
                    print(f"\nStarting transcription with {model_name} (size: {model_data['size']})")
                    
                    start_time_transcription = time.time()
                    result = model_data["model"].transcribe(
                        audio_array,
                        language="en",
                        fp16=False  # Set to False for better compatibility
                    )
                    end_time_transcription = time.time()
                    
                    transcription_time = end_time_transcription - start_time_transcription
                    text = result["text"].strip()
                    
                    if model_name == "base":
                        print(f"Raw transcription ({transcription_time:.2f} sec to process. WILL USE THIS ONE): {text}")
                        base_transcription = text
                    else:
                        print(f"Raw transcription ({transcription_time:.2f} sec to process): {text}")
                    
                    transcription_results.append({
                        "model": model_name,
                        "time": transcription_time,
                        "text": text
                    })
                
                print("-" * 60)
                
                # Use the base model transcription
                if base_transcription:
                    print(f"\nYou said: {base_transcription}")
                    return base_transcription.lower()
                else:
                    print("No speech detected in recording")
                    if attempt < max_attempts - 1:
                        speak("I didn't hear anything. Could you please try again?")
            
        except Exception as e:
            print(f"Error during recording: {e}")
            if attempt < max_attempts - 1:
                speak("I'm having trouble hearing you. Let's try once more.")
    
    # If we get here, all attempts failed
    return None

#%% GPT Integration Functions

def process_with_gpt(user_text, context):
    """
    Use GPT to interpret user responses and extract structured information
    """
    try:
        # Create a prompt based on the context and user text
        if context == "name":
            prompt = f"""
            The user is responding to "What's your name?" and said: "{user_text}"
            Extract their first name only, ignoring filler words like "uh", "um", etc.
            If there are multiple names, pick the one that seems most likely to be their first name.
            Return ONLY the name as plain text with proper capitalization, without any prefixes or labels.
            Just the name itself, nothing else - no "Name: " prefix.
            """
        elif context == "loan_interest":
            prompt = f"""
            The user is responding to "Are you looking for a loan?" and said: "{user_text}"
            Determine if their response is affirmative (yes) or negative (no).
            Return ONLY "yes" or "no".
            """
        elif context == "state":
            prompt = f"""
            The user is responding to "What state do you live in?" and said: "{user_text}"
            Extract the US state name with proper capitalization.
            Return ONLY the state name with proper capitalization.
            """
        elif context == "house_value":
            prompt = f"""
            The user is responding to "What is the value of your house?" and said: "{user_text}"
            Extract the numeric value of the house.
            Handle formats like "$X", "X dollars", "X million", "Xmm", etc.
            Return the value as a number (in dollars, not millions).
            Return ONLY the numeric value with no text or symbols.
            """
        elif context == "rate_acceptance":
            prompt = f"""
            The user is responding to "Is this interest rate acceptable to you?" and said: "{user_text}"
            Determine if their response is affirmative (yes) or negative (no).
            Return ONLY "yes" or "no".
            """
        elif context == "sports_team":
            prompt = f"""
            The user said they're from "{user_text}" state.
            Generate a brief, cheeky comment about a popular sports team from that state.
            Keep it light-hearted and playful, about 1-2 sentences.
            Return only text and numbers, no emoji or special characters.
            """
        
        # Make the GPT API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract specific information from user input."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more deterministic responses
            max_tokens=50
        )
        
        # Extract and return the processed text
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error processing with GPT: {e}")
        return user_text  # Fallback to original text

def extract_number(text):
    """
    Extract a numeric value from text, handling different formats
    First tries GPT, then falls back to regex if GPT fails
    """
    try:
        # Try to use GPT to extract the number
        numeric_value = process_with_gpt(text, "house_value")
        
        # Process the result
        if numeric_value:
            # Remove any non-numeric characters except decimal point
            clean_value = re.sub(r"[^0-9.]", "", numeric_value)
            return float(clean_value)
    except:
        pass
    
    # Fallback to regex method
    # Remove common currency symbols and words
    text = text.replace("$", "").replace("dollars", "").replace("million", "000000")
    
    # Handle abbreviated forms
    if "k" in text.lower():
        text = text.lower().replace("k", "000")
    if "m" in text.lower() or "mm" in text.lower():
        text = re.sub(r"[mM]{1,2}", "000000", text)
        
    # Find all numbers in the text
    numbers = re.findall(r"\d+\.?\d*", text)
    
    if numbers:
        return float(numbers[0])
    return None

def calculate_loan_details(house_value):
    """
    Calculate loan amount and rate
    """
    # Calculate 80% of house value
    loan_amount = house_value * 0.8
    
    # Generate random interest rate between 6% and 7.5% in 0.25% increments
    possible_rates = [6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5]
    interest_rate = random.choice(possible_rates)
    
    return loan_amount, interest_rate

def generate_rate_discount():
    """
    Generate a random discount in 5 bp increments from 5 bps to 30 bps
    """
    possible_discounts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    return random.choice(possible_discounts)

#%% Main Conversation Flow

def run_loan_conversation():
    # Dictionary to store all information for the final table
    loan_data = {
        "call_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": "",
        "state": "",
        "house_value": 0,
        "loan_amount": 0,
        "offered_rate": 0,
        "rate_acceptable": "",
        "exception_given": 0,
        "exception_rate": 0,
        "exception_acceptable": "N/A"
    }
    
    # Introduction
    speak("Hi, what's your name?")
    
    # Get name
    name_response = listen_for_response()
    if not name_response:
        speak("I didn't catch your name after several attempts. Let's try again later.")
        return
    
    # Use GPT to extract name
    first_name = process_with_gpt(name_response, "name")
    if not first_name or first_name.lower() == name_response.lower():  # If GPT returns unchanged text
        # Fallback to simpler method
        name_parts = name_response.split()
        if len(name_parts) >= 1:
            first_name = name_parts[0].capitalize()
        else:
            first_name = "there"  # Fallback
    
    # Store the name
    loan_data["name"] = first_name
    
    # Greet by name and ask about loan
    speak(f"Hi, {first_name}. Are you looking for a loan?")
    
    # Get loan interest confirmation
    loan_interest_response = listen_for_response()
    if not loan_interest_response:
        speak("I didn't catch that after several attempts. Let's start over.")
        return
    
    # Use GPT to determine if response is affirmative
    loan_interest = process_with_gpt(loan_interest_response, "loan_interest")
    if loan_interest.lower() != "yes":
        speak("It seems you're not interested in a loan right now. Feel free to come back when you need one.")
        return
    
    # Ask about state
    speak("Ok great. I just need to collect some information from you. To start, what state do you live in?")
    
    # Get state
    state_response = listen_for_response()
    if not state_response:
        speak("I didn't catch your state after several attempts. Let's try again later.")
        return
    
    # Use GPT to extract state
    state = process_with_gpt(state_response, "state")
    loan_data["state"] = state
    
    # Get sports team comment
    sports_comment = process_with_gpt(state, "sports_team")
    
    # Respond with the state and sports comment
    speak(f"Thank you. You live in {state}. {sports_comment}")
    
    # Ask about house value
    speak("And what is the value of your house?")
    
    # Get house value
    house_value_response = listen_for_response()
    if not house_value_response:
        speak("I didn't catch the house value after several attempts. Let's try again later.")
        return
    
    # Extract numeric value from response using GPT
    house_value = extract_number(house_value_response)
    if not house_value:
        speak("I couldn't understand the house value. Please try again with a number.")
        return
    
    # Store house value
    loan_data["house_value"] = house_value
    
    # Calculate loan details
    loan_amount, interest_rate = calculate_loan_details(house_value)
    loan_data["loan_amount"] = loan_amount
    loan_data["offered_rate"] = interest_rate
    
    # Format currency values for speech
    if loan_amount >= 1000000:
        formatted_loan = f"{loan_amount/1000000:.2f} million dollars"
    else:
        formatted_loan = f"{loan_amount:,.2f} dollars"
        
    if house_value >= 1000000:
        formatted_house = f"{house_value/1000000:.2f} million dollars"
    else:
        formatted_house = f"{house_value:,.2f} dollars"
    
    # Present results
    speak(f"Got it. For a house valued at {formatted_house}, at 80% loan-to-value, I calculate the loan amount to be {formatted_loan} and we can offer you a {interest_rate}% interest rate.")
    
    # Ask if the rate is acceptable
    speak("Is this interest rate acceptable to you?")
    
    # Get rate acceptance
    rate_response = listen_for_response()
    if not rate_response:
        speak("I didn't catch your response about the rate. Let's assume it's acceptable and continue.")
        loan_data["rate_acceptable"] = "yes"
    else:
        rate_acceptance = process_with_gpt(rate_response, "rate_acceptance")
        loan_data["rate_acceptable"] = rate_acceptance
        
        # If rate is not acceptable, offer a discount
        if rate_acceptance.lower() == "no":
            # Generate a random discount
            discount = generate_rate_discount()
            loan_data["exception_given"] = discount
            
            # Calculate new rate
            new_rate = interest_rate - discount
            loan_data["exception_rate"] = new_rate
            
            # Present the new rate
            speak(f"I understand. I can offer you a special discount of {discount} percentage points, bringing your rate down to {new_rate}%.")
            speak("Is this new rate acceptable to you?")
            
            # Get response about new rate
            new_rate_response = listen_for_response()
            if not new_rate_response:
                speak("I didn't catch your response about the new rate. Let's assume it's acceptable and continue.")
                loan_data["exception_acceptable"] = "yes"
            else:
                new_rate_acceptance = process_with_gpt(new_rate_response, "rate_acceptance")
                loan_data["exception_acceptable"] = new_rate_acceptance
                
                if new_rate_acceptance.lower() == "yes":
                    speak("Great! I'll process your application with the discounted rate.")
                else:
                    speak("I understand. Unfortunately, that's the best rate we can offer at this time.")
        else:
            speak("Great! I'll process your application with the standard rate.")
    
    # Display the structured table
    table_data = [
        ["Time / date of call", loan_data["call_datetime"]],
        ["Name of Applicant", loan_data["name"]],
        ["State", loan_data["state"]],
        ["House value", f"${loan_data['house_value']:,.2f}"],
        ["Loan amount", f"${loan_data['loan_amount']:,.2f}"],
        ["Offered interest rate", f"{loan_data['offered_rate']}%"],
        ["Whether that rate is acceptable", loan_data["rate_acceptable"]]
    ]
    
    # Add exception information if applicable
    if loan_data["rate_acceptable"] == "no":
        table_data.extend([
            ["Exception given", f"{loan_data['exception_given']} percentage points"],
            ["Exception rate", f"{loan_data['exception_rate']}%"],
            ["Whether exception rate is acceptable", loan_data["exception_acceptable"]]
        ])
    
    # Print the table
    print("\n\n" + "=" * 60)
    print("LOAN APPLICATION SUMMARY")
    print("=" * 60)
    print(tabulate(table_data, tablefmt="grid"))
    print("=" * 60 + "\n")
    
    # End conversation
    speak("Thank you for using our service. A summary of your application has been generated. Is there anything else you'd like to know?")

#%% Main Execution

if __name__ == "__main__":
    print("Loan Assistant Voice System (Powered by Whisper - Multi-Model Comparison)")
    print("Press Ctrl+C to exit\n")
    
    # List available audio devices
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show input devices
            print(f"  Device {i}: {device['name']} (Inputs: {device['max_input_channels']})")
    
    # Ask user to select device (optional - sounddevice usually picks the right default)
    try:
        selected_device = int(input("\nEnter device number (or press Enter for default): "))
        sd.default.device = selected_device
    except ValueError:
        print("Using default audio device")
    
    try:
        # Test audio setup
        print("\nTesting audio setup...")
        test_duration = 1
        test_audio = sd.rec(int(test_duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        print("Audio setup successful!\n")
        
        # Start the conversation
        run_loan_conversation()
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your audio device settings.")