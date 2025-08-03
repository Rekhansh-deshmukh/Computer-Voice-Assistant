import customtkinter as ctk
import threading
import sounddevice as sd
import vosk
import numpy as np
import json
import pyttsx3
import queue
import time
import requests # For making HTTP requests to the LLM

# --- CONFIGURATION ---
VOSK_MODEL_PATH = "vosk-model-en-in-0.5"
WAKE_WORD = "computer"
SAMPLERATE = 16000
CHANNELS = 1
BLOCKSIZE = 4000
COMMAND_TIMEOUT = 5.0 # 5 seconds of no new words

# --- NEW: LLM Configuration ---
# IMPORTANT: Change this URL to your local LLM's API endpoint.
# This example is for Ollama running the llama3 model.
LLM_ENDPOINT = "http://localhost:11434/"
LLM_MODEL = "llama3" # The model name your LLM server uses

class VoiceAssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- TTS ENGINE AND QUEUE SETUP ---
        self.engine = pyttsx3.init()
        self.speak_queue = queue.Queue()
        self.speaker_thread = threading.Thread(target=self._speaker_thread_worker, daemon=True)
        self.speaker_thread.start()

        # --- WINDOW SETUP ---
        self.title("LLM Voice Assistant")
        self.geometry("500x350")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- STATE VARIABLES ---
        self.listening_thread = None
        self.is_listening = False
        self.in_command_window = False
        self.last_word_time = 0
        self.command_audio_buffer = []

        # --- WIDGETS ---
        self.status_label = ctk.CTkLabel(self, text="Status: Idle", font=("Arial", 16))
        self.status_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.log_textbox = ctk.CTkTextbox(self, state="disabled", font=("Arial", 12))
        self.log_textbox.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.toggle_button = ctk.CTkButton(self, text="Start Listening", command=self.toggle_listening)
        self.toggle_button.grid(row=2, column=0, padx=20, pady=20)

    # --- TTS METHODS ---
    def speak(self, text):
        self.speak_queue.put(text)

    def _speaker_thread_worker(self):
        while True:
            try:
                text_to_speak = self.speak_queue.get()
                self.engine.say(text_to_speak)
                self.engine.runAndWait()
                self.speak_queue.task_done()
            except Exception as e:
                print(f"Error in speaker thread: {e}")

    # --- UI & CONTROL METHODS ---
    def update_status(self, text):
        self.status_label.configure(text=f"Status: {text}")

    def log_message(self, message, speak_message=True):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", f"{message}\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")
        if speak_message:
            self.speak(message)

    def toggle_listening(self):
        if self.is_listening: self.stop_listening()
        else: self.start_listening()

    def start_listening(self):
        self.is_listening = True
        self.toggle_button.configure(text="Stop Listening")
        self.listening_thread = threading.Thread(target=self.main_listener_loop, daemon=True)
        self.listening_thread.start()

    def stop_listening(self):
        self.is_listening = False
        self.in_command_window = False
        self.toggle_button.configure(text="Start Listening")
        self.update_status("Idle")

    # --- REWRITTEN: CORE LISTENING LOGIC (Unchanged from previous) ---
    def main_listener_loop(self):
        try:
            vosk_model = vosk.Model(VOSK_MODEL_PATH)
            recognizer = vosk.KaldiRecognizer(vosk_model, SAMPLERATE)
            recognizer.SetWords(True)
        except Exception as e:
            self.log_message(f"Initialization Error: {e}")
            self.stop_listening()
            return

        self.update_status(f"Listening for '{WAKE_WORD}'...")
        with sd.InputStream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype='int16', channels=CHANNELS) as stream:
            while self.is_listening:
                try:
                    data, overflowed = stream.read(BLOCKSIZE)
                    is_final = recognizer.AcceptWaveform(data.tobytes())
                    partial_result = json.loads(recognizer.PartialResult())
                    partial_text = partial_result.get('partial', '')

                    if self.in_command_window:
                        self.command_audio_buffer.append(data)
                        if partial_text: self.last_word_time = time.time()
                        
                        if is_final:
                            final_result = json.loads(recognizer.Result())
                            command_text = final_result.get('text', '')
                            if command_text:
                                self.log_message(f"> Query: {command_text}", speak_message=False)
                                self.process_command(command_text)
                            self.in_command_window = False
                            self.update_status(f"Listening for '{WAKE_WORD}'...")
                        elif time.time() - self.last_word_time > COMMAND_TIMEOUT:
                            self.log_message("Timeout reached, processing command...", speak_message=False)
                            full_command_audio = np.concatenate(self.command_audio_buffer)
                            recognizer.AcceptWaveform(full_command_audio.tobytes())
                            final_result = json.loads(recognizer.Result())
                            command_text = final_result.get('text', '')
                            if command_text:
                                self.log_message(f"> Query: {command_text}", speak_message=False)
                                self.process_command(command_text)
                            else:
                                self.log_message("I heard something, but could not understand.")
                            self.in_command_window = False
                            self.command_audio_buffer = []
                            self.update_status(f"Listening for '{WAKE_WORD}'...")
                    else: # Listening for wake word
                        if WAKE_WORD in partial_text:
                            self.in_command_window = True
                            self.last_word_time = time.time()
                            self.command_audio_buffer = []
                            self.update_status("Speak your query now...")
                            self.log_message(f"'{WAKE_WORD}' detected.")
                            recognizer.Reset()
                except Exception as e:
                    self.log_message(f"Error during listening loop: {e}")
    
    # --- NEW: LLM Query Function ---
    def query_llm(self, prompt):
        self.log_message("Sending query to LLM...", speak_message=False)
        self.update_status("Querying LLM...")
        
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False # We want the full response at once
        }
        
        try:
            response = requests.post(LLM_ENDPOINT, json=payload, timeout=60)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            response_json = response.json()
            return response_json.get("response", "No response text found in LLM output.")
            
        except requests.exceptions.RequestException as e:
            error_message = f"Error connecting to LLM: {e}"
            print(error_message)
            return error_message

    # --- REWRITTEN: COMMAND PROCESSING METHOD ---
    def process_command(self, text):
        # Forward the recognized text to the LLM
        response = self.query_llm(text)
        
        if response:
            self.log_message(f"LLM Response: {response}", speak_message=False)
            self.speak(response)

if __name__ == "__main__":
    app = VoiceAssistantApp()
    app.mainloop()