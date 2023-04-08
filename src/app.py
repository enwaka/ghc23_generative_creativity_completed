import soundfile as sf
import sounddevice as sd
import openai
import os
import whisper
from dotenv import load_dotenv
from flask import Flask, render_template

#load environment variables
load_dotenv()

# Select from the following models: "tiny", "base", "small", "medium", "large"
model = whisper.load_model("base")

app = Flask(__name__)
@app.route("/home")
def index():
    return render_template("/input.html")      

@app.route("/record", methods = ['POST', 'GET'])
def voice_rec():
    fs = 44100
    duration = 5
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait() 
    
    sf.write("audio_file.mp3", myrecording, fs)
    generated_lyrics = transcribe()
    return render_template("input.html", generated_lyrics=generated_lyrics)


def transcribe():
    print("Transcribing audio to text...")
    audio = "audio_file.mp3"
    options = {"fp16": False, "task": "transcribe"}
    results = model.transcribe(audio, **options)

    print("The transcribed text is...")
    print(results["text"])

    generated_lyrics = generate_lyrics(results["text"])
    return generated_lyrics


def generate_lyrics(text):
    openai.api_key = os.environ.get('OPEN_API_KEY')
    
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Write a music lyrics: {text}",
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text