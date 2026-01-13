import os
from flask import Flask, request, render_template, send_from_directory
from pydub import AudioSegment
import yt_dlp
import sys
from openai import OpenAI
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

load_dotenv()  # loads .env into environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in .env")

# api_key = os.getenv("AIzaSyB6fvsaJI8h7II7OT3NrBvz25GGKYIVukw")
sys.path.append(os.path.join(os.path.dirname(__file__),"Shazam"))
from app_matching import recognize_uploaded_song
app = Flask(__name__)


SAVE_DIR = "recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

MUSIC_DIR = "music"
os.makedirs(MUSIC_DIR, exist_ok=True)


def get_next_filename(base_name="recording", extension=".mp3"):
    counter = 1
    while True:
        filename = f"{base_name}{counter}{extension}"
        if not os.path.exists(os.path.join(SAVE_DIR, filename)):
            return filename
        counter += 1


def make_unique_filename(filename):
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(SAVE_DIR, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1
    return new_filename


def get_song_metadata_gemini(song_name):
    """
    Use the new OpenAI/Gemini API client (>=1.0) to get singer/metadata.
    """
    client = genai.Client(api_key=api_key)

    prompt = f"Give me the singer of the song '{song_name}' and any additional info like album or release year."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=GenerateContentConfig(
                thinking_config=ThinkingConfig(thinking_budget=512)
            )
        )
        # New API: messages are in response.choices[0].message.content
        content = response.candidates[0].content.parts[0].text
        return content
    except Exception as e:
        return f"Error fetching metadata: {str(e)}"


@app.route("/")
def index():
    return render_template("index.html", title="Record Audio")


@app.route("/save_audio", methods=["POST"])
def save_audio():
    audio_file = request.files["audio_data"]
    user_filename = request.form.get("filename", "").strip()

    if not user_filename:
        filename = get_next_filename()
    else:
        if not user_filename.lower().endswith(".mp3"):
            user_filename += ".mp3"
        filename = make_unique_filename(user_filename)

    temp_path = os.path.join(SAVE_DIR, "temp.webm")
    audio_file.save(temp_path)

    sound = AudioSegment.from_file(temp_path, format="webm")
    sound.export(os.path.join(SAVE_DIR, filename), format="mp3")
 
    os.remove(temp_path)

    return f"Saved as {filename} - <a href='/download/{filename}' target='_blank'>Download</a>"


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(SAVE_DIR, filename, as_attachment=True)


@app.route("/youtube")
def youtube_page():
    return render_template("youtube.html", title="YouTube Download")


@app.route("/download_youtube", methods=["POST"])
def download_youtube():
    youtube_url = request.form.get("youtube_url", "").strip()
    if not youtube_url:
        return "Please provide a YouTube link."

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(MUSIC_DIR, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            song_title = info.get('title', 'downloaded_song')
        return f"Downloaded and saved in 'music' folder as {song_title}.mp3"
    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/upload")
def upload_page():
    return render_template("upload.html", title = "Upload Audio")

@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file uploaded"
    file = request.files["file"]
    if file.filename =="":
        return "No file selected"
    
    user_filename = request.form.get("filename","").strip()
    if not user_filename:
        user_filename = get_next_filename()
    else:
        if not user_filename.lower().endswith(".mp3"):
            user_filename += ".mp3"
        user_filename = make_unique_filename(user_filename)
    print("DEBUG >>> user_filename =", user_filename, type(user_filename))
    temp_path = os.path.join(SAVE_DIR,file.filename)
    file.save(temp_path)

    name, ext = os.path.splitext(file.filename)
    ext = ext.lstrip(".").lower()

    # If already mp3, just save directly
    if ext == "mp3":
        os.rename(temp_path, os.path.join(SAVE_DIR, user_filename))
    else:
        try:
            sound = AudioSegment.from_file(temp_path, format=ext)
            sound.export(os.path.join(SAVE_DIR, user_filename), format="mp3")
            os.remove(temp_path)
        except Exception as e:
            os.remove(temp_path)
            return f"Error converting file: {str(e)}"

    return render_template("file_saved.html", filename=user_filename)

@app.route("/recognize")
def recognize_page():
    return render_template("recognize.html", title="Song Recognition")


@app.route("/recognize_file", methods=["POST"])
def recognize_file():
    if "file" not in request.files:
        return "No file uploaded."

    file = request.files["file"]
    if file.filename == "":
        return "No file selected."

    temp_path = os.path.join(SAVE_DIR, "temp_recognize.mp3")
    file.save(temp_path)

    # Prepare database songs
    # database_songs = {}
    # for f in os.listdir(SAVE_DIR):
    #     database_songs[f] = os.path.join(SAVE_DIR, f)
    # for f in os.listdir(MUSIC_DIR):
    #     database_songs[f] = os.path.join(MUSIC_DIR, f)

    # Call your fingerprinting function
    try:
        best_match = recognize_uploaded_song(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return f"Error recognizing song: {str(e)}"

    os.remove(temp_path)

    if best_match:
        metadata = get_song_metadata_gemini(best_match)
        html_metadata = metadata.replace("*", "").replace(".", "").replace("\n", "<br>")
        return f"""
        <h3>Best match: {best_match}</h3>
        <h4>Metadata:</h4>
        <p>{html_metadata}</p>
        <a href='/'>Back to Home</a>
        """
        # return render_template(
        #     "recognize_result.html",
        #     best_match=best_match,
        #     metadata=html_metadata
        # )
    else:
        return f"No match found. <br><a href='/'>Back to Home</a>"
        # return render_template(
        #     "recognize_result.html",
        #     best_match=None,
        #     metadata=""
        # )


if __name__ == "__main__":
    app.run(debug=True)
