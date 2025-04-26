from flask import Flask, render_template, Response, jsonify
import cv2
from deepface import DeepFace

app = Flask(__name__)

# Load Haar cascade for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion-based treatment suggestions
treatment_suggestions = {
    "happy": "Keep smiling! Spread joy by talking to a loved one.",
    "sad": "Listen to music or take a deep breath.",
    "angry": "Take a walk or try a 5-minute meditation.",
    "fear": "Try deep breathing. Everything will be okay.",
    "surprise": "Enjoy the moment! Write down your thoughts.",
    "neutral": "Take a short break. A cup of tea might help!",
    "disgust": "Relax and watch something light-hearted."
}

# Function to generate video frames for the webcam feed
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get the detected emotion and suggestion
@app.route('/get_emotion')
def get_emotion():
    cap = cv2.VideoCapture(0)  # Open the webcam
    ret, frame = cap.read()    # Capture a frame
    if ret:
        try:
            # Analyze the frame for emotions
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            suggestion = treatment_suggestions.get(dominant_emotion, "Stay positive!")
            return jsonify({'emotion': dominant_emotion, 'suggestion': suggestion})
        except Exception as e:
            print("Error analyzing emotion:", e)
            return jsonify({'emotion': 'Unknown', 'suggestion': 'Error detecting emotion.'})
    else:
        return jsonify({'emotion': 'Unknown', 'suggestion': 'No frame captured.'})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)