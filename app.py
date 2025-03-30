from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load Hugging Face model for emotion analysis
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Emoji mapping for detected emotions
emoji_map = {
    "joy": "😃",
    "anger": "😠",
    "surprise": "😲",
    "sadness": "😢",
    "fear": "😨",
    "disgust": "🤢",
    "neutral": "😐"
}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Analyze emotion using AI model
    results = emotion_classifier(message)
    
    # Get the most confident emotion
    top_emotion = max(results[0], key=lambda x: x['score'])

    emotion_label = top_emotion["label"]
    confidence = round(top_emotion["score"] * 100, 2)
    emoji = emoji_map.get(emotion_label.lower(), "❓")

    return jsonify({
        "emotion": emotion_label,
        "confidence": confidence,
        "emoji": emoji
    })


if __name__ == '__main__':
    PORT = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT, debug=True)
