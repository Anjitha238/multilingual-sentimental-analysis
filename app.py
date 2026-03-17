from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect

app = Flask(__name__)
CORS(app)

# Multilingual sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text")
    lang = data.get("language")

    # Auto detect language if needed
    if lang == "auto":
        lang = detect(text)

    translated_text = text

    # Translate only if not English
    if lang != "en":
        translated_text = GoogleTranslator(
            source=lang, target="en"
        ).translate(text)

    result = sentiment_pipeline(translated_text)[0]

    stars = int(result["label"][0])
    # score = round(result["score"] * 100, 2)
    score = round((result["score"] * 100), 2)

    if stars <= 2:
        sentiment = "negative"
    elif stars == 3:
        sentiment = "neutral"
    else:
        sentiment = "positive"

    return jsonify({
        "original_text": text,
        "detected_language": lang,
        "translated_text": translated_text,
        "sentiment": sentiment,
        "confidence": score
    })

if __name__ == "__main__":
    app.run(debug=True)
