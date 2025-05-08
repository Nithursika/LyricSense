from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load scikit-learn model and vectorizer
model = joblib.load("trained_model/genre_classifier_model.pkl")
vectorizer = joblib.load("trained_model/tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')  # Make sure you have this HTML

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    lyrics = data.get("lyrics", "")
    if not lyrics.strip():
        return jsonify({"error": "Please provide lyrics"}), 400

    try:
        X = vectorizer.transform([lyrics])
        prediction = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        response = {
            "predicted_genre": prediction,
            "probabilities": dict(zip(model.classes_, probs.round(4)))
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
