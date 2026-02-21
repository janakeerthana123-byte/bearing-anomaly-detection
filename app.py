from flask import Flask, request, jsonify
import numpy as np
import joblib
from scipy.stats import kurtosis, skew
from config import FEATURE_NAMES, WINDOW_SIZE, SAMPLING_FREQUENCY

# ----------------------------------------
# Initialize Flask App
# ----------------------------------------
app = Flask(__name__)

# ----------------------------------------
# Load Trained Model
# ----------------------------------------
model = joblib.load("bearing_anomaly_model.pkl")


# ----------------------------------------
# Feature Extraction Function
# ----------------------------------------
def extract_features_from_signal(signal):

    signal = np.array(signal)

    # Ensure correct window size
    if len(signal) != WINDOW_SIZE:
        raise ValueError(f"Signal must contain {WINDOW_SIZE} samples")

    # ----- Time Domain -----
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms
    kurt = kurtosis(signal)
    sk = skew(signal)

    # ----- Frequency Domain -----
    fft_values = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_values)
    freq = np.fft.fftfreq(len(signal), d=1/SAMPLING_FREQUENCY)

    positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
    positive_freq = freq[:len(freq)//2]

    spectral_energy = np.sum(positive_magnitude**2)
    max_fft = np.max(positive_magnitude)
    dominant_freq = positive_freq[np.argmax(positive_magnitude)]

    features = [
        rms, peak, crest_factor,
        kurt, sk,
        spectral_energy, max_fft, dominant_freq
    ]

    return np.array(features).reshape(1, -1)


# ----------------------------------------
# Prediction Endpoint
# ----------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if "signal" not in data:
            return jsonify({"error": "No signal provided"}), 400

        signal = data["signal"]

        features = extract_features_from_signal(signal)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()

        status = "Healthy" if prediction == 0 else "Fault Detected"

        return jsonify({
            "status": status,
            "confidence": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Bearing Anomaly Detection API Running"
# ----------------------------------------
# Run Server
# ----------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)