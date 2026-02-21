from flask import Flask, request, render_template
import numpy as np
import joblib
from scipy.stats import kurtosis, skew
from config import WINDOW_SIZE, SAMPLING_FREQUENCY

app = Flask(__name__)

model = joblib.load("bearing_anomaly_model.pkl")


def extract_features_from_signal(signal):

    signal = np.array(signal)

    if len(signal) != WINDOW_SIZE:
        raise ValueError(f"Signal must contain exactly {WINDOW_SIZE} values")

    # Time-domain
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms
    kurt = kurtosis(signal)
    sk = skew(signal)

    # Frequency-domain
    fft_values = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_values)
    freq = np.fft.fftfreq(len(signal), d=1/SAMPLING_FREQUENCY)

    positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
    positive_freq = freq[:len(freq)//2]

    spectral_energy = np.sum(positive_magnitude**2)
    max_fft = np.max(positive_magnitude)
    dominant_freq = positive_freq[np.argmax(positive_magnitude)]

    features = np.array([
        rms, peak, crest_factor,
        kurt, sk,
        spectral_energy, max_fft, dominant_freq
    ]).reshape(1, -1)

    return features


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_input = request.form["signal"]
        signal = list(map(float, raw_input.split(",")))

        features = extract_features_from_signal(signal)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()

        status = "Healthy" if prediction == 0 else "Fault Detected"

        return render_template(
            "index.html",
            prediction=status,
            confidence=round(float(probability), 3)
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction="Error",
            confidence=str(e)
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)