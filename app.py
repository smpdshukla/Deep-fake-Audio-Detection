import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import os
import matplotlib.pyplot as plt
import io
import requests
from pathlib import Path
import re

# =========================
# 🔧 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Deepfake Audio Detection Dashboard",
    layout="wide",
)

# =========================
# 🎨 THEME (peach + pink)
# =========================
st.markdown(
    """
    <style>
    .stApp{
        background-color: #FFF8E7; /* light peach */
        color: #000000;
    }
    .main-title{
        font-size: 36px;
        font-weight: bold;
        color: #C2185B; /* deep pink */
        text-align: center;
        border-bottom: 4px solid #FF9800; /* orange underline */
        padding-bottom: 5px;
        margin-bottom: 20px;
    }
    .sub-title{
        font-size: 20px;
        color: #E91E63; /* pink accent */
        text-align: center;
        margin-bottom: 30px;
    }
    .sidebar-title{
        font-size: 16px;
        font-weight: bold;
        color: #000000 !important;
        margin-bottom: 8px;
    }
    section[data-testid="stSidebar"]{
        background-color: #ffd8a8 !important; /* peachy sidebar */
    }
    [data-testid="stSidebar"]{
        color: #c71585  !important;
    }
    section[data-testid="stSidebar"] label{
        color: #c2185b !important;
        font-weight: bold;
    }
    .result-box{
        background-color: #FCE4EC; /* light pink */
        border: 2px solid #C2185B;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #C2185B;
        margin-top: 20px;
    }
    /* Buttons */
    .stButton > button {
        background-color: #C2185B !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #a01549 !important;
        color: #fff !important;
    }
    </style>    
     """,
    unsafe_allow_html=True,
)
st.markdown(""" 
<style>
    [data-testid="stNotificationContent"]{
        color: #c2185b !important;
        font-weight: bold;
    }
    .stSuccess[data-testid="stNotificationContent"]{
            color: #2e7d32 !important;
            font-weight: bold;
    }
</style>
 """, unsafe_allow_html=True)

# =========================
# 📚 SIDEBAR (About + Upload)
# =========================
st.sidebar.title("📖 About the App")

st.sidebar.markdown(
    """
    <div class="sidebar-info-box" style="
        background-color:#ffe6f0;
        border:2px solid #ff66a3;
        border-radius:10px;
        padding:12px;
        margin-bottom:15px;
        color:#2a2a2a;
        font-size:15px;
        line-height:1.5;">
      <h4 style="color:#cc0066;margin-top:0;margin-bottom:8px;">Deepfake Audio Detection</h4>
      <p style="color: #c71585;">
        This app identifies whether an audio file is <b>Real</b> (human voice) or <b>Deepfake</b> (AI-generated voice)
        using a <b>stacked ensemble</b>:
      </p>
      <ul style="padding-left:20px;margin:0;color: #c71585;">
        <li>CNN, RNN, BiLSTM, GRU</li>
        <li>Logistic Regression (meta-classifier)</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("""
    <style>
    .sidebar-title {
        color: #c71585 !important;
        font-weight: bold;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


st.sidebar.markdown(
    '<div class="sidebar-title">Upload Audio (Local or URL)</div>',
    unsafe_allow_html=True
)


# --- Upload controls (Local + URL without ffmpeg) ---
st.markdown(""" 
    <style>
    /* Disable pointer events for the whole dropzone except the Browse button */
    [data-testid="stFileUploaderDropzone"]{
        pointer-events: none !important; 
    }
    [data-testid="stFileUploaderDropzone"] button {
        pointer-events: auto !important;
        } </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    [data-testid="stTooltipIcon"] svg {
        fill: #c71585 !important;
    }</style> 
""", unsafe_allow_html=True)
ALLOWED_EXTS = {"wav"}  # keep formats that work without ffmpeg reliably

uploaded_file = st.sidebar.file_uploader(
    "Choose a file (WAV)",
    type=list(ALLOWED_EXTS),
    help="Drag & drop or browse a local file.",
)

st.sidebar.markdown("**Or** paste a direct URL (WAV)")
audio_url = st.sidebar.text_input("Audio URL", placeholder="https://example.com/sample.wav")
fetch_btn = st.sidebar.button("Fetch from URL")

# =========================
# 📂 File handling helpers
# =========================

MODEL_DIR = os.path.dirname(__file__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\.-]", "_", name)
    return name[:120] if len(name) > 120 else name

def save_bytes_to_disk(b: bytes, name_hint: str) -> str:
    fname = sanitize_filename(name_hint)
    if "." not in Path(fname).name:
        fname += ".wav"
    path = os.path.join(UPLOAD_DIR, fname)
    with open(path, "wb") as f:
        f.write(b)
    return path

def load_from_url(url: str):
    if not url:
        return None, None, "Please paste a valid URL."
    ext = Path(url.split("?")[0]).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTS:
        return None, None, f"Unsupported URL file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTS)}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return io.BytesIO(r.content), Path(url.split("?")[0]).name, None
    except Exception as e:
        return None, None, f"Failed to fetch URL: {e}"

# =========================
# 🧠 Models (definitions + loading)
# =========================
def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_bilstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(128),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource
def load_models():
    # Build architectures (must match training-time shapes)
    cnn_model = create_cnn_model((120, 1))
    rnn_model = create_rnn_model((120, 1))
    bilstm_model = create_bilstm_model((120, 1))
    gru_model = create_gru_model((120, 1))

    # Load weights
    cnn_model.load_weights(os.path.join(MODEL_DIR, "cnn.h5"))
    rnn_model.load_weights(os.path.join(MODEL_DIR, "rnn.h5"))
    bilstm_model.load_weights(os.path.join(MODEL_DIR, "bilstm.h5"))
    gru_model.load_weights(os.path.join(MODEL_DIR, "gru.h5"))

    # XGBoost
    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(MODEL_DIR, "model_xgb.json"))

    # Meta model (LogReg)
    meta_model = LogisticRegression()
    try:
        meta_model.coef_ = np.load(os.path.join(MODEL_DIR, "meta_model_coef.npy"))
        meta_model.intercept_ = np.load(os.path.join(MODEL_DIR, "meta_model_intercept.npy"))
        meta_model.classes_ = np.array([0, 1])
    except FileNotFoundError:
        st.error("Meta-model files are missing. Please ensure 'meta_model_coef.npy' and 'meta_model_intercept.npy' exist.")
    return cnn_model, bilstm_model, rnn_model, gru_model, xgb_model, meta_model

cnn_model, bilstm_model, rnn_model, gru_model, xgb_model, meta_model = load_models()

# =========================
# 🎚️ Feature extraction
# =========================
def extract_features(file_path, sr=16000, n_mfcc=40):
    """
    Extract features from the audio file: MFCCs + delta + delta-delta.
    Return mean-pooled feature vector.
    """
    audio_data, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features = np.vstack((mfcc, delta_mfcc, delta2_mfcc))
    return np.mean(features.T, axis=0)

def get_weighted_predictions(models, features_rnn, features_cnn, features_scaled, weights):
    predictions = []
    for model_name, model in models.items():
        if model_name in ['rnn', 'bilstm', 'gru']:
            pred = model.predict(features_rnn, verbose=0).flatten() * weights[model_name]
        elif model_name == 'cnn':
            pred = model.predict(features_cnn, verbose=0).flatten() * weights[model_name]
        elif model_name == 'xgb':
            pred = model.predict(xgb.DMatrix(features_scaled)) * weights[model_name]
        predictions.append(pred)
    return np.sum(predictions, axis=0)

# =========================
# 🧭 HEADER
# =========================
st.markdown('<div class="main-title">Deepfake Audio Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Using Stacking Model for Classification</div>', unsafe_allow_html=True)

# =========================
# 📥 Resolve input source (local or URL)
# =========================
file_path = None
audio_bytes_for_player = None
audio_mime = None

if uploaded_file is not None:
    ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTS:
        st.error(f"Unsupported file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTS)}")
    else:
        # Save to disk
        file_path = save_bytes_to_disk(uploaded_file.read(), uploaded_file.name)
        audio_bytes_for_player = open(file_path, "rb").read()
        audio_mime = "audio/wav" if ext == "wav" else ("audio/flac" if ext == "flac" else "audio/mp3")

elif fetch_btn and audio_url:
    buf, name, err = load_from_url(audio_url)
    if err:
        st.error(err)
    else:
        ext = Path(name).suffix.lower().lstrip(".")
        file_path = save_bytes_to_disk(buf.getvalue(), name)
        audio_bytes_for_player = open(file_path, "rb").read()
        audio_mime = "audio/wav" if ext == "wav" else ("audio/flac" if ext == "flac" else "audio/mp3")
        st.success("✅ Audio fetched from the web.")

# =========================
# 🔎 Process, visualize, predict
# =========================
if file_path:
    # 1) Audio player
    st.markdown("### 🎧 Play Uploaded Audio")
    st.audio(audio_bytes_for_player, format=audio_mime)

    # Progress bar
    progress_bar = st.progress(0)
    progress_bar.progress(20)

    # 2) Load waveform for plots
    try:
        audio_data, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        st.error(f"Failed to load audio for analysis: {e}")
        st.stop()

    progress_bar.progress(40)

    # 3) Waveform
    st.markdown("### Waveform of Uploaded Audio")
    fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio_data, sr=sr, ax=ax_wave, color="#FF9800")
    ax_wave.set_title("Waveform")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    st.pyplot(fig_wave)

    progress_bar.progress(60)

    # 4) MFCCs
    st.markdown("### MFCCs of Uploaded Audio")
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    fig, ax = plt.subplots(figsize=(10, 4))
    mfcc_display = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax, cmap='magma')
    ax.set_title("MFCCs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC Coefficients")
    plt.colorbar(mfcc_display, format='%+2.0f dB', ax=ax)
    st.pyplot(fig)

    progress_bar.progress(75)

    # 5) Feature extraction
    features = extract_features(file_path)
    features_rnn = features.reshape(1, -1, 1)
    features_cnn = features.reshape(1, -1, 1)
    features_scaled = features.reshape(1, -1)

    # 6) Ensemble prediction
    models = {
        'rnn': rnn_model,
        'cnn': cnn_model,
        'bilstm': bilstm_model,
        'gru': gru_model,
        'xgb': xgb_model
    }
    weights = {
        'rnn': 0.35,
        'cnn': 1.0,
        'bilstm': 1.0,
        'gru': 0.35,
        'xgb': 0.5
    }

    weighted_predictions = get_weighted_predictions(models, features_rnn, features_cnn, features_scaled, weights)

    progress_bar.progress(90)

    # 7) Meta-model decision
    try:
        final_prediction = meta_model.predict(weighted_predictions.reshape(-1, 1))[0]
        probs = meta_model.predict_proba(weighted_predictions.reshape(-1, 1))[0]
        confidence = float(np.max(probs))

        label = "Real" if final_prediction == 0 else "Fake"
        if label == "Real":
            bg_color, border_color, text_color = "#d4edda", "#28a745", "#155724"  # greenish
        else:
            bg_color, border_color, text_color = "#f8d7da", "#dc3545", "#721c24"  # reddish

        st.markdown(
            f"""
            <div style="
                border: 2px solid {border_color};
                border-radius: 10px;
                padding: 15px;
                background-color: {bg_color};
                color: {text_color};
                font-size: 18px;
                font-weight: bold;
                text-align: center;">
                Classification Result: {label}<br>
                Confidence Score: {confidence*100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )
    except AttributeError:
        st.error("Meta-model is not properly loaded. Please ensure the meta-model files are available.")

    progress_bar.progress(100)

    # 8) Accuracy chart (static example)
    st.subheader("Model Accuracy")
    models_list = ["CNN", "RNN", "BiLSTM", "GRU", "XGBoost", "Stacked Model"]
    accuracy = [95, 90, 92, 89, 67, 94]
    fig_acc, ax = plt.subplots(figsize=(8, 5))
    ax.barh(models_list, accuracy, color='skyblue')
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model Accuracy")
    ax.invert_yaxis()
    for i, v in enumerate(accuracy):
        ax.text(v + 1, i, f"{v}%", va='center')
    st.pyplot(fig_acc)
else:
    st.markdown("""
    <div style="
        background-color: #2196F3; 
        color: white; 
        padding: 10px; 
        border-radius: 8px;">
        Upload a local WAV file or paste a direct URL and click <b>Fetch from URL</b> to begin.
    </div>
    </style> """,unsafe_allow_html=True)
    


















