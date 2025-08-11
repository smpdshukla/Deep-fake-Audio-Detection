import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import os
import matplotlib.pyplot as plt

st.sidebar.title("📖 About the Project")
st.sidebar.info(
    """
    **Deepfake Audio Detection**

    This app helps identify whether an audio file is **Real** (human voice) 
    or **Deepfake** (AI-generated voice).  
    using a **stacked ensemble model** combining:
    - CNN, RNN, BiLSTM, GRU  
    - Logistic Regression (meta-classifier)

    """
)

# Define the path to model directory
MODEL_DIR = os.path.dirname(__file__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Custom CSS for styling
# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp{
        background-color: #FFF8E7;
        color: #000000;
    }
    .main-title{
        font-size: 36px;
        font-weight: bold;
        color: #C2185B;
        text-align: center;
        border-bottom: 4px solid #FF9800;
        padding-bottom: 5px;
        margin-bottom: 20px;
    }
    .sub-title{
        font-size: 20px;
        color: #E91E63;
        text-align: center;
        margin-bottom: 30px;
    }
    section[data-testid="stSidebar"]{
        backfroun-color: #E1BEE7;
    }
    .result-box{
        background-color: #FCE4EC;
        border: 2px solid #C2185B;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #C2185B;
        margin-top: 20px;
    }
    </style>
     """,
    unsafe_allow_html=True,
)
# Custom sidebar style
st.markdown(
    """
    <style>
    /* Sidebar gradient background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFD6E8, #FFEFF7); /* Light pink gradient */
        color: #2E2E2E;
    }

    /* Sidebar headings */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #8E44AD; /* Purple shade */
    }

    /* Info box inside sidebar */
    [data-testid="stSidebar"] .stAlert {
        background-color: #FFF0F5 !important;
        border-radius: 12px;
        border: 1px solid #F78FB3;
    }

    /* File uploader style */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #FFF7FA;
        border: 2px dashed #FF8AB4;
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App title
st.markdown('<div class="main-title">Deepfake Audio Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Using Stacking Model for Classification</div>', unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.markdown('<div class="sidebar-title">Upload Audio File</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Choose a WAV file", type=["wav","mp3"])

# Define the CNN model architecture
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
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the RNN model architecture
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the BiLSTM model architecture
def create_bilstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the GRU model architecture
def create_gru_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(128),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Load the pre-trained models
@st.cache_resource
def load_models():
    # Load CNN model
    cnn_model = create_cnn_model((120, 1))
    cnn_model.load_weights(os.path.join(MODEL_DIR, "cnn.h5"))

    # Load RNN model
    rnn_model = create_rnn_model((120, 1))
    rnn_model.load_weights(os.path.join(MODEL_DIR, "rnn.h5"))

    # Load BiLSTM model
    bilstm_model = create_bilstm_model((120, 1))
    bilstm_model.load_weights(os.path.join(MODEL_DIR, "bilstm.h5"))

    # Load GRU model
    gru_model = create_gru_model((120, 1))
    gru_model.load_weights(os.path.join(MODEL_DIR, "gru.h5"))

    # Load XGBoost model
    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(MODEL_DIR, "model_xgb.json"))

    # Load meta-model (Logistic Regression)
    meta_model = LogisticRegression()
    try:
        meta_model.coef_ = np.load(os.path.join(MODEL_DIR, "meta_model_coef.npy"))
        meta_model.intercept_ = np.load(os.path.join(MODEL_DIR, "meta_model_intercept.npy"))
        meta_model.classes_ = np.array([0, 1])
    except FileNotFoundError:
        st.error("Meta-model files are missing. Please save the meta-model after training.")

    return cnn_model, bilstm_model, rnn_model, gru_model, xgb_model, meta_model

cnn_model, bilstm_model, rnn_model, gru_model, xgb_model, meta_model = load_models()

# Define the feature extraction function
def extract_features(file_path, sr=16000, n_mfcc=40):
    """
    Extract features from the audio file.
    Features include MFCCs, delta MFCCs, and delta-delta MFCCs.
    """
    # Load the audio file
    audio_data, _ = librosa.load(file_path, sr=sr)
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    
    # Compute delta and delta-delta MFCCs
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Combine all features
    features = np.vstack((mfcc, delta_mfcc, delta2_mfcc))
    
    # Return the mean of the features across time
    return np.mean(features.T, axis=0)

# Define the function to get weighted predictions
def get_weighted_predictions(models, features_rnn, features_cnn, features_scaled, weights):
    predictions = []
    for model_name, model in models.items():
        if model_name in ['rnn', 'bilstm', 'gru']:
            pred = model.predict(features_rnn).flatten() * weights[model_name]
        elif model_name == 'cnn':
            pred = model.predict(features_cnn).flatten() * weights[model_name]
        elif model_name == 'xgb':
            pred = model.predict(xgb.DMatrix(features_scaled)) * weights[model_name]
        predictions.append(pred)
    return np.sum(predictions, axis=0)

# Main logic for the app
if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Progress bar
    progress_bar = st.progress(0)
    progress_bar.progress(10)

    # Load audio
    audio_data, sr = librosa.load(file_path, sr=16000)
    progress_bar.progress(30)
    # Audio player
    st.markdown("### 🎧 Play Uploaded Audio")
    st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")

    # Display waveform
    st.markdown("### Waveform of Uploaded Audio")
    fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio_data, sr=sr, ax=ax_wave, color="#FF9800")
    ax_wave.set_title("Waveform")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    st.pyplot(fig_wave)
    progress_bar.progress(50)

    # Display MFCCs
    st.markdown("### MFCCs of Uploaded Audio")
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    fig, ax = plt.subplots(figsize=(10, 4))
    mfcc_display = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax, cmap='magma')
    ax.set_title("MFCCs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC Coefficients")
    plt.colorbar(mfcc_display, format='%+2.0f dB', ax=ax)
    st.pyplot(fig)
    progress_bar.progress(70)

    # Feature extraction
    features = extract_features(file_path)
    features_rnn = features.reshape(1, -1, 1)
    features_cnn = features.reshape(1, -1, 1)
    features_scaled = features.reshape(1, -1)

    # Models and weights
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

    # Get predictions
    weighted_predictions = get_weighted_predictions(models, features_rnn, features_cnn, features_scaled, weights)
    progress_bar.progress(90)

    # Final prediction
    try:
        final_prediction = meta_model.predict(weighted_predictions.reshape(-1, 1))[0]
        probs = meta_model.predict_proba(weighted_predictions.reshape(-1, 1))[0]  # Get probabilities
        confidence = float(np.max(probs))  # Highest probability as confidence

        label = "Real" if final_prediction == 0 else "Fake"
        if label == "Real":
            bg_color, border_color, text_color = "#d4edda", "#28a745", "#155724"  # light green
        else:
            bg_color, border_color, text_color = "#f8d7da", "#dc3545", "#721c24"  # light red

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
            """, unsafe_allow_html=True    
        )
    except AttributeError:
        st.error("Meta-model is not properly loaded. Please ensure the meta-model files are available.")
    progress_bar.progress(100)

    #Accuracy chart
    models = ["CNN", "RNN", "BiLSTM", "GRU", "XGBoost", "Stacked Model"]
    accuracy = [95, 90, 92, 89, 67, 94]
    fig,  ax = plt.subplots(figsize=(8,5))
    ax.barh(models,accuracy,color='skyblue')
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model Accuracy")
    ax.invert_yaxis()
    for i,v in enumerate(accuracy):
        ax.text(v+1,i,f"{v}%",va='center')
    st.subheader("Model Accuracy")

    st.pyplot(fig)





