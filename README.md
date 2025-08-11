## 🎙 Deep-fake-Audio-Detection
Deep fake Audio Detection is  AI/ML based project that uses classification models to detect whether audio is real or a machine generated.
The pipeline includes:
1. **Audio Preprocessing:** Converts raw audio into machine-readable features. It extracts Mel-Frequency Cepstral Coefficients (MFCCs), which help the model capture nuances in human speech.
2. **Feature Normalization:** Applies a pre-trained scaler to normalize features across different audio samples, ensuring consistency and reducing model bias caused by amplitude variations or background noise.  
3. **Model Prediction:** Multiple deep learning models are used in parallel,**CNN**,**RNN**,**BiLSTM**,**GRU**,**XGBoost** for the prediction.
  4. **Ensemble Learning:** Outputs from all base models are combined using a **Logistic Regression meta-classifier**, which assigns optimal weights,and **XGBoost**, which captures complex non-linear relationships.
  5. **Web App Integration:** A user-friendly Streamlit interface allows non-technical users to upload audio files and instantly receive predictions. 
  6. **Performance:** Deep learning models like **CNN** and **BiLSTM** excel individually (95% and 92% accuracy respectively), but combining them through **stacking** significantly improves robustness. Our final ensemble model achieved **~94% overall accuracy**, outperforming individual models while remaining efficient enough for real-world deployment.

## 🚀 Features
1.**Deepfake vs. Real Detection:** Identifies whether an uploaded audio clip is genuine human speech or synthetically generated using deepfake technology.  
2. **Advanced Ensemble Learning:** Combines strengths of CNN, RNN, BiLSTM, GRU, and XGBoost to deliver high-accuracy predictions.  
3. **Cross-Platform Deployment:** Works seamlessly as a Streamlit interface, depending on the use case.  
4.**User-Friendly Interface:** Non-technical users can easily upload audio files without requiring command-line interaction.  
5. **Scalable and Extendable:** The architecture is modular, allowing easy integration with new models, datasets, or cloud deployment platforms.  

## 📦 Dataset
1. **Source:** for-2seconds dataset taken from FoR dataset.(Publicly available speech datasets & synthetic deepfake audio.)
  
2. **Temporary Testing Dataset:**
   
3. **Follow Path:** **temp/for-2seconds/testing/**
    
4. Contains short 2-second audio clips for quick evaluation.
  
5. real/ → Human voice samples.
  
6. fake/ → AI-generated voice samples.
 
7. **Supported Formats:** .wav (default), .mp3 and .flac (when enabled in uploader).

## 📊 Models Used

| Model               | Type             | Role in Stack                |
|---------------------|------------------|------------------------------|
| CNN                 | Deep Learning    | Base model                   |
| RNN                 | Deep Learning    | Base model                   |
| BiLSTM              | Deep Learning    | Base model                   |
| GRU                 | Deep Learning    | Base model                   |
| XGBoost             | ML Algorithm     | Base model                   |
| Logistic Regression | ML Algorithm     | Meta-classifier              |

✅ Meta-classifier combines all base models’ predictions for final classification.


## 🛠 Tech Stack & Tools
- **Languages:** Python 3.9+  
- **Deep Learning:** TensorFlow, Keras  
- **Machine Learning:** XGBoost, Scikit-learn, Logistic Regression (meta-classifier)  
- **Audio Processing:** Librosa, SoundFile  
- **Data Handling:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **UI & Deployment:** Streamlit 
- **Environment:** Jupyter Notebook, Anaconda

## ▶️ Usage
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. 2. Download the dataset consisting of **Fake** and **Real** audio files for testing:  
   [📂 Fake-or-Real (FoR) Dataset on Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4.Upload an audio file via the interface.

5.View prediction — whether the audio is Real or Deepfake.


## 📉 Visual Outputs in App
- 🎵 Waveform of uploaded audio
- 🎼 MFCC feature plot
- 📊 Stacked model performance chart
- 🗂 Final result in a highlight box – "Real" or "Fake" with confidence %


## 📈 Future Improvements
1.Support for multilingual audio.

2.Real-time streaming detection.

3.Cloud deployment for scalability.

4.Integration of Explainable AI (XAI) for transparency.

