# Deep-fake-Audio-Detection
Deep fake Audio Detection is  AI/ML based project that uses classification models to detect whether audio is real or a machine generated.
The pipeline includes:
1. **Audio Preprocessing:** Converts raw audio into machine-readable features. It extracts Mel-Frequency Cepstral Coefficients (MFCCs), spectral contrast, and chroma features, which help the model capture nuances in human speech and synthetic voice generation patterns.  
2. **Feature Normalization:** Applies a pre-trained scaler to normalize features across different audio samples, ensuring consistency and reducing model bias caused by amplitude variations or background noise.  
3. **Model Prediction:** Multiple deep learning models are used in parallel,**CNN**,**RNN**,**BiLSTM**,**GRU**,**XGBoost** for the prediction.
  4. **Ensemble Learning:** Outputs from all base models are combined using a **Linear Regression meta-classifier**, which assigns optimal weights,and **XGBoost**, which captures complex non-linear relationships.
  5. **Web App Integration:** A user-friendly Streamlit interface allows non-technical users to upload audio files and instantly receive predictions. This makes the solution accessible for real-world use cases like media verification, forensic investigations, and online identity protection.
  6. **Performance:** Deep learning models like **CNN** and **BiLSTM** excel individually (95% and 92% accuracy respectively), but combining them through **stacking** significantly improves robustness and generalizability. Our final ensemble model achieved **~94% overall accuracy**, outperforming individual models while remaining efficient enough for real-world deployment.


## 🚀 Features
1.**Deepfake vs. Real Detection:** Identifies whether an uploaded audio clip is genuine human speech or synthetically generated using deepfake technology.  
2. **Advanced Ensemble Learning:** Combines strengths of CNN, RNN, BiLSTM, GRU, and XGBoost to deliver high-accuracy predictions.  
3. **Cross-Platform Deployment:** Works seamlessly as a Flask web application or a Streamlit interface, depending on the use case.  
4.**User-Friendly Interface:** Non-technical users can easily upload audio files without requiring command-line interaction.  
5. **Scalable and Extendable:** The architecture is modular, allowing easy integration with new models, datasets, or cloud deployment platforms.  

## ▶️ Usage
Run the Flask/Streamlit app.
python app.py or streamlit run app.py.
Upload an audio file via the interface.
View prediction — whether the audio is Real or Deepfake.

## 📊 Models Used
**1.CNN**: Captures spectral features from audio signals.
**2.RNN & GRU**: Models sequential dependencies in audio.
**3.BiLSTM**: Learns temporal context in both directions.
**4.XGBoost:** Meta-classifier combining deep learning model outputs.

## 📈 Future Improvements
1.Support for multilingual audio.
2.Real-time streaming detection.
3.Cloud deployment for scalability.
4.Integration of Explainable AI (XAI) for transparency.

