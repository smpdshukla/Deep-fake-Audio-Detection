# Deep-fake-Audio-Detection
Deep fake Audio Detection is  AI/ML based project that uses classification models to detect whether audio is real or a machine generated.
The pipeline includes:
1. **Audio Preprocessing:** Converts raw audio into machine-readable features. It extracts Mel-Frequency Cepstral Coefficients (MFCCs), spectral contrast, and chroma features, which help the model capture nuances in human speech and synthetic voice generation patterns.  
2. **Feature Normalization:** Applies a pre-trained scaler to normalize features across different audio samples, ensuring consistency and reducing model bias caused by amplitude variations or background noise.  
3. **Model Prediction:** Multiple deep learning models are used in parallel:  
  - **CNN** learns spatial and spectral audio representations.  
  - **RNN** models sequential data patterns.  
  - **BiLSTM** captures both past and future audio context for more robust learning.  
  - **GRU** provides a lightweight yet effective sequence model.  
  4. **Ensemble Learning:** An **XGBoost meta-classifier** aggregates the predictions from all deep learning models, reducing overfitting and boosting classification accuracy.  
  5. **Web App Integration:** A user-friendly Streamlit interface allows non-technical users to upload audio files and instantly receive predictions. This makes the solution accessible for real-world use cases like media verification, forensic investigations, and online identity protection.  

## 🚀 Features
1.**Deepfake vs. Real Detection:** Identifies whether an uploaded audio clip is genuine human speech or synthetically generated using deepfake technology.  
2. **Advanced Ensemble Learning:** Combines strengths of CNN, RNN, BiLSTM, GRU, and XGBoost to deliver high-accuracy predictions.  
3. **Cross-Platform Deployment:** Works seamlessly as a Flask web application or a Streamlit interface, depending on the use case.  
4.**User-Friendly Interface:** Non-technical users can easily upload audio files without requiring command-line interaction.  
5. **Scalable and Extendable:** The architecture is modular, allowing easy integration with new models, datasets, or cloud deployment platforms.  

