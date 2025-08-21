# EVALUATING DEEP LEARNING MODELS FOR EARTHQUAKE MAGNITUDE PREDICTION

---

## 📌 Project Title: Evaluating Deep Learning Models for Earthquake Magnitude Prediction

**GitHub Repository**: [EVALUATING-DEEP-LEARNING-MODELS-FOR-EARTHQUAKE-MAGNITUDE-PREDICTION](https://github.com/3m0r9/EVALUATING-DEEP-LEARNING-MODELS-FOR-EARTHQUAKE-MAGNITUDE-PREDICTION)

---

### 🧠 Overview

This research project investigates the effectiveness of deep learning models for predicting earthquake magnitudes using seismic datasets from Indonesia (BMKG) and Japan (JMA). The goal is to develop a predictive system that can assist in early warning mechanisms and reduce disaster impacts through timely forecasting.

---

### 🗂 Dataset

* **BMKG (Indonesia)**: Real earthquake event logs
* **JMA (Japan)**: Clean, structured seismic data
* **Features Used**:

  * Latitude, longitude
  * Depth, date-time
  * Magnitude (target)
  * Seismic wave features

---

### 🏗️ Models Implemented

| Model Type                 | Architecture Details                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------ |
| **CNN-GRU**                | Combines convolutional feature extraction with GRU for temporal dependencies         |
| **CNN-BiLSTM + Attention** | Adds bidirectional LSTM with an attention mechanism for better context understanding |

---

### ⚙️ Methodology

1. **Data Preprocessing**

   * Normalization with MinMaxScaler
   * Time-sequence formatting
   * Window-based sliding for temporal structure

2. **Training & Validation**

   * Separate experiments on JMA and BMKG datasets
   * 70:30 train-test split
   * Epochs: 100+
   * Optimizer: Adam
   * Loss Function: MSE (Mean Squared Error)

3. **Evaluation Metrics**

   * Mean Absolute Error (MAE)
   * Root Mean Squared Error (RMSE)
   * R² Score
   * Visualization: Line plot comparison of actual vs predicted magnitudes

---

### 📈 Results Summary

| Metric         | CNN-GRU                                                 | CNN-BiLSTM + Attention |
| -------------- | ------------------------------------------------------- | ---------------------- |
| MAE (JMA)      | 0.075                                                   | 0.067                  |
| RMSE (JMA)     | 0.107                                                   | 0.099                  |
| R² Score (JMA) | 0.91                                                    | **0.93**               |
| Best Accuracy  | CNN-BiLSTM + Attention outperformed across all datasets |                        |

The **CNN-BiLSTM with Attention** showed superior generalization and lower error rates, making it suitable for real-time earthquake prediction systems.

---

### 🧩 Visual Insights

* Comparative line charts of actual vs. predicted values show tight alignment, especially in the CNN-BiLSTM + Attention model.
* Confusion matrices and scatter plots further supported accuracy in extreme magnitude values.

---

### 🎯 Conclusion

Deep learning, especially attention-based architectures, holds promise for **early detection and forecasting of natural disasters**. The research proves that a carefully designed neural network can effectively approximate complex geophysical phenomena such as earthquake magnitude prediction.

---

### 🔍 Future Work

* Extend to multivariate forecasting using oceanographic data
* Deploy real-time alert systems using streaming data (Kafka, Spark)
* Improve robustness through hybrid ensembles

