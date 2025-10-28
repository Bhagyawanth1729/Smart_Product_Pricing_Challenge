# 🧠 Smart Product Pricing using Supervised Autoencoder + Stacking Ensemble

Predicting optimal product prices is critical for e-commerce competitiveness. This project tackles the challenge of **accurate price prediction** using a multimodal approach, combining textual descriptions, visual features, and structured product attributes without relying on external pricing history.

---

## 📋 Problem Statement & Key Idea

E-commerce platforms require accurate, dynamic pricing models. The goal is to predict a product's price based *only* on its intrinsic attributes:
1.  **Textual Description:** Title, description, etc.
2.  **Image Representation:** Product photos.
3.  **Structured Attributes:** Metadata, quantity, etc.

### Key Idea
We unify these disparate data types into a robust predictive framework using a **Supervised Autoencoder** for effective image feature compression and a **Stacking Ensemble** for final price regression.

---

## 🏗️ Model Architecture & Approach Overview

This solution leverages the strengths of deep learning for feature extraction and traditional gradient-boosting models for superior regression performance.

### 1️⃣ Feature Engineering

| Feature Type | Description | Initial Technique | Output Dimension |
| :--- | :--- | :--- | :--- |
| **Structured** | Product metadata, item quantity. | `StandardScaler` normalization. | ~5 |
| **Text** | Combined Title, Description, Quantity. | Pre-trained text embeddings (e.g., Sentence-BERT). | 384 |
| **Image** | CNN embeddings extracted from product images (e.g., ResNet). | Compressed using **Supervised Autoencoder**. | **512** |
| **Final Feature Vector** | Concatenation of all modalities. | | **901-dim** |

### 2️⃣ Supervised Autoencoder (Image Feature Compression)

This component is crucial for creating a compact, highly informative image representation that is optimized for the *reconstruction* of the original image features **and** the final *price prediction*.

* **Architecture:**
    * **Encoder:** $2048 \rightarrow 1024 \rightarrow 512$
    * **Decoder:** $512 \rightarrow 1024 \rightarrow 2048$
    * **Regression Head:** $512 \rightarrow 1$ (for price prediction)
* **Loss Function:** A multi-task loss to balance reconstruction quality and predictive power:
    $$\text{Total Loss} = 0.5 \times \text{MSE}(\text{reconstruction}) + 0.5 \times \text{MSE}(\text{price prediction})$$
* **Training Results:** The final AE loss reduced significantly from $0.63 \rightarrow \mathbf{0.116}$, demonstrating excellent convergence and effective feature learning.

### 3️⃣ Stacking Ensemble Regressor

The concatenated 901-dim feature vector is fed into a two-layer stacking ensemble for the final prediction.

| Layer | Model Type | Specific Models | Purpose |
| :--- | :--- | :--- | :--- |
| **Base Models** | Heterogeneous Regressors | `LightGBM Regressor`, `XGBoost Regressor`, `MLP Regressor` | Capture different non-linearities in the feature space. |
| **Meta-Model** | Final Aggregator | `LightGBM Regressor` | Trained on **5-fold OOF (Out-Of-Fold) predictions** from the base models to learn how to optimally blend their outputs. |

---

## 📊 Performance Summary

The model was trained on **75,000 products** using a **Colab T4 GPU**.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **R² Score** | **0.9143** | Explains over **91%** of the variance in product price, indicating strong generalization. |
| **RMSE (Log-Space)** | **0.2758** | Low error in the log-transformed price space. |
| **SMAPE** | **21.74 %** | Competitive performance for a complex e-commerce price prediction task without external data. |

### 🧮 Evaluation Metric: SMAPE

The model is evaluated using the Symmetric Mean Absolute Percentage Error (SMAPE):

$$
\text{SMAPE} = \frac{1}{n} \sum \frac{|\hat{y} - y| (|\hat{y}| + |y|) / 2}{} \times 100
$$

A lower SMAPE indicates superior performance. Our $\mathbf{21.74\%}$ result is submission-ready.

---

## 📦 Files & Inference Pipeline

### Files Generated

* `stacked_model_supervised_autoencoder.pkl`: The complete serialized model (scalers, AE weights, base/meta models).
* `test_out.csv`: The final predicted prices for the challenge submission.

### Inference Pipeline

1.  Load model, scalers, and autoencoder weights.
2.  Pre-process/Scale structured and text features.
3.  Pass **raw image features** through the trained **Autoencoder's Encoder** to get the 512-dim embedding.
4.  Concatenate all features (901-dim).
5.  Generate base-model predictions.
6.  Feed base predictions to the Meta-Model to predict the final price.
7.  Save predictions to `test_out.csv`.

---

## 🚀 Conclusion

The **Supervised Autoencoder + Stacking Ensemble** methodology successfully addressed the challenge of multimodal product price prediction. The architecture demonstrated:
* Effective feature learning from high-dimensional image data.
* Strong generalization with $\mathbf{R^2 \approx 0.91}$.
* Competitive e-commerce prediction accuracy ($\mathbf{SMAPE \approx 21.7\%}$).

This performance is robust and ready for the ML Challenge 2025 leaderboard.

---
## 👨‍💻 Team & Contact

| Name | Role | Email |
| :--- | :--- | :--- |
| **Bhagyawanth** | Lead Data Scientist / Model Architect | [bhagyawanthningappa@gmail.com |
| **S Dhiraj** | Feature Engineering Specialist | singurudhiraj@gmail.com|
| **Jonan Puro** | Data Preprocessing & Pipeline Engineer | jotharrison@gmail.com |

## 👨‍💻 Author

**Bhagyawanth**
* B.Tech Computer Science | Data Analyst & ML Enthusiast
* **Tools:** Python, PyTorch, scikit-learn, LightGBM, XGBoost
* **Email:** [Bhagyawanthningappa@gmail.com]
