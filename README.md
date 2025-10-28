# Solution for ML Challenge 2025: Smart Product Pricing

[cite_start]A project by **Ctrl + Alt + Learn** [cite: 2]

---

### Executive Summary

[cite_start]This repository contains the solution for the "Smart Product Pricing" challenge[cite: 1]. [cite_start]We developed a **multimodal pricing predictor** designed to accurately estimate product prices by learning from a combination of product metadata, textual descriptions, and image features[cite: 5].

[cite_start]The core of our approach is a **Supervised Autoencoder + Stacking Ensemble**[cite: 5]. [cite_start]This pipeline intelligently extracts deep representations from images, aligns them with text and structured data, and fuses them through a robust ensemble learning model to achieve high-precision price estimation[cite: 6].

---

## 🚀 Methodology

[cite_start]Our end-to-end pipeline converts raw, multimodal data into a final price prediction through five key stages[cite: 9].

### 1. Data Preprocessing
* [cite_start]**Metadata:** Cleaned and standardized[cite: 10].
* [cite_start]**Text:** Converted into high-dimensional embeddings using **Sentence-BERT**[cite: 10].
* [cite_start]**Images:** Processed through a pre-trained CNN to extract **2048-D feature vectors**[cite: 10].

### 2. Supervised Autoencoder
[cite_start]To make the raw image features more meaningful, we trained a specialized autoencoder (2048 $\rightarrow$ 1024 $\rightarrow$ 512 $\rightarrow$ 1024 $\rightarrow$ 2048)[cite: 18, 19, 20]. [cite_start]This module is *supervised*—it's trained with a **joint loss function** that forces the 512-D bottleneck representation to learn two things simultaneously[cite: 22]:
1.  [cite_start]**Visual Reconstruction ($L_{recon}$):** How to reconstruct the original image feature[cite: 20].
2.  [cite_start]**Price Alignment ($L_{reg}$):** How to predict the (log) price from the bottleneck vector[cite: 21].

[cite_start]This ensures the final 512-D embedding captures both visual content and pricing cues[cite: 23].

### 3. Feature Fusion
[cite_start]The outputs from the preprocessing and autoencoder stages are concatenated into a single, unified feature matrix[cite: 12]:

[cite_start]$X_{full} = [ X_{structured} \mid X_{text} \mid X_{image(bottleneck)} ]$ [cite: 26]

[cite_start]This matrix is then standardized to ensure all modalities have a balanced influence[cite: 27].

### 4. Base Model Training (Level 0)
[cite_start]Three diverse regressors are trained in parallel on the complete, fused dataset using 5-fold cross-validation[cite: 13]:
* [cite_start]**LightGBM** (lr=0.05, 31 leaves) [cite: 30]
* **XGBoost** (max_depth=5, lr=0.05) [cite: 31]
* [cite_start]**MLP** (2 hidden layers: 128-64, ReLU) [cite: 32]

### 5. Stacking Ensemble (Level 1)
The out-of-fold (OOF) predictions from the three base models are saved and used as new "meta-features." [cite_start]A final **LightGBM meta-model** (300 estimators, lr=0.03) is trained on these meta-features[cite: 14, 33]. [cite_start]This two-level stacking architecture learns the optimal way to combine the predictions, correcting for individual model errors and boosting overall robustness[cite: 34].

---

## 📈 Model Performance

[cite_start]Our stacked ensemble significantly outperformed the individual base models, demonstrating the effectiveness of the feature fusion and diverse model aggregation[cite: 38].

### [cite_start]Final Validation Metrics [cite: 36]
| Metric | Score |
| :--- | :--- |
| **RMSE** | **0.2758** |
| **R²** | **0.9143** |
| **SMAPE** | **21.74%** |

### [cite_start]Base Model RMSE (for comparison) [cite: 36]
* **LightGBM:** 0.3165
* **XGBoost:** 0.3361
* **MLP:** 0.3499

---

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/smart-product-pricing.git](https://github.com/your-username/smart-product-pricing.git)
    cd smart-product-pricing
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ How to Run

### 1. Data
Place your raw data (e.g., `train.csv`, `test.csv`, and the `images/` directory) into the `/data/` folder.

### 2. Training
To run the full end-to-end pipeline (preprocessing, autoencoder training, and ensemble training), execute:
```bash
python train.py
