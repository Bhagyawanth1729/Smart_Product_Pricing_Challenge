🧠 Project Title:

Smart Product Pricing using Supervised Autoencoder + Stacking Ensemble

📋 Problem Statement

E-commerce platforms rely heavily on accurate price prediction for optimal competitiveness and profitability. The goal of this challenge is to predict the price of products using their textual descriptions, image representations, and structured attributes — without relying on any external pricing data.

🧰 Approach Overview

This solution integrates deep learning feature extraction and traditional machine learning regression using a stacking ensemble.

Key Idea:

Combine multiple data modalities — text, image, and structured product attributes — into a unified predictive model via:

Supervised Autoencoder (for learning compact image embeddings)

Stacked Ensemble of LightGBM, XGBoost, and MLP models

LightGBM Meta-Model trained on out-of-fold predictions

🏗️ Model Architecture
1️⃣ Feature Engineering
Feature Type	Description	Technique
Structured Features	Product metadata, item quantity, etc.	StandardScaler normalization
Text Features	Combined title, description, and item pack quantity	Pre-trained text embeddings (384-dim) + StandardScaler
Image Features	CNN embeddings extracted from product images	Compressed using a Supervised Autoencoder to 512-dim

All three feature types are concatenated → 901-dim final feature vector per product.

2️⃣ Supervised Autoencoder (Image Feature Compression)

Encoder: 2048 → 1024 → 512

Decoder: 512 → 1024 → 2048

Regression Head: 512 → 1

Loss Function:

Total Loss = 0.5 * MSE(reconstruction) + 0.5 * MSE(price prediction)


Training: 50 epochs, batch size 256, Adam optimizer (lr = 1e-3)

✅ Final AE loss reduced from 0.63 → 0.116 (excellent convergence)

3️⃣ Stacking Ensemble (Supervised Learning Stage)

Base Models:

LightGBM Regressor

XGBoost Regressor

MLP Regressor

Meta-Model:

LightGBM Regressor trained on 5-fold OOF (out-of-fold) predictions.

⚙️ Training Setup
Component	Setting
Frameworks	PyTorch, scikit-learn, LightGBM, XGBoost
Hardware	Colab GPU (T4)
Data Size	75,000 products
Loss Metrics	RMSE, R², SMAPE
Random Seed	42
📊 Performance Summary (Training Results)
Metric	Value	Interpretation
RMSE (Meta-Model)	0.2758	Low log-space error
R² Score	0.9143	Explains 91% of price variance
SMAPE	21.74 %	Strong performance for e-commerce price prediction
📦 Files Generated
File	Description
stacked_model_supervised_autoencoder.pkl	Saved full model (scalers, autoencoder weights, base models, meta model)
test_out.csv	Final predicted prices for submission
test_pca_transformer.pkl (optional)	PCA reducer (if used for dimensionality reduction)
🧩 Testing / Inference Pipeline

Load saved model and scalers

Scale structured, text, and image features from test data

Pass image features through trained autoencoder encoder

Concatenate all features

Generate base-model predictions → feed to meta-model

Predict final prices → save as test_out.csv

🧮 Evaluation Metric

The model is evaluated using SMAPE (Symmetric Mean Absolute Percentage Error):

𝑆
𝑀
𝐴
𝑃
𝐸
=
1
𝑛
∑
∣
𝑦
𝑝
𝑟
𝑒
𝑑
−
𝑦
𝑡
𝑟
𝑢
𝑒
∣
(
∣
𝑦
𝑝
𝑟
𝑒
𝑑
∣
+
∣
𝑦
𝑡
𝑟
𝑢
𝑒
∣
)
/
2
×
100
SMAPE=
n
1
	​

∑
(∣y
pred
	​

∣+∣y
true
	​

∣)/2
∣y
pred
	​

−y
true
	​

∣
	​

×100

Lower SMAPE indicates better performance.

🚀 Conclusion

✅ The Supervised Autoencoder + Stacking Ensemble approach achieved:

Strong generalization (R² ≈ 0.91)

Competitive SMAPE (≈ 21.7%)

Excellent balance of interpretability and predictive power.

This performance is acceptable and submission-ready for the ML Challenge 2025 leaderboard.

👨‍💻 Author

Bhagyawanth
B.Tech Computer Science | Data Analyst & ML Enthusiast
📧 Email: [your email here]
🧠 Tools: Python, PyTorch, scikit-learn, LightGBM, XGBoost
