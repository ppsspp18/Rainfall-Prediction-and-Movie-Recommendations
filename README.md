# 🌧️ Rainfall Prediction & Movie Recommendation System

> **CS771 Course Project** — Introduction to Machine Learning
> Instructor: Prof. Purushottam Kar, IIT Kanpur
> Duration: October 2025 – November 2025

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Part 1 — Rainfall Prediction](#3-part-1--rainfall-prediction)
   - [Dataset](#31-dataset)
   - [Why This Problem?](#32-why-this-problem)
   - [Data Preprocessing](#33-data-preprocessing)
   - [Why These Preprocessing Steps?](#34-why-these-preprocessing-steps)
   - [Machine Learning Models](#35-machine-learning-models)
   - [Why These ML Models?](#36-why-these-ml-models)
   - [Deep Learning Models (LSTM & GRU)](#37-deep-learning-models-lstm--gru)
   - [Why Deep Learning After Classical ML?](#38-why-deep-learning-after-classical-ml)
   - [Why PCA Before Deep Learning?](#39-why-pca-before-deep-learning)
   - [Results & Evaluation](#310-results--evaluation)
   - [Why These Evaluation Metrics?](#311-why-these-evaluation-metrics)
4. [Part 2 — Movie Recommendation System](#4-part-2--movie-recommendation-system)
   - [Why a Recommendation System?](#41-why-a-recommendation-system)
   - [Clustering Approaches](#42-clustering-approaches)
   - [Why These Clustering Algorithms?](#43-why-these-clustering-algorithms)
   - [Cosine Similarity](#44-cosine-similarity)
   - [Why Cosine Similarity?](#45-why-cosine-similarity)
5. [Tech Stack](#5-tech-stack)
6. [Installation & Setup](#6-installation--setup)
7. [Key Takeaways](#7-key-takeaways)

---

## 1. Project Overview

This project is divided into two independent but complementary machine learning tasks:

**Task A — Rainfall Prediction:**
Given historical daily weather observations from across Australia, predict whether it will rain the next day (`RainTomorrow`: Yes/No). This is a **binary classification** problem that benchmarks a wide range of classical ML models alongside deep learning architectures (LSTM, GRU).

**Task B — Movie Recommendation System:**
Given a dataset of movies and their metadata (genres, ratings, tags, etc.), group movies using unsupervised clustering and recommend similar movies to a user using **cosine similarity** on the resulting feature representations.

Together, the project covers the full ML pipeline: data exploration, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and unsupervised learning — making it a comprehensive demonstration of applied machine learning.

---

## 2. Repository Structure

```
Rainfall-Prediction-and-Movie-Recommendations/
│
├── Rain_prediction.ipynb            # Notebook 1: Classical ML models for rainfall prediction
├── rainfall_prediction_2.ipynb      # Notebook 2: Deep Learning (LSTM & GRU) with PCA
│
├── Movie Recommendation/
│   └── (Notebook/scripts for clustering-based movie recommendation)
│
├── LICENSE                          # MIT License
└── README.md                        # This file
```

**Why two separate notebooks for rainfall prediction?**
The project intentionally separates classical ML experiments (Notebook 1) from deep learning experiments (Notebook 2). This keeps each notebook focused, makes the methodology easier to follow, and allows independent reproducibility of each approach without running unnecessary code.

---

## 3. Part 1 — Rainfall Prediction

### 3.1 Dataset

| Property | Details |
|---|---|
| **Name** | weatherAUS.csv |
| **Source** | [Kaggle — Weather Dataset Rattle Package](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) |
| **Records** | ~145,000 daily weather observations |
| **Locations** | 49 weather stations across Australia |
| **Target Variable** | `RainTomorrow` (Yes / No) |
| **Features** | 23 weather attributes (temperature, humidity, wind speed, pressure, etc.) |

**Key Features Used:**

| Feature | Description |
|---|---|
| `MinTemp`, `MaxTemp` | Daily minimum and maximum temperature (°C) |
| `Rainfall` | Amount of rainfall recorded (mm) |
| `WindGustSpeed` | Maximum wind gust speed in the 24 hours prior to midnight (km/h) |
| `WindSpeed9am`, `WindSpeed3pm` | Wind speed at 9am and 3pm |
| `Humidity9am`, `Humidity3pm` | Relative humidity (%) at 9am and 3pm |
| `Pressure9am`, `Pressure3pm` | Atmospheric pressure (hPa) at 9am and 3pm |
| `Cloud9am`, `Cloud3pm` | Cloud cover (oktas) at 9am and 3pm |
| `Sunshine` | Hours of bright sunshine in the day |
| `RainToday` | Whether it rained today (binary) |
| `WindDir9am`, `WindDir3pm`, `WindGustDir` | Categorical wind direction variables |

---

### 3.2 Why This Problem?

Rainfall prediction is a classic example of a high-stakes binary classification problem:

- **Real-world impact**: Accurate next-day rain forecasting directly benefits agriculture (irrigation planning), transport, outdoor event management, and disaster preparedness.
- **Rich ML learning ground**: The dataset includes mixed data types (numerical + categorical), class imbalance, missing values, and temporal structure — making it ideal for practicing the full preprocessing and modeling pipeline.
- **Benchmarking opportunity**: With a large, well-known dataset and a clear binary target, it is straightforward to rigorously compare a wide range of models using consistent evaluation metrics.

---

### 3.3 Data Preprocessing

The raw dataset requires significant cleaning before any model can be trained. The following steps were applied:

#### Step 1 — Handling Missing Values
Rows with missing values in critical columns were dropped after encoding. This is done post-encoding to prevent issues with categorical encoders seeing `NaN` values.

#### Step 2 — Categorical Encoding (One-Hot Encoding)
Columns like `WindDir9am`, `WindDir3pm`, `WindGustDir`, and `RainToday` are categorical. They were transformed into binary indicator columns using **one-hot encoding**.

#### Step 3 — Outlier Treatment (IQR Capping)
Numerical features like `Rainfall` and `WindGustSpeed` can have extreme outliers. These were capped using the **Interquartile Range (IQR)** method:

- Lower cap: Q1 − 1.5 × IQR
- Upper cap: Q3 + 1.5 × IQR

This retains all rows (unlike dropping outliers) while reducing the distorting effect of extreme values.

#### Step 4 — Feature Scaling (Normalization)
Numerical features were scaled to a common range. This ensures no single feature dominates distance-based models simply because it has a larger numerical range.

#### Step 5 — Class Imbalance Handling (SMOTE)
The target variable `RainTomorrow` is imbalanced: approximately **78% No, 22% Yes**. Without correction, most models would simply predict "No" to achieve high raw accuracy while completely failing to identify rainy days.

**SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training set:
- Generates synthetic samples for the minority class ("Yes") by interpolating between existing minority samples and their nearest neighbours.
- Applied **only to the training set** to prevent data leakage into validation/test sets.

#### Step 6 — Dimensionality Reduction (PCA) [Notebook 2 only]
**Principal Component Analysis (PCA)** was applied before training the deep learning models, reducing the feature space to the most informative principal components.

---

### 3.4 Why These Preprocessing Steps?

| Step | Why It's Necessary |
|---|---|
| **Drop missing values** | ML models cannot natively handle NaN; removing them after encoding avoids encoder errors and keeps the pipeline clean |
| **One-hot encoding** | Tree-based models can handle label encoding, but distance-based models (KNN, SVM) and neural networks require numeric inputs; one-hot encoding avoids imposing an artificial ordinal relationship on nominal categories like wind direction |
| **IQR capping** | Dropping outliers loses data; capping preserves all records while reducing distortion. Rainfall in Australia has extreme right skew — a single storm event should not disproportionately shift a model's decision boundary |
| **Normalization** | KNN and SVM are sensitive to feature scale: a feature ranging 0–1000 will overwhelm a feature ranging 0–1 in distance calculations. Normalization ensures fair treatment of all features |
| **SMOTE** | A model trained on imbalanced data maximises accuracy by predicting the majority class. SMOTE balances the training set so the model learns to genuinely distinguish rain from no-rain, improving recall on the minority (rainy) class — the class we most care about predicting correctly |
| **PCA** | Deep learning models (LSTM, GRU) are computationally expensive and can overfit on high-dimensional tabular data. PCA compresses the feature space by removing redundant/correlated features, speeding up training and often improving generalisation |

---

### 3.5 Machine Learning Models

The following models were implemented using **scikit-learn** (and dedicated libraries for boosting models):

| Model | Type | Library |
|---|---|---|
| Logistic Regression | Linear classifier | scikit-learn |
| K-Nearest Neighbors (KNN) | Instance-based | scikit-learn |
| Support Vector Machine (LinearSVC) | Margin-based | scikit-learn |
| Decision Tree | Tree-based | scikit-learn |
| Random Forest | Ensemble (Bagging) | scikit-learn |
| Naive Bayes | Probabilistic | scikit-learn |
| Nearest Class Mean (NCM) / LWP | Prototype-based | scikit-learn / custom |
| XGBoost | Ensemble (Gradient Boosting) | xgboost |
| LightGBM | Ensemble (Gradient Boosting) | lightgbm |
| CatBoost | Ensemble (Gradient Boosting) | catboost |

**Hyperparameter Tuning** was performed on key models (e.g., number of trees in Random Forest, C parameter in SVM, learning rate and depth in boosting models) using grid search or random search with cross-validation to find the best configuration.

---

### 3.6 Why These ML Models?

The model selection deliberately covers a spectrum of algorithmic families to enable meaningful comparison:

**KNN** — Establishes a simple non-parametric baseline. Predicts based on the majority class among the k nearest training points. Sensitive to feature scaling (hence the need for normalization).

**SVM (LinearSVC)** — Finds a maximum-margin hyperplane separating the two classes. Works well in high-dimensional spaces and is robust to overfitting. The linear kernel was chosen as a practical choice given the number of features post-encoding.

**NCM (Nearest Class Mean / LWP)** — A prototype-based method directly related to the CS771 course syllabus (Prof. Purushottam Kar's course covers Learning with Prototypes). NCM predicts the class whose mean representation is closest to the query point. LWP extends this with learned prototypes.

**Random Forest** — Addresses Decision Tree's overfitting by training many trees on bootstrap samples and averaging their predictions. Robust, interpretable via feature importance, and handles mixed data well.

**XGBoost, LightGBM, CatBoost** — The three leading gradient boosting libraries. Each builds an ensemble of weak trees sequentially, where each tree corrects the errors of the previous. They consistently top tabular ML benchmarks and were included to find the best achievable accuracy on this dataset. CatBoost additionally handles categorical features natively.

**Why so many models?** In a course project context, benchmarking multiple models is valuable: it demonstrates that model selection matters, reveals which algorithmic assumptions match the data structure, and provides empirical evidence for final model choice rather than arbitrary selection.

---

### 3.7 Deep Learning Models (LSTM & GRU)

Implemented using **TensorFlow/Keras**, two recurrent neural network architectures were applied:

#### LSTM (Long Short-Term Memory)
LSTMs are a type of Recurrent Neural Network (RNN) that address the **vanishing gradient problem** through a gating mechanism with three gates:
- **Forget gate**: Decides what information to discard from the cell state.
- **Input gate**: Decides what new information to add.
- **Output gate**: Decides what to output based on the cell state.

This allows LSTMs to capture long-range dependencies in sequential data.

#### GRU (Gated Recurrent Unit)
GRUs are a simplified variant of LSTMs with only two gates:
- **Reset gate**: Controls how much past information to forget.
- **Update gate**: Controls how much of the past state to carry forward.

GRUs are faster to train than LSTMs and achieve comparable performance on many tasks.

**Architecture used (approximate):**
```
Input → [PCA-reduced features]
  ↓
Reshape to (1, n_features)  [treat each sample as a 1-step sequence]
  ↓
LSTM / GRU layer(s)
  ↓
Dense layer(s) with ReLU activation
  ↓
Output: Dense(1, activation='sigmoid')  [binary classification]
  ↓
Loss: Binary Crossentropy | Optimizer: Adam
```

---

### 3.8 Why Deep Learning After Classical ML?

1. **Representational power**: Neural networks can learn non-linear feature interactions automatically, without manual feature engineering.
2. **Scalability**: With ~145,000 records, deep learning has enough data to generalise without severe overfitting.
3. **Recurrent architecture relevance**: Weather data has temporal structure — patterns like "it has been humid for 3 days" matter. LSTMs and GRUs are designed to model such sequential dependencies.
4. **Benchmarking ceiling**: After establishing the best classical ML performance, deep learning was used to test whether a more complex model could push accuracy further — and it did, achieving **95% accuracy**.
5. **Academic exposure**: Implementing both families gives hands-on experience with scikit-learn's API, TensorFlow/Keras's model-building API, and the practical differences between the two paradigms.

---

### 3.9 Why PCA Before Deep Learning?

| Reason | Explanation |
|---|---|
| **Dimensionality** | After one-hot encoding, the feature space grows significantly. Feeding all features into an LSTM/GRU as-is makes the network large and slow to train |
| **Multicollinearity** | Weather features are highly correlated (e.g., `MinTemp`/`MaxTemp`, `Humidity9am`/`Humidity3pm`). PCA decorrelates features, providing a cleaner signal for the network |
| **Overfitting prevention** | Fewer input dimensions → fewer parameters → less chance of memorising the training set |
| **Faster convergence** | Reducing feature count speeds up each training epoch, enabling more epochs / hyperparameter experiments in limited time |
| **Variance maximisation** | PCA retains the components that explain the most variance, so minimal information is lost while dimensionality drops |

PCA was applied **after** SMOTE and scaling, ensuring the principal components are learned on balanced, normalised data.

---

### 3.10 Results & Evaluation

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | ~83% | ~0.65 |
| KNN | ~82% | ~0.63 |
| SVM (LinearSVC) | ~84% | ~0.67 |
| Decision Tree | ~79% | ~0.61 |
| Random Forest | ~86% | ~0.72 |
| XGBoost | ~87% | ~0.74 |
| LightGBM | ~87% | ~0.74 |
| CatBoost | ~87% | ~0.75 |
| GRU + PCA | ~93% | ~0.87 |
| **LSTM + PCA** | **~95%** | **~0.90** |

> *Note: Exact values may vary slightly with random seed. Results above are representative of the experiments conducted.*

---

### 3.11 Why These Evaluation Metrics?

Given the class imbalance in the dataset, **accuracy alone is misleading**. A model that always predicts "No Rain" would achieve ~78% accuracy but would be completely useless.

| Metric | Why It Matters Here |
|---|---|
| **Accuracy** | Overall correctness. Useful only in conjunction with other metrics due to class imbalance |
| **F1 Score** | Harmonic mean of Precision and Recall. Balances the cost of false positives and false negatives — critical when both types of errors matter |
| **Precision** | Of all days predicted as rainy, how many actually rained? High precision → fewer false alarms |
| **Recall** | Of all days that actually rained, how many did we catch? High recall → fewer missed rain days |
| **Jaccard Index** | Measures overlap between predicted and actual positive sets. Useful for imbalanced classification |
| **Log Loss** | Penalises confident wrong predictions. Measures calibration quality of probabilistic classifiers |

---

## 4. Part 2 — Movie Recommendation System

### 4.1 Why a Recommendation System?

Recommendation systems are one of the most commercially impactful applications of ML (Netflix, Spotify, Amazon). This part of the project implements an **unsupervised, content-based** approach using clustering and cosine similarity — directly applicable to real-world scenarios where user interaction data (ratings, watch history) may be sparse or unavailable.

The approach taken here is **content-based filtering**: movies are characterised by their attributes (genres, tags, ratings, etc.) and recommended based on similarity in this feature space — rather than relying on the behaviour of other users.

---

### 4.2 Clustering Approaches

Three clustering algorithms were applied and compared:

#### K-Means Clustering
Partitions movies into **K pre-specified clusters** by iteratively:
1. Assigning each movie to the nearest centroid.
2. Recomputing centroids as the mean of all assigned points.

The optimal K is determined using the **Elbow Method** (plotting inertia vs. K) or **Silhouette Score**.

#### Agglomerative (Hierarchical) Clustering
A **bottom-up** hierarchical approach:
1. Each movie starts as its own cluster.
2. The two closest clusters are merged iteratively.
3. This continues until a stopping criterion (e.g., desired number of clusters or distance threshold) is met.

The result is a **dendrogram** that reveals the hierarchical relationship between movies, allowing flexible cluster cutoffs.

**Linkage criteria used:** Ward linkage (minimises within-cluster variance at each merge step).

#### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
A **density-based** algorithm that:
1. Identifies **core points** (points with ≥ `min_samples` neighbours within radius `ε`).
2. Expands clusters from core points.
3. Labels points that are neither core nor reachable as **noise/outliers**.

DBSCAN does **not** require specifying K in advance and naturally identifies outlier movies that don't fit any cluster.

---

### 4.3 Why These Clustering Algorithms?

| Algorithm | Why Included |
|---|---|
| **K-Means** | Fast, scalable, and interpretable. The natural starting point for clustering. Works well when clusters are roughly spherical and similar in size |
| **Agglomerative** | Produces a dendrogram revealing hierarchical genre/theme relationships between movies. Does not require K upfront (can choose cutoff after seeing the tree). Captures non-spherical cluster shapes |
| **DBSCAN** | Identifies niche/outlier movies that don't cleanly belong to any group. Robust to irregular cluster shapes. Important for recommendation: "weird" outlier movies might be gems for specific tastes |

Using all three allows comparison: if all three algorithms agree on which movies cluster together, we have high confidence in those groupings. Disagreements highlight ambiguous cases (movies that blend genres or are genuinely unique).

---

### 4.4 Cosine Similarity

After clustering, **cosine similarity** is used to rank and recommend movies within the same cluster as a query movie.

For two movie feature vectors **A** and **B**:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

This gives a value between 0 (completely dissimilar) and 1 (identical direction / profile).

**How a recommendation is generated:**
1. User queries a movie (e.g., "Toy Story").
2. The system identifies which cluster that movie belongs to.
3. Cosine similarity is computed between the query movie and all other movies in the same cluster.
4. Top-N most similar movies are returned as recommendations.

---

### 4.5 Why Cosine Similarity?

| Reason | Explanation |
|---|---|
| **Magnitude-invariant** | Cosine similarity measures the *angle* between vectors, not their magnitude. A movie with 1000 ratings and one with 10 ratings can still be meaningfully compared if their genre profiles are similar |
| **Sparse data friendly** | Movie feature vectors (especially genre or tag-based) are often sparse. Cosine similarity handles sparse vectors naturally |
| **Interpretable scale** | Output is bounded [0, 1] (or [-1, 1] for signed features), making it easy to set recommendation thresholds |
| **Standard in NLP & RecSys** | Cosine similarity is the industry standard for content-based similarity in both text (TF-IDF) and recommendation contexts |
| **Complements clustering** | Clustering narrows the candidate pool; cosine similarity ranks within it. This two-stage approach is efficient and produces contextually relevant recommendations |

---

## 5. Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Language** | Python 3.x |
| **Data Manipulation** | pandas, NumPy |
| **Visualisation** | matplotlib, seaborn |
| **Classical ML** | scikit-learn |
| **Gradient Boosting** | XGBoost, LightGBM, CatBoost |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **Deep Learning** | TensorFlow / Keras |
| **Dimensionality Reduction** | scikit-learn (PCA) |
| **Clustering** | scikit-learn (K-Means, Agglomerative, DBSCAN) |
| **Similarity** | scikit-learn (cosine_similarity) |
| **Notebook Environment** | Jupyter Notebook |

---

## 6. Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Clone the Repository

```bash
git clone https://github.com/ppsspp18/Rainfall-Prediction-and-Movie-Recommendations.git
cd Rainfall-Prediction-and-Movie-Recommendations
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn \
            xgboost lightgbm catboost imbalanced-learn tensorflow jupyter
```

### Download the Dataset

Download `weatherAUS.csv` from [Kaggle](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) and place it in the root directory (or update the file path in the notebooks accordingly).

### Run the Notebooks

```bash
jupyter notebook
```

Open:
- `Rain_prediction.ipynb` — for classical ML experiments
- `rainfall_prediction_2.ipynb` — for deep learning (LSTM/GRU) experiments
- `Movie Recommendation/` — for the clustering-based recommendation system

---

## 7. Key Takeaways

**On Rainfall Prediction:**
- **Preprocessing is critical**: SMOTE had a larger positive effect on F1 score than switching between many of the classical ML models.
- **Ensemble methods dominate classical ML**: Random Forest, XGBoost, LightGBM, and CatBoost all significantly outperformed simpler models like Logistic Regression and Naive Bayes.
- **Deep learning with PCA achieved the best results**: LSTM + PCA reached 95% accuracy and F1 of 0.90, demonstrating that recurrent architectures can capture weather's temporal structure better than classical models when given clean, compact input.
- **Hyperparameter tuning matters**: Well-tuned classical models (CatBoost ~87%) came surprisingly close to the deep learning models, highlighting that tuning often matters more than model choice.

**On Movie Recommendations:**
- **No single clustering algorithm wins universally**: K-Means is fastest, Agglomerative reveals hierarchy, and DBSCAN finds outliers. The right choice depends on the use case.
- **Cosine similarity is robust for content-based filtering**: It handles the sparse, high-dimensional nature of movie feature vectors gracefully.
- **Clustering + similarity is a practical architecture**: It scales well — clustering reduces the search space from thousands to dozens, and cosine similarity ranks within that small pool efficiently.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
