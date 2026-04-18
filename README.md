This project is an **End-to-End Machine Learning Solution** designed to predict the likelihood of Autism Spectrum Disorder (ASD) in adults.

It includes a comprehensive data science pipeline—from exploratory data analysis and model training to a functional web application for real-time predictions.

Below is a structured `README.md` for your GitHub repository.

--
# 🧠 Autism Spectrum Disorder (ASD) Prediction System

This repository features a complete machine learning project aimed at predicting ASD using behavioral and demographic data. It includes a **Jupyter Notebook** for training and evaluating models and a **Streamlit Web App** for interactive deployment.

## 🚀 Project Overview

Predicting ASD typically involves lengthy clinical assessments. This project leverages an XGBoost-based classification model to provide a fast, data-driven assessment based on 10 behavioral questions (A1-A10) and basic demographic factors.

### Key Components:
- **Exploratory Data Analysis (EDA):** Insights into data distribution and feature correlations.
- **Preprocessing:** Handling missing values, fuzzy country name matching, and SMOTE for class imbalance.
- **Machine Learning:** Comparison of Random Forest, Gradient Boosting, and XGBoost.
- **Deployment:** A Streamlit interface for user interaction and model inference.

## 📁 Project Structure

* `Final Project.ipynb`: The core research notebook containing data cleaning, model training, and performance metrics.
* `app.py`: Streamlit application script for the user interface.
* `best_model.pkl`: The saved XGBoost model with the highest accuracy.
* `scaler.pkl`: StandardScaler object used to normalize numerical features.
* `encoders.pkl`: LabelEncoders for processing categorical data like gender and ethnicity.

## 📊 Technical Stack

- **Languages:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
- **Deployment:** Streamlit
- **Text Processing:** FuzzyWuzzy (for cleaning country data)

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/autism-prediction-app.git
   cd autism-prediction-app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you include `streamlit`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, and `fuzzywuzzy` in your requirements file).*

3. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## 🧠 How It Works

The application takes 20 inputs, including:
- **A1-A10 Scores:** Binary responses to specific behavioral questions.
- **Demographics:** Age, Gender, Ethnicity, and Country of residence.
- **Medical History:** History of jaundice or family history of autism.

The input is then encoded, scaled, and passed through the XGBoost model to output a prediction and the associated probability.

## 📈 Model Performance
The final model was optimized using `RandomizedSearchCV` and evaluated on metrics including:
- **Accuracy**
- **Recall** (Prioritized to minimize false negatives in medical screening).

---
