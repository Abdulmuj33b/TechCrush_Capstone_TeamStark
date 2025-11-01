# Heart Disease Risk Prediction

🫀 Heart Disease Predictor — Team Stark (TechCrush Bootcamp)

📘 Project Overview

Team Stark proudly presents the Heart Disease Predictor, a machine learning project developed during the TechCrush Data Science Bootcamp. The aim is simple but powerful: to leverage data-driven insights and predictive modeling to support early detection of heart disease and promote health awareness across Africa.

Our model analyzes key clinical indicators and predicts the likelihood of heart disease, empowering healthcare professionals and individuals to take preventive measures earlier.

Tagline: Prediction for Heart Health in Africa

🧩 Project Structure

Team_Stark_Heart_Disease_Predictor/
│
├── Team_Stark_Heart_Disease_Predictor.ipynb   # Main project notebook
├── data/                                       # Contains raw and cleaned datasets
│   ├── heart.xls
│   └── cleaned.heart.csv
├── models/                                     # Trained model files (e.g., XGBoost.joblib)
├── images/                                     # Visualizations and plots
└── README.md                                   # Project documentation

🧠 Data Description

The dataset (heart.xls) contains patient health records with features like:

Age, Sex, Cholesterol, Resting BP

Fasting Blood Sugar, Chest Pain Type, Max Heart Rate

Exercise-Induced Angina, ST Depression

Target (0 = No Heart Disease, 1 = Heart Disease)

After cleaning and preprocessing, the data was stored as cleaned.heart.csv for reproducibility.

🔍 Exploratory Data Analysis (EDA)

Our EDA covered:

Statistical summaries and missing value checks

Correlation analysis to identify key predictors

Distribution plots for age, cholesterol, and max heart rate

Class imbalance checks

Key Insights:

Chest pain type and cholesterol levels showed significant association with heart disease.

A mild class imbalance was handled using SMOTE (Synthetic Minority Oversampling Technique).

📊 Placeholder: Insert EDA plots (histograms, correlation heatmap, etc.)

⚙️ Model Development

Three core models were trained and evaluated:

Logistic Regression — Simple and interpretable baseline.

Random Forest — Ensemble method capturing nonlinear relationships.

XGBoost — Gradient boosting algorithm providing the best performance.

The modeling pipeline included:

Data preprocessing with ColumnTransformer

Stratified K-Fold cross-validation

Hyperparameter tuning with grid search

Model saving via joblib

📈 Placeholder: Insert Model Comparison Bar Chart

📊 Model Evaluation

Model

Accuracy

ROC-AUC

F1-Score

Logistic Regression

0.78

0.80

0.76

Random Forest

0.81

0.82

0.79

XGBoost (Best)

0.84

0.83

0.82

Best Model: ✅ XGBoost (ROC-AUC ≈ 0.83)

🧩 Placeholder: Insert Confusion Matrix, ROC Curve, Precision-Recall Curve

💾 Model Saving & Deployment

The trained XGBoost model was saved in /models/XGBoost.joblib.

Ready for integration into a Streamlit web app for real-time predictions.

📦 Placeholder: Insert sample Streamlit interface screenshot

🧰 Tools & Libraries Used

Languages: Python 3.x

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, XGBoost, joblib

Environment: Google Colab

🚀 How to Run the Project Locally

Clone the repository:

git clone https://github.com/<yourusername>/Team_Stark_Heart_Disease_Predictor.git
cd Team_Stark_Heart_Disease_Predictor

Install dependencies:

pip install -r requirements.txt

Run the notebook:

jupyter notebook Team_Stark_Heart_Disease_Predictor.ipynb

(Optional) Launch the Streamlit app:

streamlit run app.py

🔮 Future Improvements

Integration with electronic health records (EHR)

Deployment as an API or mobile health assistant

Incorporation of real-time vitals and wearable data

👩🏽‍💻 Team Credits — Team Stark

Institution: TechCrush Data Science BootcampTagline: Prediction for Heart Health in Africa

Member

Role

Samuel Egwe

Team Leader

Dorcas Adegbohun

Data Analyst

Kingsley Obidi

ML Engineer

Ekezia Gloria

Research & Documentation

Abdulmujeeb Abidoye

Data Preprocessing

Iberedem Umanah

Feature Engineering

Abdulafeez Ismail

Model Evaluation

Nimota Yusuf

Data Visualization

Israel Oyebade

Testing & Validation

Iorlumun

Quality Assurance

Blessing Elaigwu

EDA & Insights

Kelechi

Support Developer

Zion

Report Writing

Sunmisola Lawal

Presentation Design

Emmanuel Ogundaini

Assistant Developer

Felicia Davids

Reviewer

💬 Acknowledgements

Special thanks to TechCrush Data Science Bootcamp for guidance, mentorship, and providing the platform to learn, collaborate, and innovate for Africa’s health future most especially Miss Gift Upwek.

“When data meets determination, impact happens.” — Team Stark
