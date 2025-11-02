📰 Medium Write-Up — Team Stark’s Journey

Title: When Data Meets the Heart - Team Stark’s Journey to Predicting Heart Disease

In Africa, the conversation around heart health is often drowned by louder challenges - malaria, typhoid, and other infectious diseases. But quietly, cardiovascular diseases have crept up the ranks, becoming one of the top causes of mortality. That silent crisis inspired a group of data science enthusiasts at TechCrush Bootcamp to take action. We called ourselves Team Stark, and we set out to use data to make a difference.

💡 The Spark

Our mission was clear: “Prediction for Heart Health in Africa.”  We wanted to build a machine learning model that could predict heart disease early — especially for people who may not have access to regular health checkups.  The idea was simple: let the data tell a story about risk, and let the model serve as a guide for prevention.

🧠 The Data Chronicles

We began with the heart disease dataset — a structured collection of health indicators like age, cholesterol, blood pressure, and more.  Before touching any model, we dove deep into Exploratory Data Analysis (EDA) to understand what the data was really saying.

We cleaned, visualized, and debated over each pattern.  Chest pain type, cholesterol, and resting blood pressure stood out as important predictors. We noticed some imbalance in the target classes, and that’s where SMOTE (Synthetic Minority Oversampling Technique) came to our rescue, helping balance the scale for fairer training.

⚙️ Building the Engine

Once the data was ready, we moved into model training. We started simple — Logistic Regression — to get a baseline. Then came Random Forest, a solid performer. But the real star of the show was XGBoost - the gradient boosting powerhouse that captured complex patterns and gave us the best performance.

After several rounds of cross-validation and hyperparameter tuning, XGBoost achieved an impressive ROC-AUC of 0.83. That was the moment we knew we had something reliable.

📊 Insights Beyond the Code

Our visualizations revealed striking insights:

People with higher cholesterol and lower exercise tolerance showed higher risk.

Certain chest pain types had strong correlation with heart disease presence.

Age remained a consistent predictor -reminding us that lifestyle and early intervention matter.

We created and saved our plots — ROC curves, confusion matrices, and feature importance charts — which all painted a picture of a model that wasn’t just accurate but interpretable.

💾 The Finish Line

We wrapped up by saving our trained XGBoost model using joblib, making it ready for integration into a Streamlit app. The vision? A simple interface where users could input health data and get instant risk feedback.  It’s a prototype now, but we see it as a step toward community health empowerment.

🤝 The Power of Collaboration

Team Stark wasn’t just about coding — it was about teamwork, late-night brainstorming sessions, and helping each other grow. Each member brought something unique to the table:

Analysts visualized the unseen.

Engineers fine-tuned the models.

Designers crafted the presentations.

Writers turned numbers into narratives.

Together, we didn’t just build a model — we built a learning experience.

🔮 Lessons and Next Steps

This project taught us that data science is not just about algorithms — it’s about empathy, context, and purpose.  Our next goal is to deploy this model in real-world scenarios, perhaps through health outreach programs or collaborations with clinics.

We plan to enhance the model further by integrating real-time vitals, wearable data, and more African-specific health metrics.

💬 Final Thoughts

We started as students; we finished as innovators.  Team Stark’s Heart Disease Predictor is more than a project - it’s a vision of how African talent can harness AI to solve African problems.

“When data meets determination, impact happens.” - Team Stark, TechCrush Data Science Bootcamp

