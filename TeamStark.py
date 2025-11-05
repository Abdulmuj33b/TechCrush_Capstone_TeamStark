import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import tempfile

# Page configuration
st.set_page_config(
    page_title="HeartGuard - Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .high-risk {
        background-color: #ffebee;
        border-left-color: #ff4b4b;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .vital-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .validation-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Validation Functions
def validate_patient_inputs(age, sex, trestbps, chol, thalach, oldpeak, ca, thal):
    """
    Validate patient inputs and return warnings if any values are outside typical ranges
    """
    warnings = []
    
    # Age validation
    if age < 20 or age > 100:
        warnings.append("Error: Age outside typical adult range (20-100 years)")
    elif age < 30:
        warnings.append("Info: Patient is relatively young for heart disease assessment")
    elif age > 70:
        warnings.append("Warning: Advanced age - increased baseline risk")
    
    # Blood pressure validation
    if trestbps < 80:
        warnings.append("Error: Very low resting blood pressure (hypotension)")
    elif trestbps < 90:
        warnings.append("Warning: Low resting blood pressure")
    elif trestbps > 140:
        warnings.append("Warning: Elevated resting blood pressure (Stage 1 Hypertension)")
    elif trestbps > 180:
        warnings.append("Error: Very high resting blood pressure (Hypertensive Crisis)")
    
    # Cholesterol validation
    if chol < 100:
        warnings.append("Error: Very low cholesterol level")
    elif chol < 125:
        warnings.append("Warning: Low cholesterol level")
    elif chol > 240:
        warnings.append("Warning: High cholesterol level (Hypercholesterolemia)")
    elif chol > 300:
        warnings.append("Error: Very high cholesterol level")
    
    # Maximum heart rate validation
    max_predicted_hr = 220 - age  # Rough estimate
    if thalach > max_predicted_hr + 20:
        warnings.append("Warning: Maximum heart rate exceeds predicted maximum for age")
    elif thalach < 100:
        warnings.append("Warning: Low maximum heart rate achieved")
    elif thalach < (max_predicted_hr * 0.85):
        warnings.append("Info: Submaximal exercise heart rate")
    
    # ST Depression validation
    if oldpeak < 0:
        warnings.append("Error: ST depression cannot be negative")
    elif oldpeak > 4:
        warnings.append("Warning: Significant ST depression detected")
    elif oldpeak > 6:
        warnings.append("Error: Very high ST depression - consider immediate medical attention")
    
    # Number of major vessels
    if ca > 3:
        warnings.append("Warning: High number of major vessels affected")
    
    return warnings

def validate_input_completeness(input_data):
    """
    Check if all required fields are filled
    """
    missing_fields = []
    for field, value in input_data.items():
        if value is None or (isinstance(value, (int, float)) and np.isnan(value)):
            missing_fields.append(field)
    
    return missing_fields

def calculate_data_quality_score(input_dict):
    """Calculate a data quality score based on input validity"""
    score = 100
    deductions = []
    
    # Check each parameter and deduct points for outliers
    if not (20 <= input_dict['age'] <= 100):
        score -= 20
        deductions.append("Age outside typical range")
    
    if not (80 <= input_dict['trestbps'] <= 200):
        score -= 15
        deductions.append("BP outside typical range")
    
    if not (100 <= input_dict['chol'] <= 400):
        score -= 15
        deductions.append("Cholesterol outside typical range")
    
    if input_dict['thalach'] < 60:
        score -= 10
        deductions.append("Very low max heart rate")
    
    if input_dict['oldpeak'] > 6:
        score -= 10
        deductions.append("Extreme ST depression")
    
    return max(score, 0), deductions

def display_vital_status(age, trestbps, chol, thalach, oldpeak):
    """Display color-coded vital status"""
    st.sidebar.markdown("### Vital Status")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        # Blood Pressure
        if 90 <= trestbps <= 120:
            st.success("BP: Normal")
        elif 121 <= trestbps <= 139:
            st.warning("BP: Elevated")
        else:
            st.error("BP: Abnormal")
        
        # Cholesterol
        if chol < 200:
            st.success("Chol: Normal")
        elif 200 <= chol <= 239:
            st.warning("Chol: Borderline")
        else:
            st.error("Chol: High")
    
    with col2:
        # Heart Rate
        max_expected = 220 - age
        if thalach >= max_expected * 0.85:
            st.success("Max HR: Good")
        else:
            st.warning("Max HR: Low")
        
        # ST Depression
        if oldpeak <= 1.0:
            st.success("ST Depression: Normal")
        elif 1.0 < oldpeak <= 2.0:
            st.warning("ST Depression: Mild")
        else:
            st.error("ST Depression: Significant")

def show_input_guidelines():
    """Show input guidelines in an expander"""
    with st.sidebar.expander("Input Guidelines"):
        st.markdown("""
        **Typical Clinical Ranges:**
        - **Age:** 30-80 years (most studies)
        - **Resting BP:** 90-140 mmHg
        - **Cholesterol:** 125-240 mg/dl
        - **Max Heart Rate:** 120-200 bpm
        - **ST Depression:** 0-4 mm
        
        **Clinical Notes:**
        - **Chest Pain Types:** 
          0 = Typical Angina, 1 = Atypical, 
          2 = Non-anginal, 3 = Asymptomatic
        - **ST Slope:** 
          0 = Upsloping, 1 = Flat, 2 = Downsloping
        - **Major Vessels:** 0-4 (fluoroscopy results)
        - **Thalassemia:** 
          1 = Normal, 2 = Fixed Defect, 
          3 = Reversible Defect
        """)

# Load model with multiple fallback options
@st.cache_resource
def load_model():
    try:
        # Try local paths first
        local_paths = [
            'models/final_xgb_optuna.pkl',
            './final_xgb_optuna.pkl',
            'final_xgb_optuna.pkl'
        ]
        
        for path in local_paths:
            if os.path.exists(path):
                st.success(f"Model loaded from local: {path}")
                return joblib.load(path)
        
        # If local not found, download from GitHub
        st.info("Downloading model from GitHub...")
        
        # Raw GitHub URL
        github_raw_url = "https://raw.githubusercontent.com/Abdulmuj33b/TechCrush_Capstone_TeamStark/main/models/final_xgb_optuna.pkl"
        
        # Download the file
        response = requests.get(github_raw_url, stream=True)
        response.raise_for_status()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            temp_path = tmp_file.name
        
        # Load the model from temporary file
        model = joblib.load(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        st.success("Model downloaded and loaded successfully from GitHub!")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("""
        Troubleshooting tips:
        1. Make sure the model file exists in your GitHub repository
        2. Ensure the file is committed and pushed to GitHub
        3. Check that the file path in the repository is correct
        """)
        return None

# Initialize model
model = load_model()

# App header
st.markdown('<h1 class="main-header">HeartGuard Pro</h1>', unsafe_allow_html=True)
st.markdown("### Advanced Heart Disease Risk Assessment Tool")

# Sidebar with patient information
with st.sidebar:
    st.header("Patient Information")
    st.markdown("---")
    
    # Personal Information
    st.subheader("Personal Details")
    age = st.slider("Age", 20, 100, 50, help="Patient's age in years")
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    
    st.markdown("---")
    
    # Medical History
    st.subheader("Medical Examination")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cp = st.selectbox(
            "Chest Pain Type", 
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina", 
                2: "Non-anginal Pain",
                3: "Asymptomatic"
            }[x]
        )
        
        trestbps = st.number_input(
            "Resting BP (mm Hg)", 
            min_value=80, max_value=200, value=120,
            help="Resting blood pressure"
        )
        
        chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=100, max_value=600, value=200,
            help="Serum cholesterol level"
        )
        
        fbs = st.selectbox(
            "Fasting Blood Sugar", 
            options=[0, 1],
            format_func=lambda x: "Normal" if x == 0 else "High (>120 mg/dl)"
        )
    
    with col2:
        restecg = st.selectbox(
            "Resting ECG", 
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x]
        )
        
        thalach = st.slider(
            "Max Heart Rate", 
            min_value=60, max_value=220, value=150,
            help="Maximum heart rate achieved"
        )
        
        exang = st.selectbox(
            "Exercise Angina", 
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
    
    st.markdown("---")
    
    # Exercise Test Results
    st.subheader("Exercise Test")
    
    oldpeak = st.slider(
        "ST Depression", 
        min_value=0.0, max_value=6.0, value=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest"
    )
    
    slope = st.selectbox(
        "ST Slope", 
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat", 
            2: "Downsloping"
        }[x]
    )
    
    ca = st.selectbox(
        "Major Vessels", 
        options=[0, 1, 2, 3, 4],
        help="Number of major vessels colored by fluoroscopy"
    )
    
    thal = st.selectbox(
        "Thalassemia", 
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Normal",
            2: "Fixed Defect", 
            3: "Reversible Defect"
        }[x]
    )

    # Real-time validation
    st.markdown("---")
    st.subheader("Input Validation")
    
    # Run validation
    validation_warnings = validate_patient_inputs(
        age, sex, trestbps, chol, thalach, oldpeak, ca, thal
    )
    
    # Display validation results
    if validation_warnings:
        for warning in validation_warnings:
            if warning.startswith("Error:"):
                st.error(warning)
            elif warning.startswith("Warning:"):
                st.warning(warning)
            else:
                st.info(warning)
    else:
        st.success("All inputs within expected ranges")
    
    # Display vital status
    display_vital_status(age, trestbps, chol, thalach, oldpeak)
    
    # Show input guidelines
    show_input_guidelines()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Risk Assessment")
    
    # Create input data dictionary
    input_dict = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    
    if st.button("Calculate Heart Disease Risk", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not available. Please check if the model file is properly installed.")
        else:
            # Check for missing fields
            missing_fields = validate_input_completeness(input_dict)
            
            if missing_fields:
                st.error(f"Missing required fields: {', '.join(missing_fields)}")
                st.stop()
            
            # Calculate data quality score
            data_quality_score, quality_issues = calculate_data_quality_score(input_dict)
            
            # Show validation summary
            if validation_warnings or quality_issues:
                st.subheader("Validation Summary")
                
                if data_quality_score < 90:
                    st.warning(f"Data Quality Score: {data_quality_score}%")
                
                if validation_warnings:
                    for warning in validation_warnings:
                        st.markdown(f'<div class="validation-warning">{warning}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
            
            # Create DataFrame for prediction
            input_data = pd.DataFrame([[
                age, sex, cp, trestbps, chol, fbs, restecg, 
                thalach, exang, oldpeak, slope, ca, thal
            ]], columns=[
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ])
            
            # Make prediction
            with st.spinner("Analyzing patient data..."):
                try:
                    prediction_proba = model.predict_proba(input_data)[:, 1]
                    probability_of_disease = prediction_proba[0] * 100
                    
                    # Display results
                    st.subheader("Prediction Result")
                    
                    if probability_of_disease >= 50:
                        risk_class = "high-risk"
                        risk_level = "HIGH RISK"
                        risk_message = "High likelihood of heart disease detected"
                        recommendation = "**Recommendation:** Please consult a cardiologist for further evaluation and treatment planning."
                    else:
                        risk_class = "low-risk" 
                        risk_level = "LOW RISK"
                        risk_message = "Low likelihood of heart disease"
                        recommendation = "**Recommendation:** Maintain healthy lifestyle with regular check-ups."
                    
                    st.markdown(f"""
                    <div class="prediction-box {risk_class}">
                        <h2>{risk_level}: {probability_of_disease:.1f}%</h2>
                        <p><strong>{risk_message}</strong></p>
                        <p>{recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk gauge visualization
                    st.subheader("Risk Level")
                    risk_percentage = probability_of_disease / 100
                    st.progress(float(risk_percentage))
                    st.caption(f"Risk Score: {probability_of_disease:.1f}%")
                    
                    # Risk factors summary
                    st.subheader("Risk Factor Analysis")
                    risk_col1, risk_col2 = st.columns(2)
                    
                    with risk_col1:
                        st.write("**Major Risk Factors:**")
                        major_factors = []
                        
                        if age > 55:
                            major_factors.append(f"Age ({age} years)")
                        if trestbps > 140:
                            major_factors.append("Hypertension")
                        if chol > 240:
                            major_factors.append("High Cholesterol")
                        if fbs == 1:
                            major_factors.append("Elevated Blood Sugar")
                        if exang == 1:
                            major_factors.append("Exercise-Induced Angina")
                        if oldpeak > 2.0:
                            major_factors.append("Significant ST Depression")
                        
                        for factor in major_factors:
                            st.write(f"• {factor}")
                        
                        if not major_factors:
                            st.write("No major risk factors identified")
                    
                    with risk_col2:
                        st.write("**Contributing Factors:**")
                        contributing_factors = []
                        
                        if cp in [1, 2, 3]:
                            contributing_factors.append("Atypical Chest Pain")
                        if restecg in [1, 2]:
                            contributing_factors.append("ECG Abnormalities")
                        if slope == 2:
                            contributing_factors.append("Downsloping ST Segment")
                        if ca > 0:
                            contributing_factors.append(f"{ca} Major Vessel(s) Affected")
                        if thal == 3:
                            contributing_factors.append("Reversible Thalassemia")
                        
                        for factor in contributing_factors:
                            st.write(f"• {factor}")
                        
                        if not contributing_factors:
                            st.write("No additional contributing factors")
                    
                    # Clinical recommendations based on risk level
                    st.subheader("Clinical Recommendations")
                    
                    if probability_of_disease >= 70:
                        st.error("""
                        **Urgent Action Recommended:**
                        - Immediate cardiology consultation
                        - Consider stress testing or angiography
                        - Lifestyle modification imperative
                        - Regular monitoring of vital signs
                        """)
                    elif probability_of_disease >= 50:
                        st.warning("""
                        **Moderate Risk Management:**
                        - Schedule cardiology follow-up
                        - Consider non-invasive testing
                        - Implement lifestyle changes
                        - Monitor risk factors regularly
                        """)
                    else:
                        st.success("""
                        **Preventive Measures:**
                        - Maintain healthy lifestyle
                        - Regular exercise regimen
                        - Balanced diet
                        - Annual health check-ups
                        """)
                        
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.info("Please check your input values and try again.")

            # Final disclaimer
            st.markdown("---")
            st.info("""
            **Medical Disclaimer:** 
            This prediction is based on machine learning models and should be used as a supplementary tool only. 
            Always consult healthcare professionals for medical diagnosis and treatment decisions. 
            In case of emergency symptoms (chest pain, shortness of breath), seek immediate medical attention.
            """)

with col2:
    st.header("Feature Importance")
    st.markdown("""
    <div class="feature-card">
    <strong>Key Predictive Factors:</strong>
    <ul>
    <li><strong>Age & Gender</strong> - Baseline risk factors</li>
    <li><strong>Cholesterol Levels</strong> - Lipid profile impact</li>
    <li><strong>Blood Pressure</strong> - Cardiovascular strain</li>
    <li><strong>Chest Pain Type</strong> - Symptom patterns</li>
    <li><strong>Exercise Response</strong> - Functional capacity</li>
    <li><strong>ST Depression</strong> - Ischemia indicator</li>
    <li><strong>Major Vessels</strong> - Anatomical assessment</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("About This Tool")
    st.markdown("""
    <div class="feature-card">
    **HeartGuard Pro** uses advanced machine learning to assess heart disease risk based on clinical parameters.
    
    **Model Characteristics:**
    - Algorithm: XGBoost with Optuna optimization
    - Accuracy: ~85% (validation)
    - Features: 13 clinical parameters
    - Training: UCI Heart Disease Dataset
    
    **Last Updated:** January 2024
    **Version:** 2.0 with Enhanced Validation
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("Emergency Info")
    st.markdown("""
    <div style="background-color: #ffebee; padding: 1rem; border-radius: 10px; border-left: 4px solid #ff4b4b;">
    <strong>Seek Immediate Medical Attention for:</strong>
    <ul>
    <li>Chest pain or pressure</li>
    <li>Shortness of breath</li>
    <li>Radiating arm/jaw pain</li>
    <li>Sudden dizziness</li>
    <li>Severe palpitations</li>
    </ul>
    <strong>Emergency Contact:</strong> 911 or local emergency services
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "HeartGuard Pro v2.0 | Medical AI Tool | For educational and screening purposes only"
    "</div>", 
    unsafe_allow_html=True
)
