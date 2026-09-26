
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "models/career_recommendation_xgboost.pkl"
ENCODER_PATH = "models/target_encoder.pkl"

st.set_page_config(
    page_title="CareerFit - AI Career Guidance",
    page_icon="🎯",
    layout="wide",
)

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return pipeline, label_encoder

def normalize_multivalue(value):
    return ", ".join(
        [x.strip() for x in value.split(",") if x.strip()]
    )

def recommend_top_careers(
    education,
    academic_stream,
    cgpa,
    skills,
    interests,
    favorite_subjects,
    personality,
    top_n=3,
):
    student = pd.DataFrame([{
        "Education": education,
        "Academic Stream": academic_stream,
        "CGPA": cgpa,
        "Skills": normalize_multivalue(skills),
        "Interests": normalize_multivalue(interests),
        "Favorite Subjects": normalize_multivalue(favorite_subjects),
        "Personality": normalize_multivalue(personality),
    }])

    probabilities = pipeline.predict_proba(student)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_n]

    return [
        (
            label_encoder.inverse_transform([i])[0],
            float(probabilities[i])
        )
        for i in top_indices
    ]

st.title("🎯 CareerFit")
st.subheader("AI-Powered Career Guidance Expert System")
st.write(
    "Find career options that match your education, skills, interests, "
    "favorite subjects, personality, and academic profile."
)

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    st.error(
        "Trained model files are missing. Run `python train_model.py` first."
    )
    st.stop()

pipeline, label_encoder = load_artifacts()

with st.form("career_form"):
    st.markdown("### 👨‍🎓 Student Profile")

    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "Education",
            ["Bachelor", "Master", "Diploma", "PhD", "Other"]
        )

        academic_stream = st.text_input(
            "Academic Stream",
            placeholder="e.g. Computer Science"
        )

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=8.0,
            step=0.1
        )

        personality = st.text_input(
            "Personality",
            placeholder="e.g. Analytical, Creative, Logical"
        )

    with col2:
        skills = st.text_area(
            "Skills",
            placeholder="Python, Machine Learning, SQL"
        )

        interests = st.text_area(
            "Interests",
            placeholder="Artificial Intelligence, Data Science"
        )

        favorite_subjects = st.text_area(
            "Favorite Subjects",
            placeholder="Mathematics, Computer Science"
        )

    submitted = st.form_submit_button(
        "🚀 Predict My Career",
        use_container_width=True
    )

if submitted:
    required = {
        "Academic Stream": academic_stream,
        "Skills": skills,
        "Interests": interests,
        "Favorite Subjects": favorite_subjects,
        "Personality": personality,
    }

    missing = [name for name, value in required.items() if not value.strip()]

    if missing:
        st.warning(
            "Please complete: " + ", ".join(missing)
        )
    else:
        results = recommend_top_careers(
            education=education,
            academic_stream=academic_stream,
            cgpa=cgpa,
            skills=skills,
            interests=interests,
            favorite_subjects=favorite_subjects,
            personality=personality,
            top_n=3,
        )

        st.markdown("## 🏆 Top Career Recommendations")

        cols = st.columns(3)

        for rank, ((career, probability), col) in enumerate(
            zip(results, cols), start=1
        ):
            with col:
                st.metric(
                    f"#{rank} Recommendation",
                    career,
                    f"{probability * 100:.2f}%"
                )
                st.progress(float(probability))

        st.markdown("### 📊 Recommendation Details")

        result_df = pd.DataFrame(
            [
                {
                    "Rank": rank,
                    "Career": career,
                    "Probability": f"{probability * 100:.2f}%"
                }
                for rank, (career, probability) in enumerate(
                    results, start=1
                )
            ]
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "These percentages are the XGBoost model's predicted class "
            "probabilities for the submitted profile."
        )

st.markdown("---")
st.caption(
    "CareerFit | Machine Learning: XGBoost | "
    "Preprocessing: scikit-learn | UI: Streamlit"
)
