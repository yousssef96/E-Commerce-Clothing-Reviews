import pandas as pd
import streamlit as st

from src.config import ONEHOT_COLS
from src.load import load_model

st.set_page_config(page_title="Review Recommender", page_icon="👗")


@st.cache_resource         
def get_model():
    return load_model()


model = get_model()

encoder = (model.named_steps["preprocess"]
           .named_transformers_["onehot_pipeline"]
           .named_steps["onehot"])
options = {col: [str(c) for c in cats] for col, cats in zip(ONEHOT_COLS, encoder.categories_)}

st.title("Will this review recommend the product?")
st.write("Enter a review and the reviewer's details. The model predicts whether they recommend the item.")

with st.form("review_form"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", min_value=18, max_value=100, value=35)
    feedback = c2.number_input("Positive Feedback Count", min_value=0, max_value=500, value=0)

    c3, c4, c5 = st.columns(3)
    division = c3.selectbox("Division", options["Division Name"])
    department = c4.selectbox("Department", options["Department Name"])
    class_name = c5.selectbox("Class", options["Class Name"])

    review = st.text_area("Review text", height=150,
                          placeholder="Love the fabric, but it runs small...")
    submitted = st.form_submit_button("Predict")

if submitted:
    if not review.strip():
        st.warning("Please write a review first.")
        st.stop()

    row = pd.DataFrame([{
        "Age": float(age),
        "Positive Feedback Count": float(feedback),
        "Division Name": division,
        "Department Name": department,
        "Class Name": class_name,
        "Review Text": review,
    }])

    proba = float(model.predict_proba(row)[0, 1])

    if proba >= 0.5:
        st.success("Likely recommends the product")
    else:
        st.error("Likely does not recommend the product")

    st.metric("Probability of recommending", f"{proba:.1%}")
    st.progress(proba)