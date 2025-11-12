import streamlit as st
import pickle

st.set_page_config(page_title="Disease Identification", page_icon="🧬")
st.title("🧬 Disease Identification System")
st.write("Predict Cholera, Jaundice, or Typhoid from symptoms using trained ML models.")

# Load model + scaler
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Predict helper
def predict(model, scaler, inputs):
    scaled = scaler.transform([inputs])
    return model.predict(scaled)[0]

# Sidebar selection
disease = st.sidebar.radio("Select Disease", ["Cholera", "Jaundice", "Typhoid"])

# Disease-specific forms
if disease == "Cholera":
    st.subheader("🧪 Cholera Symptoms")
    symptoms = {
        "Fever": "fever",
        "Headaches and Nausea": "headaches_and_nausea",
        "Vomiting": "vomiting",
        "Fatigue": "fatigue",
        "Severe Dehydration": "severe_dehydration",
        "Diarrhea": "diarrhea",
        "Pain in Abdomen": "pain_in_abdomen"
    }
    inputs = [1 if st.checkbox(symptom) else 0 for symptom in symptoms]
    if st.button("Check for Cholera"):
        model, scaler = load_model("models/cholera_model.pkl")
        result = predict(model, scaler, inputs)
        st.success("✅ Cholera Detected" if result else "❌ No Cholera")

elif disease == "Jaundice":
    st.subheader("🧪 Jaundice Symptoms")
    symptoms = {
        "Fever": "fever",
        "Chills": "chills",
        "Abdominal Pain": "abdominal_pain",
        "Yellowing of Skin": "yellowing_of_skin",
        "Yellowing of Eyes": "yellowing_of_eyes",
        "Dark Colored Urine": "dark_colored_urine"
    }
    inputs = [1 if st.checkbox(symptom) else 0 for symptom in symptoms]
    if st.button("Check for Jaundice"):
        model, scaler = load_model("models/jaundice_model.pkl")
        result = predict(model, scaler, inputs)
        st.success("✅ Jaundice Detected" if result else "❌ No Jaundice")

elif disease == "Typhoid":
    st.subheader("🧪 Typhoid Symptoms")
    symptoms = {
        "Fever": "fever",
        "Chills": "chills",
        "Bloating": "bloating",
        "Red Dots on Skin": "red_dots_on_skin",
        "Loss of Appetite": "loss_of_appetite"
    }
    inputs = [1 if st.checkbox(symptom) else 0 for symptom in symptoms]
    if st.button("Check for Typhoid"):
        model, scaler = load_model("models/typhoid_model.pkl")
        result = predict(model, scaler, inputs)
        st.success("✅ Typhoid Detected" if result else "❌ No Typhoid")
