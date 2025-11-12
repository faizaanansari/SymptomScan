import streamlit as st
import pickle

st.set_page_config(page_title="🧬 Disease Predictor", page_icon="🧪", layout="centered")

# Title banner
st.markdown("""
    <div style='background: linear-gradient(to right, #36d1dc, #5b86e5); padding: 1rem; border-radius: 10px; text-align: center'>
        <h1 style='color: white;'>🧬 Disease Identification System</h1>
        <p style='color: white; font-size: 1.1rem;'>Predict Cholera, Jaundice, or Typhoid from symptoms</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Load model
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Predict function
def predict(model, scaler, inputs):
    scaled = scaler.transform([inputs])
    return model.predict(scaled)[0]

# Sidebar
st.sidebar.title("🧪 Select Disease")
disease = st.sidebar.radio("", ["Cholera", "Jaundice", "Typhoid"])

# Common layout for symptoms
def symptom_form(symptom_labels, model_file, disease_name):
    st.subheader(f"🩺 Symptoms for {disease_name}")
    cols = st.columns(2)
    inputs = []
    for i, label in enumerate(symptom_labels):
        with cols[i % 2]:
            inputs.append(1 if st.checkbox(label) else 0)

    if st.button(f"🔍 Predict {disease_name}"):
        model, scaler = load_model(model_file)
        result = predict(model, scaler, inputs)
        if result == 1:
            st.success(f"✅ {disease_name} Detected!")
        else:
            st.info(f"❌ No {disease_name} Detected.")

# Cholera
if disease == "Cholera":
    symptom_form(
        [
            "Fever",
            "Headaches and Nausea",
            "Vomiting",
            "Fatigue",
            "Severe Dehydration",
            "Diarrhea",
            "Pain in Abdomen"
        ],
        "models/cholera_model.pkl",
        "Cholera"
    )

# Jaundice
elif disease == "Jaundice":
    symptom_form(
        [
            "Fever",
            "Chills",
            "Abdominal Pain",
            "Yellowing of Skin",
            "Yellowing of Eyes",
            "Dark Colored Urine"
        ],
        "models/jaundice_model.pkl",
        "Jaundice"
    )

# Typhoid
elif disease == "Typhoid":
    symptom_form(
        [
            "Fever",
            "Chills",
            "Bloating",
            "Red Dots on Skin",
            "Loss of Appetite"
        ],
        "models/typhoid_model.pkl",
        "Typhoid"
    )

# Footer
st.markdown("""
<hr style="margin-top: 2rem;">
<div style='text-align: center; color: grey; font-size: 0.9rem'>
    Built with ❤️ using Streamlit · Mini Project by Abdullah .
</div>
""", unsafe_allow_html=True)
