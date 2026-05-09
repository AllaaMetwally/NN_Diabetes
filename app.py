import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# إعدادات الصفحة
st.set_page_config(page_title="Diabetes Neural Network Expert", layout="centered")

@st.cache_resource
def load_assets():
    scaler = pickle.load(open('scaler.sav', 'rb'))
    nn_model = load_model('diabetes_nn_model.h5')
    return scaler, nn_model

try:
    scl, model = load_assets()
except Exception:
    st.error("⚠️ ملفات الموديل غير موجودة.")
    st.stop()

# العنوان الرئيسي والفرعي الجديد
st.title(" 🩺Clinical Decision Support System (CDSS)")
st.markdown("""
    *This **CDSS** leverages a **Multi-Layer Perceptron (MLP)** neural network to assist healthcare 
    providers in early diabetes detection through advanced pattern recognition.*
""")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    preg = st.number_input("Pregnancies", 0, 20, 3)
    glu = st.number_input("Glucose Level (mg/dL)", 0, 200, 120)
    bp = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70)
    skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
with col2:
    ins = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("BMI (weight in kg/(height in m)^2)", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age (years)", 0, 120, 30)

if st.button("Generate Full Health Report"):
    input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    std_data = scl.transform(input_data)
    prob = model.predict(std_data)[0][0]
    prediction = 1 if prob > 0.5 else 0
    st.markdown("---")

    # 1. قسم النتيجة
    st.subheader("📊 AI Diagnosis")
    if prediction == 1:
        st.error(f"⚠️ Prediction: Diabetic Risk Detected (Confidence: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Prediction: Healthy / Low Risk (Confidence: {(1-prob)*100:.2f}%)")
    
    st.progress(float(prob))
    st.markdown("---")

    # 2. جدول المؤشرات (Risk Indicators)
    st.subheader("🔬 Key Risk Indicators")
    indicators = {
        "Factor": ["Glucose", "BMI", "Blood Pressure", "Age"],
        "Value": [glu, bmi, bp, age],
        "Status": [
            "✅ Normal" if glu < 140 else " High (Prediabetes/Diabetes)",
            "✅ Normal" if bmi < 25 else " Overweight/Obese",
            "✅ Normal" if bp < 80 else " High Blood Pressure",
            "✅ Normal" if age < 45 else " Age-related Risk"
        ]
    }
    st.table(pd.DataFrame(indicators))
    st.markdown("---")

    # 3. قسم النصائح الطبية الذكية (Personalized Advice)
    st.subheader("💡 Personalized Health Recommendations")
    
    advices = []

    
    # نصائح بناءً على الجلوكوز
    if glu >= 140:
        advices.append("**Dietary Control:** Reduce refined sugars and carbohydrates. Focus on high-fiber foods.")
    
    # نصائح بناءً على BMI
    if bmi >= 25:
        advices.append("**Weight Management:** A 5-10% weight loss can significantly reduce diabetes risk.")
        advices.append("**Physical Activity:** Aim for 150 minutes of moderate aerobic activity per week.")
    
    # نصائح بناءً على ضغط الدم
    if bp >= 80:
        advices.append("**Sodium Intake:** Reduce salt consumption and monitor blood pressure daily.")

    # نصائح بناءً على السن والوراثة
    if age > 45 or dpf > 0.5:
        advices.append("**Regular Screening:** Since you have non-modifiable risk factors, annual check-ups are essential.")

    # نصيحة عامة بناءً على الموديل
    if prediction == 1:
        advices.append(" **Urgent:** Your AI profile matches diabetic patterns. Please see an endocrinologist for a Lab A1C test.")
    else:
        advices.append(" **Maintain:** Your profile looks good! Keep a balanced lifestyle to prevent future risk.")

    # عرض النصائح في شكل نقاط (Bullets)
    for advice in advices:
        st.write(f"- {advice}")

    st.markdown("---")
    st.caption("⚠️ Disclaimer: This AI tool is for educational purposes. Always consult a certified medical professional.")