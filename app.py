import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# إعدادات الصفحة
st.set_page_config(page_title="Diabetes Detection - Neural Network", layout="centered")

# دالة لتحميل الموديل والسكيلر
@st.cache_resource
def load_assets():
    # تحميل السكيلر (ضروري جداً لتوحيد المقاييس)
    scaler = pickle.load(open('scaler.sav', 'rb'))
    # تحميل موديل الشبكة العصبية
    nn_model = load_model('diabetes_nn_model.h5')
    return scaler, nn_model

try:
    scl, model = load_assets()
except Exception as e:
    st.error("⚠️ تأكدي من وجود ملفات 'scaler.sav' و 'diabetes_nn_model.h5' في نفس الفولدر")
    st.stop()

# واجهة المستخدم
st.title("🧠 Diabetes Prediction (Deep Learning)")
st.write("This system uses a **Neural Network (MLP)** to predict diabetes risk.")
st.markdown("---")

# تقسيم المدخلات لعمودين عشان الشكل يكون أرتب
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", 0, 20, 3)
    glu = st.number_input("Glucose Level", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 150, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    ins = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 0, 120, 30)

if st.button("Analyze with Neural Network"):
    # 1. تحويل المدخلات لمصفوفة
    input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    
    # 2. عمل Scaling (خطوة حاسمة للشبكات العصبية)
    std_data = scl.transform(input_data)
    
    # 3. التنبؤ (الموديل بيطلع احتمالية بين 0 و 1)
    prediction_prob = model.predict(std_data)
    prediction = (prediction_prob > 0.5).astype("int32")

    # 4. عرض النتيجة
    st.markdown("### Result:")
    if prediction[0][0] == 0:
        st.success(f"✅ The person is likely **Healthy**. (Confidence: {(1-prediction_prob[0][0])*100:.2f}%)")
    else:
        st.error(f"⚠️ The person is likely **Diabetic**. (Confidence: {prediction_prob[0][0]*100:.2f}%)")
    
    # عرض الـ Probability Slider
    st.info("Risk Probability Level:")
    st.progress(float(prediction_prob[0][0]))