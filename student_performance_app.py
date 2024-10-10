import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load('best_model.pkl')

# ตั้งค่าหน้าเว็บ
st.title("Student Performance Prediction")
st.write("ทำนายผลการเรียนของนักเรียนโดยใช้โมเดลที่ผ่านการฝึกมาแล้ว")

# สร้างช่องรับข้อมูลจากผู้ใช้
study_hours = st.number_input("Study Hours per Week | จำนวนชั่วโมงที่นักเรียนใช้ในการเรียนต่อสัปดาห์", min_value=0.0, max_value=100.0, step=0.1)
attendance_rate = st.number_input("Attendance Rate | อัตราการเข้าชั้นเรียนของนักศึกษา (%):", min_value=0.0, max_value=100.0, step=0.1, value=75.0)
previous_grades = st.number_input("Previous Grades | คะแนนจากการสอบครั้งที่แล้ว (0 - 100):", min_value=0.0, max_value=100.0, step=0.1, value=70.0)
extra_activities = st.selectbox("Participation in Extracurricular Activities | การเข้าร่วมกิจกรรมนอกหลักสูตร ", ["No", "Yes"])
parent_education = st.selectbox("Parent Education Level | ระดับการศึกษาของผู้ปกครอง", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])


# การแปลงค่าข้อมูลที่กรอกเพื่อให้เข้ากับโมเดล
extra_activities_encoded = 1 if extra_activities == "Yes" else 0
parent_education_encoded = ["High School", "Associate", "Bachelor", "Master", "Doctorate"].index(parent_education)

# สร้าง array ของข้อมูล
input_data = np.array([[study_hours, attendance_rate, previous_grades, extra_activities_encoded, parent_education_encoded]])

# เพิ่มปุ่มเพื่อทำการทำนาย
if st.button("Predict"):
    # ทำการทำนายผล
    prediction = model.predict(input_data)
    # แสดงผลการทำนาย
    if prediction[0] == 1:
        st.success("ผลการทำนาย: นักเรียนมีโอกาสที่จะ 'ผ่าน'")
    else:
        st.error("ผลการทำนาย: นักเรียนมีโอกาสที่จะ 'ไม่ผ่าน'")
