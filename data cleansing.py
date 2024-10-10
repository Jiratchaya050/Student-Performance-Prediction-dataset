# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = '/content/Student Performance Prediction Dataset.csv'  # สำหรับ Colab, ให้อัปโหลดไฟล์ไปที่ Colab
data = pd.read_csv(file_path)

# Step 1: Data Cleansing
# ตรวจสอบค่าที่ขาดหายไป
print("Missing values in each column:\n", data.isnull().sum())

# กำจัดข้อมูลที่มีค่าขาดหายไป (หากมี)
data = data.dropna()

# ตรวจสอบและเปลี่ยนข้อมูลประเภทตัวอักษรเป็นประเภทตัวเลข (ถ้ามี)
# ตัวอย่างการแปลงข้อมูล categorical เป็น numerical (เปลี่ยนตามความเหมาะสมของชุดข้อมูลจริง)
data['Participation in Extracurricular Activities'] = data['Participation in Extracurricular Activities'].map({'Yes': 1, 'No': 0})
data['Parent Education Level'] = data['Parent Education Level'].map({'High School': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4})

# Step 2: Feature Selection
# เลือก Features (X) และ Target (y)
X = data[['Study Hours per Week', 'Attendance Rate', 'Previous Grades', 'Participation in Extracurricular Activities', 'Parent Education Level']]
y = data['Passed']

# Step 3: Data Transformation
# การใช้ Standard Scaler เพื่อปรับข้อมูลให้เหมาะสม
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-Test Split
# แยกข้อมูลเป็น Train (80%) และ Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Save Train and Test sets to CSV
# สร้าง DataFrame จากข้อมูลที่แยกออกมา
train_data = pd.DataFrame(X_train, columns=X.columns)
train_data['Passed'] = y_train.reset_index(drop=True)

test_data = pd.DataFrame(X_test, columns=X.columns)
test_data['Passed'] = y_test.reset_index(drop=True)

# บันทึกข้อมูล Train และ Test เป็นไฟล์ CSV
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("Train and test sets saved as 'train_data.csv' and 'test_data.csv'")
