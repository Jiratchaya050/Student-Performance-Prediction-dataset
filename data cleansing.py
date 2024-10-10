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

# Function to detect outliers using IQR
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

# Apply outlier detection to the specific numerical columns
outliers_study_hours = detect_outliers_iqr(data['Study Hours per Week'])
outliers_attendance_rate = detect_outliers_iqr(data['Attendance Rate'])
outliers_previous_grades = detect_outliers_iqr(data['Previous Grades'])

# Combine the results into one DataFrame
outliers_df = pd.DataFrame({
    'Study Hours Outlier': outliers_study_hours,
    'Attendance Rate Outlier': outliers_attendance_rate,
    'Previous Grades Outlier': outliers_previous_grades
})

# Keep only rows that do not have any outliers in any of the three columns
data_clean = data[~outliers_df.any(axis=1)]

# Shuffle the cleaned dataset
df_clean_shuffled = data_clean.sample(frac=1, random_state=42).reset_index(drop=True)

# Output the cleaned and shuffled data
print("Cleaned and Shuffled Data (Outliers Removed):")
print(df_clean_shuffled)

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
