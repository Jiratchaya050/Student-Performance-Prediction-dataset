# Import libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the preprocessed train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Separate features and target
X_train = train_data.drop(columns=['Passed'])
y_train = train_data['Passed']
X_test = test_data.drop(columns=['Passed'])
y_test = test_data['Passed']

# Step 1: Set up hyperparameter grids for each algorithm
# Random Forest Hyperparameters
rf_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# K-Nearest Neighbors Hyperparameters
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Step 2: Define models with GridSearchCV
rf_model = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy')

# Step 3: Train models
print("Training Random Forest model...")
rf_model.fit(X_train, y_train)

imputer = SimpleImputer(strategy='mean') # or 'median', 'most_frequent'
X_train_imputed = imputer.fit_transform(X_train)

print("Training K-Nearest Neighbors model...")
knn_model.fit(X_train_imputed, y_train)

# Step 4: Evaluate models on test data
# Best model selection based on GridSearchCV
rf_best = rf_model.best_estimator_
knn_best = knn_model.best_estimator_

# Impute missing values in X_test using the same imputer fitted on X_train
X_test_imputed = imputer.transform(X_test)

# Impute missing values in X_test using the same imputer fitted on X_train
X_test_imputed = imputer.transform(X_test)

# Predictions
rf_pred = rf_best.predict(X_test)
knn_pred = knn_best.predict(X_test_imputed) # Use imputed X_test for KNN

# Evaluation metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)

rf_report = classification_report(y_test, rf_pred)
knn_report = classification_report(y_test, knn_pred)

# Confusion matrices
rf_cm = confusion_matrix(y_test, rf_pred)
knn_cm = confusion_matrix(y_test, knn_pred)

# Step 5: Choose the best model and save it
best_model = rf_best if rf_accuracy > knn_accuracy else knn_best
joblib.dump(best_model, 'best_student_performance_model.pkl')

# Step 6: Display results
# Print classification reports
print("Random Forest Classification Report:\n", rf_report)
print("K-Nearest Neighbors Classification Report:\n", knn_report)

# Show accuracy comparison
model_names = ['Random Forest', 'K-Nearest Neighbors']
accuracies = [rf_accuracy, knn_accuracy]

plt.figure(figsize=(8, 6))
sns.barplot(x=model_names, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# Show confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title("Random Forest Confusion Matrix")
axs[0].set_xlabel("Predicted")
axs[0].set_ylabel("Actual")

sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Blues", ax=axs[1])
axs[1].set_title("K-Nearest Neighbors Confusion Matrix")
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

