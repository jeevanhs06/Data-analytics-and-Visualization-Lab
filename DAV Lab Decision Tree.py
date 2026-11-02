#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Decision Tree Classifier - Student Pass/Fail Prediction

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 2: Create Dataset
data = {
    "Study_Hours": [1, 2, 4, 5, 1, 2, 4, 5],
    "Attendance": [50, 60, 65, 70, 80, 85, 90, 95],
    "Result": [
        "Fail", "Fail", "Fail", "Fail",
        "Pass", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Step 3: Separate Input Features and Target
X = df[["Study_Hours", "Attendance"]]
y = df["Result"]

# Step 4: Split Dataset into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining Records:")
print(len(X_train))

print("\nTesting Records:")
print(len(X_test))

# Step 5: Create Decision Tree Classifier
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=2,
    random_state=42
)

# Step 6: Train the Model
model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")

# Step 7: Visualize the Decision Tree
plt.figure(figsize=(16, 9), dpi=150)

plot_tree(
    model,
    feature_names=["Study_Hours", "Attendance"],
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    proportion=False,
    precision=2,
    fontsize=12
)

plt.title("Decision Tree - Student Pass/Fail Prediction", fontsize=16)
plt.tight_layout()
plt.show()

# Step 8: Predict Results for Test Data
y_pred = model.predict(X_test)

print("\nActual Results:")
print(y_test.values)

print("\nPredicted Results:")
print(y_pred)

# Step 9: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Step 10: Generate Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Fail", "Pass"]
)

print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"]
)

plt.xlabel("Predicted Result")
plt.ylabel("Actual Result")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Step 11: Classification Report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Fail", "Pass"],
        zero_division=0
    )
)

# Step 12: Predict Result for a New Student
new_student = pd.DataFrame({
    "Study_Hours": [4],
    "Attendance": [85]
})

prediction = model.predict(new_student)

print("\nNew Student Details:")
print(new_student)

print("\nPredicted Student Result:", prediction[0])


# In[ ]:




