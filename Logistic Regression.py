#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Step 2: Create Dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6,
                    1, 2, 3, 4, 5, 6],

    "Result": [
        "Fail", "Fail", "Fail", "Fail", "Pass", "Pass",
        "Fail", "Fail", "Fail", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

print("STUDENT DATASET")
print(df)

# Step 3: Separate Input and Output
X = df[["Study_Hours"]]
y = df["Result"]

# Step 4: Split Dataset into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# Step 5: Create Logistic Regression Model
model = LogisticRegression()

# Step 6: Train the Model
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# Step 7: Display Learned Coefficients
print("\nLEARNED MODEL PARAMETERS")

print("Intercept:", model.intercept_[0])
print("Coefficient:", model.coef_[0][0])

# Step 8: Predict Results for Test Data
y_pred = model.predict(X_test)

print("\nACTUAL RESULTS:")
print(y_test.values)

print("\nPREDICTED RESULTS:")
print(y_pred)

# Step 9: Calculate Prediction Probabilities
y_prob = model.predict_proba(X_test)

print("\nPREDICTED PROBABILITIES:")
print(y_prob)

# Step 10: Compare Actual and Predicted Results
comparison = pd.DataFrame({
    "Study_Hours": X_test["Study_Hours"].values,
    "Actual_Result": y_test.values,
    "Predicted_Result": y_pred
})

print("\nACTUAL VS PREDICTED RESULTS")
print(comparison)

# Step 11: Visualize Logistic Regression Curve
study_hours = np.linspace(
    df["Study_Hours"].min(),
    df["Study_Hours"].max(),
    200
).reshape(-1, 1)

# Find the column corresponding to Pass
pass_index = list(model.classes_).index("Pass")

# Calculate probability of Pass
probabilities = model.predict_proba(study_hours)[:, pass_index]

plt.figure(figsize=(9, 5))

# Plot actual results
plt.scatter(
    df["Study_Hours"],
    df["Result"].map({"Fail": 0, "Pass": 1}),
    label="Actual Results"
)

# Plot Logistic Regression curve
plt.plot(
    study_hours,
    probabilities,
    label="Logistic Regression Curve"
)

# Display decision threshold
plt.axhline(
    y=0.5,
    linestyle="--",
    label="Threshold = 0.5"
)

plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression - Pass/Fail Prediction")

plt.yticks(
    [0, 0.5, 1],
    ["Fail (0)", "0.5", "Pass (1)"]
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 12: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL EVALUATION METRICS")

print("Accuracy:", accuracy * 100, "%")

# Step 13: Generate Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Fail", "Pass"]
)

print("\nCONFUSION MATRIX")
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

# Step 14: Calculate Precision
precision = precision_score(
    y_test,
    y_pred,
    pos_label="Pass",
    zero_division=0
)

print("\nPrecision:", precision)

# Step 15: Calculate Recall
recall = recall_score(
    y_test,
    y_pred,
    pos_label="Pass",
    zero_division=0
)

print("Recall:", recall)

# Step 16: Calculate F1-Score
f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Pass",
    zero_division=0
)

print("F1-Score:", f1)

# Step 17: Classification Report
print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Fail", "Pass"],
        zero_division=0
    )
)

# Step 18: Predict Result for a New Student
new_student = pd.DataFrame({
    "Study_Hours": [5]
})

prediction = model.predict(new_student)

probability = model.predict_proba(new_student)

print("\nNEW STUDENT DETAILS")
print(new_student)

print("\nProbability of Fail:", probability[0][0])
print("Probability of Pass:", probability[0][1])

print("\nPREDICTED STUDENT RESULT:", prediction[0])


# In[ ]:




