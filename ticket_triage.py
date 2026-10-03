# ============================================================
# Customer Support Ticket Classification
# AI Use Case: Automatically classify support tickets
# Categories:
#   1. Billing
#   2. Technical Issue
#   3. Account Access
#   4. General Inquiry
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# STEP 1: Create a Small Dataset
# ============================================================

data = {
    "message": [
        # Billing
        "Why was I charged twice this month?",
        "I was charged an incorrect amount",
        "How can I get a refund?",
        "My credit card was charged twice",
        "I have a question about my invoice",
        "Why is my monthly bill so high?",
        "I want to cancel my payment",
        "There is an extra charge on my account",
        "My payment failed but money was deducted",
        "Can you explain this charge on my bill?",
        "I need a copy of my invoice",
        "When will my refund arrive?",
        "My subscription payment is incorrect",
        "I was billed for a service I did not purchase",
        "How do I update my billing information?",
        "The invoice amount is wrong",
        "I need help with my payment",
        "Why did you charge my card?",
        "Can I change my payment method?",
        "I have a problem with my invoice",

        # Technical Issue
        "The application keeps crashing",
        "The website is not loading",
        "I cannot upload a file",
        "The app freezes when I open it",
        "The system shows an error message",
        "The application stopped working",
        "I am getting a technical error",
        "The website is extremely slow",
        "My dashboard is not displaying correctly",
        "The application crashes whenever I upload a document",
        "I cannot download my report",
        "The page keeps refreshing",
        "The software is not responding",
        "There is a bug in the application",
        "The website gives me a server error",
        "My files are not uploading",
        "The application has stopped responding",
        "I am unable to generate a report",
        "The system is showing an unexpected error",
        "The website does not work properly",

        # Account Access
        "I forgot my password",
        "I cannot log into my account",
        "How do I reset my password?",
        "My account is locked",
        "I cannot access my account",
        "I forgot my username",
        "My login credentials are not working",
        "I am unable to sign in",
        "How can I unlock my account?",
        "I cannot remember my password",
        "My account has been locked",
        "The login page is rejecting my password",
        "I need help accessing my account",
        "I cannot sign into the application",
        "How do I recover my account?",
        "My password reset link is not working",
        "I lost access to my account",
        "I am having trouble logging in",
        "Why can't I access my account?",
        "Please help me reset my login",

        # General Inquiry
        "What are your customer support hours?",
        "How can I contact customer support?",
        "Where can I find your documentation?",
        "What services do you provide?",
        "Can you tell me more about your company?",
        "What are your business hours?",
        "Where can I find more information?",
        "How does your service work?",
        "Do you provide customer support on weekends?",
        "Where can I find the user guide?",
        "What features are available?",
        "Can you explain your services?",
        "How can I contact your team?",
        "Where can I find your FAQ?",
        "What is your support schedule?",
        "Can you provide more information?",
        "What products do you offer?",
        "How do I learn more about your service?",
        "Where is the documentation?",
        "Can you tell me about your support options?"
    ],

    "category": (
        ["Billing"] * 20 +
        ["Technical Issue"] * 20 +
        ["Account Access"] * 20 +
        ["General Inquiry"] * 20
    )
}

df = pd.DataFrame(data)

print("=" * 60)
print("DATASET")
print("=" * 60)

print(df.head())

print("\nTotal samples:", len(df))

print("\nClass distribution:")
print(df["category"].value_counts())

# ============================================================
# STEP 2: Split Dataset
# ============================================================

X = df["message"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# STEP 3: Create AI Classification Pipeline
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

# ============================================================
# STEP 4: Train the Model
# ============================================================

model.fit(X_train, y_train)

print("\nModel training completed.")

# ============================================================
# STEP 5: Make Predictions
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# STEP 6: Calculate Accuracy
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.2%}")

# ============================================================
# STEP 7: Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ============================================================
# STEP 8: Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title("Customer Support Ticket Classification")
plt.tight_layout()

plt.show()

# ============================================================
# STEP 9: Test New Customer Tickets
# ============================================================

new_tickets = [
    "I cannot log into my account",
    "Why was my credit card charged twice?",
    "The application keeps crashing",
    "What are your customer support hours?",
    "I forgot my password",
    "The website is showing an error",
    "I need a refund for my payment",
    "Where can I find your documentation?"
]

predictions = model.predict(new_tickets)

print("\n" + "=" * 60)
print("NEW TICKET PREDICTIONS")
print("=" * 60)

for ticket, prediction in zip(new_tickets, predictions):

    print("\nTicket:")
    print(ticket)

    print("Predicted Category:")
    print(prediction)

# ============================================================
# STEP 10: Interactive Prediction
# ============================================================

print("\n" + "=" * 60)
print("INTERACTIVE CUSTOMER SUPPORT CLASSIFIER")
print("=" * 60)

while True:

    user_input = input(
        "\nEnter a customer support message "
        "(type 'exit' to stop): "
    )

    if user_input.lower() == "exit":
        print("Program ended.")
        break

    prediction = model.predict([user_input])[0]

    # Probability/confidence
    probabilities = model.predict_proba([user_input])[0]

    max_probability = max(probabilities)

    print("\nPredicted Category:", prediction)
    print(f"Confidence: {max_probability:.2%}")