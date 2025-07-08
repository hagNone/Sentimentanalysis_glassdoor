from .preprocessing import df_sampled
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

''' Map sentiment labels to binary values '''
df_sampled['label'] = df_sampled['sentiment'].map({'positive': 1, 'negative': 0})

''' Convert text data to TF-IDF features '''
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_sampled['cleaned_text'])
y = df_sampled['label']



'''Split the dataset into training and testing sets'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


'''Train a Logistic Regression model'''
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


'''Evaluate the model's performance'''
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Logistic Regression Evaluation
y_pred_lr = model.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
precision_lr, recall_lr, f1_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='binary')

print(f"Accuracy: {acc_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print(f"F1-score: {f1_lr:.4f}")

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6,5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Logistic Regression - Confusion Matrix')
plt.show()


'''Train a Naive Bayes model'''
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


'''Evaluate the Naive Bayes model's performance'''
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_nb = nb_model.predict(X_test)

acc_nb = accuracy_score(y_test, y_pred_nb)
precision_nb, recall_nb, f1_nb, _ = precision_recall_fscore_support(y_test, y_pred_nb, average='binary')

print(f"Naive Bayes Classifier Results:")
print(f"Accuracy: {acc_nb:.4f}")
print(f"Precision: {precision_nb:.4f}")
print(f"Recall: {recall_nb:.4f}")
print(f"F1-score: {f1_nb:.4f}")

cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6,5))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Naive Bayes)')
plt.show()


'''Train a Support Vector Machine model'''
from sklearn.svm import LinearSVC

svm_model = LinearSVC(max_iter=5000, random_state=42)
svm_model.fit(X_train, y_train)


'''Evaluate the SVM model's performance'''
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_svm = svm_model.predict(X_test)

acc_svm = accuracy_score(y_test, y_pred_svm)
precision_svm, recall_svm, f1_svm, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='binary')

print(f"SVM Classifier Results:")
print(f"Accuracy: {acc_svm:.4f}")
print(f"Precision: {precision_svm:.4f}")
print(f"Recall: {recall_svm:.4f}")
print(f"F1-score: {f1_svm:.4f}")

cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6,5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (SVM)')
plt.show()


'''Aggregate results for all models'''
results = {
    'Logistic Regression': {'accuracy': acc_lr, 'precision': precision_lr, 'recall': recall_lr, 'f1': f1_lr},
    'Naive Bayes': {'accuracy': acc_nb, 'precision': precision_nb, 'recall': recall_nb, 'f1': f1_nb},
    'SVM': {'accuracy': acc_svm, 'precision': precision_svm, 'recall': recall_svm, 'f1': f1_svm},
}

import pandas as pd
metrics_df = pd.DataFrame(results).T  # transpose for easier plotting
metrics_df


'''Plotting the performance metrics of all models'''
import matplotlib.pyplot as plt
import seaborn as sns

metrics_df.plot(kind='bar', figsize=(12,7))
plt.title('Comparison of Models on Sentiment Classification')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.grid(axis='y')
plt.show()


'''testing '''
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\r\n|\n|\t', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

''' Function to predict sentiment using the classical model '''
def predict_classical_model(review_text, model, tfidf_vectorizer):
    cleaned = preprocess_text(review_text)
    vectorized = tfidf_vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    sentiment = 'positive' if pred == 1 else 'negative'
    return sentiment

''' Function to test the model with user input '''
def test_review_input():
    review = input("Enter your review text:\n")

    print("Choose model to test:")
    print("1 - Logistic Regression")
    print("2 - Naive Bayes")
    print("3 - SVM")

    choice = input("Enter choice number: ").strip()

    if choice == '1':
        sentiment = predict_classical_model(review, model, tfidf)
    elif choice == '2':
        sentiment = predict_classical_model(review, nb_model, tfidf)
    elif choice == '3':
        sentiment = predict_classical_model(review, svm_model, tfidf)
    else:
        print("Invalid choice!")
        return

    print(f"Predicted sentiment: {sentiment}")


test_review_input() 


