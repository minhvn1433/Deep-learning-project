import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

from google.colab import drive
drive.mount('/content/drive')

#load data
train_data = pd.read_excel('/content/drive/MyDrive/Dataset/train.xlsx')
test_data = pd.read_excel('/content/drive/MyDrive/Dataset/test.xlsx')
X_train = train_data['Sentence']
y_train = train_data['Emotion']
X_test = test_data['Sentence']
y_test = test_data['Emotion']

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
#Grid search
svm_model = SVC()
param_grid = {
    'C': [5,10,20],
    'kernel': [ 'rbf'],
}

grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search_svm.fit(X_train_tfidf, y_train_encoded)
best_params_svm = grid_search_svm.best_params_
print(f"Best Hyperparameters: {best_params_svm}")
best_svm_model = grid_search_svm.best_estimator_
y_pred_svm = best_svm_model.predict(X_test_tfidf)
y_pred_original_svm = label_encoder.inverse_transform(y_pred_svm)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_original_svm)
precision_svm = precision_score(y_test, y_pred_original_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_original_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_original_svm, average='weighted')

print(f"\nSVM Metrics:")
print(f"Accuracy: {accuracy_svm * 100:.2f}%")
print(f"Precision: {precision_svm:.2f}")
print(f"Recall: {recall_svm:.2f}")
print(f"F1 Score: {f1_svm:.2f}")

# Additional information with classification report
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_original_svm))