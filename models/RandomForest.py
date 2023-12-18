import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from google.colab import drive
drive.mount('/content/drive')
drive.mount('/content/drive')
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
rf_model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search_rf.fit(X_train_tfidf, y_train_encoded)
best_params_rf = grid_search_rf.best_params_
print(f"Best Hyperparameters: {best_params_rf}")
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test_tfidf)
y_pred_original_rf = label_encoder.inverse_transform(y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_original_rf)
precision_rf = precision_score(y_test, y_pred_original_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_original_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_original_rf, average='weighted')
print(f"\nRandom Forest Metrics:")
print(f"Accuracy: {accuracy_rf * 100:.2f}%")
print(f"Precision: {precision_rf:.2f}")
print(f"Recall: {recall_rf:.2f}")
print(f"F1 Score: {f1_rf:.2f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_original_rf))