import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from google.colab import drive
drive.mount('/content/drive')
# Load data
train_data = pd.read_excel('/content/drive/MyDrive/Dataset/train.xlsx')
test_data = pd.read_excel('/content/drive/MyDrive/Dataset/test.xlsx')
X_train = train_data['Sentence']
y_train = train_data['Emotion']
X_test = test_data['Sentence']
y_test = test_data['Emotion']

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), seed=42)
param_grid = {
    'max_depth': [ 5],
    'learning_rate': [0.1],
    'n_estimators': [250,300,350,400],
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train_encoded)
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")
best_xgb_model = grid_search.best_estimator_
y_pred = best_xgb_model.predict(X_test_tfidf)
y_pred_original = label_encoder.inverse_transform(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_original)
precision = precision_score(y_test, y_pred_original, average='weighted')
recall = recall_score(y_test, y_pred_original, average='weighted')
f1 = f1_score(y_test, y_pred_original, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Additional information with classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_original))
