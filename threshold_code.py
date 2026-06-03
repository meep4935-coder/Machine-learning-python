import joblib
import sklearn
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# 1. Load the dataset
dataset = pd.read_excel(r"C:\Users\WongK2\Downloads\Reviewed_Investment_Database_2025-03-11.xlsx")

# FIX 1: Safely drop empty row footprints to prevent cleaning crashes
dataset = dataset.dropna(subset=['project description'])
print(f"Loaded dataset cleanly with {len(dataset)} actual rows.")

# 2. Combine project title and project description safely
dataset['project title'] = dataset['project title'].fillna('').astype(str)
dataset['project description'] = dataset['project description'].fillna('').astype(str)
dataset['full project words'] = dataset['project title'] + " " + dataset['project description']

# 3. Label mapping
clean_tech_labels = [
    'air / environment / remediation', 'other', 'biofuels / bioenergy / bioproducts', 
    'energy efficiency', 'mining / processing / materials / manufacturing / industry', 
    'precision agriculture / forestry / biodiversity', 'renewable / non-emitting energy', 
    'smart grid / energy storage', 'transportation', 'waste and recycling', 'water and waste water'
]
dataset['label'] = dataset['sub sector review'].apply(lambda x: "Clean Tech" if x in clean_tech_labels else "Not Clean Tech")

# 4. Processing function for dataset 
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

dataset['full project words'] = dataset['full project words'].apply(preprocess_text)

# 5. Split train/test sets
xtrain, xtest, ytrain, ytest = train_test_split(
    dataset['full project words'],
    dataset['label'],
    test_size=0.2,
    random_state=42
)

# 6. Pipeline configuration
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(128,),
        activation='relu',
        solver='adam',
        alpha=0.01,
        learning_rate='constant',
        batch_size=16,
        max_iter=50,
        random_state=42
    ))
])

print("Training Neural Network... (This may take a moment)")
pipeline.fit(xtrain, ytrain)

# 7. Predict & Evaluated with Kev's Threshold
def predict_with_threshold(text, threshold=0.75):
    probs = pipeline.predict_proba([text])[0]
    max_index = np.argmax(probs)
    max_prob = probs[max_index]
    predicted_label = pipeline.classes_[max_index]
    
    if max_prob < threshold:
        return 'Not Clean Tech', max_prob
    else:
        return predicted_label, max_prob

y_pred = []
for text in xtest:
    label, conf = predict_with_threshold(text, threshold=0.75)
    y_pred.append(label)

print("\n--- Model Classification Report ---")
print(classification_report(ytest, y_pred))

# FIX 2: Fixed the title syntax so it does not lock down matplotlib variables
cm = confusion_matrix(ytest, y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
print("\nShowing Confusion Matrix plot. Close the window to save the final file...")
plt.show()

# 8. Save final pkl output to your folder path
CTvsnotCT = r'C:\Users\WongK2\Downloads\threshold_code.pkl'
joblib.dump(pipeline, CTvsnotCT)
print(f"\nSuccess! File successfully generated and stored at:\n{CTvsnotCT}")