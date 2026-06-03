import joblib
import sklearn
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# 1. Load the dataset
dataset = pd.read_excel(r"C:\Users\WongK2\Downloads\Reviewed_Investment_Database_2025-03-11.xlsx")

# Drop rows with empty descriptions or empty sub-sectors
dataset = dataset.dropna(subset=['project description', 'sub sector review'])

# 2. Filter dataset for Stage 2 (Clean Tech sub-sectors ONLY)
clean_tech_labels = [
    'air / environment / remediation', 'other', 'biofuels / bioenergy / bioproducts', 
    'energy efficiency', 'mining / processing / materials / manufacturing / industry', 
    'precision agriculture / forestry / biodiversity', 'renewable / non-emitting energy', 
    'smart grid / energy storage', 'transportation', 'waste and recycling', 'water and waste water'
]
dataset = dataset[dataset['sub sector review'].isin(clean_tech_labels)]
print(f"Loaded dataset cleanly with {len(dataset)} Clean Tech rows for Stage 2.")

# 3. Combine project title and description safely
dataset['project title'] = dataset['project title'].fillna('').astype(str)
dataset['project description'] = dataset['project description'].fillna('').astype(str)
dataset['full project words'] = dataset['project title'] + " " + dataset['project description']

# 4. Processing function for data set 
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

print("Cleaning data text fields...")
dataset['full project words'] = dataset['full project words'].apply(preprocess_text)

# 5. Split the dataset directly in memory (No slow Excel writing/reading mid-script)
xtrain, xtest, ytrain, ytest = train_test_split(
    dataset['full project words'],
    dataset['sub sector review'],
    test_size=0.20,
    random_state=42
)

# 6. Bundle everything into a unified Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words=None)), # stop_words=None since custom cleaning already handled it
    ('clf', MLPClassifier(
        hidden_layer_sizes=(128,),
        activation='relu',
        solver='adam',
        alpha=0.01,
        learning_rate='constant',
        batch_size=256,       # Optimized for speed
        max_iter=30,          # Optimized iterations
        random_state=42,
        verbose=True          # Tracks live iteration scores
    ))
])

print("Training Stage 2 Sub-Sector Neural Network...")
pipeline.fit(xtrain, ytrain)

# 7. Evaluate and report metrics
y_pred_class = pipeline.predict(xtest)

print("\n--- Stage 2 Model Classification Report ---")
print(classification_report(ytest, y_pred_class))

# 8. Save the final Stage 2 model file
subsector_file = r"C:\Users\WongK2\Downloads\subsector.final.pkl"
joblib.dump(pipeline, subsector_file)
print(f"\nSuccess! Stage 2 file successfully generated and stored at:\n{subsector_file}")

# 9. Confidence level calculation and results export
y_pred_prob = pipeline.predict_proba(xtest)
classes_list = list(pipeline.classes_)

results = []
for i in range(len(y_pred_class)):
    predicted_label = y_pred_class[i]
    class_index = classes_list.index(predicted_label)
    confidence_score = y_pred_prob[i][class_index]
    
    sample_result = {
        'Project Description': xtest.iloc[i], # .iloc prevents index key errors
        'Sub Sector': ytest.iloc[i],
        'Predicted': predicted_label,
        'Confidence': f'{confidence_score:.4f}'
    }
    results.append(sample_result)
    
results_df = pd.DataFrame(results)
output_excel_path = r"C:\Users\WongK2\Downloads\subsector_test_results.xlsx"
results_df.to_excel(output_excel_path, index=False)
print(f"Test prediction spreadsheet exported to: {output_excel_path}")

# 10. Draw Confusion Matrix
cm = confusion_matrix(ytest, y_pred_class)
plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.title('Stage 2 Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
print("\nShowing Confusion Matrix plot. Close the window to completely finish execution.")
plt.show()