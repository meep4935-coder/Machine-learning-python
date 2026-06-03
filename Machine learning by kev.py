"""Created June 2nd 2026 by @author Kev.Wo
Machine learning program in service for ISED CTCGB
This program consist of two algorithms that sort and list given data
For more information contact branch manager"""

import joblib
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# =============================================================================
# CLEAN TECH ML PIPELINE 
# =============================================================================
# This script runs a two-stage machine learning classification pipeline:
#
#   Stage 1 — Is this project Clean Tech or Not Clean Tech?
#             Uses a trained model (threshold_code.pkl)
#             Only labels a project "Clean Tech" if confidence >= 75%
#
#   Stage 2 — What sub-sector does this Clean Tech project belong to?
#             Uses a second trained model (subsector.final.pkl)
#             Only assigns a sub-sector if confidence >= 25%
#             Otherwise labels it "other"
#
# INPUT:  An Excel file with a column for project description and/or project title
# OUTPUT: An Excel file with predicted labels and confidence scores added
#
# credits: RuggeroG's trained models (threshold & subsector)
# =============================================================================

#below configure the path directory to your own locally saved data

# Your new central folder path:
folder_path = r"C:\Users\WongK2\OneDrive - ISED-ISDE\Desktop\ML python"

# The spreadsheet you want to classify (Update filename if different):
input_excel = rf"{folder_path}\Translated_data_TEST.xlsx" 

# Where the final classified results will save:
output_excel = rf"{folder_path}\PythonML_results.xlsx"

#below is the algorithm, kindly do not edit unless authorized

#trained model files sitting in the folder:
stage1_model_path = rf"{folder_path}\threshold_code.pkl"
stage2_model_path = rf"{folder_path}\subsector.final.pkl"

 

print("Loading AI models from project folder...")
stage1_model = joblib.load(stage1_model_path)
stage2_model = joblib.load(stage2_model_path)

print("Loading dataset to classify...")
dataset = pd.read_excel(input_excel)

# Ensure text columns are clean strings
dataset['project title'] = dataset['project title'].fillna('').astype(str)
dataset['project description'] = dataset['project description'].fillna('').astype(str)
dataset['full project words'] = dataset['project title'] + " " + dataset['project description']


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

print("Preprocessing project text fields...")
processed_texts = dataset['full project words'].apply(preprocess_text)

print("Classifying data records via Two-Stage Pipeline...")

final_categories = []
stage1_confidences = []
stage2_confidences = []

for text in processed_texts:
    # --- STAGE 1: Clean Tech vs Not Clean Tech ---
    s1_probs = stage1_model.predict_proba([text])[0]
    s1_classes = list(stage1_model.classes_)
    
    s1_max_idx = np.argmax(s1_probs)
    s1_label = s1_classes[s1_max_idx]
    s1_conf = s1_probs[s1_max_idx]
    
    stage1_confidences.append(s1_conf)
    
    # Apply strict 0.75 threshold restriction
    if s1_label == "Not Clean Tech" or s1_conf < 0.75:
        final_categories.append("Not Clean Tech")
        stage2_confidences.append(np.nan) 
    else:
        # --- STAGE 2: Specific Sub-Sector Specialist ---
        s2_probs = stage2_model.predict_proba([text])[0]
        s2_classes = list(stage2_model.classes_)
        
        s2_max_idx = np.argmax(s2_probs)
        s2_label = s2_classes[s2_max_idx]
        s2_conf = s2_probs[s2_max_idx]
        
        final_categories.append(s2_label)
        stage2_confidences.append(s2_conf)

dataset['AI_Predicted_Subsector'] = final_categories
dataset['Stage1_Filter_Confidence'] = stage1_confidences
dataset['Stage2_Subsector_Confidence'] = stage2_confidences

if 'full project words' in dataset.columns:
    dataset = dataset.drop(columns=['full project words'])

dataset.to_excel(output_excel, index=False)
print(f"\nPipeline Successful! Your classified file is saved at:\n{output_excel}")