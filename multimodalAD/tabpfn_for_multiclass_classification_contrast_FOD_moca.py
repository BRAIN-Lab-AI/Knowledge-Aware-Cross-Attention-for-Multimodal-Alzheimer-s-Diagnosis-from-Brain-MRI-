# %%
# TabPFN script adapted for ALBEF Multimodal Features (Tabular + Image)

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from pathlib import Path
import shap 
import json 

# --- IMPORTANT PATH VARIABLES ---
# *** ACTION REQUIRED: UPDATE THIS PATH ***
ALBEF_FEATURE_PATH = '/content/drive/MyDrive/PhD KFUPM/Deep Learning/Term paper/Submission/multimodalAD/ALBEF/output/train_ADNI_2025-11-23_21-00-51' 

# Path to your main tabular data file
CSV_FILE_PATH = '/content/drive/MyDrive/PhD KFUPM/Deep Learning/Term paper/MOCA_data_18.csv'
# ---------------------------------

# --- 1. Load Tabular Data and Split ---

df = pd.read_csv(CSV_FILE_PATH)
train_data, temp_data = train_test_split(df, test_size=(1 - 0.6), random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

data_train = train_data.copy()
extra_rows = []
for _, one_data in train_data.iterrows():
    if one_data['Diagnosis'] == 2:
        extra_rows.append(one_data)
    elif one_data['Diagnosis'] == 3:
        extra_rows.extend([one_data] * 3)

if extra_rows:
    extra_df = pd.DataFrame(extra_rows)
    data_train = pd.concat([data_train, extra_df], ignore_index=True)

data_val = validation_data.copy()
data_test = test_data.copy()

print(f"Tabular Train Shape (Augmented): {data_train.shape}")


# --- 2. Load Multimodal (ALBEF) Features ---

try:
    tr_features_albef = np.load(os.path.join(ALBEF_FEATURE_PATH, 'tr_feat_fusioned.npz'))['fusioned']
    val_features_albef = np.load(os.path.join(ALBEF_FEATURE_PATH, 'val_feat_fusioned.npz'))['fusioned']
    test_features_albef = np.load(os.path.join(ALBEF_FEATURE_PATH, 'test_feat_fusioned.npz'))['fusioned']

    tr_diagnosis_albef = np.load(os.path.join(ALBEF_FEATURE_PATH, 'tr_label.npz'))['label']
    val_diagnosis_albef = np.load(os.path.join(ALBEF_FEATURE_PATH, 'val_label.npz'))['label']
    test_diagnosis_albef = np.load(os.path.join(ALBEF_FEATURE_PATH, 'test_label.npz'))['label']
    
except FileNotFoundError as e:
    print(f"\n--- CRITICAL ERROR: ALBEF Feature files not found. ---")
    print(f"Please ensure {ALBEF_FEATURE_PATH} is correct and rerun train_ADNI.py.")
    sys.exit(1)

# Combine Val and Test ALBEF features/labels
test_features_albef = np.concatenate([val_features_albef, test_features_albef], axis=0)
test_diagnosis_albef = np.concatenate([val_diagnosis_albef, test_diagnosis_albef], axis=0)

# Re-augment the ALBEF features to match the tabular data augmentation
def re_augment_albef_features(features, labels, original_data_train):
    
    num_features = len(features)
    num_original_rows = len(original_data_train)
    
    if num_features != num_original_rows:
        print(f"WARNING: Image features ({num_features}) and Tabular rows ({num_original_rows}) size mismatch.")
    
    extra_features = []
    
    for idx in range(num_features):
        try:
            row_diagnosis = original_data_train.iloc[idx]['Diagnosis']
            feature_row = features[idx]
            
            if row_diagnosis == 2:
                extra_features.append(feature_row)
            elif row_diagnosis == 3:
                extra_features.extend([feature_row] * 3)
        except IndexError:
             print(f"Warning: Skipping index {idx} during feature matching.")
             continue

    if extra_features:
        features_augmented = np.vstack([features, np.array(extra_features)])
        
        augmented_labels = []
        for idx in range(num_features):
            row_diagnosis = original_data_train.iloc[idx]['Diagnosis']
            augmented_labels.append(labels[idx])
            
            if row_diagnosis == 2:
                augmented_labels.append(labels[idx])
            elif row_diagnosis == 3:
                augmented_labels.extend([labels[idx]] * 3)
        
        labels_augmented = np.array(augmented_labels)
    else:
        features_augmented = features
        labels_augmented = labels

    return features_augmented, labels_augmented

# Re-augment ALBEF features for training set
tr_features_albef, tr_diagnosis_albef = re_augment_albef_features(tr_features_albef, tr_diagnosis_albef, train_data)


# --- 4. Final Multimodal Data Preparation ---

def preprocess_tabular_data(df_tab, df_albef_features, expected_diagnosis):
    
    # 1. Map 'Sex' to numbers and select numerical columns
    df_tab['Sex'] = df_tab['Sex'].map({'M': 0, 'F': 1})
    df_tab_numeric = df_tab.select_dtypes(include=['number']).drop(columns=['Diagnosis'], errors='ignore')

    num_albef_rows = df_albef_features.shape[0]
    
    # CRITICAL: Truncate tabular data to match the available image features
    if len(df_tab_numeric) > num_albef_rows:
        df_tab_numeric = df_tab_numeric.iloc[:num_albef_rows].reset_index(drop=True)
    elif len(df_tab_numeric) < num_albef_rows:
        print("CRITICAL: Tabular data is unexpectedly smaller than ALBEF features. Padding with NaN.")
        
    # 2. Combine features (ALBEF features are DataFrame columns 7 onwards)
    df_albef_features_df = pd.DataFrame(df_albef_features, index=df_tab_numeric.index)
    df_combined = pd.concat([df_tab_numeric, df_albef_features_df], axis=1)
    
    # 3. Handle NaNs
    nan_mask = np.sum(df_combined.isna().to_numpy(), axis=1) <= 5
    
    df_combined_filtered = df_combined[nan_mask].reset_index(drop=True)
    df_combined_filtered = df_combined_filtered.ffill()
    
    # Filter the diagnosis labels based on the same mask
    y_filtered = expected_diagnosis[nan_mask]

    # Final data conversion
    X = df_combined_filtered.to_numpy()
    
    # 4. Get Feature Names (FIXED LOGIC)
    feature_names = df_combined_filtered.columns.tolist()

    return X, y_filtered, feature_names

# Get combined, clean datasets
X_train, y_train, train_feature_names = preprocess_tabular_data(data_train, tr_features_albef, tr_diagnosis_albef)
# For the test data, concatenate val and test tabular data first, then merge features
test_tabular_data = pd.concat([data_val, data_test], ignore_index=True)
X_test, y_test, test_feature_names = preprocess_tabular_data(test_tabular_data, test_features_albef, test_diagnosis_albef)


print(f"\nFinal Training Data Shape (X, y): {X_train.shape}, {y_train.shape}")
print(f"Final Testing Data Shape (X, y): {X_test.shape}, {y_test.shape}")

# --- 5. Standardize/Scale Features (Required for TabPFN) ---

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 6. TabPFN Classification ---

print("\n--- Running TabPFN Classification ---")

TabPFNclf = TabPFNClassifier(ignore_pretraining_limits=True, device='cuda') 
TabPFNclf.fit(X_train, y_train)

# Predict labels
predictions = TabPFNclf.predict(X_test)
prediction_probabilities = TabPFNclf.predict_proba(X_test)

# --- 7. Report Metrics ---

print("\n--- PERFORMANCE METRICS (TabPFN) ---")

print("Accuracy", accuracy_score(y_test, predictions))
conf_matrix = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:\n", conf_matrix)
print("ROC AUC (OVR):", roc_auc_score(y_test, prediction_probabilities, multi_class="ovr"))

y_true = y_test
y_pred = predictions
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

print("\nDetailed Multi-class Metrics:")
print("Precision (per class):", precision)
print("Recall (per class):", recall)
print("F1-score (per class):", f1)
print("Balanced Accuracy:", balanced_acc)
print("Matthews Correlation Coefficient (MCC):", mcc)


# --- 8. SHAP ANALYSIS (FINAL FIX FOR INDEXING) ---
try:
    import shap
    print("\n--- Starting SHAP Feature Importance Analysis (KernelExplainer) ---")

    # The TabPFN model was trained on 771 features. We MUST give the explainer a function 
    # that handles 771 features, even if SHAP only varies the first N features.
    
    # 1. Define the number of features to explain (the most important ones)
    N_SHAP_FEATURES = 10 
    
    # 2. Define the background data using ONLY the features we care about (first 10)
    X_train_shap = X_train[:, :N_SHAP_FEATURES]
    X_test_shap = X_test[:, :N_SHAP_FEATURES]
    
    N_BACKGROUND = min(50, X_train_shap.shape[0]) 
    if X_train_shap.shape[0] < N_BACKGROUND:
        N_BACKGROUND = X_train_shap.shape[0]

    background_indices = np.random.choice(X_train_shap.shape[0], N_BACKGROUND, replace=False)
    background = X_train_shap[background_indices]

    # 3. Create a wrapper function that takes the 10 SHAP features and pads them 
    #    back up to 771 features (with zeros) for the TabPFN model to process.
    def shap_model_wrapper(X_input_10_features):
        # X_input_10_features has shape (M, 10), where M is N_BACKGROUND or N_TEST
        
        # Create a zero array of the full size (M, 771)
        X_full = np.zeros((X_input_10_features.shape[0], X_train.shape[1]))
        
        # Paste the 10 features back into the start of the array
        X_full[:, :N_SHAP_FEATURES] = X_input_10_features
        
        # Call the actual TabPFN model (which expects 771 features)
        return TabPFNclf.predict_proba(X_full)

    # 4. Initialize KernelExplainer with the wrapper function and the 10-feature background
    explainer = shap.KernelExplainer(shap_model_wrapper, background)

    # 5. Calculate SHAP values using the 10-feature test data
    shap_values_raw = explainer.shap_values(X_test_shap) 
    
    # --- Visualization Data Saving ---
    
    RUNS_DIR = os.path.join(ALBEF_FEATURE_PATH, 'shap_runs')
    Path(RUNS_DIR).mkdir(exist_ok=True)
    
    np.save(os.path.join(RUNS_DIR, 'shap_values_raw.npy'), np.array(shap_values_raw, dtype=object))
    np.save(os.path.join(RUNS_DIR, 'shap_test_data.npy'), X_test_shap)
    
    # Feature names must correspond to the 10 features used
    truncated_feature_names = train_feature_names[:N_SHAP_FEATURES]
    with open(os.path.join(RUNS_DIR, 'feature_names.txt'), 'w') as f:
        json.dump(truncated_feature_names, f)

    print(f"\nSHAP analysis arrays saved to {RUNS_DIR}")
    print("NOTE: SHAP calculation is now running using the model wrapper.")


except ImportError:
    print("\n--- SHAP Analysis Skipped ---")
    print("Please install the SHAP library ('!pip install shap') to enable feature importance analysis.")