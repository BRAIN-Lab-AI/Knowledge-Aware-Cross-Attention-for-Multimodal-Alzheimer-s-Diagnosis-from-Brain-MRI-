import os
import json
import numpy as np

# Define paths
base_path = '/content/drive/MyDrive/PhD KFUPM/Deep Learning/Term paper/Submission/multimodalAD/ALBEF/data/'
json_path = os.path.join(base_path, 'ADNI.json')

# Ensure directory exists
if not os.path.exists(base_path):
    os.makedirs(base_path, exist_ok=True)

# Load data
with open(json_path, 'r') as f:
    data = json.load(f)

# --- MODIFIED SPLIT: 60% Train, 20% Val, 20% Test ---
total_len = len(data)
train_end = int(total_len * 0.6)
val_end = int(total_len * 0.8)

train = data[:train_end]
# Augmentation for Train
# for each cdr = 0.5, duplicate 1 time
train += [per for per in train if per['cdr'] == 0.5]
# for each cdr >= 1, duplicate 5 times
train += [per for per in train if per['cdr'] >= 1.] * 5

val = data[train_end:val_end]
test = data[val_end:]
# ----------------------------------------------------

# Subject ID extraction for overlap check
train_subs = set([per['mri'].split('/')[-3] for per in train])
val_subs = set([per['mri'].split('/')[-3] for per in val])
test_subs = set([per['mri'].split('/')[-3] for per in test])

# make sure no overlap
# Note: If this fails, your ADNI.json isn't sorted by subject. 
# You might need to shuffle/group by subject before splitting.
try:
    assert len(train_subs.intersection(val_subs)) == 0
    assert len(train_subs.intersection(test_subs)) == 0
    assert len(val_subs.intersection(test_subs)) == 0
    print("✅ Overlap check passed.")
except AssertionError:
    print("❌ Assertion Failed: Data leakage detected! Subjects appear in multiple splits.")
    print(f"Train-Val overlap: {len(train_subs.intersection(val_subs))}")
    print(f"Train-Test overlap: {len(train_subs.intersection(test_subs))}")
    print(f"Val-Test overlap: {len(val_subs.intersection(test_subs))}")

print(len(train), len(val), len(test))

train_cdr = {0.: 0, 0.5: 0, 1.: 0}
val_cdr = {0.: 0, 0.5: 0, 1.: 0}
test_cdr = {0.: 0, 0.5: 0, 1.: 0}

for per in train:
    cdr = float(per['cdr'])
    cdr = cdr if cdr < 1. else 1.
    train_cdr[cdr] += 1

for per in val:
    cdr = float(per['cdr'])
    cdr = cdr if cdr < 1. else 1.
    val_cdr[cdr] += 1

for per in test:
    cdr = float(per['cdr'])
    cdr = cdr if cdr < 1. else 1.
    test_cdr[cdr] += 1

print("Train CDR:", train_cdr)
print("Val CDR:", val_cdr)
print("Test CDR:", test_cdr)

# save to json
train_path = os.path.join(base_path, 'ADNI_train.json')
val_path = os.path.join(base_path, 'ADNI_val.json')
test_path = os.path.join(base_path, 'ADNI_test.json')

with open(train_path, 'w') as f:
    json.dump(train, f)

with open(val_path, 'w') as f:
    json.dump(val, f)

with open(test_path, 'w') as f:
    json.dump(test, f)

print(f"Files saved to {base_path}")