import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# === Step 1: Load and clean training data ===
train_data = pd.read_csv("prepared_training_data.csv")

# Fill missing and unknown values
for col in ['gender', 'language', 'vendor_tag_name']:
    if col not in train_data.columns:
        train_data[col] = 'unknown'
train_data = train_data.fillna('unknown')

# Encode categorical columns
le_gender = LabelEncoder()
le_language = LabelEncoder()
le_vendor_tag = LabelEncoder()

train_data['gender'] = le_gender.fit_transform(train_data['gender'].astype(str))
train_data['language'] = le_language.fit_transform(train_data['language'].astype(str))
train_data['vendor_tag_name'] = le_vendor_tag.fit_transform(train_data['vendor_tag_name'].astype(str))

# Convert numeric columns safely to float
for col in ['order_count', 'avg_total', 'avg_rating', 'avg_distance', 'is_favorite']:
    train_data[col] = pd.to_numeric(train_data[col], errors='coerce').fillna(0.0)

# Features for model
feature_cols = ['order_count', 'avg_total', 'avg_rating', 'avg_distance',
                'is_favorite', 'gender', 'language', 'vendor_tag_name']

X_train = train_data[feature_cols]
y_train = train_data['label']

# === Step 2: Train the model ===
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# === Step 3: Load test data ===
test_customers = pd.read_csv("test_customers.csv")
test_locations = pd.read_csv("test_locations.csv")
vendors = pd.read_csv("vendors.csv")

# Prepare vendor_tag_name
if 'vendor_tag_name' not in vendors.columns:
    vendors['vendor_tag_name'] = 'unknown'
vendors['vendor_tag_name'] = vendors['vendor_tag_name'].fillna('unknown')
vendors['vendor_tag_name'] = le_vendor_tag.transform(vendors['vendor_tag_name'].astype(str))

# Merge customers and locations
test_pairs = pd.merge(test_customers, test_locations, on='customer_id')
test_pairs['gender'] = le_gender.transform(['unknown'] * len(test_pairs))
test_pairs['language'] = le_language.transform(['unknown'] * len(test_pairs))

# === Step 4: Generate test combinations FAST ===
print("⚡ Generating test combinations (fast)...")
test_pairs['key'] = 1
vendors['key'] = 1
test_df = pd.merge(test_pairs, vendors[['id', 'vendor_tag_name', 'key']], on='key').drop(columns='key')
test_df.rename(columns={'id': 'vendor_id'}, inplace=True)

# Add default feature values
test_df['order_count'] = 0.0
test_df['avg_total'] = 0.0
test_df['avg_rating'] = 0.0
test_df['avg_distance'] = 0.0
test_df['is_favorite'] = 0.0

# Make sure all numeric test features are float
for col in ['order_count', 'avg_total', 'avg_rating', 'avg_distance', 'is_favorite']:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0.0)

# === Step 5: Predict scores ===
X_test = test_df[feature_cols]
print("🔍 Predicting best vendors per location...")
test_df['score'] = model.predict_proba(X_test)[:, 1]

# === Step 6: Get top-1 vendor per customer-location ===
test_df['cid_loc'] = test_df['customer_id'].astype(str) + ' X ' + test_df['location_number'].astype(str)
top_df = test_df.sort_values(by=['cid_loc', 'score'], ascending=[True, False])
top1_df = top_df.groupby('cid_loc').first().reset_index()

# === Step 7: Save final submission ===
top1_df['cid_x_loc_x_vendor'] = top1_df.apply(
    lambda x: f"{x['customer_id']} X {x['location_number']} X {x['vendor_id']}", axis=1
)
submission = top1_df[['cid_x_loc_x_vendor', 'score']]
submission.to_csv("final_submission.csv", index=False)

print("✅ final_submission.csv generated successfully!")
