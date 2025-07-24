Restaurant Recommendation System Documentation

Objective
The goal of this project is to build a recommendation engine that predicts which restaurants (vendors) a customer is likely to order from, based on historical order data, customer profiles, and vendor characteristics.

Input Datasets
File Name	Description
orders.csv	Past orders including customer, vendor, location, totals, etc.
train_customers.csv	Customer demographic info (gender, language)
train_locations.csv	Locations where each customer placed orders
vendors.csv	Vendor metadata including tag name
test_customers.csv	Customers for which predictions are required
test_locations.csv	Customer locations for prediction

Step-by-Step Process
1. Data Loading and Preprocessing
Loaded all CSVs using pandas.
Checked and handled mixed data types, missing values.
Merged orders.csv with train_customers.csv, train_locations.csv, and vendors.csv.
2. Feature Engineering
Created aggregated features for each (customer_id, location_number, vendor_id): - order_count: Number of past orders - avg_total: Average grand_total spent - avg_rating: Average vendor rating - avg_distance: Average delivery distance - is_favorite: Max value (0 or 1) from past orders - Joined with categorical features: gender, language, vendor_tag_name
3. Encoding Categorical Variables
Used LabelEncoder to transform gender, language, and vendor_tag_name into numeric format.
Filled missing values with “unknown” before encoding.
4. Numeric Conversion & Cleaning
Converted all numerical features to float type using pd.to_numeric(..., errors='coerce').
Filled any remaining NaNs with 0.0.
5. Model Training
Used XGBClassifier from XGBoost.
Input features: order_count, avg_total, avg_rating, avg_distance, is_favorite, encoded gender, language, and vendor_tag_name
Model was trained on binary target (1 = preferred vendor, 0 = others).
6. Generating Test Set (Fast Cartesian Product)
Created combinations of (customer_id, location_number) and all vendor_ids using pandas merge().
Joined with test_customers.csv and vendors.csv to populate necessary features.
Filled features with default values (0.0) where historical data is unavailable.
7. Prediction
Used model.predict_proba(X_test) to generate prediction scores.
Selected top vendor per (customer_id, location_number) by sorting by predicted probability.
8. Final Submission File
Created a column cid_x_loc_x_vendor in format customer_id X location_number X vendor_id
Saved top predictions to final_submission.csv

Final Output Format
cid_x_loc_x_vendor	score
10001 X 1 X 2009	0.9483
10001 X 2 X 1015	0.9145

Tools Used
Python 3.10+
pandas
numpy
scikit-learn (LabelEncoder)
xgboost
tqdm (for progress bars)

Notes
All intermediate steps were optimized for performance.
Cleaned and typecasted all data to avoid KeyError, ValueError, and object dtype issues in XGBoost.
Output format strictly follows the assignment PDF specification.

Author
This solution was implemented by Upendra Challa.
