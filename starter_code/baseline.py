import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def run_baseline():
    print("Loading data...")
    # Load training data
    train_path = os.path.join('..', 'data', 'train.csv')
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run generate_data.py first.")
        return

    train = pd.read_csv(train_path)
    
    # Prepare features and target
    # distinct from id
    features = [c for c in train.columns if c not in ['id', 'target']]
    X = train[features]
    y = train['target']
    
    # Internal validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest baseline...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Validation score
    val_preds = clf.predict(X_val)
    val_score = f1_score(y_val, val_preds, average='macro')
    print(f"Internal Validation F1 Score: {val_score:.4f}")
    
    # Retrain on full training set for submission
    print("Retraining on full training set...")
    clf.fit(X, y)
    
    # Predict on Test
    print("Predicting on test set...")
    test_path = os.path.join('..', 'data', 'test.csv')
    test = pd.read_csv(test_path)
    X_test = test[features]
    test_preds = clf.predict(X_test)
    
    # Save Submission
    os.makedirs('../submissions', exist_ok=True)
    submission_path = os.path.join('..', 'submissions', 'sample_submission.csv')
    
    submission_df = pd.DataFrame({
        'id': test['id'],
        'target': test_preds
    })
    
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    run_baseline()
