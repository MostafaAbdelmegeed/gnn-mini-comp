import pandas as pd
from sklearn.metrics import f1_score
import sys
import os

def score_submission(submission_file):
    print(f"Scoring {submission_file}...")
    
    try:
        submission = pd.read_csv(submission_file)
    except Exception as e:
        print(f"Error reading submission file: {e}")
        return

    # Load Ground Truth
    truth_path = os.path.join('data', 'test_labels.csv')
    if not os.path.exists(truth_path):
        print(f"Error: {truth_path} not found.")
        return
        
    truth = pd.read_csv(truth_path)
    
    # Sort both to ensure alignment by ID
    submission = submission.sort_values('id').reset_index(drop=True)
    truth = truth.sort_values('id').reset_index(drop=True)
    
    # Check if IDs match
    if not submission['id'].equals(truth['id']):
        print("Error: Submission IDs do not match test set IDs.")
        return

    # Calculate Score
    score = f1_score(truth['target'], submission['target'], average='macro')
    print(f"FINAL SCORE (Macro F1): {score:.4f}")
    return score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scoring_script.py <path_to_submission.csv>")
        # Default for testing
        default_sub = os.path.join('submissions', 'sample_submission.csv')
        if os.path.exists(default_sub):
            print(f"No argument provided. Using default: {default_sub}")
            score_submission(default_sub)
        else:
            sys.exit(1)
    else:
        score_submission(sys.argv[1])
