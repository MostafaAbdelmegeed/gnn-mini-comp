import pandas as pd
import os
import glob
from scoring_script import score_submission
from datetime import datetime

def update_leaderboard():
    submission_files = glob.glob(os.path.join('submissions', '*.csv'))
    results = []
    
    print(f"Found {len(submission_files)} submissions.")
    
    for sub in submission_files:
        basename = os.path.basename(sub)
        print(f"Processing {basename}...")
        try:
            score = score_submission(sub)
            if score is not None:
                # Use file modification time as date
                mod_time = os.path.getmtime(sub)
                date_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
                results.append({
                    'User/Submission': basename,
                    'Macro F1 Score': score,
                    'Date': date_str
                })
        except Exception as e:
            print(f"Failed to score {basename}: {e}")

    # Sort by Score (Desc)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='Macro F1 Score', ascending=False).reset_index(drop=True)
        df.index += 1 # Rank starts at 1
        df.index.name = 'Rank'
        df = df.reset_index()
        
        # Format markdown table
        markdown_table = df.to_markdown(index=False, floatfmt=".4f")
    else:
        markdown_table = "No valid submissions found."

    # Update LEADERBOARD.md
    with open('LEADERBOARD.md', 'w') as f:
        f.write("# GNN Challenge Leaderboard\n\n")
        f.write(markdown_table)
    
    print("LEADERBOARD.md updated.")

if __name__ == "__main__":
    update_leaderboard()
