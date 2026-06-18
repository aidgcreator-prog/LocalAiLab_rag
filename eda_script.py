import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

file_path = '/test_data.csv'
output_dir = '/generated_plots/'
log_file = '/eda_results.txt'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    with open(log_file, 'w') as f:
        df = pd.read_csv(file_path)
        
        f.write("--- DATA OVERVIEW ---\n")
        f.write("First 5 rows:\n")
        f.write(df.head().to_string())
        f.write("\n\nSummary Information:\n")
        f.write(str(df.info()))
        
        num_cols = ['Age', 'Income', 'Credit_Score', 'Employment_Years', 'Loan_Amount', 'Monthly_Payment']
        existing_num_cols = [col for col in num_cols if col in df.columns]
        
        f.write("\n\n--- DESCRIPTIVE STATISTICS ---\n")
        stats = df[existing_num_cols].describe().loc[['mean', '50%', 'std', 'min', 'max']]
        stats.rename(index={'50%': 'median'}, inplace=True)
        f.write(stats.to_string())
        
        f.write("\n\n--- CORRELATION MATRIX ---\n")
        correlation_matrix = df[existing_num_cols].corr()
        f.write(correlation_matrix.to_string())
        
        f.write("\n\n--- GENERATING VISUALIZATIONS ---\n")
        sns.set_theme(style="whitegrid")
        
        if 'Age' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['Age'], kde=True, color='skyblue')
            plt.title('Distribution of Age')
            plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
            plt.close()
            f.write("Saved: age_distribution.png\n")
        
        if 'Income' in df.columns and 'Loan_Amount' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='Income', y='Loan_Amount', alpha=0.6)
            plt.title('Income vs Loan Amount')
            plt.savefig(os.path.join(output_dir, 'income_vs_loan_amount.png'))
            plt.close()
            f.write("Saved: income_vs_loan_amount.png\n")
            
        if 'Credit_Score' in df.columns and 'Loan_Approved' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='Loan_Approved', y='Credit_Score', palette='Set2')
            plt.title('Credit Score by Loan Approval Status')
            plt.savefig(os.path.join(output_dir, 'credit_score_by_approval.png'))
            plt.close()
            f.write("Saved: credit_score_by_approval.png\n")

        f.write("\n--- ANALYSIS COMPLETE ---")

except Exception as e:
    with open(log_file, 'a') as f_err:
        f_err.write(f"\nAn error occurred: {str(e)}")
