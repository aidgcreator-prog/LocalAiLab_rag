"""
Data Analysis Framework for Admission Data
This script outlines the typical analysis steps that would be performed on admission data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """
    Load and explore the admission data
    """
    # This would be the actual data loading step
    # df = pd.read_csv('admission_data.csv')
    
    # For demonstration, we'll create a sample dataset
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'GRE_Score': np.random.normal(310, 20, n_samples).astype(int),
        'GPA': np.random.normal(3.5, 0.5, n_samples),
        'Research': np.random.binomial(1, 0.4, n_samples),
        'Sex': np.random.choice(['Male', 'Female'], n_samples),
        'Admitted': np.random.binomial(1, 0.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    df['GRE_Score'] = np.clip(df['GRE_Score'], 250, 350)
    
    print("Dataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df

def analyze_distributions(df):
    """
    Analyze distributions of key variables
    """
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # GRE Score distribution
    axes[0,0].hist(df['GRE_Score'], bins=20, alpha=0.7, color='skyblue')
    axes[0,0].set_title('GRE Score Distribution')
    axes[0,0].set_xlabel('GRE Score')
    
    # GPA distribution
    axes[0,1].hist(df['GPA'], bins=20, alpha=0.7, color='lightgreen')
    axes[0,1].set_title('GPA Distribution')
    axes[0,1].set_xlabel('GPA')
    
    # Research experience distribution
    df['Research'].value_counts().plot(kind='bar', ax=axes[1,0], color=['lightcoral', 'lightblue'])
    axes[1,0].set_title('Research Experience Distribution')
    axes[1,0].set_xlabel('Research Experience (0=No, 1=Yes)')
    
    # Admission status distribution
    df['Admitted'].value_counts().plot(kind='bar', ax=axes[1,1], color=['lightpink', 'lightyellow'])
    axes[1,1].set_title('Admission Status Distribution')
    axes[1,1].set_xlabel('Admitted (0=No, 1=Yes)')
    
    plt.tight_layout()
    plt.savefig('generated_plots/distribution_analysis.png')
    plt.close()
    
    print("Distribution analysis completed")

def analyze_correlations(df):
    """
    Analyze correlations between variables
    """
    # Create correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(8, 6))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.savefig('generated_plots/correlation_analysis.png')
    plt.close()
    
    print("Correlation analysis completed")

def analyze_by_category(df):
    """
    Analyze admission patterns by different categories
    """
    # Admissions by gender
    gender_admission = df.groupby('Sex')['Admitted'].mean()
    
    plt.figure(figsize=(10, 6))
    gender_admission.plot(kind='bar', color=['lightblue', 'lightcoral'])
    plt.title('Admission Rate by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Admission Rate')
    plt.xticks(rotation=0)
    plt.savefig('generated_plots/gender_admission.png')
    plt.close()
    
    # Admissions by research experience
    research_admission = df.groupby('Research')['Admitted'].mean()
    
    plt.figure(figsize=(10, 6))
    research_admission.plot(kind='bar', color=['lightgreen', 'lightyellow'])
    plt.title('Admission Rate by Research Experience')
    plt.xlabel('Research Experience (0=No, 1=Yes)')
    plt.ylabel('Admission Rate')
    plt.xticks(rotation=0)
    plt.savefig('generated_plots/research_admission.png')
    plt.close()
    
    print("Category analysis completed")

def build_prediction_model(df):
    """
    Build a simple prediction model
    """
    # Prepare data for modeling
    # For demonstration, we'll use all numeric features
    features = ['GRE_Score', 'GPA', 'Research']
    X = df[features]
    y = df['Admitted']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_[0]
    })
    print("\nFeature Importance (Coefficients):")
    print(feature_importance.sort_values('Coefficient', key=abs, ascending=False))
    
    return model

def main():
    """
    Main function to run all analyses
    """
    print("Starting Data Analysis for Admission Data")
    print("=" * 50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Analyze distributions
    analyze_distributions(df)
    
    # Analyze correlations
    analyze_correlations(df)
    
    # Analyze by categories
    analyze_by_category(df)
    
    # Build prediction model
    model = build_prediction_model(df)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()