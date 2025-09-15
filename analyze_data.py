import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better-looking plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def clean_numeric(value):
    """Convert string numbers with % and $ to float"""
    if pd.isna(value) or value == '':
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    value = value.replace(',', '').replace('$', '').replace('%', '')
    try:
        return float(value)
    except ValueError:
        return np.nan

def visualize_raw_data(df, sheet_name):
    """Visualize raw data"""
    print(f"\n=== Raw Data Analysis for {sheet_name} ===")
    
    # 1. Display basic info
    print(f"\nShape: {df.shape}")
    print("\nData types:")
    print(df.dtypes.value_counts())
    
    # 2. Visualize missing values
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title(f'Missing Values - {sheet_name}')
    plt.tight_layout()
    plt.savefig(f'visualizations/{sheet_name}_raw_missing_values.png')
    plt.close()
    
    # 3. Visualize numeric data distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Plot distributions of numeric columns
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):  # Ensure we don't exceed the number of subplots
                try:
                    sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].tick_params(axis='x', rotation=45)
                except Exception as e:
                    print(f"Could not plot {col}: {str(e)}")
        
        # Remove any empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.savefig(f'visualizations/{sheet_name}_raw_distributions.png')
        plt.close()
        print(f"\nSaved raw data distributions to visualizations/{sheet_name}_raw_distributions.png")
    
    # 4. Correlation heatmap for numeric columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title(f'Correlation Heatmap - {sheet_name}')
        plt.tight_layout()
        plt.savefig(f'visualizations/{sheet_name}_correlation_heatmap.png')
        plt.close()
        print(f"Saved correlation heatmap to visualizations/{sheet_name}_correlation_heatmap.png")

def visualize_cleaned_data(df, sheet_name):
    """Visualize data after cleaning"""
    print(f"\n=== Cleaned Data Analysis for {sheet_name} ===")
    
    # 1. Display basic info after cleaning
    print(f"\nShape after cleaning: {df.shape}")
    print("\nData types after cleaning:")
    print(df.dtypes.value_counts())
    
    # 2. Visualize missing values after cleaning
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title(f'Missing Values After Cleaning - {sheet_name}')
    plt.tight_layout()
    plt.savefig(f'visualizations/{sheet_name}_cleaned_missing_values.png')
    plt.close()
    
    # 3. Visualize numeric data distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Plot distributions of numeric columns
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):  # Ensure we don't exceed the number of subplots
                try:
                    sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].tick_params(axis='x', rotation=45)
                except Exception as e:
                    print(f"Could not plot {col}: {str(e)}")
        
        # Remove any empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.savefig(f'visualizations/{sheet_name}_cleaned_distributions.png')
        plt.close()
        print(f"\nSaved cleaned data distributions to visualizations/{sheet_name}_cleaned_distributions.png")
    
    # 4. Correlation heatmap for numeric columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title(f'Correlation Heatmap - {sheet_name}')
        plt.tight_layout()
        plt.savefig(f'visualizations/{sheet_name}_correlation_heatmap.png')
        plt.close()
        print(f"Saved correlation heatmap to visualizations/{sheet_name}_correlation_heatmap.png")

def clean_data(df):
    """Clean the dataframe"""
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # 1. Drop completely empty rows and columns
    df_clean = df_clean.dropna(how='all')
    df_clean = df_clean.loc[:, (df_clean != 0).any(axis=0)]
    
    # 2. Clean column names
    df_clean.columns = [str(col).strip() for col in df_clean.columns]
    
    # 3. Convert numeric columns
    for col in df_clean.select_dtypes(include=['object']).columns:
        try:
            # Try to convert to numeric, coerce errors to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        except:
            pass
    
    # 4. Handle missing values
    # For numeric columns, fill with median (more robust to outliers)
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
    
    return df_clean

def load_and_analyze(file_path):
    print(f"Loading file: {file_path}")
    
    # Get all sheet names
    xls = pd.ExcelFile(file_path)
    print("\nAvailable sheets in the Excel file:")
    for i, sheet in enumerate(xls.sheet_names, 1):
        print(f"{i}. {sheet}")
    
    # Create visualizations directory if it doesn't exist
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # Process each sheet
    for sheet_name in xls.sheet_names:
        print(f"\n{'='*50}")
        print(f"Processing sheet: {sheet_name}")
        print(f"{'='*50}")
        
        # Load raw data
        raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
        
        # Visualize raw data
        print("\n[STEP 1/3] Visualizing raw data...")
        visualize_raw_data(raw_df, sheet_name)
        
        # Clean data
        print("\n[STEP 2/3] Cleaning data...")
        clean_df = clean_data(raw_df)
        
        # Visualize cleaned data
        print("\n[STEP 3/3] Visualizing cleaned data...")
        visualize_cleaned_data(clean_df, sheet_name)
        
        # Save cleaned data to CSV
        clean_df.to_csv(f'cleaned_{sheet_name}.csv', index=False)
        print(f"\nSaved cleaned data to cleaned_{sheet_name}.csv")
    
    print("\nAnalysis complete! Check the 'visualizations' folder for all generated plots.")
    
    # Basic visualizations for the last processed dataframe
    if 'df' in locals() and 'numeric_cols' in locals() and numeric_cols:
        try:
            plt.figure(figsize=(12, 6))
            df[numeric_cols].hist(bins=20, figsize=(15, 10))
            plt.tight_layout()
            plt.savefig('numeric_columns_distribution.png')
            print("\nSaved distribution plots for numeric columns as 'numeric_columns_distribution.png'")
        except Exception as e:
            print(f"\nCould not create distribution plots: {str(e)}")
    
    return df, sheets

if __name__ == "__main__":
    file_path = "Fountain Forward Analytics Project Manager - Practice Data Set.xlsx"
    df = load_and_analyze(file_path)
