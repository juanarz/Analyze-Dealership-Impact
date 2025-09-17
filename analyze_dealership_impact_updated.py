import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set style for better-looking plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def get_all_sheets(filepath: str) -> List[str]:
    """Get all sheet names from the Excel file"""
    try:
        xl = pd.ExcelFile(filepath)
        return xl.sheet_names
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return []

def load_and_clean_data(filepath: str, month: str) -> pd.DataFrame:
    """Load and clean data from a specific sheet in the Excel file
    
    Args:
        filepath: Path to the Excel file
        month: Name of the sheet to load
        
    Returns:
        DataFrame with cleaned data for the specified month
    """
    try:
        # Read the Excel file, skipping the first row and using the second row as header
        df = pd.read_excel(filepath, sheet_name=month, header=1)
        
        # Clean column names (remove extra spaces and special characters)
        df.columns = df.columns.str.strip()
        
        # Filter for CORE Website Leads only (case insensitive and strip whitespace)
        if 'Lead Source' in df.columns:
            df['Lead Source'] = df['Lead Source'].astype(str).str.strip()
            core_leads = df[df['Lead Source'].str.upper() == 'CORE WEBSITE LEAD'].copy()
        else:
            core_leads = df.copy()
            print(f"Warning: 'Lead Source' column not found in {month}, using all rows")
        
        # Convert percentage strings to float
        percent_cols = [col for col in core_leads.columns if '%' in str(col)]
        for col in percent_cols:
            if core_leads[col].dtype == 'object':
                core_leads[col] = core_leads[col].astype(str).str.rstrip('%').astype('float') / 100.0
        
        # Ensure numeric columns are numeric
        numeric_cols = [
            'Total Leads', 'Good Leads', 'Bad Leads', 'Duplicate Leads',
            'Sold in Timeframe', 'Total Gross', 'Total Cost', 'Cost Per Lead',
            'Cost Per Sold', 'Profit', 'Appts Set', 'Appts Scheduled',
            'Appts Confirmed', 'Appts Shown'
        ]
        
        for col in numeric_cols:
            if col in core_leads.columns:
                core_leads[col] = pd.to_numeric(core_leads[col], errors='coerce')
        
        # Calculate total cost if not present
        if 'Total Cost' not in core_leads.columns and 'Cost Per Lead' in core_leads.columns and 'Total Leads' in core_leads.columns:
            core_leads['Total Cost'] = core_leads['Cost Per Lead'] * core_leads['Total Leads']
        
        # Calculate profit if not present
        if 'Profit' not in core_leads.columns and 'Total Gross' in core_leads.columns and 'Total Cost' in core_leads.columns:
            core_leads['Profit'] = core_leads['Total Gross'] - core_leads['Total Cost']
        
        # Add month column for reference
        core_leads['Month'] = month
        
        return core_leads
    
    except Exception as e:
        print(f"Error loading data for {month}: {str(e)}")
        return pd.DataFrame()

def calculate_kpis(df: pd.DataFrame, month: str, marketing_budget: float = 0) -> Dict[str, float]:
    """Calculate key performance indicators for a given month
    
    Args:
        df: DataFrame containing the data for the month
        month: Name of the month being processed
        marketing_budget: Optional marketing budget to use if cost data is missing
        
    Returns:
        Dictionary containing all calculated KPIs
    """
    if df.empty:
        print(f"No data available for {month}")
        return {}
    
    # Basic metrics
    kpis = {
        'Month': month,
        'Total Leads': df['Total Leads'].sum() if 'Total Leads' in df.columns else 0,
        'Good Leads': df['Good Leads'].sum() if 'Good Leads' in df.columns else 0,
        'Bad Leads': df['Bad Leads'].sum() if 'Bad Leads' in df.columns else 0,
        'Duplicate Leads': df['Duplicate Leads'].sum() if 'Duplicate Leads' in df.columns else 0,
        'Cars Sold': df['Sold in Timeframe'].sum() if 'Sold in Timeframe' in df.columns else 0,
        'Total Gross': df['Total Gross'].sum() if 'Total Gross' in df.columns else 0,
    }
    
    # Calculate or get cost metrics
    if 'Cost Per Lead' in df.columns and not df['Cost Per Lead'].isna().all() and df['Cost Per Lead'].mean() > 0:
        kpis['Cost Per Lead'] = df['Cost Per Lead'].mean()
    
    if 'Cost Per Sold' in df.columns and not df['Cost Per Sold'].isna().all() and df['Cost Per Sold'].mean() > 0:
        kpis['Cost Per Sold'] = df['Cost Per Sold'].mean()
    
    # Calculate total cost using the best available data
    if 'Total Cost' in df.columns and not df['Total Cost'].isna().all() and df['Total Cost'].sum() > 0:
        kpis['Total Cost'] = df['Total Cost'].sum()
    elif 'Cost Per Lead' in kpis and kpis['Total Leads'] > 0:
        kpis['Total Cost'] = kpis['Cost Per Lead'] * kpis['Total Leads']
    elif marketing_budget > 0:
        kpis['Total Cost'] = marketing_budget
    else:
        kpis['Total Cost'] = 0
    
    # Calculate cost per metrics if not already available
    if 'Cost Per Lead' not in kpis and kpis['Total Leads'] > 0:
        kpis['Cost Per Lead'] = kpis['Total Cost'] / kpis['Total Leads']
    
    if 'Cost Per Sold' not in kpis and kpis['Cars Sold'] > 0:
        kpis['Cost Per Sold'] = kpis['Total Cost'] / kpis['Cars Sold']
    
    # Calculate profit
    if 'Profit' in df.columns and not df['Profit'].isna().all():
        kpis['Profit'] = df['Profit'].sum()
    else:
        kpis['Profit'] = kpis['Total Gross'] - kpis['Total Cost']
    
    # Calculate ROI
    kpis['ROI %'] = (kpis['Profit'] / kpis['Total Cost']) * 100 if kpis['Total Cost'] > 0 else 0
    
    # Calculate Average Gross per Car
    if kpis['Cars Sold'] > 0:
        kpis['Avg Gross'] = kpis['Total Gross'] / kpis['Cars Sold']
    else:
        kpis['Avg Gross'] = 0
    
    # Calculate conversion rates
    kpis['Lead Conversion Rate %'] = (kpis['Cars Sold'] / kpis['Total Leads'] * 100) if kpis['Total Leads'] > 0 else 0
    kpis['Good Lead %'] = (kpis['Good Leads'] / kpis['Total Leads'] * 100) if kpis['Total Leads'] > 0 else 0
    
    # Get funnel metrics
    funnel_metrics = {
        'Appts Set': 'Appts Set',
        'Appts Scheduled': 'Appts Scheduled',
        'Appts Confirmed': 'Appts Confirmed',
        'Appts Shown': 'Appts Shown'
    }
    
    for k, v in funnel_metrics.items():
        if v in df.columns:
            kpis[k] = df[v].sum()
    
    # Calculate funnel conversion rates
    if 'Appts Set' in kpis and kpis['Total Leads'] > 0:
        kpis['Appt Set Rate %'] = (kpis['Appts Set'] / kpis['Total Leads']) * 100
    
    if 'Appts Shown' in kpis and kpis['Appts Set'] > 0:
        kpis['Appt Show Rate %'] = (kpis['Appts Shown'] / kpis['Appts Set']) * 100
    
    if kpis['Cars Sold'] > 0 and 'Appts Shown' in kpis and kpis['Appts Shown'] > 0:
        kpis['Close Rate %'] = (kpis['Cars Sold'] / kpis['Appts Shown']) * 100
    
    return kpis
    
    # Calculate conversion and ROI metrics
    lead_conversion_rate = (cars_sold / total_leads) * 100 if total_leads > 0 else 0
    roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0
    
    # Appointment funnel metrics
    appt_metrics = {}
    for metric in ['Appts Set', 'Appts Scheduled', 'Appts Confirmed', 'Appts Shown']:
        if metric in df.columns:
            appt_metrics[f'{metric}'] = df[metric].sum()
    
    # Calculate funnel conversion rates
    funnel_rates = {}
    funnel_steps = [('Appts Set', 'Appts Scheduled'), 
                   ('Appts Scheduled', 'Appts Confirmed'),
                   ('Appts Confirmed', 'Appts Shown')]
    
    for current, next_step in funnel_steps:
        if current in df.columns and next_step in df.columns:
            current_total = df[current].sum()
            next_total = df[next_step].sum()
            rate = (next_total / current_total * 100) if current_total > 0 else 0
            funnel_rates[f'{current} to {next_step} %'] = rate
    
    # Compile all metrics
    kpis = {
        'Month': month,
        'Total Leads': total_leads,
        'Good Leads': good_leads,
        'Bad Leads': bad_leads,
        'Duplicate Leads': duplicate_leads,
        'Cars Sold': cars_sold,
        'Total Gross': total_gross,
        'Total Cost': total_cost,
        'Cost Per Lead': cost_per_lead,
        'Cost Per Sold': cost_per_sold,
        'Lead Conversion Rate %': lead_conversion_rate,
        'ROI %': roi,
        'Profit': total_profit,
        'Good Lead Rate %': (good_leads / total_leads * 100) if total_leads > 0 else 0,
        'Bad Lead Rate %': (bad_leads / total_leads * 100) if total_leads > 0 else 0,
        'Duplicate Rate %': (duplicate_leads / total_leads * 100) if total_leads > 0 else 0,
        **appt_metrics,
        **funnel_rates,
        'Avg Days to Sale': df['Avg Days to Sale'].mean() if 'Avg Days to Sale' in df.columns else 0,
        'Avg Days to Appt Set': df['Avg Days to Appt Set'].mean() if 'Avg Days to Appt Set' in df.columns else 0,
        'Initial Visits': df['Initial Visits'].sum() if 'Initial Visits' in df.columns else 0,
        'Be Back Visits': df['Be Back Visits'].sum() if 'Be Back Visits' in df.columns else 0
    }
    
    return kpis

def create_comparison_plot(kpis_list: List[Dict[str, float]], metric: str, ylabel: str, 
                         title: str, filename: str, output_dir: str = 'visualizations'):
    """Create a bar plot comparing metrics across months
    
    Args:
        kpis_list: List of KPI dictionaries for each month
        metric: The metric to plot
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename (without extension)
        output_dir: Directory to save the plot
    """
    if not kpis_list:
        print(f"No data to plot for {metric}")
        return
    
    # Prepare data
    months = [kpi.get('Month', '') for kpi in kpis_list]
    values = [kpi.get(metric, 0) for kpi in kpis_list]
    
    # Skip if no valid data
    if not any(values):
        print(f"No valid data for {metric}")
        return
    
    # Create the plot
    plt.figure(figsize=(max(10, len(months) * 1.5), 6))
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({'Month': months, 'Value': values})
    ax = sns.barplot(data=plot_df, x='Month', y='Value', hue='Month', palette='viridis', legend=False)
    
    # Add title and labels
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    
    # Add value labels on the bars
    for i, v in enumerate(values):
        if pd.isna(v):
            continue
            
        if '%' in metric or '%' in ylabel:
            label = f"{v:.1f}%"
        elif any(c in ylabel for c in ['$', 'Cost', 'Gross', 'Profit']):
            label = f"${v:,.0f}" if v >= 1000 else f"${v:.2f}"
        else:
            label = f"{v:,.0f}" if v >= 1000 else f"{v:.1f}"
            
        # Position the label above the bar
        ax.text(i, v, label, ha='center', va='bottom')
    
    # Adjust layout and save
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), bbox_inches='tight')
    plt.close()

def create_funnel_plots(kpis_list: List[Dict[str, float]], output_dir: str = 'visualizations'):
    """Create funnel plots showing both conversion rates and absolute volumes
    
    Args:
        kpis_list: List of KPI dictionaries for each month
        output_dir: Directory to save the plots
    """
    if not kpis_list:
        print("No data available for funnel plots")
        return
    
    # Define the funnel stages and their labels
    stages = ['Appts Set', 'Appts Scheduled', 'Appts Confirmed', 'Appts Shown']
    stage_labels = ['Set', 'Scheduled', 'Confirmed', 'Shown']
    
    # Prepare data for conversion rate plot
    plt.figure(figsize=(12, 8))
    
    # Colors for each month
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(kpis_list)))
    
    # Plot conversion rates for each month
    for idx, kpis in enumerate(kpis_list):
        month = kpis.get('Month', f'Month {idx+1}')
        rates = []
        
        # Calculate conversion rates for each stage
        for i, stage in enumerate(stages):
            if i == 0:
                base = kpis.get('Total Leads', 1)
            else:
                prev_stage = stages[i-1]
                base = kpis.get(prev_stage, 1)
                
            current = kpis.get(stage, 0)
            rate = (current / base * 100) if base > 0 else 0
            rates.append(rate)
        
        # Plot the line for this month
        plt.plot(stage_labels, rates, 'o-', label=month, color=colors[idx], linewidth=2.5)
    
    # Add labels and legend
    plt.title('Appointment Funnel Conversion Rates by Month', fontsize=14, pad=20)
    plt.xlabel('Funnel Stage', fontsize=12, labelpad=10)
    plt.ylabel('Conversion Rate (%)', fontsize=12, labelpad=10)
    plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the conversion rate plot
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'funnel_conversion_rates.png'), bbox_inches='tight')
    plt.close()
    
    # Create a second plot for absolute volumes
    plt.figure(figsize=(12, 8))
    
    # Set the width of the bars
    bar_width = 0.8 / len(kpis_list)
    
    for idx, kpis in enumerate(kpis_list):
        month = kpis.get('Month', f'Month {idx+1}')
        values = [kpis.get(stage, 0) for stage in stages]
        
        # Calculate positions for each bar group
        x = np.arange(len(stages)) + idx * bar_width - (len(kpis_list) - 1) * bar_width / 2
        
        # Plot bars for this month
        bars = plt.bar(x, values, bar_width, label=month, color=colors[idx])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if height is greater than 0
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
    
    # Add labels and legend
    plt.title('Appointment Funnel Volumes by Month', fontsize=14, pad=20)
    plt.xlabel('Funnel Stage', fontsize=12, labelpad=10)
    plt.ylabel('Number of Appointments', fontsize=12, labelpad=10)
    plt.xticks(range(len(stages)), stage_labels)
    plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Save the volume plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'funnel_volumes.png'), bbox_inches='tight')
    plt.close()

def main():
    # File path to the Excel file
    filepath = 'Fountain Forward Analytics Project Manager - Practice Data Set.xlsx'
    
    # Get all sheet names from the Excel file
    sheet_names = get_all_sheets(filepath)
    if not sheet_names:
        print(f"Error: No sheets found in {filepath}")
        return
    
    print(f"Found {len(sheet_names)} sheets: {', '.join(sheet_names)}")
    
    # Marketing budgets by month (can be expanded as needed)
    marketing_budgets = {
        'February': 57660,  # February marketing budget
        'May': 54595,       # May marketing budget
        # Add other months as needed
    }
    
    # Process each sheet and collect KPIs
    all_kpis = []
    all_data = []
    
    for month in sheet_names:
        print(f"\nProcessing {month}...")
        
        # Load and clean the data
        month_data = load_and_clean_data(filepath, month)
        if month_data.empty:
            print(f"Skipping {month} - no valid data found")
            continue
            
        all_data.append(month_data)
        
        # Get marketing budget for this month
        budget = marketing_budgets.get(month, 0)
        if budget > 0:
            print(f"Using marketing budget for {month}: ${budget:,.2f}")
        
        # Calculate KPIs
        month_kpis = calculate_kpis(month_data, month, budget)
        if month_kpis:
            all_kpis.append(month_kpis)
    
    if not all_kpis:
        print("No valid data found in any sheet")
        return
    
    # Create a comparison DataFrame
    comparison_df = pd.DataFrame(all_kpis)
    
    # Reorder columns to put important metrics first
    cols = ['Month', 'Total Leads', 'Good Leads', 'Bad Leads', 'Duplicate Leads',
            'Cars Sold', 'Total Gross', 'Total Cost', 'Profit', 'ROI %',
            'Lead Conversion Rate %', 'Cost Per Lead', 'Cost Per Sold']
    
    # Add any additional columns that weren't in our initial list
    remaining_cols = [col for col in comparison_df.columns if col not in cols]
    comparison_df = comparison_df[cols + remaining_cols]
    
    # Save the comparison to a CSV file
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'dealership_kpi_comparison.csv')
    comparison_df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"\nComparison data saved to '{csv_path}'")
    
    # Create visualizations for each metric
    metrics_to_plot = [
        ('Cars Sold', 'Number of Cars', 'Total Cars Sold', 'cars_sold'),
        ('Total Gross', 'Total Gross ($)', 'Total Gross Revenue', 'total_gross'),
        ('Avg Gross', 'Average Gross ($)', 'Average Gross per Car', 'avg_gross'),
        ('Total Cost', 'Total Cost ($)', 'Total Marketing Cost', 'total_cost'),
        ('Profit', 'Profit ($)', 'Total Profit', 'total_profit'),
        ('ROI %', 'ROI (%)', 'Return on Investment', 'roi'),
        ('Lead Conversion Rate %', 'Conversion Rate (%)', 'Lead to Sale Conversion Rate', 'conversion_rate'),
        ('Cost Per Lead', 'Cost Per Lead ($)', 'Average Cost Per Lead', 'cost_per_lead'),
        ('Cost Per Sold', 'Cost Per Sold ($)', 'Average Cost Per Car Sold', 'cost_per_sold'),
        ('Appt Set Rate %', 'Appointment Set Rate (%)', 'Lead to Appointment Set Rate', 'appt_set_rate'),
        ('Appt Show Rate %', 'Appointment Show Rate (%)', 'Appointment Set to Show Rate', 'appt_show_rate'),
        ('Close Rate %', 'Close Rate (%)', 'Appointment to Sale Conversion', 'close_rate')
    ]
    
    # Create a directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Generate comparison plots for each metric
    for metric, ylabel, title, filename in metrics_to_plot:
        if metric in all_kpis[0]:  # Check if metric exists in the first month's KPIs
            create_comparison_plot(
                all_kpis, 
                metric, 
                ylabel, 
                f"{title} by Month", 
                filename,
                output_dir=vis_dir
            )
    
    # Create funnel visualizations
    create_funnel_plots(all_kpis, output_dir=vis_dir)
    
    # Print summary
    print("\n=== Analysis Complete ===")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    
    # Print key insights comparing first and last month
    if len(all_kpis) >= 2:
        first = all_kpis[0]
        last = all_kpis[-1]
        
        print("\nKey Insights (First vs Last Month):")
        print(f"1. Cars Sold: {first.get('Cars Sold', 0)} → {last.get('Cars Sold', 0)} ({((last.get('Cars Sold', 0)/first.get('Cars Sold', 1))-1)*100:+.1f}%)")
        print(f"2. Lead Conversion: {first.get('Lead Conversion Rate %', 0):.1f}% → {last.get('Lead Conversion Rate %', 0):.1f}%")
        print(f"3. ROI: {first.get('ROI %', 0):.1f}% → {last.get('ROI %', 0):.1f}%")
        print(f"4. Cost Per Lead: ${first.get('Cost Per Lead', 0):.2f} → ${last.get('Cost Per Lead', 0):.2f}")
        print(f"5. Cost Per Sold: ${first.get('Cost Per Sold', 0):.2f} → ${last.get('Cost Per Sold', 0):.2f}")
        
        # Print summary of all months
        print("\nSummary of All Months:")
        print(comparison_df[['Month', 'Total Leads', 'Cars Sold', 'Total Gross', 'Total Cost', 'Profit', 'ROI %']].to_string(index=False))

if __name__ == "__main__":
    main()
