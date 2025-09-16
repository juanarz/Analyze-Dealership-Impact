import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for better-looking plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_clean_data(filepath, month):
    """Load and clean data from Excel file"""
    try:
        # Read the Excel file, skipping the first row and using the second row as header
        df = pd.read_excel(filepath, sheet_name=month, header=1)
        
        # Print column names for debugging
        print(f"\nColumns in {month} data:")
        print("\n".join([f"- {col}" for col in df.columns]))
        
        # Clean column names (remove extra spaces and special characters)
        df.columns = df.columns.str.strip()
        
        # Filter for CORE Website Leads only (case insensitive and strip whitespace)
        df['Lead Source'] = df['Lead Source'].astype(str).str.strip()
        core_leads = df[df['Lead Source'].str.upper() == 'CORE WEBSITE LEAD'].copy()
        
        # Print sample data for cost-related columns
        cost_cols = [col for col in core_leads.columns if 'cost' in col.lower() or 'price' in col.lower() or 'spend' in col.lower()]
        print("\nCost-related columns found:", cost_cols)
        if cost_cols:
            print("\nSample cost data:")
            print(core_leads[cost_cols].head())
        
        # Convert percentage strings to float
        percent_cols = [col for col in core_leads.columns if '%' in str(col)]
        for col in percent_cols:
            if core_leads[col].dtype == 'object':
                core_leads[col] = core_leads[col].astype(str).str.rstrip('%').astype('float') / 100.0
        
        # Ensure numeric columns are numeric
        numeric_cols = ['Total Leads', 'Good Leads', 'Bad Leads', 'Duplicate Leads', 
                       'Sold in Timeframe', 'Total Gross', 'Total Cost', 'Cost Per Lead', 
                       'Cost Per Sold', 'Profit']
        
        for col in numeric_cols:
            if col in core_leads.columns:
                core_leads[col] = pd.to_numeric(core_leads[col], errors='coerce')
        
        # If Total Cost is missing but Cost Per Lead is available, calculate it
        if 'Total Cost' not in core_leads.columns and 'Cost Per Lead' in core_leads.columns and 'Total Leads' in core_leads.columns:
            core_leads['Total Cost'] = core_leads['Cost Per Lead'] * core_leads['Total Leads']
        
        # If Profit is missing but Total Gross and Total Cost are available, calculate it
        if 'Profit' not in core_leads.columns and 'Total Gross' in core_leads.columns and 'Total Cost' in core_leads.columns:
            core_leads['Profit'] = core_leads['Total Gross'] - core_leads['Total Cost']
        
        return core_leads
    
    except Exception as e:
        print(f"Error loading data for {month}: {str(e)}")
        return pd.DataFrame()

def calculate_kpis(df, month, marketing_budget):
    """Calculate key performance indicators for a given month"""
    if df.empty:
        print(f"No data available for {month}")
        return {}
    
    # Basic metrics
    total_leads = df['Total Leads'].sum() if 'Total Leads' in df.columns else 0
    good_leads = df['Good Leads'].sum() if 'Good Leads' in df.columns else 0
    bad_leads = df['Bad Leads'].sum() if 'Bad Leads' in df.columns else 0
    duplicate_leads = df['Duplicate Leads'].sum() if 'Duplicate Leads' in df.columns else 0
    
    # Sales metrics
    cars_sold = df['Sold in Timeframe'].sum() if 'Sold in Timeframe' in df.columns else 0
    total_gross = df['Total Gross'].sum() if 'Total Gross' in df.columns else 0
    
    # Calculate total cost - prioritize marketing budget if provided
    if marketing_budget > 0:
        total_cost = marketing_budget
        print(f"Using provided marketing budget as Total Cost: ${total_cost:,.2f}")
    elif 'Total Cost' in df.columns and df['Total Cost'].sum() > 0:
        total_cost = df['Total Cost'].sum()
        print(f"Using Total Cost from data: ${total_cost:,.2f}")
    elif 'Cost Per Lead' in df.columns and total_leads > 0 and df['Cost Per Lead'].mean() > 0:
        total_cost = df['Cost Per Lead'].mean() * total_leads
        print(f"Calculated Total Cost from Cost Per Lead: ${total_cost:,.2f}")
    else:
        total_cost = 0
        print("Warning: No valid cost data found. Using $0 for cost calculations.")
    
    # Ensure we have a positive cost for calculations
    if total_cost <= 0:
        print("Warning: Total cost is zero or negative. Some metrics may be affected.")
        if month == 'February':
            total_cost = 57660  # Fallback to hardcoded February budget
            print(f"Using fallback February budget: ${total_cost:,.2f}")
        elif month == 'May':
            total_cost = 54595  # Fallback to hardcoded May budget
            print(f"Using fallback May budget: ${total_cost:,.2f}")
    
    # Calculate cost metrics
    cost_per_lead = total_cost / total_leads if total_leads > 0 else 0
    cost_per_sold = total_cost / cars_sold if cars_sold > 0 else 0
    
    # Calculate profit if not available
    if 'Profit' in df.columns:
        total_profit = df['Profit'].sum()
    else:
        total_profit = total_gross - total_cost
    
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

def create_comparison_plot(data_feb, data_may, metric, ylabel, title, filename):
    """Create a bar plot comparing February and May metrics"""
    if not data_feb or not data_may:
        print(f"Insufficient data to create plot for {metric}")
        return
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Month': ['February', 'May'],
        'Value': [data_feb.get(metric, 0), data_may.get(metric, 0)]
    })
    
    plt.figure()
    
    # Create barplot with hue parameter to avoid deprecation warning
    ax = sns.barplot(x='Month', y='Value', hue='Month', data=plot_data, 
                    palette=['#1f77b4', '#ff7f0e'], legend=False)
    
    # Add value labels on top of bars
    for i, v in enumerate(plot_data['Value']):
        if isinstance(v, (int, float)):
            # Format the value based on its magnitude
            if abs(v) >= 1000:
                label = f"${v/1000:,.1f}K" if ylabel.startswith('$') else f"{v/1000:,.1f}K"
            else:
                label = f"${v:,.2f}" if ylabel.startswith('$') else f"{v:,.2f}"
            ax.text(i, v + (max(plot_data['Value']) * 0.02), label, ha='center')
    
    plt.title(title, pad=20)
    plt.xlabel('')
    plt.ylabel(ylabel)
    
    # Adjust layout to prevent title cutoff
    plt.tight_layout()
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(f'visualizations/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_funnel_plot(data_feb, data_may):
    """Create a funnel plot for the appointment process"""
    if not data_feb or not data_may:
        print("Insufficient data to create funnel plot")
        return
    
    funnel_steps = ['Appts Set', 'Appts Scheduled', 'Appts Confirmed', 'Appts Shown']
    
    # Filter out steps that don't have data
    funnel_steps = [step for step in funnel_steps 
                   if step in data_feb and step in data_may 
                   and data_feb[step] > 0 and data_may[step] > 0]
    
    if not funnel_steps:
        print("No valid funnel data available")
        return
    
    feb_values = [data_feb[step] for step in funnel_steps]
    may_values = [data_may[step] for step in funnel_steps]
    
    # Calculate conversion rates
    feb_rates = [100]  # Start with 100% for the first step
    may_rates = [100]
    
    for i in range(1, len(funnel_steps)):
        feb_rate = (feb_values[i] / feb_values[0]) * 100 if feb_values[0] > 0 else 0
        may_rate = (may_values[i] / may_values[0]) * 100 if may_values[0] > 0 else 0
        feb_rates.append(feb_rate)
        may_rates.append(may_rate)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot February funnel
    plt.subplot(1, 2, 1)
    plt.title('February - Appointment Funnel')
    sns.barplot(x=feb_rates, y=funnel_steps, color='#1f77b4')
    plt.xlabel('Conversion Rate (%)')
    
    # Add value labels
    for i, v in enumerate(feb_rates):
        plt.text(v + 2, i, f"{v:.1f}%")
    
    # Plot May funnel
    plt.subplot(1, 2, 2)
    plt.title('May - Appointment Funnel')
    sns.barplot(x=may_rates, y=funnel_steps, color='#ff7f0e')
    plt.xlabel('Conversion Rate (%)')
    
    # Add value labels
    for i, v in enumerate(may_rates):
        plt.text(v + 2, i, f"{v:.1f}%")
    
    plt.tight_layout()
    plt.savefig('visualizations/appointment_funnel_comparison.png', bbox_inches='tight')
    plt.close()

def main():
    # File paths and marketing budgets
    filepath = 'Fountain Forward Analytics Project Manager - Practice Data Set.xlsx'
    marketing_budgets = {
        'February': 57660,  # February marketing budget
        'May': 54595        # May marketing budget
    }
    
    # Load and clean data
    print("Loading February data...")
    feb_data = load_and_clean_data(filepath, 'February')
    
    print("Loading May data...")
    may_data = load_and_clean_data(filepath, 'May')
    
    if feb_data.empty or may_data.empty:
        print("Error: Could not load one or both datasets. Please check the file paths and sheet names.")
        return
    
    # Calculate KPIs using provided marketing budgets
    print("\nCalculating KPIs...")
    feb_kpis = calculate_kpis(feb_data, 'February', marketing_budgets['February'])
    may_kpis = calculate_kpis(may_data, 'May', marketing_budgets['May'])
    
    # Create a comparison DataFrame
    comparison = pd.DataFrame([feb_kpis, may_kpis])
    
    # Save the comparison to a CSV file
    comparison.to_csv('dealership_kpi_comparison.csv', index=False)
    print("\nComparison data saved to 'dealership_kpi_comparison.csv'")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Create comparison plots for key metrics
    metrics_to_plot = [
        ('Cars Sold', 'Number of Cars', 'Cars Sold Comparison', 'cars_sold_comparison'),
        ('Total Gross', 'Total Gross ($)', 'Total Gross Revenue Comparison', 'total_gross_comparison'),
        ('Lead Conversion Rate %', 'Conversion Rate (%)', 'Lead Conversion Rate Comparison', 'conversion_rate_comparison'),
        ('ROI %', 'ROI (%)', 'Return on Investment Comparison', 'roi_comparison'),
        ('Cost Per Lead', 'Cost ($)', 'Cost Per Lead Comparison', 'cost_per_lead_comparison'),
        ('Cost Per Sold', 'Cost ($)', 'Cost Per Car Sold Comparison', 'cost_per_sold_comparison')
    ]
    
    print("\nGenerating visualizations...")
    for metric, ylabel, title, filename in metrics_to_plot:
        if metric in feb_kpis and metric in may_kpis:
            create_comparison_plot(feb_kpis, may_kpis, metric, ylabel, title, filename)
    
    # Create appointment funnel visualization
    create_funnel_plot(feb_kpis, may_kpis)
    
    # Print summary
    print("\n=== Analysis Complete ===")
    print(f"Visualizations have been saved to the 'visualizations' directory.")
    print(f"\nKey Insights:")
    print(f"1. Cars Sold: {feb_kpis.get('Cars Sold', 0)} (Feb) → {may_kpis.get('Cars Sold', 0)} (May)")
    print(f"2. Lead Conversion Rate: {feb_kpis.get('Lead Conversion Rate %', 0):.1f}% → {may_kpis.get('Lead Conversion Rate %', 0):.1f}%")
    print(f"3. ROI: {feb_kpis.get('ROI %', 0):.1f}% → {may_kpis.get('ROI %', 0):.1f}%")
    print(f"4. Cost Per Lead: ${feb_kpis.get('Cost Per Lead', 0):.2f} → ${may_kpis.get('Cost Per Lead', 0):.2f}")
    print(f"5. Cost Per Sold: ${feb_kpis.get('Cost Per Sold', 0):.2f} → ${may_kpis.get('Cost Per Sold', 0):.2f}")

if __name__ == "__main__":
    main()
