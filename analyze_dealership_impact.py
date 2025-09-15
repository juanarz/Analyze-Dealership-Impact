import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def calculate_kpis(df, month, marketing_budget):
    """Calculate key performance indicators for a given month"""
    # Clean and filter for CORE Website Leads only (case insensitive and strip whitespace)
    df['Lead Source'] = df['Lead Source'].astype(str).str.strip()
    core_leads = df[df['Lead Source'].str.upper() == 'CORE WEBSITE LEAD'].copy()
    
    # Print debug info
    print(f"\n=== {month} ===")
    print(f"Total rows in dataset: {len(df)}")
    print(f"Rows with 'CORE Website Lead': {len(core_leads)}")
    
    # Basic metrics
    total_leads = len(core_leads)
    cars_sold = core_leads['Sold in Timeframe'].sum()
    total_gross = core_leads['Total Gross'].sum()
    
    # Helper function to safely get column values
    def safe_get(col, default=0):
        return core_leads[col].sum() if col in core_leads.columns else default
    
    def safe_mean(col, default=0):
        return core_leads[col].mean() if col in core_leads.columns else default
    
    # Lead Quality Metrics
    if 'Lead Status' in core_leads.columns:
        good_leads = (core_leads['Lead Status'] == 'Good').sum()
        bad_leads = (core_leads['Lead Status'] == 'Bad').sum()
    else:
        good_leads = 0
        bad_leads = 0
    
    duplicate_leads = safe_get('Dup')
    
    # Conversion Metrics
    total_appts_shown = safe_get('Appts Shown')
    total_initial_visits = safe_get('Initial Visits', 0)
    
    # Efficiency Metrics
    avg_days_lead_to_sale = safe_mean('Days Lead to Sale')
    avg_days_lead_to_appt = safe_mean('Days Lead to Appt')
    
    # Appointment Funnel Metrics
    appts_set = safe_get('Appts Set')
    appts_scheduled = safe_get('Appts Scheduled')
    appts_confirmed = safe_get('Appts Confirmed')
    
    # Visit Engagement
    total_visits = safe_get('Total Visits', 0)
    be_back_visits = safe_get('Be Back Visits', 0)
    
    # Calculate KPIs
    cost_per_car = marketing_budget / cars_sold if cars_sold > 0 else 0
    avg_gross = total_gross / cars_sold if cars_sold > 0 else 0
    
    # Lead Quality Rates
    good_lead_rate = (good_leads / total_leads) * 100 if total_leads > 0 else 0
    bad_lead_rate = (bad_leads / total_leads) * 100 if total_leads > 0 else 0
    duplicate_rate = (duplicate_leads / total_leads) * 100 if total_leads > 0 else 0
    
    # Conversion Rates
    lead_to_sale_cr = (cars_sold / total_leads) * 100 if total_leads > 0 else 0
    lead_to_appt_cr = (total_appts_shown / total_leads) * 100 if total_leads > 0 else 0
    
    # Appointment Funnel Rates
    appt_set_rate = (appts_set / total_leads) * 100 if total_leads > 0 else 0
    appt_scheduled_rate = (appts_scheduled / appts_set) * 100 if appts_set > 0 else 0
    appt_confirmed_rate = (appts_confirmed / appts_scheduled) * 100 if appts_scheduled > 0 else 0
    appt_shown_rate = (total_appts_shown / appts_confirmed) * 100 if appts_confirmed > 0 else 0
    
    # Visit Engagement Rates
    initial_visit_rate = (total_initial_visits / total_visits) * 100 if total_visits > 0 else 0
    be_back_rate = (be_back_visits / total_visits) * 100 if total_visits > 0 else 0
    visit_to_sale_cr = (cars_sold / total_visits) * 100 if total_visits > 0 else 0
    
    # Profitability Metrics
    gross_margin_per_lead = total_gross / total_leads if total_leads > 0 else 0
    
    return {
        'Month': month,
        'Marketing Budget': marketing_budget,
        'Cars Sold': cars_sold,
        'Total Gross': total_gross,
        'Cost Per Car': cost_per_car,
        'Average Gross': avg_gross,
        'ROI': (total_gross - marketing_budget) / marketing_budget * 100 if marketing_budget > 0 else 0,
        
        # Lead Quality
        'Total Leads': total_leads,
        'Good Lead Rate %': good_lead_rate,
        'Bad Lead Rate %': bad_lead_rate,
        'Duplicate Rate %': duplicate_rate,
        
        # Conversion Metrics
        'Lead to Sale CR %': lead_to_sale_cr,
        'Lead to Appt CR %': lead_to_appt_cr,
        
        # Efficiency Metrics
        'Avg Days Lead to Sale': avg_days_lead_to_sale,
        'Avg Days Lead to Appt': avg_days_lead_to_appt,
        
        # Appointment Funnel
        'Appt Set Rate %': appt_set_rate,
        'Appt Scheduled Rate %': appt_scheduled_rate,
        'Appt Confirmed Rate %': appt_confirmed_rate,
        'Appt Shown Rate %': appt_shown_rate,
        
        # Visit Engagement
        'Initial Visit Rate %': initial_visit_rate,
        'Be Back Rate %': be_back_rate,
        'Visit to Sale CR %': visit_to_sale_cr,
        
        # Profitability
        'Gross Margin per Lead': gross_margin_per_lead,
        'Profit per Lead': (total_gross - marketing_budget) / total_leads if total_leads > 0 else 0,
        'Profit per Sale': (total_gross - marketing_budget) / cars_sold if cars_sold > 0 else 0,
        'Cost Efficiency': cost_per_car / avg_gross if avg_gross > 0 else 0
    }

def create_comparison_plot(data_feb, data_may, metric, ylabel, title, filename):
    """Create a bar plot comparing February and May metrics"""
    months = ['February', 'May']
    values = [data_feb[metric], data_may[metric]]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(months, values, color=['#1f77b4', '#ff7f0e'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if metric in ['Cost Per Car', 'Total Gross', 'Average Gross']:
            plt.text(bar.get_x() + bar.get_width()/2., height + (0.02 * max(values)),
                   f'${height:,.2f}',
                   ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + (0.02 * max(values)),
                   f'{height:,.0f}',
                   ha='center', va='bottom')
    
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel(ylabel)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'visualizations/{filename}.png')
    plt.close()

def main():
    # Load the cleaned data
    feb_df = pd.read_csv('cleaned_February.csv')
    may_df = pd.read_csv('cleaned_May.csv')
    
    # Marketing budgets
    feb_budget = 57660
    may_budget = 54595
    
    # Calculate KPIs for both months
    feb_kpis = calculate_kpis(feb_df, 'February', feb_budget)
    may_kpis = calculate_kpis(may_df, 'May', may_budget)
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame([feb_kpis, may_kpis])
    print("\n=== Performance Summary ===")
    print(summary_df.to_string(index=False))
    
    # Create visualizations
    create_comparison_plot(feb_kpis, may_kpis, 'Cars Sold', 'Number of Cars', 
                          'Cars Sold in February vs May', 'cars_sold_comparison')
    
    create_comparison_plot(feb_kpis, may_kpis, 'Total Gross', 'Total Gross ($)', 
                          'Total Gross from CORE Leads', 'total_gross_comparison')
    
    create_comparison_plot(feb_kpis, may_kpis, 'Average Gross', 'Average Gross ($)', 
                          'Average Gross per Car', 'avg_gross_comparison')
    
    create_comparison_plot(feb_kpis, may_kpis, 'Cost Per Car', 'Cost Per Car ($)', 
                          'Marketing Cost Per Car Sold', 'cost_per_car_comparison')
    
    # Define metrics to compare
    metrics = [
        # Basic Metrics
        ('Cars Sold', '', '0', ''),
        ('Total Gross', '$', '0,.0f', '$'),
        ('Average Gross', '$', '0,.2f', '$'),
        ('Cost Per Car', '$', '0,.2f', '$'),
        ('ROI', '', '0.1f', '%'),
        
        # Lead Quality
        ('Good Lead Rate %', '', '0.1f', '%'),
        ('Bad Lead Rate %', '', '0.1f', '%'),
        ('Duplicate Rate %', '', '0.1f', '%'),
        
        # Conversion Metrics
        ('Lead to Sale CR %', '', '0.1f', '%'),
        ('Lead to Appt CR %', '', '0.1f', '%'),
        
        # Efficiency Metrics
        ('Avg Days Lead to Sale', '', '0.1f', ''),
        ('Avg Days Lead to Appt', '', '0.1f', ''),
        
        # Profitability
        ('Gross Margin per Lead', '$', '0.2f', ''),
        ('Profit per Lead', '$', '0.2f', ''),
        ('Profit per Sale', '$', '0.2f', ''),
        ('Cost Efficiency', '', '0.2f', '')
    ]
    
    # Calculate improvements
    improvement = {'Metric': [], 'February': [], 'May': [], 'Change': []}
    
    for metric, prefix, format_spec, suffix in metrics:
        improvement['Metric'].append(metric)
        
        # Format February value
        feb_val = feb_kpis.get(metric, 0)
        if isinstance(feb_val, (int, float)):
            feb_fmt = f"{prefix}{feb_val:>{format_spec}}{suffix}"
        else:
            feb_fmt = str(feb_val)
        improvement['February'].append(feb_fmt)
        
        # Format May value
        may_val = may_kpis.get(metric, 0)
        if isinstance(may_val, (int, float)):
            may_fmt = f"{prefix}{may_val:>{format_spec}}{suffix}"
        else:
            may_fmt = str(may_val)
        improvement['May'].append(may_fmt)
        
        # Calculate and format change
        if isinstance(feb_val, (int, float)) and isinstance(may_val, (int, float)) and feb_val != 0:
            if '%' in metric or 'Rate' in metric or 'ROI' in metric:
                change = may_val - feb_val
                change_fmt = f"{change:+.1f}% points"
            else:
                change = ((may_val - feb_val) / feb_val) * 100
                change_fmt = f"{change:+.1f}%"
        else:
            change_fmt = "N/A"
        improvement['Change'].append(change_fmt)
    
    print("\n=== Performance Comparison ===")
    print(pd.DataFrame(improvement).to_string(index=False))

if __name__ == "__main__":
    main()
