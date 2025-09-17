import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Fountain Forward Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .positive-delta {
        color: #27ae60;
    }
    .negative-delta {
        color: #e74c3c;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def load_and_prepare_data():
    """Load and prepare data from Excel file"""
    try:
        # Load data from Excel
        file_path = "Fountain Forward Analytics Project Manager - Practice Data Set.xlsx"
        
        # Read all sheets
        xls = pd.ExcelFile(file_path)
        
        # Initialize list to store processed data for each month
        all_months_data = []
        
        for sheet_name in xls.sheet_names:
            # Skip sheets that don't contain relevant data
            if sheet_name in ['Sheet1', 'Sheet2', 'Sheet3']:
                continue
                
            # Read the sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Process the data (this part needs to be adjusted based on your actual data structure)
            # The following is a placeholder - you'll need to adjust the column names and calculations
            # based on your actual data structure
            
            # Example processing (replace with actual data processing logic)
            total_leads = len(df) if not df.empty else 0
            total_appts_set = df['Appointment Set'].sum() if 'Appointment Set' in df.columns else 0
            total_sales = df['Sale'].sum() if 'Sale' in df.columns else 0
            
            # Calculate conversion rates (example - adjust based on your funnel stages)
            conversion_rate = (total_sales / total_leads * 100) if total_leads > 0 else 0
            
            # Calculate costs (example - adjust based on your data)
            total_cost = 50000  # Placeholder - replace with actual calculation
            profit = 100000     # Placeholder - replace with actual calculation
            
            # Append monthly data
            month_data = {
                'Month': sheet_name,
                'Total Leads': total_leads,
                'Total Appts Set': total_appts_set,
                'Total Sales': total_sales,
                'Conversion Rate': conversion_rate,
                'Total Cost': total_cost,
                'Profit': profit,
                'Cost per Lead': total_cost / total_leads if total_leads > 0 else 0,
                'Cost per Sale': total_cost / total_sales if total_sales > 0 else 0
            }
            
            all_months_data.append(month_data)
        
        # Create a DataFrame with all months data
        if all_months_data:
            return pd.DataFrame(all_months_data)
        else:
            st.error("No valid data found in the Excel file.")
            return None
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def display_kpi_metrics(df):
    """Display KPI metrics in a row of cards"""
    if df is None or df.empty:
        return
    
    # Get the latest month's data
    latest_month = df.iloc[-1]
    
    # Calculate deltas if we have more than one month of data
    if len(df) > 1:
        prev_month = df.iloc[-2]
        delta_leads = latest_month['Total Leads'] - prev_month['Total Leads']
        delta_appts = latest_month['Total Appts Set'] - prev_month['Total Appts Set']
        delta_sales = latest_month['Total Sales'] - prev_month['Total Sales']
        delta_conversion = latest_month['Conversion Rate'] - prev_month['Conversion Rate']
        delta_cost = latest_month['Total Cost'] - prev_month['Total Cost']
        delta_profit = latest_month['Profit'] - prev_month['Profit']
    else:
        delta_leads = delta_appts = delta_sales = delta_conversion = delta_cost = delta_profit = 0
    
    # Create columns for metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Helper function to format deltas
    def format_delta(value, is_percent=False):
        if is_percent:
            return f"{value:+.1f}%"
        return f"{value:+,.0f}"
    
    # Metric 1: Total Leads
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Leads</div>
                <div class="metric-value">{latest_month['Total Leads']:,.0f}</div>
                <div class="{'positive-delta' if delta_leads >= 0 else 'negative-delta'}">
                    {format_delta(delta_leads)} from last month
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Metric 2: Total Appts Set
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Appts Set</div>
                <div class="metric-value">{latest_month['Total Appts Set']:,.0f}</div>
                <div class="{'positive-delta' if delta_appts >= 0 else 'negative-delta'}">
                    {format_delta(delta_appts)} from last month
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Metric 3: Total Sales
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Sales</div>
                <div class="metric-value">{latest_month['Total Sales']:,.0f}</div>
                <div class="{'positive-delta' if delta_sales >= 0 else 'negative-delta'}">
                    {format_delta(delta_sales)} from last month
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Metric 4: Conversion Rate
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Conversion Rate</div>
                <div class="metric-value">{latest_month['Conversion Rate']:.1f}%</div>
                <div class="{'positive-delta' if delta_conversion >= 0 else 'negative-delta'}">
                    {format_delta(delta_conversion, True)} from last month
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Metric 5: Total Cost
    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Cost</div>
                <div class="metric-value">${latest_month['Total Cost']/1000:,.1f}K</div>
                <div class="{'positive-delta' if delta_cost >= 0 else 'negative-delta'}">
                    {format_delta(delta_cost/1000, False)}K from last month
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Metric 6: Profit
    with col6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Profit</div>
                <div class="metric-value">${latest_month['Profit']/1000:,.1f}K</div>
                <div class="{'positive-delta' if delta_profit >= 0 else 'negative-delta'}">
                    {format_delta(delta_profit/1000, False)}K from last month
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def plot_funnel_chart(df):
    """Create and display funnel chart"""
    if df is None or df.empty:
        return
    
    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1, 
        cols=2, 
        column_widths=[0.5, 0.5],
        subplot_titles=('Funnel Volumes', 'Conversion Rates'),
        horizontal_spacing=0.15
    )
    
    # Funnel stages (adjust based on your actual data)
    stages = ['Total Leads', 'Total Appts Set', 'Total Sales']
    stage_labels = ['Leads', 'Appts Set', 'Sales']
    
    # Colors for each month
    colors = px.colors.qualitative.Plotly
    
    # Add traces for each month
    for i, (_, row) in enumerate(df.iterrows()):
        # Funnel volumes (left subplot)
        fig.add_trace(
            go.Funnel(
                y=stage_labels,
                x=[row[stage] for stage in stages],
                name=row['Month'],
                textinfo="value",
                textposition="inside",
                marker=dict(color=colors[i % len(colors)]),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Calculate conversion rates
        values = [row[stage] for stage in stages]
        conversion_rates = []
        for j in range(1, len(values)):
            rate = (values[j] / values[j-1] * 100) if values[j-1] > 0 else 0
            conversion_rates.append(rate)
        
        # Add conversion rates (right subplot)
        fig.add_trace(
            go.Bar(
                x=stage_labels[1:],  # Skip the first stage for conversion rates
                y=conversion_rates,
                name=row['Month'],
                text=[f"{rate:.1f}%" for rate in conversion_rates],
                textposition='auto',
                marker_color=colors[i % len(colors)],
                opacity=0.7,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update y-axis for conversion rates
    fig.update_yaxes(title_text="Conversion Rate (%)", row=1, col=2, range=[0, 100])
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def plot_monthly_comparison(df):
    """Create and display monthly comparison charts"""
    if df is None or len(df) < 2:
        st.warning("Not enough data for monthly comparison")
        return
    
    # Create tabs for different metrics
    tab1, tab2, tab3, tab4 = st.tabs(["Cost per Lead", "Cost per Sale", "Conversion Rate", "Profit"])
    
    with tab1:
        fig = px.bar(
            df, 
            x='Month', 
            y='Cost per Lead',
            title='Cost per Lead by Month',
            text_auto='.2f',
            color='Month',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.bar(
            df, 
            x='Month', 
            y='Cost per Sale',
            title='Cost per Sale by Month',
            text_auto='.2f',
            color='Month',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.line(
            df, 
            x='Month', 
            y='Conversion Rate',
            title='Conversion Rate Trend',
            markers=True,
            text='Conversion Rate',
            color_discrete_sequence=[px.colors.qualitative.Plotly[0]]
        )
        fig.update_traces(texttemplate='%{y:.1f}%', textposition='top center')
        fig.update_yaxes(title_text='Conversion Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Create a figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bars for profit
        fig.add_trace(
            go.Bar(
                x=df['Month'],
                y=df['Profit'],
                name='Profit',
                text=df['Profit'].apply(lambda x: f"${x/1000:,.0f}K"),
                textposition='auto',
                marker_color=px.colors.qualitative.Plotly[3]
            ),
            secondary_y=False,
        )
        
        # Add line for total cost
        fig.add_trace(
            go.Scatter(
                x=df['Month'],
                y=df['Total Cost'],
                name='Total Cost',
                mode='lines+markers+text',
                text=df['Total Cost'].apply(lambda x: f"${x/1000:,.0f}K"),
                textposition='top center',
                line=dict(color=px.colors.qualitative.Plotly[1], width=3)
            ),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_layout(
            title='Profit and Cost by Month',
            yaxis_title='Profit ($)',
            yaxis2_title='Total Cost ($)',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the Streamlit app"""
    # Set title and description
    st.title('🚗 Fountain Forward Analytics Dashboard')
    st.markdown("""
    Welcome to the Fountain Forward Dealership Analytics Dashboard. 
    This dashboard provides insights into your dealership's performance metrics and conversion funnels.
    """)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_and_prepare_data()
    
    if df is None or df.empty:
        st.error("Failed to load data. Please check the Excel file and try again.")
        return
    
    # Display KPI metrics
    st.markdown("## 📊 Key Performance Indicators")
    display_kpi_metrics(df)
    
    # Display funnel chart
    st.markdown("## 📈 Funnel Analysis")
    plot_funnel_chart(df)
    
    # Display monthly comparison
    st.markdown("## 📅 Monthly Performance Comparison")
    plot_monthly_comparison(df)
    
    # Add a download button for the data
    st.download_button(
        label="📥 Download Data as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=f'dealership_metrics_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
        help="Click to download the data as a CSV file"
    )
    
    # Add a section for raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(df)

if __name__ == "__main__":
    main()
