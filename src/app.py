"""
UAV Mission Prioritization DSS - Web Interface
Interactive Dashboard using Streamlit for UAV Mission Prioritization Decision Support System using AHP and TOPSIS methods.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Import moduls
sys.path.append(os.path.dirname(__file__))
from decision_engine import DecisionSupportSystem, AHPEngine

# Page configurations
st.set_page_config(
    page_title="UAV Mission Prioritization DSS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load data from CSV files"""
    try:
        missions = pd.read_csv('data/missions.csv')
        uav_fleet = pd.read_csv('data/uav_fleet.csv')
        weather = pd.read_csv('data/weather.csv')
        
        return missions, uav_fleet, weather
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.info("Firstly run 'python src/data_collection.py' to generate the required data files.")
        st.stop()


@st.cache_data
def run_decision_analysis(_dss, custom_weights=None):
    """Run the decision analysis with optional custom weights"""
    results = _dss.run_analysis(custom_weights=custom_weights)
    return results


def create_map_visualization(df, top_n=100):
    """Map visualization of top priority missions"""
    
    # Prioritize top N missions
    plot_df = df.nsmallest(top_n, 'rank').copy()
    
    # Color by rank
    plot_df['priority_color'] = plot_df['rank']
    
    fig = px.scatter_mapbox(
        plot_df,
        lat='latitude',
        lon='longitude',
        color='priority_color',
        size='topsis_score',
        hover_data={
            'mission_id': True,
            'mission_type': True,
            'urgency_level': True,
            'rank': True,
            'topsis_score': ':.3f',
            'priority_color': False
        },
        color_continuous_scale='RdYlGn_r',
        size_max=15,
        zoom=5,
        center={'lat': 39, 'lon': 35},
        title=f'Top Priority {top_n} Mission Locations'
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        height=600,
        showlegend=False
    )
    
    return fig


def create_priority_chart(df, top_n=20):
    """Mission sorting bar chart by TOPSIS score"""
    
    plot_df = df.nsmallest(top_n, 'rank').copy()
    
    fig = px.bar(
        plot_df,
        x='topsis_score',
        y='mission_id',
        orientation='h',
        color='urgency_level',
        color_discrete_map={
            'Critical': "#680505",
            'High': "#f37e17",
            'Medium': "#238623",
            'Low': "#2987ca"
        },
        hover_data=['mission_type', 'threat_level', 'civilian_density'],
        title=f'Top priority {top_n} Mission - TOPSIS Scores'
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="TOPSIS Score",
        yaxis_title="Mission ID",
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_criteria_distribution(df):
    """Subplot of criteria distributions"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Urgency Score', 'Threat Level', 
                       'Civilian Density', 'Weather Suitability')
    )
    
    # Urgency
    fig.add_trace(
        go.Histogram(x=df['urgency_score'], name='Urgency', 
                    marker_color='#1f77b4', showlegend=False),
        row=1, col=1
    )
    
    # Threat Level
    fig.add_trace(
        go.Histogram(x=df['threat_level'], name='Threat',
                    marker_color='#ff7f0e', showlegend=False),
        row=1, col=2
    )
    
    # Civilian Density
    fig.add_trace(
        go.Histogram(x=df['civilian_density'], name='Civilian',
                    marker_color='#2ca02c', showlegend=False),
        row=2, col=1
    )
    
    # Weather Suitability
    if 'weather_suitability' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['weather_suitability'], name='Weather',
                        marker_color='#d62728', showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=500, title_text="Decision Criteria Distributions")
    
    return fig


def create_uav_status_chart(uav_df):
    """UAV Status Distribution"""
    
    # Status Counts
    status_counts = uav_df['status'].value_counts()
    
    fig1 = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.4,
        marker_colors=['#2ca02c', '#ff7f0e', '#d62728']
    )])
    
    fig1.update_layout(
        title='UAV Status Distribution',
        height=400
    )
    
    return fig1


def create_uav_fuel_chart(uav_df):
    """UAV Fuel Level Distribution by Status"""
    
    fig = px.box(
        uav_df,
        x='status',
        y='fuel_level',
        color='status',
        color_discrete_map={
            'Ready': '#2ca02c',
            'In Mission': '#ff7f0e',
            'Maintenance': '#d62728'
        },
        title='UAV Fuel Level - Status Based'
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def main():
    """Main function to run the Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">UAV Mission Prioritization DSS</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Control Panel")
    st.sidebar.markdown("---")
    
    # Load Data
    with st.spinner("Loading data..."):
        missions, uav_fleet, weather = load_data()
    
    st.sidebar.success(f"{len(missions):,} Mission loaded")
    st.sidebar.info(f"{len(uav_fleet)} UAV available")
    
    # Analyze Parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Parameters")
    
    # Use Custom Weights
    use_custom_weights = st.sidebar.checkbox("Use Custom Weights", value=False)
    
    custom_weights = None
    if use_custom_weights:
        st.sidebar.markdown("**Custom Weights** (Toplam: 1.0)")
        
        w1 = st.sidebar.slider("Urgency", 0.0, 1.0, 0.45, 0.05)
        w2 = st.sidebar.slider("Threat Level", 0.0, 1.0, 0.25, 0.05)
        w3 = st.sidebar.slider("Civilian Density", 0.0, 1.0, 0.10, 0.05)
        w4 = st.sidebar.slider("Weather Suitability", 0.0, 1.0, 0.10, 0.05)
        w5 = st.sidebar.slider("UAV Availability", 0.0, 1.0, 0.10, 0.05)
        
        total = w1 + w2 + w3 + w4 + w5
        
        if abs(total - 1.0) > 0.01:
            st.sidebar.warning(f"Total: {total:.2f} (Must be 1.0 )")
        else:
            custom_weights = {
                'urgency_score': w1,
                'threat_level': w2,
                'civilian_density': w3,
                'weather_suitability': w4,
                'uav_availability': w5
            }
    
    # Button of start analysis
    st.sidebar.markdown("---")
    run_analysis_btn = st.sidebar.button("Start Analysis", type="primary", use_container_width=True)
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "General Overview", 
        "Mission Prioritization", 
        "UAV Fleet Status",
        "Detailed Analysis"
    ])
    
    # TAB 1: General Overview
    with tab1:
        st.header("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Missions", f"{len(missions):,}")
        
        with col2:
            critical = len(missions[missions['urgency_level'] == 'Urgency'])
            st.metric("Critical Mission", critical, delta=f"{(critical/len(missions)*100):.1f}%")
        
        with col3:
            available = len(uav_fleet[uav_fleet['status'] == 'Ready'])
            st.metric("UAV Ready", available, delta=f"{(available/len(uav_fleet)*100):.1f}%")
        
        with col4:
            avg_fuel = uav_fleet[uav_fleet['status'] == 'Ready']['fuel_level'].mean()
            st.metric("Average Fuel", f"{avg_fuel:.1f}%")
        
        st.markdown("---")
        
        # Mission Type and Urgency Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Mission Type Distribution")
            mission_type_counts = missions['mission_type'].value_counts()
            fig = px.pie(
                values=mission_type_counts.values,
                names=mission_type_counts.index,
                title='Mission Types'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Urgency Level Distribution")
            urgency_counts = missions['urgency_level'].value_counts()
            fig = px.bar(
                x=urgency_counts.index,
                y=urgency_counts.values,
                color=urgency_counts.index,
                color_discrete_map={
                    'Critical': '#d62728',
                    'High': '#ff7f0e',
                    'Medium': '#2ca02c',
                    'Low': '#1f77b4'
                },
                title='Urgency Levels'
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count of Missions")
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Mission Prioritization
    with tab2:
        if run_analysis_btn or 'results' in st.session_state:
            
            if run_analysis_btn:
                with st.spinner("Decision Analysis Working (This may take a few moments...)"):
                    dss = DecisionSupportSystem()
                    dss.load_data(missions, uav_fleet, weather)
                    results = run_decision_analysis(dss, custom_weights)
                    st.session_state['results'] = results
                    st.success("Analysis Completed!")
            
            results = st.session_state['results']
            
            st.header("Prioritized Mission Results")
            
            # Filter Options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                urgency_filter = st.multiselect(
                    "Urgency Level Filter",
                    options=['Critical', 'High', 'Medium', 'Low'],
                    default=['Critical', 'High', 'Medium', 'Low']
                )
            
            with col2:
                mission_type_filter = st.multiselect(
                    "Mission Type Filter",
                    options=results['mission_type'].unique(),
                    default=results['mission_type'].unique()
                )
            
            with col3:
                top_n = st.slider("Number of tasks to show", 10, 500, 50, 10)
            
            # Apply Filters
            filtered = results[
                (results['urgency_level'].isin(urgency_filter)) &
                (results['mission_type'].isin(mission_type_filter))
            ]
            
            st.markdown(f"**Filtered Mission Counts:** {len(filtered):,}")
            
            # Map Visualization
            st.subheader("Mission Map Visualization")
            map_fig = create_map_visualization(filtered, top_n=min(top_n, 100))
            st.plotly_chart(map_fig, use_container_width=True)
            
            # Bar chart
            st.subheader("Mission Priority Chart")
            priority_fig = create_priority_chart(filtered, top_n=min(top_n, 30))
            st.plotly_chart(priority_fig, use_container_width=True)
            
            # Table
            st.subheader("Top Priority Missions Table")
            display_cols = ['rank', 'mission_id', 'mission_type', 'urgency_level', 
                          'threat_level', 'civilian_density', 'topsis_score']
            display_df = filtered.nsmallest(top_n, 'rank')[display_cols].reset_index(drop=True)
            display_df['topsis_score'] = display_df['topsis_score'].round(4)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Download Button
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Full Results as CSV",
                data=csv,
                file_name="mission_priorities.csv",
                mime="text/csv"
            )
        
        else:
            st.info("Click 'Start Analysis' in the sidebar to run the mission prioritization analysis.")
    
    # TAB 3: UAV Fleet Status
    with tab3:
        st.header("UAV Fleet Status Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uav_status_fig = create_uav_status_chart(uav_fleet)
            st.plotly_chart(uav_status_fig, use_container_width=True)
        
        with col2:
            uav_fuel_fig = create_uav_fuel_chart(uav_fleet)
            st.plotly_chart(uav_fuel_fig, use_container_width=True)
        
        # Model Distribution
        st.subheader("UAV Model Distribution")
        model_counts = uav_fleet['model'].value_counts()
        fig = px.bar(
            x=model_counts.index,
            y=model_counts.values,
            color=model_counts.index,
            title='UAV Models in Fleet'
        )
        fig.update_layout(showlegend=False, xaxis_title="Model", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Table
        st.subheader("UAV Fleet Details")
        st.dataframe(
            uav_fleet[[
                'uav_id', 'model', 'status', 'fuel_level', 
                'sensor_type', 'max_range', 'endurance'
            ]].sort_values('fuel_level', ascending=False),
            use_container_width=True,
            height=400
        )
    
    # TAB 4: Detailed Analysis
    with tab4:
        if 'results' in st.session_state:
            st.header("Detailed Statistical Analysis")
            
            results = st.session_state['results']
            
            # Criteria distributions
            st.subheader("Decision Criteria Distributions")
            criteria_fig = create_criteria_distribution(results)
            st.plotly_chart(criteria_fig, use_container_width=True)
            
            # Correlation Matrix
            st.subheader("Criteria Correlation Matrix")
            corr_cols = ['urgency_score', 'threat_level', 'civilian_density', 
                        'weather_suitability', 'uav_availability', 'topsis_score']
            corr_matrix = results[corr_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title='Correlation Matrix of Decision Criteria'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical Summary
            st.subheader("Statistical Summary of Criteria")
            st.dataframe(
                results[corr_cols].describe().round(3),
                use_container_width=True
            )
        else:
            st.info("Run the analysis first to view detailed statistics.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>UAV Mission Prioritization Decision Support System</strong></p>
        <p>AHP + TOPSIS Metodology | Information Systems Engineering Project</p>
        <p>Sinem Gokaslan - 2216001046</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()