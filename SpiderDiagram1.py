import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from math import pi, cos, sin

# Configure the page
st.set_page_config(
    page_title="Reservoir QC Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff0000;
    }
    .risk-medium {
        background-color: #fff4cc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffaa00;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #00aa00;
    }
    .category-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0px;
        border-left: 4px solid #1f77b4;
    }
    .tab-container {
        background-color: #fafafa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state with default values"""
    if 'scores' not in st.session_state:
        st.session_state.scores = {
            'Log Data QC': 5,
            'Seismic Input & Wavelet': 4,
            'Low-Freq Model': 2,
            'Rock Physics Model': 3,
            'Final Interpretation': 6,
            'Well-Tie & Blind Test': 7
        }
    
    if 'weights' not in st.session_state:
        st.session_state.weights = {
            'Log Data QC': 'Low',
            'Seismic Input & Wavelet': 'Medium',
            'Low-Freq Model': 'High',
            'Rock Physics Model': 'High',
            'Final Interpretation': 'Medium',
            'Well-Tie & Blind Test': 'Medium'
        }

def create_spider_chart(scores, weights):
    """Create the 2D spider/radar chart"""
    categories = list(scores.keys())
    score_values = list(scores.values())
    
    fig = go.Figure()
    
    # Add target area (ideal scores)
    fig.add_trace(go.Scatterpolar(
        r=[9] * len(categories) + [9],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(color='green', width=2, dash='dot'),
        name='Target (Ideal)'
    ))
    
    # Add project scores
    fig.add_trace(go.Scatterpolar(
        r=score_values + [score_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(65, 105, 225, 0.6)',
        line=dict(color='royalblue', width=3),
        name='Project Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[0, 2, 4, 6, 8, 10],
                ticktext=['0 (Poor)', '2', '4', '6', '8', '10 (Excellent)'],
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                gridcolor='lightgray'
            )
        ),
        showlegend=True,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_3d_spider_chart(scores, weights):
    """Create a 3D spider/radar chart with height representing weights"""
    categories = list(scores.keys())
    score_values = list(scores.values())
    weight_values = list(weights.values())
    
    # Convert weights to numerical values for height
    weight_map = {'Low': 1, 'Medium': 2, 'High': 3}
    height_values = [weight_map[weight] for weight in weight_values]
    
    n_categories = len(categories)
    
    # Create angles for each category
    angles = [2 * pi * i / n_categories for i in range(n_categories)]
    angles += angles[:1]  # Close the circle
    
    # Prepare data for 3D plot
    x = []
    y = []
    z = []
    colors = []
    
    # Color mapping based on score
    for i, (score, height) in enumerate(zip(score_values + [score_values[0]], height_values + [height_values[0]])):
        angle = angles[i]
        radius = score
        
        x.append(radius * cos(angle))
        y.append(radius * sin(angle))
        z.append(height)
        
        # Color based on score (red to green)
        if score < 4:
            colors.append('red')
        elif score < 7:
            colors.append('orange')
        else:
            colors.append('green')
    
    fig = go.Figure()
    
    # Add the main surface
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        colorbar_title='Score Quality',
        colorscale=[[0, 'red'], [0.5, 'orange'], [1, 'green']],
        intensity=score_values + [score_values[0]],
        opacity=0.7,
        name='QC Surface'
    ))
    
    # Add edges
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color='darkblue', width=4),
        name='Project Profile'
    ))
    
    # Add markers at each vertex
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(size=6, color=colors),
        text=categories + [categories[0]],
        textposition="middle center",
        name='Categories'
    ))
    
    # Add target surface (ideal)
    target_x = [9 * cos(angle) for angle in angles]
    target_y = [9 * sin(angle) for angle in angles]
    target_z = [2] * len(angles)  # Medium height for target
    
    fig.add_trace(go.Scatter3d(
        x=target_x, y=target_y, z=target_z,
        mode='lines',
        line=dict(color='green', width=3, dash='dash'),
        name='Target Profile'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Weight Importance',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        title="3D QC Spider Diagram<br><sub>Radius = Score, Height = Weight Importance</sub>",
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def create_3d_bars_chart(scores, weights):
    """Create a 3D bar chart showing scores and weights"""
    categories = list(scores.keys())
    score_values = list(scores.values())
    weight_values = list(weights.values())
    
    # Convert weights to numerical values for z-axis
    weight_map = {'Low': 1, 'Medium': 2, 'High': 3}
    height_values = [weight_map[weight] for weight in weight_values]
    
    # Colors based on score
    colors = []
    for score in score_values:
        if score < 4:
            colors.append('red')
        elif score < 7:
            colors.append('orange')
        else:
            colors.append('green')
    
    fig = go.Figure()
    
    # Create 3D bars
    for i, (category, score, height, color) in enumerate(zip(categories, score_values, height_values, colors)):
        fig.add_trace(go.Mesh3d(
            # Define the 8 vertices of the bar
            x=[i-0.3, i+0.3, i+0.3, i-0.3, i-0.3, i+0.3, i+0.3, i-0.3],
            y=[-0.3, -0.3, 0.3, 0.3, -0.3, -0.3, 0.3, 0.3],
            z=[0, 0, 0, 0, score, score, score, score],
            i=[0, 0, 0, 2, 4, 6],
            j=[1, 2, 3, 3, 5, 7],
            k=[2, 3, 1, 7, 6, 4],
            color=color,
            opacity=0.8,
            name=category
        ))
        
        # Add weight indicator as a smaller bar on top
        fig.add_trace(go.Mesh3d(
            x=[i-0.2, i+0.2, i+0.2, i-0.2, i-0.2, i+0.2, i+0.2, i-0.2],
            y=[-0.2, -0.2, 0.2, 0.2, -0.2, -0.2, 0.2, 0.2],
            z=[score, score, score, score, score+height*0.5, score+height*0.5, score+height*0.5, score+height*0.5],
            i=[0, 0, 0, 2, 4, 6],
            j=[1, 2, 3, 3, 5, 7],
            k=[2, 3, 1, 7, 6, 4],
            color='blue' if height == 1 else 'orange' if height == 2 else 'red',
            opacity=0.6,
            showlegend=False
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Categories',
                ticktext=categories,
                tickvals=list(range(len(categories)))
            ),
            yaxis=dict(title=''),
            zaxis=dict(title='Score & Weight', range=[0, 13]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        title="3D QC Bars<br><sub>Bar Height = Score, Top Section = Weight</sub>",
        height=600,
        showlegend=False
    )
    
    return fig

def create_bar_chart(scores, weights):
    """Create a 2D bar chart showing scores with weight-based coloring"""
    categories = list(scores.keys())
    score_values = list(scores.values())
    weight_values = list(weights.values())
    
    # Color mapping for weights
    color_map = {'Low': '#636efa', 'Medium': '#ef553b', 'High': '#00cc96'}
    colors = [color_map[weight] for weight in weight_values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=score_values,
            marker_color=colors,
            text=score_values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Score: %{y}/10<br>Weight: %{customdata}<extra></extra>',
            customdata=weight_values
        )
    ])
    
    fig.update_layout(
        title="QC Scores by Category",
        xaxis_title="Categories",
        yaxis_title="Score (0-10)",
        yaxis=dict(range=[0, 10]),
        height=400,
        showlegend=False
    )
    
    return fig

def calculate_risk_assessment(scores, weights):
    """Calculate risk assessment and generate recommendations"""
    overall_score = sum(scores.values()) / len(scores)
    
    # Count critical issues (score < 5 with high weight)
    critical_issues = []
    warning_issues = []
    
    for category, score in scores.items():
        weight = weights[category]
        
        if score < 4:
            critical_issues.append((category, score, weight))
        elif score < 6 and weight == 'High':
            critical_issues.append((category, score, weight))
        elif score < 6:
            warning_issues.append((category, score, weight))
    
    # Determine risk level
    if overall_score >= 8 and not critical_issues:
        risk_level = "LOW RISK"
        risk_class = "risk-low"
        recommendation = "✅ Project is in good condition. Continue current approach."
    elif overall_score >= 6 and len(critical_issues) <= 1:
        risk_level = "MEDIUM RISK"
        risk_class = "risk-medium"
        recommendation = "⚠️ Some areas need attention. Review recommendations below."
    else:
        risk_level = "HIGH RISK"
        risk_class = "risk-high"
        recommendation = "🚨 Immediate action required. Critical issues identified."
    
    return {
        'overall_score': overall_score,
        'risk_level': risk_level,
        'risk_class': risk_class,
        'recommendation': recommendation,
        'critical_issues': critical_issues,
        'warning_issues': warning_issues
    }

def generate_recommendations(scores, weights):
    """Generate detailed recommendations for each category"""
    recommendations = []
    
    for category, score in scores.items():
        weight = weights[category]
        
        if score < 4:
            if weight == 'High':
                rec = "🚨 CRITICAL: Immediate action required - this will severely impact results"
            else:
                rec = "🚨 HIGH PRIORITY: Address soon - foundation quality issue"
        elif score < 6:
            if weight == 'High':
                rec = "⚠️ HIGH PRIORITY: Needs improvement - high impact on final result"
            else:
                rec = "⚠️ Review recommended: Moderate impact on quality"
        elif score < 8:
            rec = "✅ Acceptable: Monitor and maintain current approach"
        else:
            rec = "⭐ Excellent: Current approach working well"
        
        recommendations.append({
            'category': category,
            'score': score,
            'weight': weight,
            'recommendation': rec
        })
    
    return recommendations

def main():
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🛢️ Reservoir Characterization QC Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.header("📊 QC Parameters")
    
    # Add reset button
    if st.sidebar.button("Reset to Default Values"):
        initialize_session_state()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Input controls for each category
    categories = [
        'Log Data QC', 'Seismic Input & Wavelet', 'Low-Freq Model',
        'Rock Physics Model', 'Final Interpretation', 'Well-Tie & Blind Test'
    ]
    
    weight_descriptions = {
        'Low': 'Foundation - must be correct but straightforward to fix',
        'Medium': 'Critical for accuracy - significant impact on results', 
        'High': 'Core translator - very high impact on final interpretation'
    }
    
    # Create input controls in sidebar
    for category in categories:
        st.sidebar.markdown(f'<div class="category-card"><h4>{category}</h4></div>', unsafe_allow_html=True)
        
        # Score slider
        new_score = st.sidebar.slider(
            f"Score (0-10)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.scores[category]),
            step=0.5,
            key=f"score_{category}"
        )
        st.session_state.scores[category] = new_score
        
        # Weight selector
        new_weight = st.sidebar.selectbox(
            "Weight",
            options=['Low', 'Medium', 'High'],
            index=['Low', 'Medium', 'High'].index(st.session_state.weights[category]),
            key=f"weight_{category}"
        )
        st.session_state.weights[category] = new_weight
        
        st.sidebar.caption(f"**{new_weight} Weight**: {weight_descriptions[new_weight]}")
        st.sidebar.markdown("---")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 2D Spider Chart", "🔄 3D Spider Chart", "📏 3D Bars", "📈 Analysis"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("2D QC Spider Diagram")
            spider_fig = create_spider_chart(st.session_state.scores, st.session_state.weights)
            st.plotly_chart(spider_fig, use_container_width=True)
        
        with col2:
            st.subheader("Scores Overview")
            bar_fig = create_bar_chart(st.session_state.scores, st.session_state.weights)
            st.plotly_chart(bar_fig, use_container_width=True)
    
    with tab2:
        st.subheader("3D QC Spider Diagram")
        st.markdown("""
        **Visualization Guide:**
        - **Radius** represents the **Score** (0-10)
        - **Height** represents the **Weight Importance** (Low=1, Medium=2, High=3)
        - **Color** indicates quality (Red=Poor, Orange=Medium, Green=Good)
        - **Dashed green line** shows the target profile
        """)
        spider_3d_fig = create_3d_spider_chart(st.session_state.scores, st.session_state.weights)
        st.plotly_chart(spider_3d_fig, use_container_width=True)
    
    with tab3:
        st.subheader("3D QC Bars Visualization")
        st.markdown("""
        **Visualization Guide:**
        - **Bar Height** represents the **Score** (0-10)
        - **Top Colored Section** represents the **Weight** (Blue=Low, Orange=Medium, Red=High)
        - **Base Color** indicates quality (Red=Poor, Orange=Medium, Green=Good)
        """)
        bars_3d_fig = create_3d_bars_chart(st.session_state.scores, st.session_state.weights)
        st.plotly_chart(bars_3d_fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Risk assessment
            risk_data = calculate_risk_assessment(st.session_state.scores, st.session_state.weights)
            
            st.subheader("Risk Assessment")
            st.markdown(f'<div class="{risk_data["risk_class"]}">', unsafe_allow_html=True)
            st.metric("Overall Score", f"{risk_data['overall_score']:.1f}/10.0")
            st.metric("Risk Level", risk_data['risk_level'])
            st.write(risk_data['recommendation'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Critical issues
            if risk_data['critical_issues']:
                st.subheader("🚨 Critical Issues")
                for category, score, weight in risk_data['critical_issues']:
                    st.error(f"**{category}**: Score {score}/10, {weight} Weight")
            
            # Warning issues
            if risk_data['warning_issues']:
                st.subheader("⚠️ Warning Issues")
                for category, score, weight in risk_data['warning_issues']:
                    st.warning(f"**{category}**: Score {score}/10, {weight} Weight")
        
        with col2:
            # Detailed recommendations table
            st.subheader("📋 Detailed Recommendations")
            
            recommendations = generate_recommendations(st.session_state.scores, st.session_state.weights)
            
            # Display as table with formatting
            for rec in recommendations:
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    st.write(f"**{rec['category']}**")
                with col2:
                    st.metric("Score", f"{rec['score']}/10", label_visibility="collapsed")
                with col3:
                    st.write(rec['recommendation'])
                st.markdown("---")
    
    # Export functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Results")
    
    if st.sidebar.button("Generate Report"):
        # Create a comprehensive report
        report_data = {
            'Category': list(st.session_state.scores.keys()),
            'Score': list(st.session_state.scores.values()),
            'Weight': list(st.session_state.weights.values()),
            'Overall_Score': risk_data['overall_score'],
            'Risk_Level': risk_data['risk_level']
        }
        
        report_df = pd.DataFrame(report_data)
        csv = report_df.to_csv(index=False)
        
        st.sidebar.download_button(
            label="Download CSV Report",
            data=csv,
            file_name="reservoir_qc_report.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
