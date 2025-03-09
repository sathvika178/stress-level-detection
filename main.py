import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from datetime import datetime

# Import custom modules
from model import StressLevelModel
from data_processor import DataProcessor
from visualization import StressVisualizer

# Set page config
st.set_page_config(
    page_title="Stress Level Detection",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state if not already initialized
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'model' not in st.session_state:
    st.session_state.model = StressLevelModel()
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'show_demo' not in st.session_state:
    st.session_state.show_demo = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Upload & Train"

# Initialize helper classes
data_processor = DataProcessor()
visualizer = StressVisualizer()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: 500;
    }
    .stAlert > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stress-metrics {
        background-color: #f1f8fe;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        color: #555;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        height: 60px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        box-shadow: 0px -2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<p class="main-header">Stress Level Detection System</p>', unsafe_allow_html=True)
st.markdown("""
This application uses machine learning to detect and classify stress levels based on environmental and physiological data.
Upload your data to train the model or use the prediction tools to analyze stress levels in real-time.
""")

# Create sidebar
with st.sidebar:
    st.title("❤️ Stress Analysis")
    
    st.markdown("### Navigation")
    selected_tab = st.radio(
        "Go to:",
        ["Upload & Train", "Prediction Dashboard", "Individual Prediction"]
    )
    
    # Update the active tab in session state
    if selected_tab != st.session_state.active_tab:
        st.session_state.active_tab = selected_tab
    
    st.markdown("---")
    
    st.markdown("### About Stress Levels")
    st.markdown("""
    The model classifies stress into 5 levels:
    - **0 - Very Low**: Minimal stress indicators
    - **1 - Low**: Mild stress within healthy range
    - **2 - Moderate**: Moderate stress levels
    - **3 - High**: Significant stress levels
    - **4 - Very High**: Severe stress, needs attention
    """)
    
    st.markdown("---")
    
    st.markdown("### Features Used")
    st.markdown("""
    The model analyzes these key data points:
    - **Humidity**: Environmental humidity (%)
    - **Temperature**: Environmental temperature (°C)
    - **Step Count**: Daily physical activity
    """)

# Tab content based on selection
if st.session_state.active_tab == "Upload & Train":
    st.markdown('<p class="sub-header">Upload Data & Train Model</p>', unsafe_allow_html=True)
    st.markdown("Upload your data file containing humidity, temperature, step count, and stress level information.")
    
    # File upload section
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        demo_data = st.checkbox("Use demonstration data instead", value=st.session_state.show_demo)
        if demo_data != st.session_state.show_demo:
            st.session_state.show_demo = demo_data
    
    with col2:
        if st.button("Download Sample Template"):
            sample_df = pd.DataFrame({
                'Humidity': [45.2, 67.8, 52.1],
                'Temperature': [24.7, 30.2, 22.5],
                'Step count': [8000, 3500, 10200],
                'Stress Level': [1, 3, 0]
            })
            
            csv = sample_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="stress_data_template.csv">Click here to download</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Process uploaded data or use demo data
    if uploaded_file is not None or st.session_state.show_demo:
        with st.spinner("Processing data..."):
            if uploaded_file is not None:
                data, message = data_processor.load_data(uploaded_file)
                if data is not None:
                    st.session_state.uploaded_data = data
                    st.success(message)
                else:
                    st.error(message)
                    st.session_state.uploaded_data = None
            
            if st.session_state.show_demo:
                st.session_state.uploaded_data = data_processor.generate_demo_data(n_samples=100)
                st.info("Using demonstration data for analysis.")
        
        # Display data if available
        if st.session_state.uploaded_data is not None:
            st.markdown("### Data Preview")
            st.dataframe(st.session_state.uploaded_data.head(10))
            
            # Data statistics
            st.markdown("### Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(st.session_state.uploaded_data))
            with col2:
                if 'Stress Level' in st.session_state.uploaded_data.columns:
                    avg_stress = round(st.session_state.uploaded_data['Stress Level'].mean(), 2)
                    st.metric("Average Stress Level", avg_stress)
            with col3:
                if 'Stress Level' in st.session_state.uploaded_data.columns:
                    mode_stress = st.session_state.uploaded_data['Stress Level'].mode()[0]
                    st.metric("Most Common Stress Level", mode_stress)
            
            # Train model button
            if st.button("Train Model"):
                with st.spinner("Training the model..."):
                    metrics = st.session_state.model.train(st.session_state.uploaded_data)
                    st.session_state.trained = True
                
                st.success("Model trained successfully!")
                
                # Display training metrics
                st.markdown("### Model Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
                    
                    # Feature importance plot
                    feature_importances = st.session_state.model.get_feature_importances()
                    fig_importance = visualizer.plot_feature_importance(feature_importances)
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                with col2:
                    # Confusion matrix as a heatmap
                    cm = metrics['confusion_matrix']
                    fig = px.imshow(
                        cm, 
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
                        y=['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
                        color_continuous_scale="Blues",
                        title="Confusion Matrix"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display data visualizations
            if 'Stress Level' in st.session_state.uploaded_data.columns:
                st.markdown("### Data Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Stress level distribution
                    fig_dist = visualizer.plot_stress_distribution(st.session_state.uploaded_data)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Feature correlations with stress
                    corr_data = st.session_state.uploaded_data[['Humidity', 'Temperature', 'Step count', 'Stress Level']].corr()['Stress Level'].drop('Stress Level').sort_values(ascending=False)
                    
                    fig = px.bar(
                        x=corr_data.index,
                        y=corr_data.values,
                        color=corr_data.values,
                        color_continuous_scale='RdBu_r',
                        labels={'x': 'Feature', 'y': 'Correlation with Stress Level'},
                        title='Feature Correlation with Stress Level'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature vs stress level plots
                st.markdown("### Feature Relationships with Stress")
                col1, col2, col3 = st.columns(3)
                
                figs = visualizer.plot_stress_factors(st.session_state.uploaded_data)
                if len(figs) >= 3:
                    with col1:
                        st.plotly_chart(figs[0], use_container_width=True)
                    with col2:
                        st.plotly_chart(figs[1], use_container_width=True)
                    with col3:
                        st.plotly_chart(figs[2], use_container_width=True)

elif st.session_state.active_tab == "Prediction Dashboard":
    st.markdown('<p class="sub-header">Prediction Dashboard</p>', unsafe_allow_html=True)
    
    if not st.session_state.trained and st.session_state.uploaded_data is None:
        st.warning("Please upload data and train the model first in the 'Upload & Train' tab.")
    else:
        # If model not trained but we have data, offer to train
        if not st.session_state.trained and st.session_state.uploaded_data is not None:
            if st.button("Train Model with Uploaded Data"):
                with st.spinner("Training the model..."):
                    metrics = st.session_state.model.train(st.session_state.uploaded_data)
                    st.session_state.trained = True
                st.success("Model trained successfully!")
        
        # If model is trained, run predictions
        if st.session_state.trained:
            # Make predictions on the uploaded data
            data = st.session_state.uploaded_data
            predictions, probabilities = st.session_state.model.predict(data)
            
            # Add predictions to the data
            data_with_pred = data.copy()
            data_with_pred['Predicted Stress'] = predictions
            
            # Store predictions in session state
            st.session_state.predictions = data_with_pred
            
            # Dashboard layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown('<div class="stress-metrics">', unsafe_allow_html=True)
                
                # Calculate average predicted stress
                avg_stress = data_with_pred['Predicted Stress'].mean()
                
                # Count stress levels
                stress_counts = data_with_pred['Predicted Stress'].value_counts().sort_index()
                
                # Highest stress count
                if len(stress_counts) > 0:
                    most_common_stress = stress_counts.idxmax()
                    most_common_count = stress_counts.max()
                    most_common_percent = (most_common_count / len(data_with_pred)) * 100
                
                    st.markdown(f"""
                    <p class="metric-label">Average Stress Level</p>
                    <p class="metric-value">{avg_stress:.1f}</p>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <p class="metric-label">Most Common Stress Level</p>
                    <p class="metric-value">{most_common_stress} ({most_common_percent:.1f}%)</p>
                    """, unsafe_allow_html=True)
                    
                    # High stress percentage (levels 3-4)
                    high_stress_count = stress_counts.get(3, 0) + stress_counts.get(4, 0)
                    high_stress_percent = (high_stress_count / len(data_with_pred)) * 100
                    
                    st.markdown(f"""
                    <p class="metric-label">High Stress Percentage</p>
                    <p class="metric-value">{high_stress_percent:.1f}%</p>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display correlation between features and predicted stress
                st.markdown("### Feature to Stress Correlation")
                
                corr_data = data_with_pred[['Humidity', 'Temperature', 'Step count', 'Predicted Stress']].corr()['Predicted Stress'].drop('Predicted Stress').sort_values(ascending=False)
                
                fig = px.bar(
                    x=corr_data.index,
                    y=corr_data.values,
                    color=corr_data.values,
                    color_continuous_scale='RdBu_r',
                    labels={'x': 'Feature', 'y': 'Correlation'},
                    title='Feature Correlation with Predicted Stress'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution of predicted stress levels
                fig_pred_dist = visualizer.plot_stress_distribution(data_with_pred.rename(columns={'Predicted Stress': 'Stress Level'}))
                if fig_pred_dist:
                    st.plotly_chart(fig_pred_dist, use_container_width=True)
                
                # Feature importance if model is trained
                if st.session_state.trained:
                    feature_importances = st.session_state.model.get_feature_importances()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Feature importance plot
                        fig_importance = visualizer.plot_feature_importance(feature_importances)
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    with col2:
                        # Show a scatter plot of the most important feature vs stress
                        most_important_feature = max(feature_importances, key=feature_importances.get)
                        
                        fig = px.scatter(
                            data_with_pred,
                            x=most_important_feature,
                            y='Predicted Stress',
                            color='Predicted Stress',
                            color_continuous_scale=visualizer.stress_colors,
                            labels={'Predicted Stress': 'Predicted Stress Level'},
                            title=f'Most Important Feature: {most_important_feature} vs Stress'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Data table with predictions
            st.markdown("### Prediction Results")
            st.dataframe(
                data_with_pred,
                column_config={
                    "Predicted Stress": st.column_config.NumberColumn(
                        "Predicted Stress Level",
                        format="%d"
                    )
                }
            )

elif st.session_state.active_tab == "Individual Prediction":
    st.markdown('<p class="sub-header">Individual Stress Level Prediction</p>', unsafe_allow_html=True)
    st.markdown("Enter environmental and physiological data to predict stress level for an individual.")
    
    if not st.session_state.trained:
        st.warning("Please train the model first in the 'Upload & Train' tab before making individual predictions.")
    else:
        # Input form for features
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                humidity = st.slider("Humidity (%)", min_value=30.0, max_value=90.0, value=50.0, step=0.1)
                temperature = st.slider("Temperature (°C)", min_value=15.0, max_value=40.0, value=25.0, step=0.1)
            
            with col2:
                step_count = st.number_input("Step Count", min_value=1000, max_value=20000, value=7500, step=100)
                
                # Time input (for reference only, not used in prediction)
                time_now = datetime.now().strftime("%H:%M")
                time_input = st.time_input("Time of measurement", value=None)
            
            submitted = st.form_submit_button("Predict Stress Level")
        
        if submitted:
            # Prepare data for prediction
            single_data = data_processor.prepare_single_prediction_data(humidity, temperature, step_count)
            
            # Make prediction
            prediction, probabilities = st.session_state.model.predict(single_data)
            
            # Display results
            stress_level = prediction[0]
            st.markdown(f"### Predicted Stress Level: {stress_level}")
            
            # Display stress level description
            description = data_processor.get_stress_level_description(stress_level)
            st.info(description)
            
            # Visual representation
            col1, col2 = st.columns(2)
            
            with col1:
                # Gauge chart
                fig_gauge = visualizer.create_gauge_chart(stress_level)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Probabilities chart
                fig_probs = visualizer.plot_prediction_probabilities(probabilities[0])
                st.plotly_chart(fig_probs, use_container_width=True)
            
            # Feature importance in this prediction
            st.markdown("### Feature Contribution to Prediction")
            
            # Get feature importances
            feature_importances = st.session_state.model.get_feature_importances()
            
            # Normalize input features for display
            humidity_norm = (humidity - 30) / 60  # Assuming range 30-90
            temp_norm = (temperature - 15) / 25  # Assuming range 15-40
            step_norm = (step_count - 1000) / 19000  # Assuming range 1000-20000
            
            # Create a dataframe for feature values and their importances
            feature_data = pd.DataFrame({
                'Feature': ['Humidity', 'Temperature', 'Step count'],
                'Value': [humidity, temperature, step_count],
                'Normalized Value': [humidity_norm, temp_norm, step_norm],
                'Importance': [
                    feature_importances.get('Humidity', 0),
                    feature_importances.get('Temperature', 0),
                    feature_importances.get('Step count', 0)
                ]
            })
            
            # Calculate contribution
            feature_data['Contribution'] = feature_data['Normalized Value'] * feature_data['Importance']
            
            # Normalize contribution to total 100%
            total_contribution = feature_data['Contribution'].abs().sum()
            if total_contribution > 0:
                feature_data['Contribution %'] = (feature_data['Contribution'] / total_contribution * 100).abs()
            else:
                feature_data['Contribution %'] = feature_data['Importance'] / feature_data['Importance'].sum() * 100
            
            # Sort by contribution
            feature_data = feature_data.sort_values('Contribution %', ascending=False)
            
            # Display as horizontal bar chart
            fig = px.bar(
                feature_data,
                y='Feature',
                x='Contribution %',
                orientation='h',
                color='Contribution %',
                color_continuous_scale='Blues',
                labels={'Contribution %': 'Contribution to Prediction (%)'},
                title='Feature Contribution to Stress Prediction'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Tips based on prediction
            st.markdown("### Stress Management Tips")
            
            if stress_level <= 1:
                st.success("""
                Your stress level is low. Keep up the good work!
                - Maintain your current routine
                - Continue with regular physical activity
                - Practice mindfulness to stay balanced
                """)
            elif stress_level == 2:
                st.info("""
                Your stress level is moderate. Consider these tips:
                - Take short breaks during the day
                - Practice deep breathing exercises
                - Ensure adequate sleep
                - Stay hydrated and maintain a balanced diet
                """)
            else:
                st.warning("""
                Your stress level is high. Consider these stress-reduction strategies:
                - Engage in physical activity to reduce stress hormones
                - Practice meditation or progressive muscle relaxation
                - Reduce caffeine and improve sleep quality
                - Consider adjusting your environment (temperature, humidity)
                - Talk to someone about your stressors
                - Take regular breaks from screens and work
                """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Stress Level Detection System | Made with ❤️ for health and wellness</p>
</div>
""", unsafe_allow_html=True)

# Run the app with the following command:
# streamlit run main.py
