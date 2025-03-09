import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st

class StressVisualizer:
    """
    Visualization utilities for the stress level detection application
    """
    
    def __init__(self):
        """Initialize the visualizer with color schemes and settings"""
        # Color scheme for stress levels (calming blue to alert red)
        self.stress_colors = ['#4575b4', '#74add1', '#fee090', '#f46d43', '#d73027']
        self.stress_level_names = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        
    def plot_stress_distribution(self, data):
        """
        Create a bar chart showing the distribution of stress levels
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data containing stress level information
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            The plotly figure object
        """
        if 'Stress Level' not in data.columns:
            return None
        
        # Count the frequency of each stress level
        stress_counts = data['Stress Level'].value_counts().sort_index()
        
        # Map stress levels to names
        level_names = [self.stress_level_names[level] for level in stress_counts.index 
                      if level < len(self.stress_level_names)]
        
        # Create the bar chart
        fig = px.bar(
            x=level_names,
            y=stress_counts.values,
            color=stress_counts.index,
            color_continuous_scale=self.stress_colors,
            labels={'x': 'Stress Level', 'y': 'Count', 'color': 'Level'},
            title='Distribution of Stress Levels'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Stress Level',
            yaxis_title='Count',
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importances):
        """
        Create a horizontal bar chart showing feature importances
        
        Parameters:
        -----------
        feature_importances : dict
            Dictionary of feature names and their importance scores
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            The plotly figure object
        """
        features = list(feature_importances.keys())
        importances = list(feature_importances.values())
        
        # Sort features by importance
        sorted_idx = np.argsort(importances)
        features = [features[i] for i in sorted_idx]
        importances = [importances[i] for i in sorted_idx]
        
        # Create the horizontal bar chart
        fig = px.bar(
            x=importances,
            y=features,
            orientation='h',
            color=importances,
            color_continuous_scale='Blues',
            labels={'x': 'Importance', 'y': 'Feature'},
            title='Feature Importance'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        
        return fig
    
    def plot_stress_factors(self, data):
        """
        Create scatter plots showing the relationship between features and stress
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data containing features and stress level information
            
        Returns:
        --------
        figs : list of plotly.graph_objects.Figure
            List of plotly figure objects
        """
        if 'Stress Level' not in data.columns:
            return []
        
        features = ['Humidity', 'Temperature', 'Step count']
        figs = []
        
        for feature in features:
            # Create scatter plot
            fig = px.scatter(
                data,
                x=feature,
                y='Stress Level',
                color='Stress Level',
                color_continuous_scale=self.stress_colors,
                labels={'Stress Level': 'Stress Level'},
                title=f'Stress Level vs {feature}'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=feature,
                yaxis_title='Stress Level',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300
            )
            
            figs.append(fig)
        
        return figs
    
    def create_gauge_chart(self, stress_level, probabilities=None):
        """
        Create a gauge chart for displaying the predicted stress level
        
        Parameters:
        -----------
        stress_level : int
            The predicted stress level (0-4)
        probabilities : array-like, optional
            Prediction probabilities for each stress level
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            The plotly figure object
        """
        # Ensure stress level is valid
        if stress_level < 0 or stress_level > 4:
            stress_level = 0
        
        # Create the gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=stress_level,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Stress Level", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 4], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': self.stress_colors[stress_level]},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 1], 'color': self.stress_colors[0]},
                    {'range': [1, 2], 'color': self.stress_colors[1]},
                    {'range': [2, 3], 'color': self.stress_colors[2]},
                    {'range': [3, 4], 'color': self.stress_colors[3]},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': stress_level
                }
            }
        ))
        
        # Add annotation for stress level name
        fig.add_annotation(
            x=0.5,
            y=0.25,
            text=self.stress_level_names[stress_level],
            font={'size': 20, 'color': self.stress_colors[stress_level]},
            showarrow=False
        )
        
        # Update layout
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        return fig
    
    def plot_prediction_probabilities(self, probabilities):
        """
        Create a bar chart showing prediction probabilities for each stress level
        
        Parameters:
        -----------
        probabilities : array-like
            Prediction probabilities for each stress level
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            The plotly figure object
        """
        # Ensure probabilities array has 5 values (for 5 stress levels)
        if len(probabilities) != 5:
            return None
        
        # Create the bar chart
        fig = px.bar(
            x=self.stress_level_names,
            y=probabilities,
            color=self.stress_level_names,
            color_discrete_sequence=self.stress_colors,
            labels={'x': 'Stress Level', 'y': 'Probability'},
            title='Prediction Probabilities for Each Stress Level'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Stress Level',
            yaxis_title='Probability',
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        
        return fig
