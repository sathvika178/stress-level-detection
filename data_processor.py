import pandas as pd
import numpy as np
import io

class DataProcessor:
    """
    Data processing utilities for the stress level detection application
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.required_columns = ['Humidity', 'Temperature', 'Step count']
        self.optional_columns = ['Stress Level']
        
    def validate_data(self, data):
        """
        Validate that the data has the required structure and types
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data to validate
            
        Returns:
        --------
        is_valid : bool
            Whether the data is valid
        message : str
            Validation message or error explanation
        """
        # Check if dataframe is empty
        if data.empty:
            return False, "The uploaded file contains no data."
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check data types
        for col in self.required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                try:
                    data[col] = pd.to_numeric(data[col])
                except:
                    return False, f"Column '{col}' contains non-numeric values."
        
        # Check for target column if present
        if 'Stress Level' in data.columns:
            # Ensure Stress Level is numeric
            if not pd.api.types.is_numeric_dtype(data['Stress Level']):
                try:
                    data['Stress Level'] = pd.to_numeric(data['Stress Level'])
                except:
                    return False, "Column 'Stress Level' contains non-numeric values."
            
            # Check stress level values are valid (e.g., 0-4)
            unique_stress_levels = set(data['Stress Level'].unique())
            valid_stress_levels = {0, 1, 2, 3, 4}  # Assuming 5 stress levels (0-4)
            
            if not unique_stress_levels.issubset(valid_stress_levels):
                return False, f"Stress Level values should be in {valid_stress_levels}, found {unique_stress_levels}"
        
        # Check for missing values
        missing_values = data[self.required_columns].isnull().sum().sum()
        if missing_values > 0:
            return False, f"The dataset contains {missing_values} missing values in required columns."
        
        return True, "Data validation successful"
    
    def load_data(self, file):
        """
        Load data from an uploaded file
        
        Parameters:
        -----------
        file : UploadedFile
            The uploaded file object
            
        Returns:
        --------
        data : pandas.DataFrame or None
            The loaded data, or None if loading failed
        message : str
            Success or error message
        """
        try:
            # Check file extension
            if file.name.endswith('.csv'):
                data = pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(file)
            else:
                return None, "Unsupported file format. Please upload a CSV or Excel file."
            
            # Validate the loaded data
            is_valid, message = self.validate_data(data)
            if not is_valid:
                return None, message
            
            return data, "Data loaded successfully"
            
        except Exception as e:
            return None, f"Error loading data: {str(e)}"
    
    def generate_demo_data(self, n_samples=100):
        """
        Generate demonstration data for the application
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        data : pandas.DataFrame
            The generated data
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate random data
        humidity = np.random.uniform(30, 80, n_samples)  # Humidity percentage
        temperature = np.random.uniform(18, 35, n_samples)  # Temperature in Celsius
        steps = np.random.randint(1000, 15000, n_samples)  # Step count
        
        # Create a logic for assigning stress levels based on the features
        stress_level = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # High humidity, high temperature, low steps might indicate high stress
            h_factor = (humidity[i] - 30) / 50  # Normalize to 0-1
            t_factor = (temperature[i] - 18) / 17  # Normalize to 0-1
            s_factor = 1 - (steps[i] - 1000) / 14000  # Normalize to 0-1 (inverse)
            
            # Weighted sum of factors
            stress_score = 0.4 * h_factor + 0.4 * t_factor + 0.2 * s_factor
            
            # Assign stress level based on score
            if stress_score < 0.2:
                stress_level[i] = 0  # Very low stress
            elif stress_score < 0.4:
                stress_level[i] = 1  # Low stress
            elif stress_score < 0.6:
                stress_level[i] = 2  # Moderate stress
            elif stress_score < 0.8:
                stress_level[i] = 3  # High stress
            else:
                stress_level[i] = 4  # Very high stress
        
        # Create DataFrame
        data = pd.DataFrame({
            'Humidity': humidity,
            'Temperature': temperature,
            'Step count': steps,
            'Stress Level': stress_level
        })
        
        return data
    
    def prepare_single_prediction_data(self, humidity, temperature, step_count):
        """
        Prepare data for a single prediction
        
        Parameters:
        -----------
        humidity : float
            Humidity percentage
        temperature : float
            Temperature in Celsius
        step_count : int
            Number of steps
            
        Returns:
        --------
        data : pandas.DataFrame
            Data formatted for prediction
        """
        data = pd.DataFrame({
            'Humidity': [humidity],
            'Temperature': [temperature],
            'Step count': [step_count]
        })
        
        return data

    def get_stress_level_description(self, level):
        """
        Get a descriptive text for a stress level
        
        Parameters:
        -----------
        level : int
            Stress level (0-4)
            
        Returns:
        --------
        description : str
            Description of the stress level
        """
        descriptions = {
            0: "Very Low Stress: You're in a relaxed state with minimal stress indicators.",
            1: "Low Stress: You're experiencing mild stress levels, but still within a healthy range.",
            2: "Moderate Stress: You're under moderate stress. Consider taking short breaks.",
            3: "High Stress: You're experiencing significant stress. It would be beneficial to engage in stress-reducing activities.",
            4: "Very High Stress: Your stress levels are very high. Immediate stress management is recommended."
        }
        
        return descriptions.get(level, "Unknown stress level")
