"""
Model Training Module for Flood Prediction

Performs:
- Exploratory Data Analysis (EDA)
- Model training (Logistic Regression, Random Forest, Gradient Boosting)
- Model evaluation and comparison
- Bootstrap validation
- Best model selection and saving
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from preprocess import preprocess_data, load_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')


class FloodPredictionModel:
    """Main class for training and evaluating flood prediction models"""
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessed_data = None
        
    def perform_eda(self, filepath='data/flood_data.csv'):
        """
        Perform Exploratory Data Analysis
        
        Args:
            filepath (str): Path to the dataset
        """
        print("\n" + "="*60)
        print("📊 EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*60)
        
        data = load_data(filepath)
        
        # Head
        print("\n📋 First 5 rows:")
        print(data.head())
        
        # Info
        print("\n📋 Dataset Info:")
        print(data.info())
        
        # Describe
        print("\n📋 Statistical Summary:")
        print(data.describe())
        
        # Missing values
        print("\n❓ Missing Values:")
        missing = data.isnull().sum()
        if missing.sum() == 0:
            print("✅ No missing values found!")
        else:
            print(missing)
        
        # Class distribution
        print("\n🎯 Flood Risk Distribution:")
        flood_dist = data['flood_risk'].value_counts()
        print(flood_dist)
        print(f"\nFlood Risk Rate: {(flood_dist.get(1, 0) / len(data) * 100):.2f}%")
        
        return data
    
    def create_visualizations(self, data, filepath='data/flood_data.csv'):
        """
        Create visualizations for EDA
        
        Args:
            data (pd.DataFrame): Dataset
            filepath (str): Path for saving plots
        """
        print("\n📈 Creating Visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 12)
        
        # 1. Correlation Heatmap
        plt.subplot(2, 3, 1)
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        # 2. Distribution plots
        feature_names = [col for col in data.columns if col != 'flood_risk']
        
        # Rainfall distribution
        plt.subplot(2, 3, 2)
        plt.hist(data['rainfall_mm'], bins=30, color='skyblue', edgecolor='black')
        plt.title('Rainfall Distribution (mm)')
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Frequency')
        
        # River Level distribution
        plt.subplot(2, 3, 3)
        plt.hist(data['river_level_m'], bins=30, color='steelblue', edgecolor='black')
        plt.title('River Level Distribution (m)')
        plt.xlabel('River Level (m)')
        plt.ylabel('Frequency')
        
        # Soil Moisture distribution
        plt.subplot(2, 3, 4)
        plt.hist(data['soil_moisture_percent'], bins=30, color='lightcoral', edgecolor='black')
        plt.title('Soil Moisture Distribution (%)')
        plt.xlabel('Soil Moisture (%)')
        plt.ylabel('Frequency')
        
        # 3. Flood vs Non-flood countplot
        plt.subplot(2, 3, 5)
        flood_counts = data['flood_risk'].value_counts()
        colors = ['green', 'red']
        plt.bar(['No Flood', 'Flood Risk'], flood_counts.values, color=colors, edgecolor='black')
        plt.title('Flood Risk Distribution')
        plt.ylabel('Count')
        for i, v in enumerate(flood_counts.values):
            plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        # 4. Temperature distribution
        plt.subplot(2, 3, 6)
        plt.hist(data['temperature_c'], bins=30, color='orange', edgecolor='black')
        plt.title('Temperature Distribution (°C)')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
        print("✅ Visualizations saved as 'eda_visualizations.png'")
        plt.close()
    
    def train_models(self, X_train, y_train):
        """
        Train multiple models
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        print("\n" + "="*60)
        print("🤖 MODEL TRAINING")
        print("="*60)
        
        # Model 1: Logistic Regression
        print("\n🔄 Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr_model
        print("✅ Logistic Regression trained")
        
        # Model 2: Random Forest
        print("\n🌲 Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        self.models['Random Forest'] = rf_model
        print("✅ Random Forest trained")
        
        # Model 3: Gradient Boosting
        print("\n📈 Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb_model
        print("✅ Gradient Boosting trained")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Args:
            X_test (np.ndarray): Testing features
            y_test (np.ndarray): Testing labels
        """
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION")
        print("="*60)
        
        results = []
        
        for model_name, model in self.models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store metrics
            self.metrics[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            results.append({
                'Model': model_name,
                'Accuracy': f'{accuracy:.4f}',
                'Precision': f'{precision:.4f}',
                'Recall': f'{recall:.4f}',
                'F1 Score': f'{f1:.4f}',
                'ROC AUC': f'{roc_auc:.4f}'
            })
            
            print(f"  ✅ Accuracy:  {accuracy:.4f}")
            print(f"  ✅ Precision: {precision:.4f}")
            print(f"  ✅ Recall:    {recall:.4f}")
            print(f"  ✅ F1 Score:  {f1:.4f}")
            print(f"  ✅ ROC AUC:   {roc_auc:.4f}")
        
        # Display comparison table
        print("\n" + "="*60)
        print("📈 MODEL COMPARISON TABLE")
        print("="*60)
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def bootstrap_validation(self, X_train, y_train, n_iterations=100):
        """
        Perform bootstrap validation for the best model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            n_iterations (int): Number of bootstrap iterations
        """
        print("\n" + "="*60)
        print("🔄 BOOTSTRAP VALIDATION")
        print("="*60)
        
        if self.best_model_name is None:
            print("❌ No model selected. Please select best model first.")
            return
        
        model_class = self.models[self.best_model_name].__class__
        accuracies = []
        
        print(f"\n🎲 Running {n_iterations} bootstrap iterations on {self.best_model_name}...")
        
        for i in range(n_iterations):
            # Resample
            X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=i)
            
            # Train model
            if model_class == LogisticRegression:
                boot_model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_class == RandomForestClassifier:
                boot_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:  # GradientBoostingClassifier
                boot_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            
            boot_model.fit(X_boot, y_boot)
            
            # Evaluate
            accuracy = boot_model.score(X_train, y_train)
            accuracies.append(accuracy)
            
            if (i + 1) % 25 == 0:
                print(f"  ✅ Completed {i + 1}/{n_iterations} iterations")
        
        # Calculate confidence intervals
        accuracies = np.array(accuracies)
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        ci_95_lower = np.percentile(accuracies, 2.5)
        ci_95_upper = np.percentile(accuracies, 97.5)
        
        print(f"\n📊 Bootstrap Validation Results:")
        print(f"  Mean Accuracy: {mean_accuracy:.4f}")
        print(f"  Std Dev: {std_accuracy:.4f}")
        print(f"  95% CI Lower: {ci_95_lower:.4f}")
        print(f"  95% CI Upper: {ci_95_upper:.4f}")
        print(f"  Confidence Interval: [{ci_95_lower:.4f}, {ci_95_upper:.4f}]")
    
    def select_best_model(self):
        """
        Select the best model based on ROC AUC score
        """
        print("\n" + "="*60)
        print("🏆 BEST MODEL SELECTION")
        print("="*60)
        
        best_score = -1
        best_name = None
        
        for model_name, metrics in self.metrics.items():
            roc_auc = metrics['roc_auc']
            if roc_auc > best_score:
                best_score = roc_auc
                best_name = model_name
        
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        print(f"\n🌟 Best Model: {best_name}")
        print(f"🌟 ROC AUC Score: {best_score:.4f}")
        print(f"🌟 Accuracy: {self.metrics[best_name]['accuracy']:.4f}")
        print(f"🌟 F1 Score: {self.metrics[best_name]['f1']:.4f}")
    
    def save_model(self, filepath='models/flood_model.pkl'):
        """
        Save the best model using joblib
        
        Args:
            filepath (str): Path to save the model
        """
        if self.best_model is None:
            print("❌ No model to save. Please train and select a model first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.best_model, filepath)
        print(f"\n✅ Model saved successfully to: {filepath}")
        print(f"   Model Type: {self.best_model_name}")
        
        # Also save the scaler
        if self.preprocessed_data and 'scaler' in self.preprocessed_data:
            scaler_path = os.path.join(os.path.dirname(filepath), 'scaler.pkl')
            joblib.dump(self.preprocessed_data['scaler'], scaler_path)
            print(f"✅ Scaler saved successfully to: {scaler_path}")
    
    def run_full_pipeline(self):
        """
        Run the complete training pipeline
        """
        print("\n" + "█"*60)
        print("█" + " "*58 + "█")
        print("█" + "  RISING WATERS - FLOOD PREDICTION MODEL TRAINING".center(58) + "█")
        print("█" + " "*58 + "█")
        print("█"*60)
        
        # Step 1: EDA
        data = self.perform_eda()
        self.create_visualizations(data)
        
        # Step 2: Preprocessing
        print("\n🔧 Preprocessing data...")
        self.preprocessed_data = preprocess_data()
        
        X_train = self.preprocessed_data['X_train']
        X_test = self.preprocessed_data['X_test']
        y_train = self.preprocessed_data['y_train']
        y_test = self.preprocessed_data['y_test']
        
        # Step 3: Model Training
        self.train_models(X_train, y_train)
        
        # Step 4: Model Evaluation
        self.evaluate_models(X_test, y_test)
        
        # Step 5: Select Best Model
        self.select_best_model()
        
        # Step 6: Bootstrap Validation
        self.bootstrap_validation(X_train, y_train, n_iterations=100)
        
        # Step 7: Save Best Model
        self.save_model()
        
        print("\n" + "█"*60)
        print("█" + " "*58 + "█")
        print("█" + "  ✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY! ✅".center(58) + "█")
        print("█" + " "*58 + "█")
        print("█"*60 + "\n")


if __name__ == "__main__":
    # Create and run the training pipeline
    trainer = FloodPredictionModel()
    trainer.run_full_pipeline()
