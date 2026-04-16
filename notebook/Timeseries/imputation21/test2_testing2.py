import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# import scipy.stats as stats
# import statsmodels.api as sm
import miceforest as mf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error


from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from tqdm import tqdm
import os



from tensorflow.keras.layers import Lambda, Dropout, Concatenate
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from sklearn.metrics import mean_absolute_error, mean_squared_error as sk_mse




from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
import warnings
import time
warnings.filterwarnings('ignore')


from sklearn.base import BaseEstimator, TransformerMixin

import random
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp

from sklearn.ensemble import RandomForestRegressor


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import datetime
import matplotlib.pyplot as plt
import os
import sys

import math 


# Read the CSV file - linked file 
file_path = "../Merged_REDCap.xlsx" 
df = pd.read_excel(file_path)

#drop dates
df = df.drop(columns=["ComplicationDate", "dob", "operation_date", "DischargeDate"]) #qol_date

#drop unused columns
df = df.drop(columns=['age_diagnosis', 'gender',
        'overall_primary_tumour', 'overall_regional_ln',
        'overall_distant_metastasis', 'neotx___notx', 'neotx___chemo',
        'neotx___rads', 'neotx___chemorads', 'neotx___immuno', 'neotx___other',
        'procedure123456', 'expectation_treatment', 'path_esoph_primtumour',
        'path_esoph_regionalln', 'path_esoph_distantmetast', 'readmission_30d',
        'postop_comp', 'los']) 


# Ordinal columns: Treat as numerical (already float)
ordinal_cols = (
    [f"gp{i}" for i in range(1, 8)] + [f"gs{i}" for i in range(1, 8)] +
    [f"ge{i}" for i in range(1, 7)] + [f"gf{i}" for i in range(1, 8)] +
    [f"a_hn{i}" for i in range(1, 6)] + ["a_hn7", "a_hn10"] +
    [f"a_e{i}" for i in range(1, 8)] + ["a_c6", "a_c2", "a_act11"]
)
# Subset for this example
ordinal_cols = [col for col in ordinal_cols if col in df.columns]
for col in ordinal_cols:
    df[col] = df[col].astype(float)

columns_to_check = ['gp1', 'gp2', 'gp3', 'gp4', 'gp5', 'gp6', 'gp7', 
                    'gs1', 'gs2', 'gs3', 'gs4', 'gs5', 'gs6', 'gs7', 
                    'ge1', 'ge2', 'ge3', 'ge4', 'ge5', 'ge6', 
                    'gf1', 'gf2', 'gf3', 'gf4', 'gf5', 'gf6', 'gf7', 
                    'a_hn1', 'a_hn2', 'a_hn3', 'a_hn4', 'a_hn5', 'a_hn7', 'a_hn10', 
                    'a_e1', 'a_e2', 'a_e3', 'a_e4', 'a_e5', 'a_e6', 'a_e7', 
                    'a_c6', 'a_c2', 'a_act11']

# Keep only rows with at least one non-missing value
df = df[df[columns_to_check].notna().any(axis=1)]

# Remove duplicates where all 44 FACT-E scores are identical for the same patient
df = df.drop_duplicates(subset=['id'] + columns_to_check, keep='first')

# Count unique patients and patients with multiple rows
unique_patients = df['id'].nunique()
patients_with_multiple = (df['id'].value_counts() > 1).sum()

print(f"Unique patients: {unique_patients}")
print(f"Patients with multiple rows: {patients_with_multiple}")

# Get the value counts for patients with multiple rows
multiple_rows_counts = df['id'].value_counts()
patients_with_multiple_rows = multiple_rows_counts[multiple_rows_counts > 1]

print("Patients with multiple rows and their counts:")
print(patients_with_multiple_rows)

# If you want to see the distribution of how many patients have 2 rows, 3 rows, etc.
print("\nDistribution of row counts:")
print(patients_with_multiple_rows.value_counts().sort_index())

# Define target variables
target_vars = ([f"gp{i}" for i in range(1, 8)] + [f"gs{i}" for i in range(1, 8)] +
                [f"ge{i}" for i in range(1, 7)] + [f"gf{i}" for i in range(1, 8)] +
                [f"a_hn{i}" for i in range(1, 6)] + ["a_hn7", "a_hn10"] +
                [f"a_e{i}" for i in range(1, 8)] + ["a_c6", "a_c2", "a_act11"])

# Calculate patient-level missingness
patient_level = df.groupby('id')[target_vars].apply(lambda x: x.notna().any()).mean()
patient_missingness = (1 - patient_level) * 100

print(f"Patient-level missingness range: {patient_missingness.min():.1f}% - {patient_missingness.max():.1f}%")

# Add these imports at the top of your file
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score

def process_for_classification(values, min_val=0, max_val=4):
    """
    Process imputed values for classification evaluation
    
    Parameters:
    -----------
    values : array-like
        Raw imputed values
    min_val : int
        Minimum allowed value (default: 0)
    max_val : int  
        Maximum allowed value (default: 4)
        
    Returns:
    --------
    numpy.ndarray
        Processed values as integers in range [min_val, max_val]
    """
    # Round to nearest integer
    rounded = np.round(values)
    # Clip to valid range
    clipped = np.clip(rounded, min_val, max_val)
    return clipped.astype(int)


def calculate_classification_metrics(y_true, y_pred, classes=None):
    """
    Calculate classification metrics for multi-class problem
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    classes : array-like, optional
        Class labels (default: [0, 1, 2, 3, 4])
        
    Returns:
    --------
    dict
        Dictionary containing classification metrics
    """
    if classes is None:
        classes = np.array([0, 1, 2, 3, 4])
    
    try:
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Macro-averaged metrics (average across classes)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted-averaged metrics (weighted by class frequency)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # For multi-class AUC, we need to binarize the labels
        try:
            # Only calculate AUC if we have more than one class present
            if len(np.unique(y_true)) > 1:
                y_true_bin = label_binarize(y_true, classes=classes)
                y_pred_bin = label_binarize(y_pred, classes=classes)
                
                # If only 2 classes present, reshape
                if y_true_bin.shape[1] == 1:
                    auc_score = roc_auc_score(y_true_bin, y_pred_bin)
                else:
                    # Multi-class AUC (macro average)
                    auc_score = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')
            else:
                auc_score = np.nan
        except:
            auc_score = np.nan
        
        # Per-class metrics
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # Calculate sensitivity (recall) and specificity for each class
        per_class_metrics = {}
        for i, class_label in enumerate(classes):
            if i < cm.shape[0] and i < cm.shape[1]:
                tp = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
                fp = cm[:, i].sum() - tp if i < cm.shape[1] else 0
                fn = cm[i, :].sum() - tp if i < cm.shape[0] else 0
                tn = cm.sum() - tp - fp - fn
                
                # Sensitivity (recall/true positive rate)
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Specificity (true negative rate)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # PPV (precision)
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                # NPV
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                per_class_metrics[f'class_{class_label}'] = {
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'ppv': ppv,
                    'npv': npv
                }
        
        # Average across classes for summary
        avg_sensitivity = np.mean([metrics['sensitivity'] for metrics in per_class_metrics.values()])
        avg_specificity = np.mean([metrics['specificity'] for metrics in per_class_metrics.values()])
        avg_ppv = np.mean([metrics['ppv'] for metrics in per_class_metrics.values()])
        avg_npv = np.mean([metrics['npv'] for metrics in per_class_metrics.values()])
        
        return {
            'accuracy': accuracy,
            'auc_multiclass': auc_score,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'avg_sensitivity': avg_sensitivity,
            'avg_specificity': avg_specificity,
            'avg_ppv': avg_ppv,
            'avg_npv': avg_npv,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm
        }
        
    except Exception as e:
        print(f"Error calculating classification metrics: {e}")
        return {
            'accuracy': np.nan,
            'auc_multiclass': np.nan,
            'precision_macro': np.nan,
            'recall_macro': np.nan,
            'precision_weighted': np.nan,
            'recall_weighted': np.nan,
            'avg_sensitivity': np.nan,
            'avg_specificity': np.nan,
            'avg_ppv': np.nan,
            'avg_npv': np.nan,
            'per_class_metrics': {},
            'confusion_matrix': None
        }
        
def apply_mice_imputation(df, columns_to_impute, validation_df=None, validation_masks=None, original_values=None):
    """
    Apply MICE imputation using miceforest package
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with missing values
    columns_to_impute : list
        List of column names to impute
    validation_df : pandas.DataFrame, optional
        Validation dataset with artificially missing values
    validation_masks : dict, optional
        Dictionary of masks for validation data
    original_values : dict, optional
        Dictionary of original values for validation
        
    Returns:
    --------
    imputed_df : pandas.DataFrame
        Data with imputed values
    validation_results : dict, optional
        Validation results if validation data provided
    """

    # Set threads for LightGBM
    os.environ['OMP_NUM_THREADS'] = '10'
    
    # # Initialize the imputation kernel
    # kernel = mf.ImputationKernel(
    #     df,
    #     datasets=1,
    #     variable_schema={
    #         col: [c for c in df.columns if c != col] for col in columns_to_impute
    #     },
    #     random_state=42  # Using fixed seed for reproducibility, can be parameterized
    # )
    
    # Drop non-numeric columns that miceforest cannot handle
    df_mice = df.select_dtypes(exclude=['object', 'datetime64[ns]', 'datetime64']).copy()

    kernel = mf.ImputationKernel(
        df_mice,
        datasets=1,
        variable_schema={
            col: [c for c in df_mice.columns if c != col] for col in columns_to_impute
        },
        random_state=42
    )
    
    # Run imputation
    for _ in tqdm(range(5), desc="MICE Imputation"):
        kernel.mice(
            iterations=1,
            verbose=False,
            num_boost_round=80,
            max_depth=10,
            num_threads=10
        )
    
    # Get imputed data
    # imputed_df = kernel.complete_data(0)
    imputed_mice = kernel.complete_data(0)
    # Put results back into a copy of the original df (which still has redcap_event_name etc.)
    imputed_df = df.copy()
    for col in columns_to_impute:
        if col in imputed_mice.columns:
            imputed_df[col] = imputed_mice[col].values
    
    # Check if there's a label encoder for redcap_event_name that needs inverse transformation
    # if 'redcap_event_name' in imputed_df.columns:
    #     try:
    #         # This is optional - only execute if le_redcap exists in the global scope
    #         if 'le_redcap' in globals():
    #             # Check if we're dealing with numeric values (could be int or float)
    #             if pd.api.types.is_numeric_dtype(imputed_df['redcap_event_name']) or \
    #                (hasattr(imputed_df['redcap_event_name'], 'cat') and pd.api.types.is_numeric_dtype(imputed_df['redcap_event_name'].cat.categories)):
    #                 imputed_df['redcap_event_name'] = globals()['le_redcap'].inverse_transform(imputed_df['redcap_event_name'].astype(int))
    #     except Exception as e:
    #         print(f"Warning: Could not inverse transform redcap_event_name: {e}")
    
    # Validate if validation data provided
    validation_results = None
    if validation_df is not None and validation_masks is not None and original_values is not None:
        validation_results = {}
        
        # Compare imputed values to real values
        for col in columns_to_impute:
            # Get indices where values were artificially set to NaN
            mask = validation_masks[col] & validation_df[col].isna()
            
            if mask.sum() == 0:
                validation_results[col] = {
                    'error': "No artificially missing values"
                }
                continue
                
            real_vals = original_values[col][mask]
            imputed_vals = imputed_df[col][mask]
            
            # Calculate continuous metrics (MAE and RMSE) - NO ROUNDING
            mae = mean_absolute_error(real_vals, imputed_vals)
            rmse = np.sqrt(mean_squared_error(real_vals, imputed_vals))
            
            # Calculate classification metrics - WITH ROUNDING
            real_vals_class = process_for_classification(real_vals)
            imputed_vals_class = process_for_classification(imputed_vals)
            
            classification_metrics = calculate_classification_metrics(real_vals_class, imputed_vals_class)
            
            validation_results[col] = {
                'mae': mae,
                'rmse': rmse,
                'accuracy': classification_metrics['accuracy'],
                'auc_multiclass': classification_metrics['auc_multiclass'],
                'avg_sensitivity': classification_metrics['avg_sensitivity'],
                'avg_specificity': classification_metrics['avg_specificity'],
                'avg_ppv': classification_metrics['avg_ppv'],
                'avg_npv': classification_metrics['avg_npv'],
                'precision_macro': classification_metrics['precision_macro'],
                'recall_macro': classification_metrics['recall_macro'],
                'real_distribution': real_vals.describe(),
                'imputed_distribution': imputed_vals.describe()
            }
    
    return imputed_df, validation_results
    
    
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import torch.distributions as dist
import time


class BayesianPCATorch(nn.Module):
    """
    PyTorch implementation of Bayesian PCA model
    """
    def __init__(self, n_features, n_components):
        super(BayesianPCATorch, self).__init__()
        self.n_features = n_features
        self.n_components = n_components
        
        # Priors for loadings (W)
        self.w_mu = nn.Parameter(torch.zeros(n_features, n_components), requires_grad=True)
        self.w_log_sigma = nn.Parameter(torch.zeros(n_features, n_components), requires_grad=True)
        
        # Noise precision (inverse variance)
        self.log_tau = nn.Parameter(torch.zeros(1), requires_grad=True)
        
    def forward(self, z):
        # Sample from approximate posterior for W
        w_sigma = torch.exp(self.w_log_sigma)
        epsilon_w = torch.randn_like(self.w_mu)
        w = self.w_mu + w_sigma * epsilon_w
        
        # Compute reconstruction
        tau = torch.exp(self.log_tau)
        reconstruction = torch.matmul(z, w.t())
        
        return reconstruction, w, tau
    
    def sample_loadings(self, n_samples=1):
        """Sample loadings (W) from the approximate posterior"""
        w_sigma = torch.exp(self.w_log_sigma)
        epsilon_w = torch.randn(n_samples, self.n_features, self.n_components, device=self.w_mu.device)
        w_samples = self.w_mu.unsqueeze(0) + w_sigma.unsqueeze(0) * epsilon_w
        return w_samples


class BayesianPCAImputer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible Bayesian PCA imputation model
    using PyTorch for GPU acceleration
    """
    
    def __init__(self,
                 n_components=5,
                 n_samples=1000,
                 batch_size=64,
                 n_epochs=100,
                 learning_rate=0.01,
                 device=None,
                 verbose=True):
        """
        Initialize PyTorch-based Bayesian PCA imputer
        
        Parameters:
        -----------
        n_components : int
            Number of principal components
        n_samples : int
            Number of posterior samples to draw
        batch_size : int
            Batch size for training
        n_epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for optimizer
        device : str or torch.device
            Device to use ('cuda' or 'cpu'), defaults to CUDA if available
        verbose : bool
            Whether to print progress
        """
        self.n_components = n_components
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.verbose = verbose
        
        # Initialize model components
        self.pca_model = None  # For standard PCA initialization
        self.scaler = None     # For data scaling
        self.model = None      # The PyTorch model
        
    def fit(self, X, y=None):
        """
        Fit Bayesian PCA model using PyTorch
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Data with missing values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Store column names and index
        self.columns = X.columns
        self.index = X.index
        
        # Create a copy of data
        X_data = X.copy()
        
        # Convert to numpy array
        X_array = X_data.values
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Simple imputation for initial values (for fitting the scaler)
        simple_imputer = SimpleImputer(strategy='mean')
        X_imputed_for_scaling = simple_imputer.fit_transform(X_array)
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X_imputed_for_scaling)
        
        # Create a mask for missing values
        self.missing_mask = np.isnan(X_array)
        
        # Perform standard PCA for initialization
        pca = PCA(n_components=self.n_components)
        pca.fit(X_scaled)
        self.pca_model = pca
        
        # Initialize PyTorch model
        n_samples, n_features = X_scaled.shape
        self.model = BayesianPCATorch(n_features, self.n_components).to(self.device)
        
        # Initialize model parameters using standard PCA
        with torch.no_grad():
            # Initialize W to PCA loadings
            self.model.w_mu.data = torch.tensor(pca.components_.T, dtype=torch.float32).to(self.device)
            
            # Initialize noise precision (tau) based on explained variance
            explained_var = pca.explained_variance_
            noise_var = np.mean(np.var(X_scaled, axis=0) - np.sum(explained_var))
            noise_var = max(noise_var, 1e-6)  # Ensure positive variance
            self.model.log_tau.data = torch.tensor([np.log(1.0 / noise_var)], dtype=torch.float32).to(self.device)
        
        # Prepare data for PyTorch training
        # Fill missing values with zeros (will be handled by the mask)
        X_for_torch = X_scaled.copy()
        X_for_torch[self.missing_mask] = 0.0
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_for_torch, dtype=torch.float32).to(self.device)
        mask_tensor = torch.tensor(~self.missing_mask, dtype=torch.float32).to(self.device)
        
        # Initialize latent variables (z) using PCA scores
        z_init = pca.transform(X_scaled)
        
        # Store the latent variables as a model parameter
        self.latent_z = nn.Parameter(torch.tensor(z_init, dtype=torch.float32, device=self.device))
        
        # Train the model
        self._train_model(X_tensor, mask_tensor)
        
        # Store final latent variables for later use
        self.z = self.latent_z.detach().cpu().numpy()
        
        return self
    
    def _train_model(self, X, mask):
        """
        Train the Bayesian PCA model using stochastic variational inference
        
        Parameters:
        -----------
        X : torch.Tensor
            Data tensor
        mask : torch.Tensor
            Mask tensor (1 for observed, 0 for missing)
        """
        # Setup optimizer - include latent_z as a parameter of self
        parameters = list(self.model.parameters()) + [self.latent_z]
        optimizer = optim.Adam(parameters, lr=self.learning_rate)
        
        # Setup scheduler for learning rate decay
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=self.verbose
        )
        
        # Training loop
        if self.verbose:
            print(f"Training Bayesian PCA model on {self.device}...")
            print(f"Data shape: {X.shape}, Components: {self.n_components}")
            print(f"Missing values: {torch.sum(mask == 0).item()} out of {X.numel()}")
            
        epoch_pbar = range(self.n_epochs)
        if self.verbose:
            epoch_pbar = tqdm(epoch_pbar, desc="Training")
            
        # For early stopping
        best_loss = float('inf')
        patience = 60  # Increased patience to 60
        patience_counter = 0
        
        # Store z for later use
        for epoch in epoch_pbar:
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, w_samples, tau = self.model(self.latent_z)
            
            # Compute loss - only for observed values
            # Likelihood term (reconstruction error)
            mse_loss = torch.sum(mask * (X - x_recon) ** 2)
            
            # Prior on z (standard normal)
            z_prior_loss = 0.5 * torch.sum(self.latent_z ** 2)
            
            # Prior on W (standard normal)
            w_mu = self.model.w_mu
            w_sigma = torch.exp(self.model.w_log_sigma)
            w_prior_loss = 0.5 * torch.sum(w_mu ** 2 + w_sigma ** 2 - torch.log(w_sigma ** 2) - 1)
            
            # Total loss
            loss = mse_loss * tau + z_prior_loss + w_prior_loss
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            # Get current loss
            current_loss = loss.item()
            
            # Update learning rate
            scheduler.step(current_loss)
            
            # Update progress bar with current loss
            if self.verbose:
                epoch_pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
            
            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
                
    def transform(self, X):
        """
        Impute missing values using Bayesian PCA
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Data with missing values
            
        Returns:
        --------
        imputed_df : pandas.DataFrame
            Data with imputed values
        """
        # Create copy to avoid modifying original
        imputed_df = X.copy()
        
        # Check if this is the same data used for fitting
        new_data = not np.array_equal(X.index, self.index) or not np.array_equal(X.columns, self.columns)
        
        if new_data:
            # For new data, we need to project onto the learned components
            # This is a simplified approach and could be improved
            
            # Create a copy of data
            X_data = X.copy()
            
            # Convert to numpy array
            X_array = X_data.values
            
            # Simple imputation for initial values
            simple_imputer = SimpleImputer(strategy='mean')
            X_imputed = simple_imputer.fit_transform(X_array)
            
            # Scale data
            X_scaled = self.scaler.transform(X_imputed)
            
            # Create a mask for missing values
            missing_mask = np.isnan(X_array)
            
            # Project data onto principal components
            z = self.pca_model.transform(X_scaled)
            
            # Get mean of W from model
            w_samples = self.model.sample_loadings(n_samples=self.n_samples)
            w_mean = w_samples.mean(dim=0).cpu().detach().numpy()
            
            # Reconstruct data
            X_reconstructed = np.dot(z, w_mean.T)
            
            # Inverse transform to original scale
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
            
            # Only replace missing values
            X_array_imputed = X_array.copy()
            X_array_imputed[missing_mask] = X_reconstructed[missing_mask]
            
            # Convert back to DataFrame
            imputed_df = pd.DataFrame(X_array_imputed, 
                                      columns=X.columns, 
                                      index=X.index)
        else:
            # For the same data used in fitting, use posterior samples
            
            # Sample from the model multiple times to get uncertainty estimates
            w_samples = self.model.sample_loadings(n_samples=self.n_samples)  # [n_samples, n_features, n_components]
            w_mean = w_samples.mean(dim=0).cpu().detach().numpy()  # [n_features, n_components]
            
            # Reconstruct data from latent variables (z)
            X_reconstructed = np.dot(self.z, w_mean.T)  # [n_samples, n_features]
            
            # Inverse transform to original scale
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
            
            # Convert to DataFrame
            X_reconstructed_df = pd.DataFrame(X_reconstructed, 
                                             columns=self.columns, 
                                             index=self.index)
            
            # Only replace missing values
            for col in imputed_df.columns:
                missing_idx = imputed_df[col].isna()
                if missing_idx.any():
                    imputed_df.loc[missing_idx, col] = X_reconstructed_df.loc[missing_idx, col].values
        
        return imputed_df
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Data with missing values
            
        Returns:
        --------
        imputed_df : pandas.DataFrame
            Data with imputed values
        """
        self.fit(X)
        return self.transform(X)


def apply_bpca_imputation(df, columns_to_impute, validation_df=None, validation_masks=None, original_values=None, display_progress=True, use_gpu=True):
    """
    Apply Bayesian PCA imputation to patient data using PyTorch
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient data with missing values
    columns_to_impute : list
        List of column names to impute
    validation_df : pandas.DataFrame, optional
        Validation dataset with artificially missing values
    validation_masks : dict, optional
        Dictionary of masks for validation data
    original_values : dict, optional
        Dictionary of original values for validation
    display_progress : bool
        Whether to display progress
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns:
    --------
    imputed_df : pandas.DataFrame
        Data with imputed values
    validation_results : dict, optional
        Validation results if validation data provided
    """
    # Extract relevant columns including potential predictors
    # Use all columns except those with excessive missing values
    threshold = 0.5  # Columns with more than 50% missing values are excluded
    # columns_to_use = [col for col in df.columns 
    #                   if df[col].isna().mean() < threshold]
    columns_to_use = [col for col in df.columns
                        if df[col].isna().mean() < threshold
                        and pd.api.types.is_numeric_dtype(df[col])]
    
    # Ensure all columns_to_impute are included
    for col in columns_to_impute:
        if col not in columns_to_use:
            columns_to_use.append(col)
    
    # Extract subset of data
    X = df[columns_to_use].copy()
    
    # Ensure all columns are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Initialize Bayesian PCA imputer
    n_components = min(5, len(columns_to_use) - 1)  # Ensure n_components is valid
    
    # Set device based on user preference and availability
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    # Determine batch size based on data size
    batch_size = min(64, len(X))  # Default 64, but smaller if dataset is tiny
    
    # Configure epochs based on data size
    n_epochs = max(100, min(300, 10000 // len(X) + 30))  # Updated minimum to 100 epochs
    
    if display_progress:
        print(f"Starting PyTorch Bayesian PCA imputation with {n_components} components")
        print(f"Using device: {device}")
        print(f"Training for up to {n_epochs} epochs with batch size {batch_size}")
        
        if device == 'cuda':
            print(f"GPU Info: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    # Determine reasonable number of posterior samples based on data size
    n_samples = 1000  # Default
    
    imputer = BayesianPCAImputer(
        n_components=n_components,
        n_samples=n_samples,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=0.01,
        device=device,
        verbose=display_progress
    )
    
    # Fit and transform with progress tracking
    start_time = None
    if display_progress:
        start_time = time.time()
        print("Training PyTorch Bayesian PCA imputation model...")
    
    X_imputed = imputer.fit_transform(X)
    
    if display_progress and start_time:
        elapsed = time.time() - start_time
        print(f"BPCA imputation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    # Create imputed dataframe
    imputed_df = df.copy()
    imputed_df[columns_to_impute] = X_imputed[columns_to_impute]
    
    # Validate if validation data provided
    validation_results = None
    if validation_df is not None and validation_masks is not None and original_values is not None:
        validation_results = {}
        validation_start_time = None
        
        if display_progress:
            validation_start_time = time.time()
            print("\nStarting validation on test data...")
        
        # Extract validation data
        X_val = validation_df[columns_to_use].copy()
        
        # Ensure all validation columns are numeric
        for col in X_val.columns:
            X_val[col] = pd.to_numeric(X_val[col], errors='coerce')
            
        # Impute validation data
        X_val_imputed = imputer.transform(X_val)
        
        # For storing overall metrics
        all_real_vals = []
        all_imputed_vals = []
        column_metrics = []
        
        # Compare imputed values to real values
        with tqdm(columns_to_impute, desc="Validating results") as pbar:
            for col in pbar:
                pbar.set_description(f"Validating {col}")
                # Get indices where values were artificially set to NaN
                mask = validation_masks[col] & validation_df[col].isna()
                
                if mask.sum() == 0:
                    validation_results[col] = {
                        'error': "No artificially missing values"
                    }
                    continue
                    
                real_vals = original_values[col][mask]
                imputed_vals = X_val_imputed[col][mask]
                
                # Collect all values for overall metrics
                all_real_vals.extend(real_vals.values)
                all_imputed_vals.extend(imputed_vals.values)
                
                # Calculate continuous metrics (MAE and RMSE) - NO ROUNDING
                mae = mean_absolute_error(real_vals, imputed_vals)
                rmse = np.sqrt(mean_squared_error(real_vals, imputed_vals))
                
                # Calculate classification metrics - WITH ROUNDING
                real_vals_class = process_for_classification(real_vals)
                imputed_vals_class = process_for_classification(imputed_vals)
                
                classification_metrics = calculate_classification_metrics(real_vals_class, imputed_vals_class)
                
                # Store metrics for summary
                column_metrics.append({
                    'column': col,
                    'mae': mae,
                    'rmse': rmse,
                    'accuracy': classification_metrics['accuracy'],
                    'count': len(real_vals)
                })
                
                validation_results[col] = {
                    'mae': mae,
                    'rmse': rmse,
                    'accuracy': classification_metrics['accuracy'],
                    'auc_multiclass': classification_metrics['auc_multiclass'],
                    'avg_sensitivity': classification_metrics['avg_sensitivity'],
                    'avg_specificity': classification_metrics['avg_specificity'],
                    'avg_ppv': classification_metrics['avg_ppv'],
                    'avg_npv': classification_metrics['avg_npv'],
                    'precision_macro': classification_metrics['precision_macro'],
                    'recall_macro': classification_metrics['recall_macro'],
                    'real_distribution': real_vals.describe(),
                    'imputed_distribution': imputed_vals.describe()
                }
                
                # Update progress
                pbar.set_postfix({"MAE": f"{mae:.4f}", "RMSE": f"{rmse:.4f}", "Acc": f"{classification_metrics['accuracy']:.4f}"})
        
        # Calculate overall metrics
        if all_real_vals:
            overall_mae = mean_absolute_error(all_real_vals, all_imputed_vals)
            overall_rmse = np.sqrt(mean_squared_error(all_real_vals, all_imputed_vals))
            
            validation_results['overall'] = {
                'mae': overall_mae,
                'rmse': overall_rmse,
                'total_values': len(all_real_vals)
            }
        
        # Print detailed summary
        if display_progress:
            validation_time = time.time() - validation_start_time if validation_start_time else 0
            total_time = validation_time + (time.time() - start_time if start_time else 0)
            
            print("\n" + "="*80)
            print(f"PYTORCH BAYESIAN PCA IMPUTATION SUMMARY (Device: {device})")
            print("="*80)
            
            # Time information
            print(f"\nTIMING INFORMATION:")
            print(f"  Training Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
            print(f"  Validation Time: {validation_time:.2f} seconds ({validation_time/60:.2f} minutes)")
            print(f"  Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            
            # Overall metrics
            if 'overall' in validation_results:
                print(f"\nOVERALL METRICS (across {validation_results['overall']['total_values']} values):")
                print(f"  MAE: {validation_results['overall']['mae']:.4f}")
                print(f"  RMSE: {validation_results['overall']['rmse']:.4f}")
            
            # Per-column metrics
            print("\nPER-COLUMN METRICS:")
            print("-"*80)
            print(f"{'Column':<20} {'MAE':<10} {'RMSE':<10} {'Accuracy':<10} {'Count':<10}")
            print("-"*80)
            
            for metric in sorted(column_metrics, key=lambda x: x['mae']):
                print(f"{metric['column']:<20} {metric['mae']:<10.4f} {metric['rmse']:<10.4f} {metric['accuracy']:<10.4f} {metric['count']:<10}")
            
            print("="*80)
    
    return imputed_df, validation_results







# Add Travis Brady's SoftImpute implementation to your code
def frob(Uold, Dsqold, Vold, U, Dsq, V):
    denom = (Dsqold ** 2).sum()
    utu = Dsq * (U.T.dot(Uold))
    vtv = Dsqold * (Vold.T.dot(V))
    uvprod = utu.dot(vtv).diagonal().sum()
    num = denom + (Dsqold ** 2).sum() - 2*uvprod
    return num / max(denom, 1e-9)


class SoftImpute:
    def __init__(self, J=2, thresh=1e-05, lambda_=0, maxit=100, random_state=None, verbose=False):
        self.J = J
        self.thresh = thresh
        self.lambda_ = lambda_
        self.maxit = maxit
        self.rs = np.random.RandomState(random_state)
        self.verbose = verbose
        self.u = None
        self.d = None
        self.v = None

    def fit(self, X):
        n, m = X.shape
        xnas = np.isnan(X)
        nz = m*n - xnas.sum()
        xfill = X.copy()
        V = np.zeros((m, self.J))
        U = self.rs.normal(0.0, 1.0, (n, self.J))
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        Dsq = np.ones((self.J, 1))
        #xfill[xnas] = 0.0
        col_means = np.nanmean(xfill, axis=0)
        np.copyto(xfill, col_means, where=np.isnan(xfill))
        ratio = 1.0
        iters = 0
        while ratio > self.thresh and iters < self.maxit:
            iters += 1
            U_old = U
            V_old = V
            Dsq_old = Dsq
            B = U.T.dot(xfill)
            if self.lambda_ > 0:
                tmp = (Dsq / (Dsq + self.lambda_))
                B = B * tmp
            Bsvd = np.linalg.svd(B.T, full_matrices=False)
            V = Bsvd[0]
            Dsq = Bsvd[1][:, np.newaxis]
            U = U.dot(Bsvd[2])
            tmp = Dsq * V.T
            xhat = U.dot(tmp)
            xfill[xnas] = xhat[xnas]
            A = xfill.dot(V).T
            Asvd = np.linalg.svd(A.T, full_matrices=False)
            U = Asvd[0]
            Dsq = Asvd[1][:, np.newaxis]
            V = V.dot(Asvd[2])
            tmp = Dsq * V.T
            xhat = U.dot(tmp)
            xfill[xnas] = xhat[xnas]
            ratio = frob(U_old, Dsq_old, V_old, U, Dsq, V)
            if self.verbose:
                print('iter: %4d ratio = %.5f' % (iters, ratio))
        self.u = U[:, :self.J]
        self.d = Dsq[:self.J]
        self.v = V[:, :self.J]
        return self

    def suv(self, vd):
        res = self.u.dot(vd.T)
        return res

    def predict(self, X, copyto=False):
        vd = self.v * np.outer(np.ones(self.v.shape[0]), self.d)
        X_imp = self.suv(vd)
        if copyto:
            np.copyto(X, X_imp, where=np.isnan(X))
        else:
            return X_imp


def apply_softimpute_imputation(df, columns_to_impute, validation_df=None, validation_masks=None, original_values=None,
                               J=None, thresh=1e-05, lambda_=0, maxit=100, random_state=42):
    """
    Apply SoftImpute imputation using Travis Brady's implementation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with missing values
    columns_to_impute : list
        List of column names to impute
    validation_df : pandas.DataFrame, optional
        Validation dataset with artificially missing values
    validation_masks : dict, optional
        Dictionary of masks for validation data
    original_values : dict, optional
        Dictionary of original values for validation
    J : int, optional
        Number of factors/rank. If None, will be estimated
    thresh : float
        Convergence threshold
    lambda_ : float
        Regularization parameter
    maxit : int
        Maximum number of iterations
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    imputed_df : pandas.DataFrame
        Data with imputed values
    validation_results : dict, optional
        Validation results if validation data provided
    """
    try:
        # Extract relevant columns including potential predictors
        # Use all columns except those with excessive missing values
        threshold = 0.5  # Columns with more than 50% missing values are excluded
        # columns_to_use = [col for col in df.columns 
        #                     if df[col].isna().mean() < threshold]
        columns_to_use = [col for col in df.columns
                        if df[col].isna().mean() < threshold
                        and pd.api.types.is_numeric_dtype(df[col])]
        
        # Ensure all columns_to_impute are included
        for col in columns_to_impute:
            if col not in columns_to_use:
                columns_to_use.append(col)
        
        # Extract subset of data
        X = df[columns_to_use].copy()
        
        # Ensure all columns are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        print(f"SoftImpute (Travis Brady): Using {len(columns_to_use)} columns, imputing {len(columns_to_impute)} columns")
        
        # Set J (rank) if not provided
        if J is None:
            J = min(X.shape[0], X.shape[1]) // 4
            J = max(2, J)  # Ensure at least rank 2
        
        print(f"SoftImpute parameters: J={J}, thresh={thresh}, lambda_={lambda_}, maxit={maxit}")
        
        # Scale the data for better numerical stability
        scaler = StandardScaler()
        
        # Initial imputation with column means for scaling
        X_for_scaling = X.copy()
        for col in X_for_scaling.columns:
            X_for_scaling[col] = X_for_scaling[col].fillna(X_for_scaling[col].mean())
        
        # Fit scaler and transform
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_for_scaling),
            columns=X.columns,
            index=X.index
        )
        
        # Restore NaN values in scaled data
        X_scaled[X.isna()] = np.nan
        
        # Initialize SoftImpute
        imputer = SoftImpute(
            J=J,
            thresh=thresh,
            lambda_=lambda_,
            maxit=maxit,
            random_state=random_state,
            verbose=True
        )
        
        # Fit and predict
        print("Training SoftImpute imputation model...")
        imputer.fit(X_scaled.values)
        X_imputed_scaled = imputer.predict(X_scaled.values)
        
        # Inverse transform to original scale
        X_imputed = pd.DataFrame(
            scaler.inverse_transform(X_imputed_scaled),
            columns=X.columns,
            index=X.index
        )
        
        # Create imputed dataframe - only replace missing values
        imputed_df = df.copy()
        for col in columns_to_impute:
            if col in X_imputed.columns:
                missing_mask = df[col].isna()
                imputed_df.loc[missing_mask, col] = X_imputed.loc[missing_mask, col]
        
        # Validate if validation data provided
        validation_results = None
        if validation_df is not None and validation_masks is not None and original_values is not None:
            validation_results = {}
            
            # Extract validation data
            X_val = validation_df[columns_to_use].copy()
            
            # Ensure all validation columns are numeric
            for col in X_val.columns:
                X_val[col] = pd.to_numeric(X_val[col], errors='coerce')
                
            # Scale validation data
            X_val_for_scaling = X_val.copy()
            for col in X_val_for_scaling.columns:
                X_val_for_scaling[col] = X_val_for_scaling[col].fillna(X_val_for_scaling[col].mean())
            
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val_for_scaling),
                columns=X_val.columns,
                index=X_val.index
            )
            
            # Restore NaN values
            X_val_scaled[X_val.isna()] = np.nan
            
            # Create a new imputer for validation data
            val_imputer = SoftImpute(
                J=J,
                thresh=thresh,
                lambda_=lambda_,
                maxit=maxit,
                random_state=random_state,
                verbose=False  # Less verbose for validation
            )
            
            # Impute validation data
            print("Imputing validation data...")
            val_imputer.fit(X_val_scaled.values)
            X_val_imputed_scaled = val_imputer.predict(X_val_scaled.values)
            
            # Inverse transform
            X_val_imputed = pd.DataFrame(
                scaler.inverse_transform(X_val_imputed_scaled),
                columns=X_val.columns,
                index=X_val.index
            )
            
            # Compare imputed values to real values
            with tqdm(columns_to_impute, desc="Validating results") as pbar:
                for col in pbar:
                    pbar.set_description(f"Validating {col}")
                    # Get indices where values were artificially set to NaN
                    mask = validation_masks[col] & validation_df[col].isna()
                    
                    if mask.sum() == 0:
                        validation_results[col] = {
                            'error': "No artificially missing values"
                        }
                        continue
                        
                    real_vals = original_values[col][mask]
                    imputed_vals = X_val_imputed.loc[mask, col]
                    
                    # Calculate continuous metrics (MAE and RMSE) - NO ROUNDING
                    mae = mean_absolute_error(real_vals, imputed_vals)
                    rmse = np.sqrt(mean_squared_error(real_vals, imputed_vals))
                    
                    # Calculate classification metrics - WITH ROUNDING
                    real_vals_class = process_for_classification(real_vals)
                    imputed_vals_class = process_for_classification(imputed_vals)
                    
                    classification_metrics = calculate_classification_metrics(real_vals_class, imputed_vals_class)
                    
                    validation_results[col] = {
                        'mae': mae,
                        'rmse': rmse,
                        'accuracy': classification_metrics['accuracy'],
                        'auc_multiclass': classification_metrics['auc_multiclass'],
                        'avg_sensitivity': classification_metrics['avg_sensitivity'],
                        'avg_specificity': classification_metrics['avg_specificity'],
                        'avg_ppv': classification_metrics['avg_ppv'],
                        'avg_npv': classification_metrics['avg_npv'],
                        'precision_macro': classification_metrics['precision_macro'],
                        'recall_macro': classification_metrics['recall_macro'],
                        'real_distribution': real_vals.describe(),
                        'imputed_distribution': pd.Series(imputed_vals).describe()
                    }
                    
                    # Update progress
                    pbar.set_postfix({"MAE": f"{mae:.4f}", "RMSE": f"{rmse:.4f}", "Acc": f"{classification_metrics['accuracy']:.4f}"})
        
        return imputed_df, validation_results
        
    except Exception as e:
        print(f"Error in SoftImpute imputation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to simple mean imputation
        print("Falling back to simple mean imputation")
        result_df = df.copy()
        for col in columns_to_impute:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                result_df[col] = result_df[col].fillna(result_df[col].mean())
        return result_df, None
    




    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Helper: build patient sequences from a flat dataframe -------------------

VISIT_ORDER = {
    'baseline_arm_1':       0,
    'preoperative_arm_1':   1,
    '1_month_postop_arm_1': 2,
    '3_months_postop_arm_1':3,
    '6_months_postop_arm_1':4,
    '1_year_postop_arm_1':  5,
    '2_years_postop_arm_1': 6,
    '3_years_postop_arm_1': 7,
    '4_years_postop_arm_1': 8,
    '5_years_postop_arm_1': 9,
}

def _build_patient_sequences(df, columns_to_impute, patient_col='id',
                             date_col='qol_date', event_col='redcap_event_name'):
    """
    Convert flat dataframe into per-patient sorted sequences.
    Ordering uses redcap_event_name visit index if available, else qol_date.
    Visit index is appended as an extra input feature to the LSTM.
    """
    feature_cols = [c for c in columns_to_impute if c in df.columns]

    # Build sort key: prefer redcap_event_name, fall back to qol_date
    if event_col in df.columns:
        sort_key = df[event_col].map(VISIT_ORDER).fillna(99).astype(int)
    else:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        min_date = dates.min()
        sort_key = (dates - min_date).dt.days.fillna(0).astype(int)

    sequences = []
    for pid, group in df.groupby(patient_col):
        idx = group.index.tolist()
        keys = sort_key[idx].values
        order = np.argsort(keys, kind='stable')
        idx_sorted = [idx[i] for i in order]
        visit_indices = keys[order]                        # e.g. [0, 3, 6]

        X = df.loc[idx_sorted, feature_cols].values.astype(float)  # [T, F]
        mask = ~np.isnan(X)

        sequences.append({
            'patient_id':    pid,
            'indices':       idx_sorted,
            'visit_indices': visit_indices,                # used as extra feature
            'X':             X,
            'mask':          mask,
        })
    return sequences, feature_cols


def _pad_sequences(sequences, n_features):
    """
    Pad sequences to the same length for batch processing.

    Returns
    -------
    X_padded  : np.ndarray [N_patients, max_T, n_features]
    mask      : np.ndarray [N_patients, max_T, n_features]  (True = observed)
    lengths   : list[int]  actual length of each patient sequence
    """
    max_T = max(len(s['indices']) for s in sequences)
    N = len(sequences)

    X_padded = np.zeros((N, max_T, n_features), dtype=np.float32)
    mask_padded = np.zeros((N, max_T, n_features), dtype=bool)

    for i, s in enumerate(sequences):
        T = len(s['indices'])
        X_padded[i, :T, :] = np.nan_to_num(s['X'], nan=0.0)
        mask_padded[i, :T, :] = s['mask']

    lengths = [len(s['indices']) for s in sequences]
    return X_padded, mask_padded, lengths


# --- LSTM model --------------------------------------------------------------
    
class LSTMImputer(nn.Module):
    """
    Simple bidirectional LSTM for longitudinal imputation.
    Input:  [batch, seq_len, n_features]  (missing values filled with 0)
    Output: [batch, seq_len, n_features]  (reconstruction for all positions)
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_layers=1, dropout=0.2):  # ← add dropout
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,  # ← LSTM internal dropout
        )
        self.dropout = nn.Dropout(dropout)  # ← add explicit dropout layer
        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)  # ← apply dropout before output
        return self.output_layer(out)


# --- Wrapper function --------------------------------------------------------

class MaskedMSELoss(nn.Module):
    """
    MSE loss that ignores missing values
    """
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        
    def forward(self, pred, target, mask):
        """
        Calculate MSE loss only on observed values
        
        Parameters:
        -----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values
        mask : torch.Tensor
            Binary mask (1 for observed, 0 for missing)
            
        Returns:
        --------
        torch.Tensor
            Masked MSE loss
        """
        # Apply mask to predictions and targets
        masked_pred = pred * mask
        masked_target = target * mask
        
        # Calculate squared error
        squared_error = (masked_pred - masked_target) ** 2
        
        # Sum squared error and count observed values
        sum_squared_error = torch.sum(squared_error)
        count = torch.sum(mask) + 1e-8  # Add small epsilon to avoid division by zero
        
        # Return MSE
        return sum_squared_error / count

def apply_lstm_imputation(df, columns_to_impute,
                          validation_df=None, validation_masks=None, original_values=None,
                          patient_col='id', date_col='qol_date',
                          hidden_dim=64, n_layers=2, epochs=100,  # ← increase to 2 layers, 100 epochs
                          learning_rate=0.001, batch_size=32, dropout=0.2):  # ← add dropout param
    """
    Apply bidirectional LSTM imputation using longitudinal patient sequences.

    Parameters
    ----------
    df : pd.DataFrame
        Data with missing values. Must contain patient_col and date_col.
    columns_to_impute : list
        FACT-E column names to impute.
    validation_df, validation_masks, original_values : optional
        Standard validation arguments (same interface as other apply_* functions).
    patient_col : str
        Column name for patient ID (default 'id').
    date_col : str
        Column name for visit date (default 'qol_date').
    hidden_dim : int
        LSTM hidden size (default 64 — keeps model small).
    n_layers : int
        Number of LSTM layers (default 1).
    epochs : int
        Training epochs (default 50).
    learning_rate : float
        Adam learning rate (default 0.001).
    batch_size : int
        Patients per batch (default 32).

    Returns
    -------
    imputed_df : pd.DataFrame
    validation_results : dict or None
    """
    print(f"LSTM Imputation: {len(df)} rows, {df[patient_col].nunique()} patients")

    # --- require id and qol_date to be present ---
    for col in [patient_col, date_col]:
        if col not in df.columns:
            print(f"  Warning: '{col}' not found. Falling back to mean imputation.")
            result_df = df.copy()
            for c in columns_to_impute:
                result_df[c] = pd.to_numeric(result_df[c], errors='coerce')
                result_df[c] = result_df[c].fillna(result_df[c].mean())
            return result_df, None

    # --- build sequences and scale -----------------------------------------
    sequences, feature_cols = _build_patient_sequences(
        df, columns_to_impute, patient_col, date_col)
    n_features = len(feature_cols)

    if n_features == 0:
        print("  No valid feature columns found.")
        return df.copy(), None

    # Compute per-feature mean/std from observed values
    all_obs = np.concatenate([s['X'] for s in sequences], axis=0)  # [total_rows, n_features]
    feat_mean = np.nanmean(all_obs, axis=0)
    feat_std  = np.nanstd(all_obs, axis=0) + 1e-8

    # Standardise X inside each sequence (in-place copy)
    sequences_scaled = []
    for s in sequences:
        X_s = (s['X'] - feat_mean) / feat_std
        X_s[~s['mask']] = 0.0          # zero-fill missing before feeding LSTM
        sequences_scaled.append({**s, 'X_scaled': X_s})

    # --- pad and create tensors --------------------------------------------
    X_padded, mask_padded, lengths = _pad_sequences(sequences_scaled, n_features)

    # Use scaled X + normalised visit index as LSTM input
    # visit index normalised to [0,1] over the 11 timepoints
    X_pad_scaled = np.zeros((len(sequences_scaled),
                             X_padded.shape[1],
                             n_features + 1), dtype=np.float32)  # +1 for visit idx
    for i, s in enumerate(sequences_scaled):
        T = len(s['indices'])
        X_pad_scaled[i, :T, :n_features] = s['X_scaled']
        X_pad_scaled[i, :T,  n_features] = s['visit_indices'] / 10.0  # normalise

    X_tensor    = torch.FloatTensor(X_pad_scaled)
    mask_tensor = torch.FloatTensor(mask_padded.astype(np.float32))   # [N, T, n_features] — unchanged, already correct
    # target: standardised observed values (zero where missing — masked out in loss)
    Y_pad_scaled = np.zeros((len(sequences_scaled),
                             X_padded.shape[1],
                             n_features), dtype=np.float32)   # target: features only
    for i, s in enumerate(sequences_scaled):
        T = len(s['indices'])
        Y_pad_scaled[i, :T, :] = s['X_scaled']
    Y_tensor = torch.FloatTensor(Y_pad_scaled)

    # --- device ------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    # --- model, optimiser, loss --------------------------------------------
    model = LSTMImputer(input_dim=n_features + 1, output_dim=n_features,
                        hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = MaskedMSELoss().to(device)   # already defined in your codebase

    dataset = TensorDataset(X_tensor, Y_tensor, mask_tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- training loop -----------------------------------------------------
    model.train()
    for epoch in tqdm(range(epochs), desc="Training LSTM"):
        epoch_loss = 0.0
        for X_b, Y_b, M_b in loader:
            X_b, Y_b, M_b = X_b.to(device), Y_b.to(device), M_b.to(device)
            pred = model(X_b)
            loss = loss_fn(pred, Y_b, M_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # --- impute: forward pass on full dataset ------------------------------
    model.eval()
    imputed_df = df.copy()

    with torch.no_grad():
        for i, s in enumerate(sequences_scaled):
            T = len(s['indices'])
            x_in = torch.FloatTensor(X_pad_scaled[i:i+1, :T, :]).to(device)
            pred = model(x_in).cpu().numpy()[0]          # [T, n_features]
            # inverse-scale
            pred_orig = pred * feat_std + feat_mean
            # clip to valid ordinal range [0, 4]
            pred_orig = np.clip(pred_orig, 0, 4)

            # only fill positions that were originally missing
            X_orig = s['X']                              # [T, n_features]
            for t, row_idx in enumerate(s['indices']):
                for f_idx, col in enumerate(feature_cols):
                    if np.isnan(X_orig[t, f_idx]):
                        imputed_df.at[row_idx, col] = pred_orig[t, f_idx]

    # --- validation (same pattern as other apply_* functions) --------------
    validation_results = None
    if validation_df is not None and validation_masks is not None and original_values is not None:
        validation_results = {}

        # Run imputation on validation_df using the trained model
        val_sequences, _ = _build_patient_sequences(
            validation_df, columns_to_impute, patient_col, date_col)

        val_sequences_scaled = []
        for s in val_sequences:
            X_s = (s['X'] - feat_mean) / feat_std
            X_s[~s['mask']] = 0.0
            val_sequences_scaled.append({**s, 'X_scaled': X_s})

        X_val_pad_scaled = np.zeros(
            (len(val_sequences_scaled),
             max(len(s['indices']) for s in val_sequences_scaled),
             n_features + 1), dtype=np.float32)
        for i, s in enumerate(val_sequences_scaled):
            T = len(s['indices'])
            X_val_pad_scaled[i, :T, :n_features] = s['X_scaled']
            X_val_pad_scaled[i, :T,  n_features] = s['visit_indices'] / 10.0

        val_imputed_df = validation_df.copy()
        with torch.no_grad():
            for i, s in enumerate(val_sequences_scaled):
                T = len(s['indices'])
                x_in = torch.FloatTensor(X_val_pad_scaled[i:i+1, :T, :]).to(device)
                pred = model(x_in).cpu().numpy()[0]
                pred_orig = np.clip(pred * feat_std + feat_mean, 0, 4)
                X_orig = s['X']
                for t, row_idx in enumerate(s['indices']):
                    for f_idx, col in enumerate(feature_cols):
                        if np.isnan(X_orig[t, f_idx]):
                            val_imputed_df.at[row_idx, col] = pred_orig[t, f_idx]

        for col in columns_to_impute:
            mask = validation_masks[col] & validation_df[col].isna()
            if mask.sum() == 0:
                validation_results[col] = {'error': "No artificially missing values"}
                continue

            real_vals    = original_values[col][mask]
            imputed_vals = val_imputed_df[col][mask]

            mae  = mean_absolute_error(real_vals, imputed_vals)
            rmse = np.sqrt(mean_squared_error(real_vals, imputed_vals))

            real_cls    = process_for_classification(real_vals)
            imputed_cls = process_for_classification(imputed_vals)
            cls_metrics = calculate_classification_metrics(real_cls, imputed_cls)

            validation_results[col] = {
                'mae': mae, 
                'rmse': rmse,
                'accuracy': cls_metrics['accuracy'],
                'auc_multiclass': cls_metrics['auc_multiclass'],
                'avg_sensitivity': cls_metrics['avg_sensitivity'],
                'avg_specificity': cls_metrics['avg_specificity'],
                'avg_ppv': cls_metrics['avg_ppv'],
                'avg_npv': cls_metrics['avg_npv'],
                'precision_macro': cls_metrics['precision_macro'],
                'recall_macro': cls_metrics['recall_macro'],
                'real_distribution': real_vals.describe(),
                'imputed_distribution': imputed_vals.describe(),
            }

    return imputed_df, validation_results





def apply_mice_longitudinal_imputation(df, columns_to_impute,
                                       validation_df=None, validation_masks=None, original_values=None,
                                       patient_col='id', date_col='qol_date'):
    """
    Time-aware MICE imputation for longitudinal data.

    Enriches the dataframe with per-patient lag features (previous-timepoint
    values) and a days-since-baseline feature before running standard MICE,
    then strips the auxiliary columns from the result.

    Parameters and return values are identical to apply_mice_imputation.
    """
    print(f"Longitudinal MICE: {len(df)} rows, {df[patient_col].nunique()} patients")

    for col in [patient_col, date_col]:
        if col not in df.columns:
            print(f"  Warning: '{col}' not found. Falling back to standard MICE.")
            return apply_mice_imputation(df, columns_to_impute,
                                         validation_df, validation_masks, original_values)

    def _add_temporal_features(frame, fit_reference=None):
        """
        Add lag-1 and days_since_baseline columns.
        fit_reference: if provided, use its min-date for days calculation
                       (ensures train/val use the same baseline).
        """
        out = frame.copy()
        dates = pd.to_datetime(out[date_col], errors='coerce')
        ref_date = fit_reference if fit_reference is not None else dates.min()
        out['_days_since_baseline'] = (dates - ref_date).dt.days.fillna(0).astype(float)

        # Sort by clinical visit order (redcap_event_name) if available,
        # else fall back to days_since_baseline
        if 'redcap_event_name' in out.columns:
            out['_visit_index'] = out['redcap_event_name'].map(VISIT_ORDER).fillna(99)
            out = out.sort_values([patient_col, '_visit_index'])
            out = out.drop(columns=['_visit_index'])
        else:
            out = out.sort_values([patient_col, '_days_since_baseline'])

        lag_cols = []
        for col_name in columns_to_impute:
            # Add lag-1, lag-2, lag-3
            for lag in [1, 2, 3]:
                lag_name = f'_lag{lag}_{col_name}'
                out[lag_name] = out.groupby(patient_col)[col_name].shift(lag)
                lag_cols.append(lag_name)

        return out, lag_cols, ref_date

    # Build augmented training frame
    df_aug, lag_cols, ref_date = _add_temporal_features(df)

    # Columns to pass into MICE (original + temporal auxiliaries)
    aux_cols = ['_days_since_baseline'] + lag_cols
    # Only keep aux cols that aren't >50% missing
    aux_cols_valid = [c for c in aux_cols if df_aug[c].isna().mean() < 0.5]

    # Build the full column set for MICE
    # (all non-date, non-id numeric columns already in df, plus aux)
    base_cols = [c for c in df_aug.columns
                 if c not in ([patient_col, date_col] + aux_cols)
                 and pd.api.types.is_numeric_dtype(df_aug[c])]
    mice_cols = list(dict.fromkeys(base_cols + aux_cols_valid))  # preserve order, no dups

    X_aug = df_aug[mice_cols].copy()
    for c in X_aug.columns:
        X_aug[c] = pd.to_numeric(X_aug[c], errors='coerce')

    print(f"  Running MICE on {len(mice_cols)} columns "
          f"({len(columns_to_impute)} target + {len(aux_cols_valid)} temporal aux)")

    os.environ['OMP_NUM_THREADS'] = '10'
    kernel = mf.ImputationKernel(
        X_aug,
        datasets=1,
        variable_schema={col: [c for c in X_aug.columns if c != col]
                         for col in columns_to_impute if col in X_aug.columns},
        random_state=42
    )
    for _ in tqdm(range(5), desc="Longitudinal MICE"):
        kernel.mice(iterations=1, verbose=False,
                    num_boost_round=80, max_depth=10, num_threads=10)

    imputed_aug = kernel.complete_data(0)

    # Put results back into a copy of the original df (strip aux cols)
    imputed_df = df.copy()
    # Restore original row order (df_aug was sorted)
    imputed_aug = imputed_aug.reindex(df.index)
    for col_name in columns_to_impute:
        if col_name in imputed_aug.columns:
            imputed_df[col_name] = imputed_aug[col_name]

    # --- validation (same pattern as other apply_* functions) --------------
    validation_results = None
    if validation_df is not None and validation_masks is not None and original_values is not None:
        validation_results = {}

        val_aug, _, _ = _add_temporal_features(validation_df, fit_reference=ref_date)
        X_val_aug = val_aug[mice_cols].copy()
        for c in X_val_aug.columns:
            X_val_aug[c] = pd.to_numeric(X_val_aug[c], errors='coerce')

        val_kernel = mf.ImputationKernel(
            X_val_aug, datasets=1,
            variable_schema={col: [c for c in X_val_aug.columns if c != col]
                             for col in columns_to_impute if col in X_val_aug.columns},
            random_state=42
        )
        for _ in range(5):
            val_kernel.mice(iterations=1, verbose=False,
                            num_boost_round=80, max_depth=10, num_threads=10)

        val_imputed_aug = val_kernel.complete_data(0).reindex(validation_df.index)

        for col_name in columns_to_impute:
            mask = validation_masks[col_name] & validation_df[col_name].isna()
            if mask.sum() == 0:
                validation_results[col_name] = {'error': "No artificially missing values"}
                continue

            real_vals    = original_values[col_name][mask]
            imputed_vals = val_imputed_aug.loc[mask, col_name]

            mae  = mean_absolute_error(real_vals, imputed_vals)
            rmse = np.sqrt(mean_squared_error(real_vals, imputed_vals))

            real_cls    = process_for_classification(real_vals)
            imputed_cls = process_for_classification(imputed_vals)
            cls_metrics = calculate_classification_metrics(real_cls, imputed_cls)

            validation_results[col_name] = {
                'mae': mae, 
                'rmse': rmse,
                'accuracy': cls_metrics['accuracy'],
                'auc_multiclass': cls_metrics['auc_multiclass'],
                'avg_sensitivity': cls_metrics['avg_sensitivity'],
                'avg_specificity': cls_metrics['avg_specificity'],
                'avg_ppv': cls_metrics['avg_ppv'],
                'avg_npv': cls_metrics['avg_npv'],
                'precision_macro': cls_metrics['precision_macro'],
                'recall_macro': cls_metrics['recall_macro'],
                'real_distribution': real_vals.describe(),
                'imputed_distribution': pd.Series(imputed_vals).describe(),
            }

    return imputed_df, validation_results





"""
MedGemma 1.5 4B LLM-based imputation for FACT-E ordinal variables.

Pipeline (mirrors Yousif et al. EMBC 2026):
    1. Compute Pearson correlations among columns_to_impute on observed pairs
    2. For each missing value, select top-k most correlated observed variables
    3. Format as structured text prompt with FACT-E item descriptions
    4. Fine-tune MedGemma-1.5-4B-IT with LoRA on (prompt -> integer) pairs
    5. Inference: decode output token, clamp to {0,1,2,3,4}
"""

FACT_E_DESCRIPTIONS = {
    'gp1': 'I have a lack of energy',
    'gp2': 'I have nausea',
    'gp3': 'Because of my physical condition, I have trouble meeting the needs of my family',
    'gp4': 'I have pain',
    'gp5': 'I am bothered by side effects of treatment',
    'gp6': 'I feel ill',
    'gp7': 'I am forced to spend time in bed',
    'gs1': 'I feel close to my friends',
    'gs2': 'I get emotional support from my family',
    'gs3': 'I get support from my friends',
    'gs4': 'My family has accepted my illness',
    'gs5': 'I am satisfied with family communication about my illness',
    'gs6': 'I feel close to my partner (or the person who is my main support)',
    'gs7': 'I am satisfied with my sex life',
    'ge1': 'I feel sad',
    'ge2': 'I am satisfied with how I am coping with my illness',
    'ge3': 'I am losing hope in the fight against my illness',
    'ge4': 'I feel nervous',
    'ge5': 'I worry about dying',
    'ge6': 'I worry that my condition will get worse',
    'gf1': 'I am able to work (include work at home)',
    'gf2': 'My work (include work at home) is fulfilling',
    'gf3': 'I am able to enjoy life',
    'gf4': 'I have accepted my illness',
    'gf5': 'I am sleeping well',
    'gf6': 'I am enjoying the things I usually do for fun',
    'gf7': 'I am content with the quality of my life right now',
    'a_hn1': 'I am able to eat the foods that I like',
    'a_hn2': 'My mouth is dry',
    'a_hn3': 'I have trouble breathing',
    'a_hn4': 'My voice has its usual quality and strength',
    'a_hn5': 'I am able to eat as much food as I want',
    'a_hn7': 'I can swallow naturally and easily',
    'a_hn10': 'I am able to communicate with others',
    'a_e1': 'I have difficulty swallowing solid foods',
    'a_e2': 'I have difficulty swallowing soft or mashed foods',
    'a_e3': 'I have difficulty swallowing liquids',
    'a_e4': 'I have pain in my chest when I swallow',
    'a_e5': 'I choke when I swallow',
    'a_e6': 'I am able to enjoy meals with family or friends',
    'a_e7': 'I wake at night because of coughing',
    'a_c6': 'I have a good appetite',
    'a_c2': 'I am losing weight',
    'a_act11': 'I have pain in my stomach area',
}

SCALE_DESC = '(0 = Not at all, 1 = A little bit, 2 = Somewhat, 3 = Quite a bit, 4 = Very much)'


def _compute_correlations(df, columns_to_impute):
    return df[columns_to_impute].corr(method='pearson')


def _select_top_k_predictors(target_col, corr_matrix, observed_cols, k=4):
    if target_col not in corr_matrix.index:
        return observed_cols[:k]
    corrs = corr_matrix[target_col].drop(labels=[target_col], errors='ignore')
    corrs = corrs[corrs.index.isin(observed_cols)]
    corrs = corrs.abs().sort_values(ascending=False)
    return corrs.index[:k].tolist()


def _build_prompt(target_col, predictor_cols, predictor_values):
    target_desc = FACT_E_DESCRIPTIONS.get(target_col, target_col)
    lines = [
        'You are predicting a missing patient-reported outcome score.',
        f'The target question is: "{target_desc}" {SCALE_DESC}',
        'Based on the most related observed answers:',
    ]
    for col, val in zip(predictor_cols, predictor_values):
        desc = FACT_E_DESCRIPTIONS.get(col, col)
        lines.append(f'  - {col} ("{desc}"): {int(val)}')
    lines.append(f'Predict the integer score (0-4) for "{target_col}":')
    return '\n'.join(lines)


def _build_training_data(df, columns_to_impute, corr_matrix, k=4, max_per_col=500, seed=42):
    rng = np.random.default_rng(seed)
    prompts, labels = [], []

    for target_col in columns_to_impute:
        obs_mask = df[target_col].notna()
        candidate_df = df[obs_mask].copy()
        if len(candidate_df) == 0:
            continue

        n = min(len(candidate_df), max_per_col)
        sample_idx = rng.choice(candidate_df.index, size=n, replace=False)

        for idx in sample_idx:
            row = df.loc[idx]
            observed_other = [c for c in columns_to_impute
                              if c != target_col and pd.notna(row[c])]
            if len(observed_other) == 0:
                continue
            predictors = _select_top_k_predictors(target_col, corr_matrix, observed_other, k=k)
            if len(predictors) == 0:
                continue
            pred_vals = [row[p] for p in predictors]
            prompts.append(_build_prompt(target_col, predictors, pred_vals))
            labels.append(str(int(row[target_col])))

    return prompts, labels


def apply_gemma_imputation(df, columns_to_impute,
                           validation_df=None, validation_masks=None, original_values=None,
                           k_predictors=4,
                           max_train_per_col=500,
                           lora_rank=32,
                           lora_alpha=64,
                           lora_dropout=0.1,
                           learning_rate=2e-5,
                           num_epochs=2,
                           batch_size=1,
                           gradient_accumulation_steps=8,
                           model_name='/home/yjkweon2/models/medgemma-1.5-4b-it',
                           seed=42):
    """
    Fine-tune MedGemma with LoRA for FACT-E ordinal imputation.

    Parameters
    ----------
    df : pd.DataFrame
    columns_to_impute : list[str]
    validation_df, validation_masks, original_values : optional
        Standard validation arguments (same interface as other apply_* functions).
    k_predictors : int
        Top-k correlated predictors per prompt (default 4, matching Yousif et al.).
    max_train_per_col : int
        Max training examples per target column (default 500).
    lora_rank : int
        LoRA rank (default 16).
    lora_alpha : int
        LoRA alpha (default 32).
    lora_dropout : float
        LoRA dropout (default 0.1).
    learning_rate : float
        AdamW learning rate (default 2e-5).
    num_epochs : int
        Training epochs (default 2).
    batch_size : int
        Per-device batch size (default 1).
    gradient_accumulation_steps : int
        Gradient accumulation steps (default 8).
    model_name : str
        HuggingFace model ID.
        Default: medgemma-1.5-4b-it (instruction-tuned version of Gemma 1.5 4B).
    seed : int

    Returns
    -------
    imputed_df : pd.DataFrame
    validation_results : dict or None
    """

    # --- Lazy imports ------------------------------------------------------
    try:
        import torch
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                  TrainingArguments, Trainer,
                                  DataCollatorForSeq2Seq,
                                  BitsAndBytesConfig)
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset as HFDataset
    except ImportError as e:
        print(f"[Gemma] Missing dependency: {e}")
        print("[Gemma] Install with: pip install --no-index transformers peft datasets accelerate bitsandbytes")
        result_df = df.copy()
        for col in columns_to_impute:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            result_df[col] = result_df[col].fillna(result_df[col].mean())
        return result_df, None

    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Gemma] Device: {device}")
    print(f"[Gemma] Model: {model_name}")
    print(f"[Gemma] Rows: {len(df)}, target columns: {len(columns_to_impute)}")

    # --- Step 1: Correlations ----------------------------------------------
    print("[Gemma] Computing Pearson correlations...")
    corr_matrix = _compute_correlations(df, columns_to_impute)

    # --- Step 2: Build training data ---------------------------------------
    print("[Gemma] Building training prompts...")
    train_prompts, train_labels = _build_training_data(
        df, columns_to_impute, corr_matrix,
        k=k_predictors, max_per_col=max_train_per_col, seed=seed)

    print(f"[Gemma] Training examples: {len(train_prompts)}")
    if len(train_prompts) == 0:
        print("[Gemma] No training data — falling back to mean imputation")
        result_df = df.copy()
        for col in columns_to_impute:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            result_df[col] = result_df[col].fillna(result_df[col].mean())
        return result_df, None

    # --- Step 3: Load tokenizer & model ------------------------------------
    print("[Gemma] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[Gemma] Loading base model...")
    load_kwargs = {'device_map': 'auto' if device == 'cuda' else None}
    if device == 'cuda':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs['quantization_config'] = bnb_config

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # --- Step 4: Attach LoRA -----------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['q_proj', 'v_proj'],  # same as Yousif et al.
        bias='none',
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # --- Step 5: Tokenize --------------------------------------------------
    max_length = 256

    def tokenize_fn(examples):
        full_texts = [p + ' ' + l for p, l in
                      zip(examples['prompt'], examples['label'])]
        tok = tokenizer(full_texts, truncation=True, max_length=max_length,
                        padding='max_length')
        tok['labels'] = tok['input_ids'].copy()
        # MedGemma requires token_type_ids; add zeros if not produced by tokenizer
        if 'medgemma' in model_name.lower() and 'token_type_ids' not in tok:
            tok['token_type_ids'] = [[0] * len(ids) for ids in tok['input_ids']]
        return tok

    hf_dataset = HFDataset.from_dict({'prompt': train_prompts, 'label': train_labels})
    tokenized = hf_dataset.map(tokenize_fn, batched=True,
                                remove_columns=['prompt', 'label'])

    # --- Step 6: Train -----------------------------------------------------
    output_dir = '/tmp/gemma_lora_imputer'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=(device == 'cuda'),
        logging_steps=50,
        save_strategy='no',
        report_to='none',
        seed=seed,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    print("[Gemma] Fine-tuning...")
    trainer.train()
    train_elapsed = time.time() - start_time
    print(f"[Gemma] Training complete in {train_elapsed:.1f}s ({train_elapsed/60:.1f} min)")

    # --- Step 7: Inference helper ------------------------------------------
    model.eval()

    def _impute_value(target_col, row, corr_matrix, k):
        observed_other = [c for c in columns_to_impute
                          if c != target_col and pd.notna(row[c])]
        if len(observed_other) == 0:
            col_mean = df[target_col].mean()
            return int(round(col_mean)) if pd.notna(col_mean) else 2

        predictors = _select_top_k_predictors(target_col, corr_matrix, observed_other, k=k)
        pred_vals = [row[p] for p in predictors]
        prompt = _build_prompt(target_col, predictors, pred_vals)

        inputs = tokenizer(prompt, return_tensors='pt',
                           truncation=True, max_length=max_length)
        if device == 'cuda':
            inputs = {k2: v.to('cuda') for k2, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3,
                                  do_sample=False, pad_token_id=tokenizer.pad_token_id)

        new_tokens = out[0][inputs['input_ids'].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        for ch in decoded:
            if ch.isdigit():
                return min(4, max(0, int(ch)))
        col_mean = df[target_col].mean()
        return int(round(col_mean)) if pd.notna(col_mean) else 2

    # --- Step 8: Impute df -------------------------------------------------
    print("[Gemma] Imputing missing values in df...")
    imputed_df = df.copy()

    for col in columns_to_impute:
        missing_idx = imputed_df.index[imputed_df[col].isna()]
        if len(missing_idx) == 0:
            continue
        for idx in missing_idx:
            row = imputed_df.loc[idx]
            val = _impute_value(col, row, corr_matrix, k_predictors)
            imputed_df.at[idx, col] = float(val)

    # --- Step 9: Validation ------------------------------------------------
    validation_results = None
    if validation_df is not None and validation_masks is not None and original_values is not None:
        print("[Gemma] Running validation...")
        validation_results = {}

        for col in columns_to_impute:
            mask = validation_masks[col] & validation_df[col].isna()
            if mask.sum() == 0:
                validation_results[col] = {'error': 'No artificially missing values'}
                continue

            val_imputed = []
            for idx in validation_df.index[mask]:
                row = validation_df.loc[idx]
                val_imputed.append(float(_impute_value(col, row, corr_matrix, k_predictors)))

            real_vals    = original_values[col][mask].values
            imputed_vals = np.array(val_imputed)

            mae  = mean_absolute_error(real_vals, imputed_vals)
            rmse = np.sqrt(mean_squared_error(real_vals, imputed_vals))

            try:
                real_cls    = process_for_classification(real_vals)
                imputed_cls = process_for_classification(imputed_vals)
                cls_metrics = calculate_classification_metrics(real_cls, imputed_cls)
            except NameError:
                cls_metrics = {k: np.nan for k in [
                    'accuracy', 'auc_multiclass', 'avg_sensitivity',
                    'avg_specificity', 'avg_ppv', 'avg_npv',
                    'precision_macro', 'recall_macro']}

            validation_results[col] = {
                'mae':             mae,
                'rmse':            rmse,
                'accuracy':        cls_metrics['accuracy'],
                'auc_multiclass':  cls_metrics['auc_multiclass'],
                'avg_sensitivity': cls_metrics['avg_sensitivity'],
                'avg_specificity': cls_metrics['avg_specificity'],
                'avg_ppv':         cls_metrics['avg_ppv'],
                'avg_npv':         cls_metrics['avg_npv'],
                'precision_macro': cls_metrics['precision_macro'],
                'recall_macro':    cls_metrics['recall_macro'],
                'real_distribution':    pd.Series(real_vals).describe(),
                'imputed_distribution': pd.Series(imputed_vals).describe(),
            }
            print(f"  {col}: MAE={mae:.4f}, RMSE={rmse:.4f}, Acc={cls_metrics['accuracy']:.4f}")

    total_elapsed = time.time() - start_time
    print(f"[Gemma] Done. Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    return imputed_df, validation_results







"""
TimesFM 2.5 imputation function for longitudinal FACT-E data.

Imputation strategy: treat each FACT-E variable as an independent univariate
time series per patient. For each missing value at visit t, observed values
from earlier visits are used as context and TimesFM forecasts forward to fill
the gap ("imputation as forecasting").

For leading-missing values (no prior context), falls back to the patient's
observed column mean, or global column mean if no observations exist.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Visit ordering (matches REDCap event names in the EGDB dataset)
# ---------------------------------------------------------------------------

VISIT_ORDER = {
    'baseline_arm_1':       0,
    'preoperative_arm_1':   1,
    '1_month_postop_arm_1': 2,
    '3_months_postop_arm_1':3,
    '6_months_postop_arm_1':4,
    '1_year_postop_arm_1':  5,
    '2_years_postop_arm_1': 6,
    '3_years_postop_arm_1': 7,
    '4_years_postop_arm_1': 8,
    '5_years_postop_arm_1': 9,
}


import os
os.environ['HF_HOME'] = '/home/yjkweon2/models/TIMESFM'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

_TIMESFM_MODEL = None  # module-level cache: model loads only once per process


# ---------------------------------------------------------------------------
# Section 1 -- Model loading
# ---------------------------------------------------------------------------
        
def _load_timesfm_model():
    """
    Load the TimesFM 2.5 (200M) PyTorch model from the HuggingFace cache and
    return it. On subsequent calls, returns the already-loaded instance.

    ForecastConfig notes:
      - max_context=64: more than enough for 11 clinical timepoints; kept small
        to avoid unnecessary memory allocation.
      - max_horizon=11: the largest gap we could ever need to forecast across.
      - normalize_inputs=True: TimesFM internally z-scores each input series,
        which is important because FACT-E values (0-4) are on a narrow scale.
      - use_continuous_quantile_head=False: we only need a point forecast, not
        a full quantile distribution. Keeps inference faster and avoids loading
        the extra 30M-parameter quantile head.
      - force_flip_invariance=False: PRO trajectories have meaningful direction
        (e.g. steady decline), so averaging forward/backward predictions would
        blur clinically relevant trends.
      - infer_is_positive=True: signals to the model that values are >= 0,
        consistent with the FACT-E 0-4 scale.
      - fix_quantile_crossing=False: only relevant when the quantile head is
        active; irrelevant here.
    """
    global _TIMESFM_MODEL
    if _TIMESFM_MODEL is not None:
        return _TIMESFM_MODEL

    try:
        import timesfm
        import torch
        torch.set_float32_matmul_precision("high")

        print("[TimesFM] Loading model from /home/yjkweon2/models/TIMESFM...")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=64,
                max_horizon=11,
                normalize_inputs=True,
                use_continuous_quantile_head=False,
                force_flip_invariance=False,
                infer_is_positive=True,
                fix_quantile_crossing=False,
            )
        )
        _TIMESFM_MODEL = model
        print("[TimesFM] Model ready.")
        return model

    except ImportError as e:
        raise ImportError(
            f"[TimesFM] Cannot import timesfm: {e}\n"
            "Install: cd ~/timesfm && pip install -e . --break-system-packages"
        )


# ---------------------------------------------------------------------------
# Section 2 -- Build per-patient sorted visit sequences
# ---------------------------------------------------------------------------

def _build_patient_visit_sequences(df, columns_to_impute,
                                   patient_col='id',
                                   event_col='redcap_event_name',
                                   date_col='qol_date'):
    """
    Convert the flat long-format dataframe into a list of per-patient dicts,
    each containing visits sorted in chronological order.

    For each patient we record:
      - indices:  the row indices (in df) in visit order
      - X:        float array [T, F] of FACT-E values (NaN where missing)
      - mask:     bool array  [T, F], True = value was observed

    Sorting priority: redcap_event_name via VISIT_ORDER, then qol_date,
    then original row order as a final fallback.
    """
    feature_cols = [c for c in columns_to_impute if c in df.columns]

    if event_col in df.columns:
        sort_key = df[event_col].map(VISIT_ORDER).fillna(99).astype(int)
    elif date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        sort_key = (dates - dates.min()).dt.days.fillna(0).astype(int)
    else:
        sort_key = pd.Series(range(len(df)), index=df.index)

    sequences = []
    for pid, group in df.groupby(patient_col):
        idx = group.index.tolist()
        keys = sort_key[idx].values
        order = np.argsort(keys, kind='stable')
        idx_sorted = [idx[i] for i in order]

        X = df.loc[idx_sorted, feature_cols].values.astype(float)  # [T, F]
        mask = ~np.isnan(X)                                          # True = observed

        sequences.append({
            'patient_id': pid,
            'indices':    idx_sorted,
            'X':          X,
            'mask':       mask,
        })
    return sequences, feature_cols


# ---------------------------------------------------------------------------
# Section 3 -- Univariate forecasting to fill one missing position
# ---------------------------------------------------------------------------

def _impute_series_with_timesfm(model, obs_vals, miss_pos):
    """
    Fill missing positions for one patient x one FACT-E variable.

    Strategy: for each missing visit t, build a context array from observed
    values at all visits strictly before t (ignoring other missing positions
    so that imputed values never cascade into further imputations).
    TimesFM forecasts 'horizon' steps ahead, where horizon = t - last_observed.

    model.forecast() returns an array of shape (1, horizon): one predicted
    value for each step from the last observed visit up to and including visit
    t. We take forecast[0, -1] -- the final step -- because that corresponds
    exactly to our target visit t. Intermediate steps are discarded.

    Example -- visits 7 and 9 missing, visits 0-6 and 8 observed:
      - Visit 7 missing: context = [obs[0], ..., obs[6]], horizon = 7-6 = 1
                         forecast shape (1, 1); forecast[0, -1] fills visit 7.
      - Visit 9 missing: context = [obs[0], ..., obs[6], obs[8]], horizon = 9-8 = 1
                         forecast shape (1, 1); forecast[0, -1] fills visit 9.
        obs[7] is excluded from context because it was not originally observed,
        preventing imputed values from influencing further imputations.

    For leading-missing positions (no prior context), returns NaN.
    Section 4 (_run_imputation) catches these NaNs and substitutes the
    patient's observed mean for that variable, or the global column mean
    if the patient has no observations at all.

    Parameters
    ----------
    model    : loaded TimesFM model
    obs_vals : dict {timepoint_index: float}  -- originally observed values only
    miss_pos : list of int  -- visit indices to fill

    Returns
    -------
    filled : dict {timepoint_index: float or nan}
    """
    filled = {}
    obs_positions = sorted(obs_vals.keys())

    for t in miss_pos:
        # Only use observations that come strictly before this missing visit
        context_positions = [p for p in obs_positions if p < t]

        if not context_positions:
            filled[t] = np.nan   # no prior context; fallback applied in Section 4
            continue

        context = np.array([obs_vals[p] for p in context_positions], dtype=np.float32)
        horizon = t - context_positions[-1]  # steps from last observed to target

        try:
            point_forecast, _ = model.forecast(horizon=horizon, inputs=[context])
            # shape (1, horizon): take the last step which is our target visit t
            filled[t] = float(point_forecast[0, -1])
        except Exception:
            filled[t] = np.nan

    return filled


# ---------------------------------------------------------------------------
# Section 4 -- Core imputation loop (separated for clean reuse in validation)
# ---------------------------------------------------------------------------

def _run_imputation(df, columns_to_impute, model,
                    patient_col, event_col, date_col):
    """
    Run the full imputation loop over all patients and variables.
    Returns an imputed copy of df.

    Separated from the public function so the validation block can call it
    on validation_df without re-entering argument handling or reloading
    the model.

    For each patient x variable:
      1. Identify observed positions (mask == True) and missing positions.
      2. Call _impute_series_with_timesfm() to get forecasted fill values.
      3. For any NaN result (leading-missing, or forecast error), substitute
         the patient's mean across observed values for that variable, falling
         back to the global column mean if the patient has no observations.
      4. Clamp to [0, 4] and round to the nearest integer (ordinal FACT-E scale).
    """
    global_col_means = {col: df[col].mean() for col in columns_to_impute}

    sequences, feature_cols = _build_patient_visit_sequences(
        df, columns_to_impute,
        patient_col=patient_col, event_col=event_col, date_col=date_col,
    )
    feat_idx = {col: i for i, col in enumerate(feature_cols)}

    imputed_df = df.copy()

    for seq in sequences:
        X    = seq['X']     # [T, F]
        mask = seq['mask']  # [T, F]

        for col in feature_cols:
            f = feat_idx[col]
            obs_vals = {t: X[t, f] for t in range(len(seq['indices'])) if mask[t, f]}
            miss_pos = [t for t in range(len(seq['indices'])) if not mask[t, f]]

            if not miss_pos:
                continue

            filled = _impute_series_with_timesfm(model, obs_vals, miss_pos)

            # Fallback for leading-missing (no prior context) or forecast errors
            patient_mean = (np.mean(list(obs_vals.values()))
                            if obs_vals else global_col_means.get(col, 2.0))

            for t, row_idx in enumerate(seq['indices']):
                if not mask[t, f]:
                    raw = filled.get(t, np.nan)
                    if np.isnan(raw):
                        raw = patient_mean
                    imputed_df.at[row_idx, col] = float(np.clip(round(raw), 0, 4))

    return imputed_df


# ---------------------------------------------------------------------------
# Section 5 -- Public function (matches apply_bpca_imputation signature)
# ---------------------------------------------------------------------------

def apply_timesfm_imputation(df, columns_to_impute,
                              validation_df=None,
                              validation_masks=None,
                              original_values=None,
                              patient_col='id',
                              event_col='redcap_event_name',
                              date_col='qol_date'):
    """
    Apply TimesFM 2.5 (200M) imputation for longitudinal FACT-E data.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format patient data with missing values across 11 timepoints.
    columns_to_impute : list of str
        FACT-E column names to impute.
    validation_df : pd.DataFrame, optional
        Copy of df with additional artificial missingness for validation.
    validation_masks : dict, optional
        {col: boolean Series} indicating which values were artificially masked.
    original_values : dict, optional
        {col: Series} of true values at the masked positions.
    patient_col : str
        Column identifying the patient (default 'id').
    event_col : str
        REDCap event name column used for visit ordering (default
        'redcap_event_name').
    date_col : str
        Fallback date column if event names are absent (default 'qol_date').

    Returns
    -------
    imputed_df : pd.DataFrame
        Copy of df with missing FACT-E values filled.
    validation_results : dict or None
        Per-column validation metrics if validation data provided, else None.
    """
    start_time = time.time()
    model = _load_timesfm_model()

    print(f"[TimesFM] Imputing {df[patient_col].nunique()} patients "
          f"x {len(columns_to_impute)} variables...")

    imputed_df = _run_imputation(df, columns_to_impute, model,
                                 patient_col, event_col, date_col)

    elapsed = time.time() - start_time
    print(f"[TimesFM] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ------------------------------------------------------------------
    # Validation block -- mirrors structure in apply_bpca_imputation
    # ------------------------------------------------------------------
    validation_results = None
    if validation_df is not None and validation_masks is not None and original_values is not None:
        print("[TimesFM] Running validation...")

        val_imputed_df = _run_imputation(validation_df, columns_to_impute, model,
                                         patient_col, event_col, date_col)

        validation_results = {}
        all_real, all_pred = [], []

        for col in columns_to_impute:
            mask_col = validation_masks[col] & validation_df[col].isna()

            if mask_col.sum() == 0:
                validation_results[col] = {'error': 'No artificially missing values'}
                continue

            real_vals = original_values[col][mask_col]
            pred_vals = val_imputed_df.loc[mask_col, col]

            all_real.extend(real_vals.values)
            all_pred.extend(pred_vals.values)

            mae  = mean_absolute_error(real_vals, pred_vals)
            rmse = np.sqrt(mean_squared_error(real_vals, pred_vals))

            real_cls = process_for_classification(real_vals)
            pred_cls = process_for_classification(pred_vals)
            cls_metrics = calculate_classification_metrics(real_cls, pred_cls)

            validation_results[col] = {
                'mae':                  mae,
                'rmse':                 rmse,
                'accuracy':             cls_metrics['accuracy'],
                'auc_multiclass':       cls_metrics['auc_multiclass'],
                'avg_sensitivity':      cls_metrics['avg_sensitivity'],
                'avg_specificity':      cls_metrics['avg_specificity'],
                'avg_ppv':              cls_metrics['avg_ppv'],
                'avg_npv':              cls_metrics['avg_npv'],
                'precision_macro':      cls_metrics['precision_macro'],
                'recall_macro':         cls_metrics['recall_macro'],
                'real_distribution':    real_vals.describe(),
                'imputed_distribution': pred_vals.describe(),
            }

        if all_real:
            overall_mae  = mean_absolute_error(all_real, all_pred)
            overall_rmse = np.sqrt(mean_squared_error(all_real, all_pred))
            print(f"[TimesFM] Overall validation -- MAE: {overall_mae:.4f}, "
                  f"RMSE: {overall_rmse:.4f}")

    return imputed_df, validation_results



















# Add these imports at the top of your file
# from TransformerImputer import apply_transformer_imputation
# from CUDATransformerImputer import apply_cudatransformer_imputation

def evaluate_real_data_imputation(df, columns_to_impute):
    """
    Evaluate imputation methods on real data with existing missing values
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your real dataset with missing values
    columns_to_impute : list
        List of column names to impute
        
    Returns:
    --------
    all_imputed_dfs : dict
        Dictionary with imputed DataFrames from each method
    original_df_with_missing : pandas.DataFrame
        Original dataframe with missing values
    execution_times : dict
        Dictionary with execution times for each method
    """
    # Define the methods to compare
    methods = {
        'MICE': apply_mice_imputation,
        'Bayesian PCA': apply_bpca_imputation,
        'SoftImpute': apply_softimpute_imputation,
        'LSTM': apply_lstm_imputation,                               
        'Longitudinal MICE': apply_mice_longitudinal_imputation,
        'Gemma':    lambda df, columns_to_impute, **kw: apply_gemma_imputation(df, columns_to_impute, model_name='/home/yjkweon2/models/gemma-2-2b-it', **kw),
        'MedGemma': lambda df, columns_to_impute, **kw: apply_gemma_imputation(df, columns_to_impute, model_name='/home/yjkweon2/models/medgemma-1.5-4b-it', **kw),
        'TimesFM': apply_timesfm_imputation   
    }
    
    # Initialize dictionaries to store results
    all_imputed_dfs = {}
    execution_times = {}
    distribution_similarity = {method: {} for method in methods}
    
    # Calculate missingness statistics before imputation
    print("Missingness statistics before imputation:")
    for col in columns_to_impute:
        missing_count = df[col].isna().sum()
        missing_percent = (missing_count / len(df)) * 100
        print(f"{col}: {missing_count} missing values ({missing_percent:.2f}%)")
    
    # Save a copy of the original data with missing values
    original_df_with_missing = df.copy()
    
    # Run each imputation method
    for method_name, method_func in methods.items():
        print(f"\nRunning {method_name} imputation...")
        
        try:
            # Time the imputation
            start_time = time.time()
            
            # Create a copy of the dataframe for this method
            method_df = df.copy()
            
            # For methods other than LSTM and Longitudinal MICE, drop the qol_date column to avoid issues
            if method_name not in ('LSTM', 'Longitudinal MICE', 'TimesFM') and 'qol_date' in method_df.columns:
                print(f"Dropping qol_date column for {method_name} method")
                method_df = method_df.drop(columns=['qol_date'])
            
            # Apply the imputation method
            imputed_df, _ = method_func(
                method_df, 
                columns_to_impute
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times[method_name] = execution_time
            
            # Store the imputed dataframe
            all_imputed_dfs[method_name] = imputed_df
            
            # Print execution time
            print(f"{method_name} completed in {execution_time:.2f} seconds")
            
            # Check if imputation was successful (no missing values in imputed columns)
            for col in columns_to_impute:
                if col not in imputed_df.columns:
                    print(f"Warning: Column {col} missing in {method_name} results")
                    continue
                    
                remaining_missing = imputed_df[col].isna().sum()
                if remaining_missing > 0:
                    print(f"Warning: {method_name} left {remaining_missing} missing values in {col}")
                
                # Measure distribution similarity between observed and imputed values
                observed_values = original_df_with_missing.loc[original_df_with_missing[col].notna(), col]
                imputed_indices = original_df_with_missing[col].isna() & ~imputed_df[col].isna()
                
                if imputed_indices.any():
                    imputed_values = imputed_df.loc[imputed_indices, col]
                    
                    if len(observed_values) > 0 and len(imputed_values) > 0:
                        # Kolmogorov-Smirnov test for distribution similarity
                        ks_stat, ks_pval = ks_2samp(observed_values, imputed_values)
                        distribution_similarity[method_name][col] = {
                            'ks_stat': ks_stat,
                            'ks_pval': ks_pval
                        }
                        print(f"{method_name} - {col} distribution similarity: KS={ks_stat:.4f}, p={ks_pval:.4f}")
                        #Lower KS statistic and higher p-values indicate better distribution preservation
                    
        except Exception as e:
            print(f"Error running {method_name}: {e}")
            import traceback
            traceback.print_exc()  # Print the full error trace for better diagnosis
    
    return all_imputed_dfs, original_df_with_missing, execution_times, distribution_similarity


def evaluate_with_sparse_validation(df, columns_to_impute, n_folds=5):

    methods = {
        'MICE': apply_mice_imputation,
        'Bayesian PCA': apply_bpca_imputation,
        'SoftImpute': apply_softimpute_imputation,
        'LSTM': apply_lstm_imputation,
        'Longitudinal MICE': apply_mice_longitudinal_imputation,
        'Gemma':    lambda df, columns_to_impute, **kw: apply_gemma_imputation(df, columns_to_impute, model_name='/home/yjkweon2/models/gemma-2-2b-it', **kw),
        'MedGemma': lambda df, columns_to_impute, **kw: apply_gemma_imputation(df, columns_to_impute, model_name='/home/yjkweon2/models/medgemma-1.5-4b-it', **kw),
        'TimesFM': apply_timesfm_imputation
    }

    results = {method: {
        'mae': [], 'rmse': [], 'accuracy': [], 'auc_multiclass': [],
        'avg_sensitivity': [], 'avg_specificity': [], 'avg_ppv': [], 'avg_npv': [],
        'precision_macro': [], 'recall_macro': [], 'time': []
    } for method in methods}

    # Filter valid columns
    valid_columns = [col for col in columns_to_impute
                     if col in df.columns and not col.startswith('qol_date_')]
    if len(valid_columns) != len(columns_to_impute):
        excluded_cols = set(columns_to_impute) - set(valid_columns)
        print(f"Warning: Excluding {len(excluded_cols)} columns: {sorted(excluded_cols)}")

    # Pre-compute per-column observed indices once
    col_observed_indices = {}
    for col in valid_columns:
        observed = df.index[df[col].notna()].tolist()
        if len(observed) >= max(5, n_folds * 2):
            col_observed_indices[col] = observed
        else:
            print(f"Skipping {col}: too few observed values ({len(observed)})")

    eligible_columns = list(col_observed_indices.keys())
    print(f"\nEligible columns for validation: {len(eligible_columns)}/{len(valid_columns)}")

    # Pre-compute fold test indices per column once
    # {col: [fold0_test_indices, fold1_test_indices, ...]}
    col_fold_indices = {}
    for col in eligible_columns:
        observed = col_observed_indices[col]
        n_observed = len(observed)
        fold_size = max(5, n_observed // n_folds)
        np.random.seed(42)
        shuffled = np.random.permutation(observed)
        n_actual_folds = min(n_folds, n_observed // fold_size)
        folds = []
        for fold in range(n_actual_folds):
            start = fold * fold_size
            end = min(start + fold_size, n_observed)
            folds.append(shuffled[start:end])
        col_fold_indices[col] = folds

    # Main loop: iterate over folds, not columns
    # Use n_folds from the column that has the most folds available
    max_folds = max(len(folds) for folds in col_fold_indices.values()) if col_fold_indices else 0

    for fold in range(max_folds):
        print(f"\n{'='*80}")
        print(f"FOLD {fold + 1}/{max_folds}")
        print(f"{'='*80}")

        # --- Mask ALL columns at once for this fold ------------------------
        df_fold = df.copy()
        
        # Store original values and track which columns participate in this fold
        fold_originals = {}  # {col: (test_indices, original_values)}
        
        for col in eligible_columns:
            folds = col_fold_indices[col]
            if fold >= len(folds):
                continue  # this column has fewer folds, skip
            
            test_indices = folds[fold]
            fold_originals[col] = {
                'test_indices': test_indices,
                'original_values': df_fold.loc[test_indices, col].copy()
            }
            # Mask this column's test indices
            df_fold.loc[test_indices, col] = np.nan

        participating_cols = list(fold_originals.keys())
        print(f"Fold {fold+1}: masking {len(participating_cols)} columns simultaneously")

        # --- Run each method ONCE on the multi-column masked dataframe -----
        for method_name, method_func in methods.items():
            print(f"\nRunning {method_name}...")

            try:
                method_df = df_fold.copy()
                if method_name not in ('LSTM', 'Longitudinal MICE', 'TimesFM') and 'qol_date' in method_df.columns:
                    method_df = method_df.drop(columns=['qol_date'])

                start_time = time.time()
                imputed_df, _ = method_func(method_df, valid_columns)  # single training run
                execution_time = time.time() - start_time

                # --- Evaluate all columns from this single imputed result --
                for col in participating_cols:
                    if col not in imputed_df.columns:
                        print(f"  Warning: {col} not found in {method_name} results. Skipping.")
                        continue

                    test_indices    = fold_originals[col]['test_indices']
                    original_values = fold_originals[col]['original_values']
                    imputed_values  = imputed_df.loc[test_indices, col]

                    # Handle any remaining missing values
                    still_missing = imputed_values.isna().sum()
                    if still_missing > 0:
                        print(f"  Warning: {method_name}/{col} has {still_missing} unimputed values")
                        valid_idx = [i for i in test_indices if not pd.isna(imputed_df.loc[i, col])]
                        if not valid_idx:
                            continue
                        original_values_filtered = df.loc[valid_idx, col]
                        imputed_values_filtered  = imputed_df.loc[valid_idx, col]
                    else:
                        original_values_filtered = original_values
                        imputed_values_filtered  = imputed_values

                    # Continuous metrics (no rounding)
                    mae  = mean_absolute_error(original_values_filtered, imputed_values_filtered)
                    rmse = np.sqrt(mean_squared_error(original_values_filtered, imputed_values_filtered))

                    # Classification metrics (with rounding)
                    real_cls    = process_for_classification(original_values_filtered)
                    imputed_cls = process_for_classification(imputed_values_filtered)
                    cls         = calculate_classification_metrics(real_cls, imputed_cls)

                    results[method_name]['mae'].append(mae)
                    results[method_name]['rmse'].append(rmse)
                    results[method_name]['accuracy'].append(cls['accuracy'])
                    results[method_name]['auc_multiclass'].append(
                        cls['auc_multiclass'] if not np.isnan(cls['auc_multiclass']) else 0)
                    results[method_name]['avg_sensitivity'].append(cls['avg_sensitivity'])
                    results[method_name]['avg_specificity'].append(cls['avg_specificity'])
                    results[method_name]['avg_ppv'].append(cls['avg_ppv'])
                    results[method_name]['avg_npv'].append(cls['avg_npv'])
                    results[method_name]['precision_macro'].append(cls['precision_macro'])
                    results[method_name]['recall_macro'].append(cls['recall_macro'])
                    results[method_name]['time'].append(execution_time)

                # Print per-fold summary for this method
                if results[method_name]['mae']:
                    recent_maes = results[method_name]['mae'][-len(participating_cols):]
                    print(f"  {method_name} fold {fold+1} avg MAE across {len(participating_cols)} cols: "
                          f"{np.mean(recent_maes):.4f} | Time: {execution_time:.2f}s")

            except Exception as e:
                print(f"  Error with {method_name}: {e}")
                import traceback
                traceback.print_exc()

    # Aggregate results
    metric_names = ['mae', 'rmse', 'accuracy', 'auc_multiclass', 'avg_sensitivity',
                    'avg_specificity', 'avg_ppv', 'avg_npv', 'precision_macro', 'recall_macro', 'time']

    for method in methods:
        if results[method]['mae']:
            for metric in metric_names:
                results[method][f'avg_{metric}'] = np.mean(results[method][metric])
                results[method][f'std_{metric}'] = np.std(results[method][metric])
        else:
            print(f"No valid results for {method}")
            for metric in metric_names:
                results[method][f'avg_{metric}'] = np.nan
                results[method][f'std_{metric}'] = np.nan

    return results





# =============================================================================
# LONGITUDINAL-SPECIFIC EVALUATION METRICS
# =============================================================================

def evaluate_trajectory_fidelity(df, imputed_dfs, columns_to_impute, 
                                 patient_col='id', event_col='redcap_event_name',
                                 n_patients_sample=400):
    """
    Evaluate how well methods preserve individual patient trajectory patterns.
    
    Strategy: For patients with ≥3 observed timepoints, mask ALL patients'
    middle visits simultaneously, run each method ONCE on the full masked
    dataset, then evaluate trajectory fidelity across all patients.
    
    This reduces training runs from (n_patients × n_methods) to just n_methods.
    """
    print("\n" + "="*80)
    print("TRAJECTORY FIDELITY EVALUATION")
    print("="*80)
    
    results = {}
    
    # --- Step 1: Identify eligible patients (≥3 timepoints) ----------------
    visit_counts = df.groupby(patient_col)[event_col].nunique()
    eligible_patients = visit_counts[visit_counts >= 3].index.tolist()
    
    if len(eligible_patients) == 0:
        print("No patients with ≥3 timepoints for trajectory evaluation")
        return results
    
    # Sample patients for speed
    if len(eligible_patients) > n_patients_sample:
        np.random.seed(42)
        eligible_patients = np.random.choice(
            eligible_patients, n_patients_sample, replace=False
        ).tolist()
    
    print(f"Evaluating {len(eligible_patients)} patients with ≥3 timepoints")
    
    # --- Step 2: Mask ALL patients' middle visits at once ------------------
    masked_data = df.copy()
    middle_visit_indices = {}  # {pid: (middle_visit_idx, original_values)}
    
    for pid in eligible_patients:
        patient_data = df[df[patient_col] == pid].copy()
        
        if len(patient_data) < 3:
            continue
        
        # Sort by visit order
        if event_col in patient_data.columns:
            patient_data['_sort_key'] = patient_data[event_col].map(VISIT_ORDER).fillna(99)
            patient_data = patient_data.sort_values('_sort_key')
        
        # Get middle visit
        mid_idx = len(patient_data) // 2
        middle_visit_idx = patient_data.index[mid_idx]
        
        # Skip if middle visit is already entirely missing
        orig_vals = df.loc[middle_visit_idx, columns_to_impute]
        if orig_vals.isna().all():
            continue
        
        # Store original values for evaluation later
        # middle_visit_indices[pid] = {
        #     'row_idx': middle_visit_idx,
        #     'original_values': orig_vals.values.copy(),  # [n_features]
        #     'obs_mask': ~np.isnan(orig_vals.values)      # True = observed
        # }
        middle_visit_indices[pid] = {
            'row_idx': middle_visit_idx,
            'original_values': orig_vals.astype(float).values.copy(),
            'obs_mask': ~orig_vals.isna().values
        }
        
        # Mask the middle visit in the shared masked dataset
        masked_data.loc[middle_visit_idx, columns_to_impute] = np.nan
    
    n_valid = len(middle_visit_indices)
    print(f"Successfully masked middle visits for {n_valid} patients")
    
    if n_valid == 0:
        print("No valid patients to evaluate after masking")
        return results
    
    # --- Step 3: Run each method ONCE on the full masked dataset -----------
    methods = [
        ('MICE',               apply_mice_imputation),
        ('Bayesian PCA',       apply_bpca_imputation),
        ('SoftImpute',         apply_softimpute_imputation),
        ('LSTM',               apply_lstm_imputation),
        ('Longitudinal MICE',  apply_mice_longitudinal_imputation),
        ('Gemma',    lambda df, columns_to_impute, **kw: apply_gemma_imputation(df, columns_to_impute, model_name='/home/yjkweon2/models/gemma-2-2b-it', **kw)),
        ('MedGemma', lambda df, columns_to_impute, **kw: apply_gemma_imputation(df, columns_to_impute, model_name='/home/yjkweon2/models/medgemma-1.5-4b-it', **kw)),
        ('TimesFM', apply_timesfm_imputation)
    ]
    
    trajectory_errors = {name: [] for name, _ in methods}
    trajectory_corrs  = {name: [] for name, _ in methods}
    
    for method_name, method_func in methods:
        print(f"\nRunning {method_name} on masked dataset...")
        
        try:
            method_df = masked_data.copy()
            
            # Drop qol_date for methods that don't need it
            if method_name not in ('LSTM', 'Longitudinal MICE', 'TimesFM') and 'qol_date' in method_df.columns:
                method_df = method_df.drop(columns=['qol_date'])
            
            # Single training run on the full masked dataset
            imputed_df, _ = method_func(method_df, columns_to_impute)
            
            # --- Step 4: Evaluate across all patients ----------------------
            for pid, info in middle_visit_indices.items():
                row_idx      = info['row_idx']
                orig_middle  = info['original_values']   # [n_features]
                obs_mask     = info['obs_mask']           # [n_features] bool
                
                if obs_mask.sum() == 0:
                    continue
                
                # Get imputed values at the masked middle visit row
                # imp_middle = imputed_df.loc[row_idx, columns_to_impute].values
                imp_middle = imputed_df.loc[row_idx, columns_to_impute].values.astype(float)
                
                # Compare only at originally-observed positions
                orig_obs = orig_middle[obs_mask]
                imp_obs  = imp_middle[obs_mask]
                
                # Skip if imputation failed (NaNs remain)
                if np.isnan(imp_obs).any():
                    continue
                
                traj_mae = np.abs(imp_obs - orig_obs).mean()
                trajectory_errors[method_name].append(traj_mae)
                
                # Trajectory correlation
                if len(orig_obs) > 1 and orig_obs.std() > 0:
                    corr = np.corrcoef(orig_obs, imp_obs)[0, 1]
                    if not np.isnan(corr):
                        trajectory_corrs[method_name].append(corr)
        
        except Exception as e:
            print(f"  Error with {method_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # --- Step 5: Aggregate and print results -------------------------------
    for method_name, _ in methods:
        errors = trajectory_errors[method_name]
        corrs  = trajectory_corrs[method_name]
        
        if errors:
            results[method_name] = {
                'trajectory_mae':         np.mean(errors),
                'trajectory_mae_std':     np.std(errors),
                'trajectory_correlation': np.mean(corrs) if corrs else np.nan,
                'n_patients':             len(errors)
            }
    
    print("\n" + "="*80)
    print("TRAJECTORY FIDELITY RESULTS")
    print("="*80)
    print(f"{'Method':<20} {'Traj MAE':<12} {'Std':<10} {'Traj Corr':<12} {'N Patients':<12}")
    print("-"*80)
    for method, metrics in sorted(results.items(), key=lambda x: x[1]['trajectory_mae']):
        print(f"{method:<20} "
              f"{metrics['trajectory_mae']:<12.4f} "
              f"{metrics['trajectory_mae_std']:<10.4f} "
              f"{metrics['trajectory_correlation']:<12.4f} "
              f"{metrics['n_patients']:<12}")
    print("="*80)
    
    return results

# def evaluate_trajectory_fidelity(df, imputed_dfs, columns_to_impute, 
#                                  patient_col='id', event_col='redcap_event_name'):
#     """
#     Evaluate how well methods preserve individual patient trajectory patterns.
    
#     Strategy: For patients with ≥3 observed timepoints, mask the middle visit
#     entirely and measure how well the imputed trajectory matches the true shape.
    
#     Returns
#     -------
#     results : dict
#         {method_name: {'trajectory_mae': float, 'trajectory_correlation': float}}
#     """
#     print("\n" + "="*80)
#     print("TRAJECTORY FIDELITY EVALUATION")
#     print("="*80)
    
#     results = {}
    
#     # Identify patients with ≥3 timepoints
#     visit_counts = df.groupby(patient_col)[event_col].nunique()
#     eligible_patients = visit_counts[visit_counts >= 3].index.tolist()
    
#     if len(eligible_patients) == 0:
#         print("No patients with ≥3 timepoints for trajectory evaluation")
#         return results
    
#     print(f"Evaluating {len(eligible_patients)} patients with ≥3 timepoints")
    
#     # For each patient, mask their middle visit and evaluate trajectory
#     trajectory_errors = {method: [] for method in imputed_dfs.keys()}
#     trajectory_corrs = {method: [] for method in imputed_dfs.keys()}
    
#     for pid in tqdm(eligible_patients[:100], desc="Trajectory evaluation"):  # limit to 100 for speed
#         patient_data = df[df[patient_col] == pid].copy()
        
#         if len(patient_data) < 3:
#             continue
            
#         # Sort by visit
#         if event_col in patient_data.columns:
#             patient_data['_sort_key'] = patient_data[event_col].map(VISIT_ORDER).fillna(99)
#             patient_data = patient_data.sort_values('_sort_key')
        
#         # Get middle visit index
#         mid_idx = len(patient_data) // 2
#         middle_visit_idx = patient_data.index[mid_idx]
        
#         # Check if this visit has any observed values
#         observed_vals = patient_data.loc[middle_visit_idx, columns_to_impute]
#         if observed_vals.isna().all():
#             continue  # skip if already all missing
        
#         # Store original values
#         original_trajectory = patient_data[columns_to_impute].values
        
#         # Create masked version (entire middle visit set to NaN)
#         masked_data = df.copy()
#         masked_data.loc[middle_visit_idx, columns_to_impute] = np.nan
        
#         # Evaluate each method
#         for method_name, method_func in [('MICE', apply_mice_imputation),
#                                           ('Bayesian PCA', apply_bpca_imputation),
#                                           ('SoftImpute', apply_softimpute_imputation),
#                                           ('LSTM', apply_lstm_imputation),
#                                           ('Longitudinal MICE', apply_mice_longitudinal_imputation)]:
#             try:
#                 # Create method-specific copy
#                 method_df = masked_data.copy()
#                 if method_name not in ('LSTM', 'Longitudinal MICE') and 'qol_date' in method_df.columns:
#                     method_df = method_df.drop(columns=['qol_date'])
                
#                 # Impute
#                 imputed_df, _ = method_func(method_df, columns_to_impute)
                
#                 # Get imputed trajectory for this patient
#                 imputed_patient = imputed_df[imputed_df[patient_col] == pid].copy()
#                 if event_col in imputed_patient.columns:
#                     imputed_patient['_sort_key'] = imputed_patient[event_col].map(VISIT_ORDER).fillna(99)
#                     imputed_patient = imputed_patient.sort_values('_sort_key')
                
#                 imputed_trajectory = imputed_patient[columns_to_impute].values
                
#                 # Compare trajectories (MAE across all timepoints and features)
#                 if imputed_trajectory.shape == original_trajectory.shape:
#                     # Only compare observed positions in original
#                     obs_mask = ~np.isnan(original_trajectory)
#                     if obs_mask.sum() > 0:
#                         traj_mae = np.abs(imputed_trajectory[obs_mask] - original_trajectory[obs_mask]).mean()
#                         trajectory_errors[method_name].append(traj_mae)
                        
#                         # Trajectory correlation (flatten and correlate)
#                         orig_flat = original_trajectory[obs_mask]
#                         imp_flat = imputed_trajectory[obs_mask]
#                         if len(orig_flat) > 1 and orig_flat.std() > 0:
#                             corr = np.corrcoef(orig_flat, imp_flat)[0, 1]
#                             trajectory_corrs[method_name].append(corr)
                            
#             except Exception as e:
#                 print(f"  Error with {method_name} for patient {pid}: {e}")
    
#     # Aggregate results
#     for method in trajectory_errors.keys():
#         if trajectory_errors[method]:
#             results[method] = {
#                 'trajectory_mae': np.mean(trajectory_errors[method]),
#                 'trajectory_correlation': np.mean(trajectory_corrs[method]) if trajectory_corrs[method] else 0,
#                 'n_patients': len(trajectory_errors[method])
#             }
    
#     # Print results
#     print("\n" + "="*80)
#     print("TRAJECTORY FIDELITY RESULTS")
#     print("="*80)
#     print(f"{'Method':<20} {'Traj MAE':<12} {'Traj Corr':<12} {'N Patients':<12}")
#     print("-"*80)
#     for method, metrics in sorted(results.items(), key=lambda x: x[1]['trajectory_mae']):
#         print(f"{method:<20} {metrics['trajectory_mae']:<12.4f} {metrics['trajectory_correlation']:<12.4f} {metrics['n_patients']:<12}")
#     print("="*80)
    
#     return results


def evaluate_temporal_smoothness(df, imputed_dfs, columns_to_impute,
                                 patient_col='id', event_col='redcap_event_name'):
    """
    Assess unrealistic discontinuities in patient trajectories.
    
    Measures the average absolute difference between consecutive timepoints.
    Lower values indicate smoother, more realistic trajectories.
    """
    print("\n" + "="*80)
    print("TEMPORAL SMOOTHNESS EVALUATION")
    print("="*80)
    
    results = {}
    
    for method_name, imputed_df in imputed_dfs.items():
        smoothness_scores = []
        
        for pid, patient_data in imputed_df.groupby(patient_col):
            if len(patient_data) < 2:
                continue
                
            # Sort by visit
            if event_col in patient_data.columns:
                patient_data['_sort_key'] = patient_data[event_col].map(VISIT_ORDER).fillna(99)
                patient_data = patient_data.sort_values('_sort_key')
            
            # Calculate consecutive differences
            trajectory = patient_data[columns_to_impute].values
            diffs = np.abs(np.diff(trajectory, axis=0))
            
            # Average across features and timepoints
            smoothness_scores.append(np.nanmean(diffs))
        
        if smoothness_scores:
            results[method_name] = {
                'mean_smoothness': np.mean(smoothness_scores),
                'std_smoothness': np.std(smoothness_scores),
                'n_patients': len(smoothness_scores)
            }
    
    # Print results
    print(f"{'Method':<20} {'Mean Smoothness':<18} {'Std':<12} {'N Patients':<12}")
    print("-"*80)
    for method, metrics in sorted(results.items(), key=lambda x: x[1]['mean_smoothness']):
        print(f"{method:<20} {metrics['mean_smoothness']:<18.4f} {metrics['std_smoothness']:<12.4f} {metrics['n_patients']:<12}")
    print("="*80)
    
    return results



# def evaluate_missing_pattern_robustness(df, columns_to_impute,
#                                         patient_col='id', event_col='redcap_event_name',
#                                         n_folds=3, n_columns_sample=35): # ← Reduced from 35 (when we use samples, Da Xu does not work...)
#     """
#     Compare method performance across different missing patterns.
#     Uses sparse cross-validation (rigorous) but samples columns for speed.
#     """
#     print("\n" + "="*80)
#     print("MISSING PATTERN ROBUSTNESS EVALUATION")
#     print("="*80)
    
#     # Classify patients by missing pattern
#     monotone_patients = []
#     intermittent_patients = []
    
#     for pid, patient_data in df.groupby(patient_col):
#         if len(patient_data) < 2:
#             continue
        
#         patient_data = patient_data.copy()
#         if event_col in patient_data.columns:
#             patient_data['_sort_key'] = patient_data[event_col].map(VISIT_ORDER).fillna(99)
#             patient_data = patient_data.sort_values('_sort_key')
        
#         visit_indices = patient_data['_sort_key'].values if '_sort_key' in patient_data.columns else np.arange(len(patient_data))
        
#         gaps = np.diff(visit_indices)
#         has_gaps = np.any(gaps > 1)
#         max_visit = max(visit_indices) if len(visit_indices) > 0 else 0
        
#        if has_gaps:
#            intermittent_patients.append(pid)
#        else:
#            min_visit = min(visit_indices)
#            if min_visit == 0 and max_visit < 9:
#                monotone_patients.append(pid)
#            else:
#                intermittent_patients.append(pid)
    
#     print(f"Monotone dropout patients: {len(monotone_patients)}")
#     print(f"Intermittent missing patients: {len(intermittent_patients)}")
    
#     if len(monotone_patients) == 0 or len(intermittent_patients) == 0:
#         print("Insufficient patients in one or both pattern groups")
#         return {}
    
#     results = {'monotone': {}, 'intermittent': {}}
    
#     # Sample columns to speed up evaluation
#     if len(columns_to_impute) > n_columns_sample:
#         np.random.seed(42)
#         sampled_cols = np.random.choice(columns_to_impute, n_columns_sample, replace=False).tolist()
#         print(f"Sampling {n_columns_sample} columns from {len(columns_to_impute)} for speed")
#     else:
#         sampled_cols = columns_to_impute
    
#     for pattern, patient_list in [('monotone', monotone_patients), 
#                                    ('intermittent', intermittent_patients)]:
#         print(f"\nEvaluating {pattern} pattern ({len(patient_list)} patients)...")
        
#         # Filter df to this patient group
#         pattern_df = df[df[patient_col].isin(patient_list)].copy()
        
#         # Run sparse validation with sampled columns
#         pattern_results = evaluate_with_sparse_validation(
#             pattern_df, sampled_cols, n_folds=n_folds
#         )
        
#         # Extract average MAE per method
#         for method, metrics in pattern_results.items():
#             if 'avg_mae' in metrics:
#                 results[pattern][method] = {
#                     'mae': metrics['avg_mae'],
#                     'rmse': metrics.get('avg_rmse', np.nan),
#                     'n_columns': len(sampled_cols)
#                 }
    
#     # Print comparison
#     print("\n" + "="*80)
#     print("MISSING PATTERN COMPARISON")
#     print("="*80)
#     print(f"{'Method':<20} {'Monotone MAE':<15} {'Intermittent MAE':<15} {'Difference':<12}")
#     print("-"*80)
    
#     all_methods = set()
#     for pattern in results.values():
#         all_methods.update(pattern.keys())
    
#     for method in sorted(all_methods):
#         mono_mae = results['monotone'].get(method, {}).get('mae', np.nan)
#         inter_mae = results['intermittent'].get(method, {}).get('mae', np.nan)
#         diff = inter_mae - mono_mae if not np.isnan(mono_mae) and not np.isnan(inter_mae) else np.nan
        
#         print(f"{method:<20} {mono_mae:<15.4f} {inter_mae:<15.4f} {diff:<12.4f}")
#     print("="*80)
    
#     return results


def evaluate_missing_pattern_robustness(df, columns_to_impute,
                                       patient_col='id', event_col='redcap_event_name'):
    """
    Compare method performance across different missing patterns:
    - Monotone dropout (patients drop out permanently after visit X)
    - Intermittent missing (sporadic attendance with returns)
    
    Returns performance metrics stratified by pattern type.
    """
    print("\n" + "="*80)
    print("MISSING PATTERN ROBUSTNESS EVALUATION")
    print("="*80)
    
    # Classify patients by missing pattern
    monotone_patients = []
    intermittent_patients = []
    
    for pid, patient_data in df.groupby(patient_col):
        if len(patient_data) < 2:
            continue
            
        # Sort by visit
        if event_col in patient_data.columns:
            patient_data['_sort_key'] = patient_data[event_col].map(VISIT_ORDER).fillna(99)
            patient_data = patient_data.sort_values('_sort_key')
        
        # Check pattern: if patient has data then stops completely = monotone
        # If patient has gaps then returns = intermittent
        visit_indices = patient_data['_sort_key'].values if '_sort_key' in patient_data.columns else range(len(patient_data))
        
        # Count gaps
        gaps = np.diff(visit_indices)
        has_gaps = np.any(gaps > 1)
        
        if has_gaps:
            intermittent_patients.append(pid)
        else:
            min_visit = min(visit_indices)
            max_visit = max(visit_indices)
            if min_visit == 0 and max_visit < 9:
                monotone_patients.append(pid)
            else:
                intermittent_patients.append(pid)
    
    print(f"Monotone dropout patients: {len(monotone_patients)}")
    print(f"Intermittent missing patients: {len(intermittent_patients)}")
    
    # Now run sparse validation separately on each group
    results = {
        'monotone': {},
        'intermittent': {}
    }
    
    for pattern, patient_list in [('monotone', monotone_patients), 
                                   ('intermittent', intermittent_patients)]:
        if len(patient_list) == 0:
            continue
            
        print(f"\nEvaluating {pattern} pattern ({len(patient_list)} patients)...")
        
        # Filter df to this patient group
        pattern_df = df[df[patient_col].isin(patient_list)].copy()
        
        # Run abbreviated validation (5 folds, fast)
        pattern_results = evaluate_with_sparse_validation(pattern_df, columns_to_impute, n_folds=5)
        
        # Extract average MAE per method
        for method, metrics in pattern_results.items():
            if 'avg_mae' in metrics:
                if method not in results[pattern]:
                    results[pattern][method] = {}
                results[pattern][method]['mae'] = metrics['avg_mae']
                results[pattern][method]['rmse'] = metrics['avg_rmse']
    
    # Print comparison
    print("\n" + "="*80)
    print("MISSING PATTERN COMPARISON")
    print("="*80)
    print(f"{'Method':<20} {'Monotone MAE':<15} {'Intermittent MAE':<15} {'Difference':<12}")
    print("-"*80)
    
    all_methods = set()
    for pattern in results.values():
        all_methods.update(pattern.keys())
    
    for method in sorted(all_methods):
        mono_mae = results['monotone'].get(method, {}).get('mae', np.nan)
        inter_mae = results['intermittent'].get(method, {}).get('mae', np.nan)
        diff = inter_mae - mono_mae if not np.isnan(mono_mae) and not np.isnan(inter_mae) else np.nan
        
        print(f"{method:<20} {mono_mae:<15.4f} {inter_mae:<15.4f} {diff:<12.4f}")
    print("="*80)
    
    return results


def plot_longitudinal_metrics(trajectory_results, smoothness_results, pattern_results):
    """
    Visualize longitudinal-specific evaluation results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Trajectory Fidelity
    if trajectory_results:
        methods = list(trajectory_results.keys())
        traj_maes = [trajectory_results[m]['trajectory_mae'] for m in methods]
        traj_corrs = [trajectory_results[m]['trajectory_correlation'] for m in methods]
        
        ax = axes[0]
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, traj_maes, width, label='Trajectory MAE', alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, traj_corrs, width, label='Trajectory Correlation', alpha=0.8, color='orange')
        ax.set_xlabel('Method')
        ax.set_ylabel('Trajectory MAE', color='blue')
        ax2.set_ylabel('Trajectory Correlation', color='orange')
        ax.set_title('Trajectory Fidelity')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    # Plot 2: Temporal Smoothness
    if smoothness_results:
        methods = list(smoothness_results.keys())
        smoothness = [smoothness_results[m]['mean_smoothness'] for m in methods]
        smoothness_std = [smoothness_results[m]['std_smoothness'] for m in methods]
        
        ax = axes[1]
        # Create bar plot with error bars
        bars = ax.bar(methods, smoothness, alpha=0.7, color='green', 
                    yerr=smoothness_std, capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black'})
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Mean Consecutive Difference', fontsize=12)
        ax.set_title('Temporal Smoothness (lower = smoother)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Set y-axis limits to zoom in on the differences
        # Start slightly below the minimum value
        min_val = min(smoothness)
        max_val = max(smoothness)
        max_std = max(smoothness_std)
        
        # Set lower limit to ~90% of minimum, upper to max + max_std with margin
        y_min = min_val * 0.85  # e.g., 0.895 * 0.90 = 0.8055
        y_max = max_val + max_std + 0.05  # Add margin above error bars
        ax.set_ylim(y_min, y_max)
        
        # Add grid for easier comparison
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 3: Missing Pattern Robustness
    if pattern_results:
        methods = []
        mono_vals = []
        inter_vals = []
        
        for method in pattern_results.get('monotone', {}).keys():
            methods.append(method)
            mono_vals.append(pattern_results['monotone'].get(method, {}).get('mae', 0))
            inter_vals.append(pattern_results['intermittent'].get(method, {}).get('mae', 0))
        
        ax = axes[2]
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, mono_vals, width, label='Monotone Dropout', alpha=0.8)
        ax.bar(x + width/2, inter_vals, width, label='Intermittent Missing', alpha=0.8)
        ax.set_xlabel('Method')
        ax.set_ylabel('MAE')
        ax.set_title('Missing Pattern Robustness')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('longitudinal_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()






def plot_execution_times(execution_times):
    """
    Plot execution times for each imputation method
    
    Parameters:
    -----------
    execution_times : dict
        Dictionary with execution times for each method
    """
    methods = list(execution_times.keys())
    times = list(execution_times.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, times, alpha=0.7, color='green')
    
    plt.title('Execution Time by Imputation Method', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 * max(times),
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('imputation_execution_times.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_distribution_similarity(distribution_similarity):
    """
    Plot KS statistics for distribution similarity
    
    Parameters:
    -----------
    distribution_similarity : dict
        Dictionary with KS test results
    """
    # Prepare data for plotting
    methods = list(distribution_similarity.keys())
    columns = list(distribution_similarity[methods[0]].keys()) if methods else []
    
    if not methods or not columns:
        print("No distribution similarity data to plot")
        return
    
    # Create a DataFrame for easier plotting
    data = []
    for method in methods:
        for col in columns:
            if col in distribution_similarity[method]:
                data.append({
                    'Method': method,
                    'Column': col,
                    'KS Statistic': distribution_similarity[method][col]['ks_stat'],
                    'p-value': distribution_similarity[method][col]['ks_pval']
                })
    
    if not data:
        print("No valid distribution similarity data to plot")
        return
    
    df_plot = pd.DataFrame(data)
    
    # Plot KS statistics (lower is better - more similar distributions)
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='Method', y='KS Statistic', hue='Column', data=df_plot)
    
    plt.title('Distribution Similarity by Method (Lower KS = More Similar)', fontsize=14)
    plt.ylabel('Kolmogorov-Smirnov Statistic', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Column', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3)
    
    plt.tight_layout()
    plt.savefig('distribution_similarity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot p-values (higher is better - can't reject null hypothesis of same distribution)
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='Method', y='p-value', hue='Column', data=df_plot)
    
    plt.title('Distribution Similarity p-values by Method (Higher = More Similar)', fontsize=14)
    plt.ylabel('p-value', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Column', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3)
    
    plt.tight_layout()
    plt.savefig('distribution_pvalues.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_imputation_histograms(original_df, imputed_dfs, columns_to_impute):
    """
    Plot histograms of imputed values compared to original observed values
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        Original data with missing values
    imputed_dfs : dict
        Dictionary of imputed DataFrames from different methods
    columns_to_impute : list
        List of column names to impute
    """
    # Filter to only include columns that exist in the original dataframe
    valid_columns = [col for col in columns_to_impute if col in original_df.columns]
    
    if not valid_columns:
        print("No valid columns found for plotting histograms")
        return
        
    print(f"Plotting histograms for columns: {valid_columns}")
    
    # Number of columns and methods
    n_cols = len(valid_columns)
    n_methods = len(imputed_dfs)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4 * n_cols))
    if n_cols == 1:
        axes = [axes]
    
    # Plot for each column
    for i, col in enumerate(valid_columns):
        ax = axes[i]
        
        # Plot original observed distribution
        observed_values = original_df[col].dropna()
        n_observed = len(observed_values)
        
        sns.histplot(observed_values, ax=ax, 
                    label=f'Original (observed, n={n_observed})', 
                    alpha=0.5, color='black', kde=True)
        
        # Plot imputed distributions (only for previously missing values)
        colors = plt.cm.tab10.colors
        for j, (method_name, imputed_df) in enumerate(imputed_dfs.items()):
            # Make sure the column exists in the imputed dataframe
            if col not in imputed_df.columns:
                print(f"Warning: Column {col} not found in {method_name} results")
                continue
                
            # Get just the imputed values (where original was missing)
            missing_mask = original_df[col].isna()
            imputed_values = imputed_df.loc[missing_mask, col].dropna()
            n_imputed = len(imputed_values)
            
            if not imputed_values.empty:
                color = colors[j % len(colors)]
                sns.histplot(imputed_values, ax=ax, 
                            label=f'{method_name} (imputed, n={n_imputed})', 
                            alpha=0.5, color=color, kde=True)
        
        # Add distribution statistics
        if not observed_values.empty:
            obs_mean = observed_values.mean()
            obs_std = observed_values.std()
            ax.axvline(obs_mean, color='black', linestyle='--', alpha=0.7)
            textstr = f'Observed: μ={obs_mean:.2f}, σ={obs_std:.2f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        ax.set_title(f'Distribution for {col}', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('imputation_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    print("Histogram plot saved to 'imputation_histograms.png'")

def plot_correlation_preservation(original_df, imputed_dfs, columns_to_impute):
    """
    Plot heatmaps showing how well each method preserves correlations
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        Original data with missing values
    imputed_dfs : dict
        Dictionary of imputed DataFrames from different methods
    columns_to_impute : list
        List of column names to impute
    """
    # Filter to only include columns that exist in all dataframes
    valid_columns = []
    for col in columns_to_impute:
        if col in original_df.columns:
            all_valid = True
            for method_name, imputed_df in imputed_dfs.items():
                if col not in imputed_df.columns:
                    all_valid = False
                    break
            if all_valid:
                valid_columns.append(col)
    
    if not valid_columns:
        print("No valid columns found for correlation analysis")
        return
        
    print(f"Analyzing correlations for columns: {valid_columns}")
    
    # Calculate original correlations (using only complete cases)
    # This is important for sparse data - we need a baseline for comparison
    complete_cases = original_df[valid_columns].dropna()
    
    if len(complete_cases) < 2:
        print("Not enough complete cases to calculate original correlations")
        # Use pairwise correlations instead
        original_corr = original_df[valid_columns].corr(method='pearson')
    else:
        original_corr = complete_cases.corr()
    
    # Calculate number of methods first
    n_methods = len(imputed_dfs)
    # Set up the figure - 2 rows, 4 columns
    total_plots = n_methods + 1  # +1 for original
    n_cols = 4
    n_rows = math.ceil(total_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))  # 4 columns * 5 width, 2 rows * 4 height
    axes = axes.flatten()  # Flatten to make indexing easier

    if n_methods == 0:
        print("No imputed dataframes to plot correlations")
        return

    # Hide unused subplots
    for idx in range(total_plots, n_rows * n_cols):
        axes[idx].set_visible(False)
    
    # Plot original correlation
    sns.heatmap(original_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('Original Correlations', fontsize=14)
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    
    # Plot correlations for each imputation method
    for i, (method_name, imputed_df) in enumerate(imputed_dfs.items()):
        imputed_corr = imputed_df[valid_columns].corr()
        sns.heatmap(imputed_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[i+1])
        axes[i+1].set_title(f'{method_name} Correlations', fontsize=14)
        axes[i+1].set_xticklabels([])
        axes[i+1].set_yticklabels([])
        axes[i+1].set_xlabel('')
        axes[i+1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('correlation_preservation.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    print("Correlation plot saved to 'correlation_preservation.png'")
    
    # Calculate and plot correlation differences
    print("Calculating correlation differences...")
    correlation_diffs = {}
    
    for method_name, imputed_df in imputed_dfs.items():
        imputed_corr = imputed_df[valid_columns].corr()
        # Calculate absolute differences between original and imputed correlations
        diff_matrix = np.abs(original_corr - imputed_corr)
        correlation_diffs[method_name] = diff_matrix
    
    # Plot correlation differences (lower is better - less difference from original)
    # Calculate grid dimensions: 2 rows, 4 columns
    n_cols = 4
    n_rows = math.ceil(n_methods / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))  # 4 columns * 5 width, 2 rows * 4 height
    axes = axes.flatten()  # Flatten to make indexing easier

    # Hide unused subplots if n_methods < 8
    for idx in range(n_methods, n_rows * n_cols):
        axes[idx].set_visible(False)
    
    for i, (method_name, diff_matrix) in enumerate(correlation_diffs.items()):
        sns.heatmap(diff_matrix, annot=False, cmap='YlOrRd', vmin=0, vmax=1, ax=axes[i])
        axes[i].set_title(f'{method_name} Correlation Differences', fontsize=12)
        
        # Hide x and y tick labels to reduce clutter
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        
        # Calculate and display average difference
        avg_diff = np.mean(diff_matrix.values)
        axes[i].text(0.5, -0.05, f'Avg Diff: {avg_diff:.4f}', 
                    horizontalalignment='center', transform=axes[i].transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('correlation_differences.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    print("Correlation difference plot saved to 'correlation_differences.png'")



def plot_validation_results(validation_results):
    """
    Plot validation results including classification metrics
    
    Parameters:
    -----------
    validation_results : dict
        Dictionary with validation results
    """
    # Prepare data for plotting
    methods = [m for m in validation_results.keys() 
              if 'avg_mae' in validation_results[m] and not np.isnan(validation_results[m]['avg_mae'])]
    
    if not methods:
        print("No valid validation results to plot")
        return
    
    # Extract metrics
    mae_means = [validation_results[m]['avg_mae'] for m in methods]
    mae_stds = [validation_results[m]['std_mae'] for m in methods]
    rmse_means = [validation_results[m]['avg_rmse'] for m in methods]
    rmse_stds = [validation_results[m]['std_rmse'] for m in methods]
    acc_means = [validation_results[m].get('avg_accuracy', 0) for m in methods]
    acc_stds = [validation_results[m].get('std_accuracy', 0) for m in methods]
    auc_means = [validation_results[m].get('avg_auc_multiclass', 0) for m in methods]
    auc_stds = [validation_results[m].get('std_auc_multiclass', 0) for m in methods]
    sens_means = [validation_results[m].get('avg_avg_sensitivity', 0) for m in methods]
    sens_stds = [validation_results[m].get('std_avg_sensitivity', 0) for m in methods]
    spec_means = [validation_results[m].get('avg_avg_specificity', 0) for m in methods]
    spec_stds = [validation_results[m].get('std_avg_specificity', 0) for m in methods]
    ppv_means = [validation_results[m].get('avg_avg_ppv', 0) for m in methods]
    ppv_stds = [validation_results[m].get('std_avg_ppv', 0) for m in methods]
    npv_means = [validation_results[m].get('avg_avg_npv', 0) for m in methods]
    npv_stds = [validation_results[m].get('std_avg_npv', 0) for m in methods]
    time_means = [validation_results[m]['avg_time'] for m in methods]
    time_stds = [validation_results[m]['std_time'] for m in methods]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    metrics_data = [
        (mae_means, mae_stds, 'MAE (Lower = Better)', 'blue'),
        (rmse_means, rmse_stds, 'RMSE (Lower = Better)', 'orange'),
        (acc_means, acc_stds, 'Accuracy (Higher = Better)', 'green'),
        (auc_means, auc_stds, 'AUC Multi-class (Higher = Better)', 'red'),
        (sens_means, sens_stds, 'Average Sensitivity (Higher = Better)', 'purple'),
        (spec_means, spec_stds, 'Average Specificity (Higher = Better)', 'brown'),
        (ppv_means, ppv_stds, 'Average PPV (Higher = Better)', 'pink'),
        (npv_means, npv_stds, 'Average NPV (Higher = Better)', 'gray'),
        (time_means, time_stds, 'Execution Time (Lower = Better)', 'cyan')
    ]
    
    for i, (means, stds, title, color) in enumerate(metrics_data):
        bars = axes[i].bar(methods, means, yerr=stds, capsize=5, alpha=0.7, color=color)
        axes[i].set_title(title, fontsize=12)
        axes[i].set_ylabel(title.split('(')[0].strip(), fontsize=10)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add values on top of bars (accounting for error bars)
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                # Add the standard deviation to position text above the error bar
                text_height = height + stds[j] + 0.005 * max(means)
                axes[i].text(bar.get_x() + bar.get_width()/2., text_height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('validation_results_with_classification.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_dataframe(imputed_dfs, validation_results, distribution_similarity, execution_times):
    """
    Create a summary dataframe of all results including classification metrics
    
    Parameters:
    -----------
    imputed_dfs : dict
        Dictionary of imputed DataFrames
    validation_results : dict
        Dictionary with validation results
    distribution_similarity : dict
        Dictionary with distribution similarity results
    execution_times : dict
        Dictionary with execution times
        
    Returns:
    --------
    summary_df : pandas.DataFrame
        DataFrame with summary of all results
    """
    # Get list of methods
    methods = list(imputed_dfs.keys())
    
    # Initialize summary data
    summary_data = []
    
    for method in methods:
        method_summary = {'Method': method}
        
        # Add validation metrics if available
        if method in validation_results and 'avg_mae' in validation_results[method]:
            method_summary['MAE'] = validation_results[method]['avg_mae']
            method_summary['RMSE'] = validation_results[method]['avg_rmse']
            method_summary['Accuracy'] = validation_results[method].get('avg_accuracy', np.nan)
            method_summary['AUC'] = validation_results[method].get('avg_auc_multiclass', np.nan)
            method_summary['Sensitivity'] = validation_results[method].get('avg_avg_sensitivity', np.nan)
            method_summary['Specificity'] = validation_results[method].get('avg_avg_specificity', np.nan)
            method_summary['PPV'] = validation_results[method].get('avg_avg_ppv', np.nan)
            method_summary['NPV'] = validation_results[method].get('avg_avg_npv', np.nan)
        else:
            method_summary['MAE'] = np.nan
            method_summary['RMSE'] = np.nan
            method_summary['Accuracy'] = np.nan
            method_summary['AUC'] = np.nan
            method_summary['Sensitivity'] = np.nan
            method_summary['Specificity'] = np.nan
            method_summary['PPV'] = np.nan
            method_summary['NPV'] = np.nan
        
        # Add execution time
        if method in execution_times:
            method_summary['Time (s)'] = execution_times[method]
        else:
            method_summary['Time (s)'] = np.nan
        
        # Add average distribution similarity metrics if available
        if method in distribution_similarity:
            ks_stats = []
            ks_pvals = []
            
            for col, results in distribution_similarity[method].items():
                if 'ks_stat' in results:
                    ks_stats.append(results['ks_stat'])
                if 'ks_pval' in results:
                    ks_pvals.append(results['ks_pval'])
            
            if ks_stats:
                method_summary['Avg KS Stat'] = np.mean(ks_stats)
            else:
                method_summary['Avg KS Stat'] = np.nan
                
            if ks_pvals:
                method_summary['Avg KS p-value'] = np.mean(ks_pvals)
            else:
                method_summary['Avg KS p-value'] = np.nan
        else:
            method_summary['Avg KS Stat'] = np.nan
            method_summary['Avg KS p-value'] = np.nan
        
        summary_data.append(method_summary)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Add ranks for each metric (1 is best)
    # Lower is better for these metrics
    for metric in ['MAE', 'RMSE', 'Time (s)', 'Avg KS Stat']:
        if metric in summary_df.columns:
            summary_df[f'{metric} Rank'] = summary_df[metric].rank()
    
    # Higher is better for these metrics
    for metric in ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Avg KS p-value']:
        if metric in summary_df.columns:
            summary_df[f'{metric} Rank'] = summary_df[metric].rank(ascending=False)
    
    # Add average rank
    rank_columns = [col for col in summary_df.columns if 'Rank' in col]
    if rank_columns:
        summary_df['Average Rank'] = summary_df[rank_columns].mean(axis=1)
        summary_df = summary_df.sort_values('Average Rank')
    
    return summary_df



def main_real_data(df, columns_to_impute):
    """
    Main function to run imputation comparison on real data with high missingness
    
        Parameters:
    -----------
    df : pandas.DataFrame
        Your real dataset with missing values
    columns_to_impute : list
        List of column names to impute (default: ge1-ge6)
        
    Returns:
    --------
    imputed_dfs : dict
        Dictionary of imputed DataFrames from each method
    summary_df : pandas.DataFrame
        DataFrame with summary of all results
    """
    print("===== Analyzing Dataset =====")
    print(f"Dataset shape: {df.shape}")

    # Filter out any derived date columns from columns_to_impute
    valid_columns = [col for col in columns_to_impute if col in df.columns and not col.startswith('qol_date_')]
    if len(valid_columns) != len(columns_to_impute):
        excluded_cols = set(columns_to_impute) - set(valid_columns)
        print(f"Warning: Excluding {len(excluded_cols)} columns that are not in the dataframe or are derived date columns:")
        print(f"  {sorted(excluded_cols)}")
        print(f"Using valid columns: {sorted(valid_columns)}")
        columns_to_impute = valid_columns

    # Show basic statistics of the dataset
    print("\nMissing value counts:")
    missing_counts = df[columns_to_impute].isna().sum()
    missing_percents = (missing_counts / len(df)) * 100
    for col, count, percent in zip(columns_to_impute, missing_counts, missing_percents):
        print(f"{col}: {count} missing values ({percent:.2f}%)")

    # Print observed values statistics
    print("\nObserved values per column:")
    for i, col in enumerate(columns_to_impute):
        observed = df[col].dropna()
        n_observed = len(observed)
        if n_observed > 0:
            print(f"{col}: {n_observed} observed values ({100-missing_percents[i]:.2f}%)")
            print(f"  Range: {observed.min()} to {observed.max()}")
            print(f"  Mean: {observed.mean():.4f}, Std: {observed.std():.4f}")
            print(f"  Unique values: {observed.nunique()} ({observed.nunique()/n_observed*100:.1f}% unique)")
        else:
            print(f"{col}: No observed values")

    print("\n===== Running imputation on real data =====")
    imputed_dfs, original_df, execution_times, distribution_similarity = evaluate_real_data_imputation(
        df, columns_to_impute
    )

    # Create a version of original_df without qol_date for plotting
    plotting_df = original_df.copy()
    if 'qol_date' in plotting_df.columns:
        print("\nRemoving qol_date for plotting functions")
        plotting_df = plotting_df.drop(columns=['qol_date'])

    # Also remove any derived date columns if they exist
    date_derived_cols = [col for col in plotting_df.columns if col.startswith('qol_date_')]
    if date_derived_cols:
        print(f"Removing derived date columns: {date_derived_cols}")
        plotting_df = plotting_df.drop(columns=date_derived_cols)

    # Plot execution times
    print("\nPlotting execution times...")
    plot_execution_times(execution_times)

    # Plot distribution similarity
    print("\nPlotting distribution similarity...")
    plot_distribution_similarity(distribution_similarity)

    # Plot histograms of imputed values
    print("\nPlotting imputation histograms...")
    plot_imputation_histograms(plotting_df, imputed_dfs, columns_to_impute)

    # Plot correlation preservation
    print("\nPlotting correlation preservation...")
    plot_correlation_preservation(plotting_df, imputed_dfs, columns_to_impute)

    print("\n===== Running sparse validation =====")
    validation_results = evaluate_with_sparse_validation(df, columns_to_impute, n_folds=5)

    # Plot validation results
    print("\nPlotting validation results...")
    plot_validation_results(validation_results)

    # Create summary dataframe
    print("\nCreating summary of results...")
    summary_df = create_summary_dataframe(
        imputed_dfs, 
        validation_results, 
        distribution_similarity, 
        execution_times
    )

    # Print summary of results
    print("\n===== Summary of Results =====")
    print(summary_df.to_string())

    # Save summary to CSV
    summary_df.to_csv('imputation_summary.csv', index=False)

    # Print recommended method based on average rank
    if 'Average Rank' in summary_df.columns and not summary_df.empty:
        best_method = summary_df.iloc[0]['Method']
        print(f"\nRecommended imputation method: {best_method}")
        
    # === NEW: Longitudinal-specific evaluation ===
    print("\n===== Running Longitudinal-Specific Evaluations =====")
    
    trajectory_results = evaluate_trajectory_fidelity(
        df, imputed_dfs, columns_to_impute, n_patients_sample=400  # ← fast
    )
    
    smoothness_results = evaluate_temporal_smoothness(
        df, imputed_dfs, columns_to_impute
    )
    
    pattern_results = evaluate_missing_pattern_robustness(
        df, columns_to_impute 
    )
    
    # Plot longitudinal metrics
    plot_longitudinal_metrics(trajectory_results, smoothness_results, pattern_results)

    return imputed_dfs#, summary_df

# Example usage
if __name__ == "__main__":
    # Load your real dataset with missing values here
    # df = pd.read_csv('your_dataset.csv')
    
    # Define columns to impute
    
    ordinal_cols = (
    [f"gp{i}" for i in range(1, 8)] + [f"gs{i}" for i in range(1, 8)] +
    [f"ge{i}" for i in range(1, 7)] + [f"gf{i}" for i in range(1, 8)] +
    [f"a_hn{i}" for i in range(1, 6)] + ["a_hn7", "a_hn10"] +
    [f"a_e{i}" for i in range(1, 8)] + ["a_c6", "a_c2", "a_act11"]
)
    columns_to_impute = ordinal_cols
    
    # Run the analysis
    # imputed_dfs, summary_df = main_real_data(df, columns_to_impute)
    imputed_dfs = main_real_data(df, columns_to_impute)
    
    # Example of how to save the best imputed dataset
    # if not summary_df.empty:
    #     worst_method = summary_df.iloc[0]['Method']
    #     print(f"\nSaving imputed dataset from {worst_method}...") 
    
    
    
    
VISIT_ORDER = {
    'baseline_arm_1':       0,
    'preoperative_arm_1':   1,
    '1_month_postop_arm_1': 2,
    '3_months_postop_arm_1':3,
    '6_months_postop_arm_1':4,
    '1_year_postop_arm_1':  5,
    '2_years_postop_arm_1': 6,
    '3_years_postop_arm_1': 7,
    '4_years_postop_arm_1': 8,
    '5_years_postop_arm_1': 9,
}

def observe_missing_patterns(df, patient_col='id', event_col='redcap_event_name'):
    monotone, intermittent = [], []

    for pid, pdata in df.groupby(patient_col):
        if len(pdata) < 2:
            continue

        pdata = pdata.copy()
        pdata['_vi'] = pdata[event_col].map(VISIT_ORDER).fillna(99)
        visit_indices = sorted(pdata['_vi'].values)

        gaps = np.diff(visit_indices)
        has_gaps = np.any(gaps > 1)
        min_visit = min(visit_indices)
        max_visit = max(visit_indices)

        if has_gaps:
            intermittent.append(pid)
        else:
            if min_visit == 0 and max_visit < 9:
                monotone.append(pid)
            else:
                intermittent.append(pid)

    total = len(monotone) + len(intermittent)
    print(f"Total patients classified: {total}")
    print(f"  Monotone dropout: {len(monotone)} ({100*len(monotone)/total:.1f}%)")
    print(f"  Intermittent:     {len(intermittent)} ({100*len(intermittent)/total:.1f}%)")

    print("\nPatients present per timepoint:")
    for visit_name, vi in sorted(VISIT_ORDER.items(), key=lambda x: x[1]):
        n = df[df[event_col] == visit_name][patient_col].nunique()
        print(f"  {visit_name:<30}: {n} patients")

    return {'monotone': monotone, 'intermittent': intermittent}

pattern_groups = observe_missing_patterns(df)
print(pattern_groups)