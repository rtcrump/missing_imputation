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
    os.environ['OMP_NUM_THREADS'] = '2' #10->2
    
    # Initialize the imputation kernel
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
    for _ in tqdm(range(5), desc="MICE Imputation"): #5->3
        kernel.mice(
            iterations=1,
            verbose=False,
            num_boost_round=80,
            max_depth=10,
            num_threads=2 #10 -> 2
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
    'surgery_arm_1':        2,
    '1_month_postop_arm_1': 3,
    '3_months_postop_arm_1':4,
    '6_months_postop_arm_1':5,
    '1_year_postop_arm_1':  6,
    '2_years_postop_arm_1': 7,
    '3_years_postop_arm_1': 8,
    '4_years_postop_arm_1': 9,
    '5_years_postop_arm_1': 10,
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
    'baseline_arm_1':        0,
    'preoperative_arm_1':    1,
    'surgery_arm_1':         2,
    '1_month_postop_arm_1':  3,
    '3_months_postop_arm_1': 4,
    '6_months_postop_arm_1': 5,
    '1_year_postop_arm_1':   6,
    '2_years_postop_arm_1':  7,
    '3_years_postop_arm_1':  8,
    '4_years_postop_arm_1':  9,
    '5_years_postop_arm_1':  10,
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











# ============================================================================
# SIMULATION STUDY 1: BOOTSTRAP STABILITY ASSESSMENT
# ============================================================================

def bootstrap_stability_assessment(df, columns_to_impute, n_bootstrap=1000, n_jobs=10, 
                                   save_checkpoint_every=20):
    """
    Assess stability of imputation methods using bootstrap resampling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataset with missingness
    columns_to_impute : list
        List of column names to impute
    n_bootstrap : int
        Number of bootstrap iterations (default: 1000)
    n_jobs : int
        Number of parallel jobs (default: 10 for your cluster)
    save_checkpoint_every : int
        Save progress every N iterations
        
    Returns:
    --------
    bootstrap_results : pandas.DataFrame
        Results from all bootstrap iterations
    bootstrap_summary : pandas.DataFrame
        Summary statistics with 95% CI
    """
    from joblib import Parallel, delayed
    import pickle
    
    methods = {
        'MICE': apply_mice_imputation,
        'Bayesian PCA': apply_bpca_imputation,
        'SoftImpute': apply_softimpute_imputation,
        'LSTM': apply_lstm_imputation,                               
        'Longitudinal MICE': apply_mice_longitudinal_imputation,
        'Gemma':    lambda df, columns_to_impute, **kw: apply_gemma_imputation(df, columns_to_impute, model_name='/home/yjkweon2/models/gemma-2-2b-it', **kw),
        #'MedGemma': lambda df, columns_to_impute, **kw: apply_gemma_imputation(df, columns_to_impute, model_name='/home/yjkweon2/models/medgemma-1.5-4b-it', **kw),
        'TimesFM': apply_timesfm_imputation
    }
    
    print(f"\n{'='*80}")
    print(f"SIMULATION 1: BOOTSTRAP STABILITY ASSESSMENT")
    print(f"{'='*80}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Methods: {list(methods.keys())}")
    print(f"Dataset: {len(df)} rows, {len(columns_to_impute)} target variables")
    
    def run_single_bootstrap(iteration, df, columns_to_impute, methods):
        """Run one bootstrap iteration for all methods"""
        np.random.seed(iteration)  # For reproducibility
        
        # Bootstrap resample with replacement
        # OLD (breaks trajectories):
        # boot_sample = df.sample(n=len(df), replace=True, random_state=iteration)
        # boot_sample = boot_sample.reset_index(drop=True)

        # NEW (resample patients, keep all their visits together):
        unique_patients = df['id'].unique()
        sampled_patients = np.random.choice(unique_patients, size=len(unique_patients), replace=True)
        boot_sample = pd.concat(
            [df[df['id'] == pid] for pid in sampled_patients],
            ignore_index=True
        )
        
        iter_results = []
        
        for method_name, method_func in methods.items():
            try:
                start_time = time.time()
                
                # Create validation data by masking 20% of observed values
                validation_df = boot_sample.copy()
                validation_masks = {}
                original_values = {}
                
                for col in columns_to_impute:
                    observed_indices = boot_sample[boot_sample[col].notna()].index.tolist()
                    n_observed = len(observed_indices)
                    
                    if n_observed > 0:
                        # Randomly select 20% of observed values for validation
                        n_to_mask = max(1, int(n_observed * 0.2))
                        mask_indices = np.random.choice(observed_indices, size=n_to_mask, replace=False)
                        
                        # Save original values
                        original_values[col] = boot_sample.loc[mask_indices, col].copy()
                        validation_masks[col] = mask_indices
                        
                        # Mask these values
                        validation_df.loc[mask_indices, col] = np.nan
                
                # Create method-specific copy
                method_df = validation_df.copy()
                
                # Drop qol_date for non-LSTM and non-longitudinal MICE methods
                if method_name not in ('LSTM', 'Longitudinal MICE', 'TimesFM') and 'qol_date' in method_df.columns:
                    method_df = method_df.drop(columns=['qol_date'])
                
                # Apply imputation once
                imputed_df, _ = method_func(method_df, columns_to_impute)
                
                execution_time = time.time() - start_time
                
                # Calculate metrics
                mae_list = []
                rmse_list = []
                
                for col in columns_to_impute:
                    if col not in imputed_df.columns or col not in original_values:
                        continue
                    
                    mask_indices = validation_masks[col]
                    if len(mask_indices) == 0:
                        continue
                    
                    real_vals = original_values[col]
                    imputed_vals = imputed_df.loc[mask_indices, col]
                    
                    mae = mean_absolute_error(real_vals, imputed_vals)
                    rmse = np.sqrt(mean_squared_error(real_vals, imputed_vals))
                    mae_list.append(mae)
                    rmse_list.append(rmse)
                
                avg_mae = np.mean(mae_list) if mae_list else np.nan
                avg_rmse = np.mean(rmse_list) if rmse_list else np.nan
                
                # Distribution similarity (KS test) - sample 5 variables
                # sample_cols = np.random.choice(columns_to_impute, 
                #                             size=min(5, len(columns_to_impute)), 
                #                             replace=False)
                ks_stats = []
                
                for col in columns_to_impute: #sample_cols
                    if col not in imputed_df.columns:
                        continue
                    
                    observed_vals = boot_sample.loc[boot_sample[col].notna(), col]
                    missing_mask = boot_sample[col].isna()
                    
                    if missing_mask.sum() > 0 and len(observed_vals) > 0:
                        imputed_vals = imputed_df.loc[missing_mask, col].dropna()
                        if len(imputed_vals) > 0:
                            ks_stat, _ = ks_2samp(observed_vals, imputed_vals)
                            ks_stats.append(ks_stat)
                
                avg_ks = np.mean(ks_stats) if ks_stats else np.nan
                
                iter_results.append({
                    'bootstrap_iter': iteration,
                    'method': method_name,
                    'mae': avg_mae,
                    'rmse': avg_rmse,
                    'ks_stat': avg_ks,
                    'execution_time': execution_time
                })
                
            except Exception as e:
                print(f"Error in bootstrap {iteration}, method {method_name}: {e}")
                iter_results.append({
                    'bootstrap_iter': iteration,
                    'method': method_name,
                    'mae': np.nan,
                    'rmse': np.nan,
                    'ks_stat': np.nan,
                    'execution_time': np.nan
                })
        
        return iter_results
    
    # Run bootstrap iterations in parallel
    print(f"\nStarting {n_bootstrap} bootstrap iterations with {n_jobs} parallel jobs...")
    print("Progress will be saved periodically.")
    
    all_results = []
    
    # Process in chunks for checkpointing
    for chunk_start in range(0, n_bootstrap, save_checkpoint_every):
        chunk_end = min(chunk_start + save_checkpoint_every, n_bootstrap)
        chunk_range = range(chunk_start, chunk_end)
        
        print(f"\nProcessing bootstrap iterations {chunk_start+1}-{chunk_end}...")
        
        chunk_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_single_bootstrap)(i, df, columns_to_impute, methods) 
            for i in chunk_range
        )
        
        # Flatten results
        for iter_result in chunk_results:
            all_results.extend(iter_result)
        
        # Save checkpoint
        checkpoint_df = pd.DataFrame(all_results)
        checkpoint_df.to_csv(f'bootstrap_checkpoint_{chunk_end}.csv', index=False)
        print(f"Checkpoint saved: bootstrap_checkpoint_{chunk_end}.csv")
    
    # Convert to DataFrame
    bootstrap_results = pd.DataFrame(all_results)
    
    # Save final results
    bootstrap_results.to_csv('bootstrap_stability_results.csv', index=False)
    print(f"\nFinal results saved: bootstrap_stability_results.csv")
    
    # Calculate summary statistics
    print("\nCalculating summary statistics...")
    
    summary_stats = []
    
    for method in bootstrap_results['method'].unique():
        method_data = bootstrap_results[bootstrap_results['method'] == method]
        
        for metric in ['mae', 'rmse', 'ks_stat', 'execution_time']:
            values = method_data[metric].dropna()
            
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                cv = std_val / mean_val if mean_val != 0 else np.nan
                
                summary_stats.append({
                    'method': method,
                    'metric': metric,
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'cv': cv
                })
    
    bootstrap_summary = pd.DataFrame(summary_stats)
    bootstrap_summary.to_csv('bootstrap_summary.csv', index=False)
    print(f"Summary statistics saved: bootstrap_summary.csv")
    
    # Print summary
    print(f"\n{'='*80}")
    print("BOOTSTRAP STABILITY SUMMARY")
    print(f"{'='*80}\n")
    
    for metric in ['mae', 'rmse', 'ks_stat', 'execution_time']:
        print(f"\n{metric.upper()}:")
        print("-" * 80)
        metric_summary = bootstrap_summary[bootstrap_summary['metric'] == metric].sort_values('mean')
        for _, row in metric_summary.iterrows():
            print(f"{row['method']:20s} Mean: {row['mean']:8.4f}  "
                  f"95% CI: [{row['ci_lower']:7.4f}, {row['ci_upper']:7.4f}]  "
                  f"CV: {row['cv']:6.4f}")
    
    return bootstrap_results, bootstrap_summary


# ============================================================================
# SIMULATION STUDY 2: COMPLETE-CASE TRUTH VALIDATION
# ============================================================================

def complete_case_simulation(df, columns_to_impute, n_simulations=1000, n_jobs=10,
                            save_checkpoint_every=20):
    """
    Evaluate imputation accuracy against known truth using complete cases
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataset
    columns_to_impute : list
        List of column names to impute
    n_simulations : int
        Number of simulation iterations (default: 1000)
    n_jobs : int
        Number of parallel jobs
    save_checkpoint_every : int
        Save progress every N iterations
        
    Returns:
    --------
    simulation_results : pandas.DataFrame
        Results from all simulation iterations
    simulation_summary : pandas.DataFrame
        Summary statistics including bias and RMSE
    """
    from joblib import Parallel, delayed
    
    methods = {
        'MICE': apply_mice_imputation,
        'Bayesian PCA': apply_bpca_imputation,
        'SoftImpute': apply_softimpute_imputation,
        'LSTM': apply_lstm_imputation,                               
        'Longitudinal MICE': apply_mice_longitudinal_imputation,
        'Gemma':    lambda df, columns_to_impute, **kw: apply_gemma_imputation(df, columns_to_impute, model_name='/home/yjkweon2/models/gemma-2-2b-it', **kw),
        #'MedGemma': lambda df, columns_to_impute, **kw: apply_gemma_imputation(df, columns_to_impute, model_name='/home/yjkweon2/models/medgemma-1.5-4b-it', **kw),
        'TimesFM': apply_timesfm_imputation
    }
    
    print(f"\n{'='*80}")
    print(f"SIMULATION 2: COMPLETE-CASE TRUTH VALIDATION")
    print(f"{'='*80}")
    
    # Extract complete cases
    # OLD (requires all 44 cols observed in same row — nearly impossible longitudinally):
    # complete_cases = df[columns_to_impute].dropna()
    # n_complete = len(complete_cases)
    # print(f"Complete cases found: {n_complete} out of {len(df)} ({n_complete/len(df)*100:.1f}%)")
    # if n_complete < 150:
    #     print(f"WARNING: Only {n_complete} complete cases. Results may be less stable.")
    #     print("Recommend having at least 150-200 complete cases for robust simulation.")
        
    # NEW (patients who have ≥1 visit where all 44 items are observed):
    fully_observed_visits = df[df[columns_to_impute].notna().all(axis=1)]
    complete_patient_ids = fully_observed_visits['id'].unique()
    complete_cases = df[df['id'].isin(complete_patient_ids)].copy()
    n_complete = len(complete_patient_ids)
    print(f"Patients with ≥1 fully observed visit: {n_complete} "
        f"({n_complete/df['id'].nunique()*100:.1f}% of patients)")
    if n_complete < 50:
        print(f"WARNING: Only {n_complete} eligible patients. Results may be less stable.")
    
    # Calculate original missingness rates for each variable
    missingness_rates = {}
    for col in columns_to_impute:
        miss_rate = df[col].isna().mean()
        missingness_rates[col] = miss_rate
        print(f"{col}: {miss_rate*100:.2f}% missingness in original data")
    
    print(f"\nSimulation iterations: {n_simulations}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Methods: {list(methods.keys())}")
    
    def run_single_simulation(iteration, complete_cases, columns_to_impute, 
                            missingness_rates, methods):
        """Run one simulation iteration for all methods"""
        np.random.seed(iteration)
        
        # Create a copy of complete cases
        truth_data = complete_cases.copy()
        
        # Induce MCAR missingness matching original rates
        sim_data = truth_data.copy()
        
        for col in columns_to_impute:
            miss_rate = missingness_rates[col]
            n_to_mask = int(len(sim_data) * miss_rate)
            
            if n_to_mask > 0:
                mask_indices = np.random.choice(sim_data.index, size=n_to_mask, replace=False)
                sim_data.loc[mask_indices, col] = np.nan
        
        iter_results = []
        
        for method_name, method_func in methods.items():
            try:
                start_time = time.time()
                
                # Create method-specific copy
                method_df = sim_data.copy()
                
                # Add back non-target columns if they exist
                # for col in df.columns:
                #     if col not in columns_to_impute and col in df.columns:
                #         # Use values from original df at the same indices
                #         method_df[col] = df.loc[sim_data.index, col]
                
                # Drop qol_date for non-LSTM and non-longitudinal MICE methods
                if method_name not in ('LSTM', 'Longitudinal MICE', 'TimesFM') and 'qol_date' in method_df.columns:
                    method_df = method_df.drop(columns=['qol_date'])
                
                # Apply imputation
                imputed_df, _ = method_func(method_df, columns_to_impute)
                
                execution_time = time.time() - start_time
                
                # Calculate bias and RMSE against known truth
                biases = []
                rmses = []
                var_results = []
                
                for col in columns_to_impute:
                    if col not in imputed_df.columns:
                        continue
                    
                    # Get missing mask
                    missing_mask = sim_data[col].isna()
                    
                    if missing_mask.sum() == 0:
                        continue
                    
                    # True values (from complete cases)
                    true_values = truth_data.loc[missing_mask, col]
                    # Imputed values
                    imputed_values = imputed_df.loc[missing_mask, col]
                    
                    # Calculate bias and RMSE
                    bias = (imputed_values - true_values).mean()
                    rmse = np.sqrt(((imputed_values - true_values) ** 2).mean())
                    mae = np.abs(imputed_values - true_values).mean()
                    
                    biases.append(bias)
                    rmses.append(rmse)
                    
                    var_results.append({
                        'simulation_iter': iteration,
                        'method': method_name,
                        'variable': col,
                        'bias': bias,
                        'rmse': rmse,
                        'mae': mae,
                        'n_missing': missing_mask.sum()
                    })
                
                # Overall summary for this iteration
                iter_results.append({
                    'simulation_iter': iteration,
                    'method': method_name,
                    'overall_bias': np.mean(biases) if biases else np.nan,
                    'overall_rmse': np.mean(rmses) if rmses else np.nan,
                    'execution_time': execution_time,
                    'n_variables_imputed': len(biases)
                })
                
                # Save variable-level results
                if iteration == 0:  # Create file on first iteration
                    var_df = pd.DataFrame(var_results)
                    var_df.to_csv('simulation_variable_results.csv', mode='w', index=False)
                else:
                    var_df = pd.DataFrame(var_results)
                    var_df.to_csv('simulation_variable_results.csv', mode='a', header=False, index=False)
                
            except Exception as e:
                print(f"Error in simulation {iteration}, method {method_name}: {e}")
                iter_results.append({
                    'simulation_iter': iteration,
                    'method': method_name,
                    'overall_bias': np.nan,
                    'overall_rmse': np.nan,
                    'execution_time': np.nan,
                    'n_variables_imputed': 0
                })
        
        return iter_results
    
    # Run simulations in parallel
    print(f"\nStarting {n_simulations} simulation iterations with {n_jobs} parallel jobs...")
    #print("This will take approximately 12-15 hours. Progress will be saved periodically.")
    
    all_results = []
    
    # Process in chunks for checkpointing
    for chunk_start in range(0, n_simulations, save_checkpoint_every):
        chunk_end = min(chunk_start + save_checkpoint_every, n_simulations)
        chunk_range = range(chunk_start, chunk_end)
        
        print(f"\nProcessing simulation iterations {chunk_start+1}-{chunk_end}...")
        
        chunk_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_single_simulation)(i, complete_cases, columns_to_impute, 
                                            missingness_rates, methods) 
            for i in chunk_range
        )
        
        # Flatten results
        for iter_result in chunk_results:
            all_results.extend(iter_result)
        
        # Save checkpoint
        checkpoint_df = pd.DataFrame(all_results)
        checkpoint_df.to_csv(f'simulation_checkpoint_{chunk_end}.csv', index=False)
        print(f"Checkpoint saved: simulation_checkpoint_{chunk_end}.csv")
    
    # Convert to DataFrame
    simulation_results = pd.DataFrame(all_results)
    
    # Save final results
    simulation_results.to_csv('simulation_complete_results.csv', index=False)
    print(f"\nFinal results saved: simulation_complete_results.csv")
    
    # Calculate summary statistics
    print("\nCalculating summary statistics...")
    
    summary_stats = []
    
    for method in simulation_results['method'].unique():
        method_data = simulation_results[simulation_results['method'] == method]
        
        for metric in ['overall_bias', 'overall_rmse', 'execution_time']:
            values = method_data[metric].dropna()
            
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                
                summary_stats.append({
                    'method': method,
                    'metric': metric,
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
    
    simulation_summary = pd.DataFrame(summary_stats)
    simulation_summary.to_csv('simulation_summary.csv', index=False)
    print(f"Summary statistics saved: simulation_summary.csv")
    
    # Print summary
    print(f"\n{'='*80}")
    print("COMPLETE-CASE SIMULATION SUMMARY")
    print(f"{'='*80}\n")
    
    for metric in ['overall_bias', 'overall_rmse', 'execution_time']:
        print(f"\n{metric.upper()}:")
        print("-" * 80)
        metric_summary = simulation_summary[simulation_summary['metric'] == metric].sort_values('mean')
        for _, row in metric_summary.iterrows():
            print(f"{row['method']:20s} Mean: {row['mean']:8.4f}  "
                  f"95% CI: [{row['ci_lower']:7.4f}, {row['ci_upper']:7.4f}]")
    
    # Also load and summarize variable-level results
    print(f"\n{'='*80}")
    print("VARIABLE-LEVEL BIAS SUMMARY (averaged across simulations)")
    print(f"{'='*80}\n")
    
    var_results = pd.read_csv('simulation_variable_results.csv')
    var_summary = var_results.groupby(['method', 'variable']).agg({
        'bias': 'mean',
        'rmse': 'mean',
        'mae': 'mean'
    }).reset_index()
    
    var_summary.to_csv('simulation_variable_summary.csv', index=False)
    
    # Print bias for each method
    for method in var_summary['method'].unique():
        method_vars = var_summary[var_summary['method'] == method].sort_values('bias', key=abs)
        print(f"\n{method}:")
        print("Variable    Bias      RMSE      MAE")
        print("-" * 50)
        for _, row in method_vars.head(10).iterrows():  # Show top 10
            print(f"{row['variable']:10s}  {row['bias']:7.4f}   {row['rmse']:7.4f}   {row['mae']:7.4f}")
    
    return simulation_results, simulation_summary


# ============================================================================
# VISUALIZATION FUNCTIONS FOR SIMULATIONS
# ============================================================================

def plot_bootstrap_results(bootstrap_results):
    """Create visualization plots for bootstrap stability results"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: MAE violin plot
    ax1 = axes[0, 0]
    sns.violinplot(data=bootstrap_results, x='method', y='mae', ax=ax1)
    ax1.set_title('Bootstrap Distribution of MAE', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: RMSE violin plot
    ax2 = axes[0, 1]
    sns.violinplot(data=bootstrap_results, x='method', y='rmse', ax=ax2)
    ax2.set_title('Bootstrap Distribution of RMSE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: KS statistic violin plot
    ax3 = axes[1, 0]
    sns.violinplot(data=bootstrap_results, x='method', y='ks_stat', ax=ax3)
    ax3.set_title('Bootstrap Distribution of KS Statistic', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Method', fontsize=12)
    ax3.set_ylabel('KS Statistic', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Execution time violin plot
    ax4 = axes[1, 1]
    sns.violinplot(data=bootstrap_results, x='method', y='execution_time', ax=ax4)
    ax4.set_title('Bootstrap Distribution of Execution Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Method', fontsize=12)
    ax4.set_ylabel('Time (seconds)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('bootstrap_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Bootstrap visualization saved: bootstrap_distributions.png")


def plot_simulation_results(simulation_results, var_summary):
    """Create visualization plots for complete-case simulation results"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Overall bias violin plot
    ax1 = axes[0, 0]
    sns.violinplot(data=simulation_results, x='method', y='overall_bias', ax=ax1)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_title('Distribution of Overall Bias (1000 Simulations)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Bias', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Overall RMSE violin plot
    ax2 = axes[0, 1]
    sns.violinplot(data=simulation_results, x='method', y='overall_rmse', ax=ax2)
    ax2.set_title('Distribution of Overall RMSE (1000 Simulations)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Variable-level bias heatmap (top 20 variables)
    ax3 = axes[1, 0]
    
    # Pivot for heatmap
    bias_pivot = var_summary.pivot(index='variable', columns='method', values='bias')
    # Select top 20 variables by absolute mean bias
    top_vars = bias_pivot.abs().mean(axis=1).nlargest(20).index
    bias_pivot_subset = bias_pivot.loc[top_vars]
    
    sns.heatmap(bias_pivot_subset, annot=False, cmap='RdBu_r', center=0, 
                ax=ax3, cbar_kws={'label': 'Bias'})
    ax3.set_title('Variable-Level Bias (Top 20 Variables)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Method', fontsize=12)
    ax3.set_ylabel('Variable', fontsize=12)
    
    # Plot 4: Method ranking by bias (bar plot)
    ax4 = axes[1, 1]
    
    # Calculate mean absolute bias for each method
    method_bias = var_summary.groupby('method')['bias'].apply(lambda x: np.abs(x).mean()).sort_values()
    
    bars = ax4.barh(method_bias.index, method_bias.values, color='steelblue')
    ax4.set_title('Mean Absolute Bias by Method', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Mean Absolute Bias', fontsize=12)
    ax4.set_ylabel('Method', fontsize=12)
    
    # Add values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('simulation_bias_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Simulation visualization saved: simulation_bias_results.png")


# ============================================================================
# MAIN EXECUTION BLOCK FOR SIMULATIONS
# ============================================================================

if __name__ == "__main__":
    
    # Define target variables
    ordinal_cols = (
        [f"gp{i}" for i in range(1, 8)] + [f"gs{i}" for i in range(1, 8)] +
        [f"ge{i}" for i in range(1, 7)] + [f"gf{i}" for i in range(1, 8)] +
        [f"a_hn{i}" for i in range(1, 6)] + ["a_hn7", "a_hn10"] +
        [f"a_e{i}" for i in range(1, 8)] + ["a_c6", "a_c2", "a_act11"]
    )
    columns_to_impute = ordinal_cols
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SIMULATION STUDY FOR MISSING DATA IMPUTATION")
    print(f"{'='*80}")
    print(f"\nDataset: {len(df)} observations, {len(columns_to_impute)} target variables")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # SIMULATION 1: BOOTSTRAP STABILITY ASSESSMENT
    # ========================================================================
    
    print("\n" + "="*80)
    print("STARTING SIMULATION 1: BOOTSTRAP STABILITY ASSESSMENT")
    print("="*80)
    
    bootstrap_results, bootstrap_summary = bootstrap_stability_assessment(
        df=df,
        columns_to_impute=columns_to_impute,
        n_bootstrap=100, #200 -> 10 -> 1000
        n_jobs=7, #10 -> 5
        save_checkpoint_every=10 #500->10
    )
    
    # Create visualizations
    plot_bootstrap_results(bootstrap_results)
    
    print("\nSimulation 1 completed successfully!")
    print("Results saved:")
    print("  - bootstrap_stability_results.csv")
    print("  - bootstrap_summary.csv")
    print("  - bootstrap_distributions.png")
    
    # ========================================================================
    # SIMULATION 2: COMPLETE-CASE TRUTH VALIDATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STARTING SIMULATION 2: COMPLETE-CASE TRUTH VALIDATION")
    print("="*80)
    
    simulation_results, simulation_summary = complete_case_simulation(
        df=df,
        columns_to_impute=columns_to_impute,
        n_simulations=100, #200->10->1000
        n_jobs=7, #10 -> 5
        save_checkpoint_every=10
    )
    
    # Load variable-level results for visualization
    var_summary = pd.read_csv('simulation_variable_summary.csv')
    
    # Create visualizations
    plot_simulation_results(simulation_results, var_summary)
    
    print("\nSimulation 2 completed successfully!")
    print("Results saved:")
    print("  - simulation_complete_results.csv")
    print("  - simulation_summary.csv")
    print("  - simulation_variable_results.csv")
    print("  - simulation_variable_summary.csv")
    print("  - simulation_bias_results.png")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("ALL SIMULATIONS COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print("  Bootstrap Stability:")
    print("    - bootstrap_stability_results.csv")
    print("    - bootstrap_summary.csv")
    print("    - bootstrap_distributions.png")
    print("\n  Complete-Case Simulation:")
    print("    - simulation_complete_results.csv")
    print("    - simulation_summary.csv")
    print("    - simulation_variable_results.csv")
    print("    - simulation_variable_summary.csv")
    print("    - simulation_bias_results.png")
    print("\nUse these results to update your manuscript with comprehensive simulation evidence.")
    print(f"\nTotal execution time: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")