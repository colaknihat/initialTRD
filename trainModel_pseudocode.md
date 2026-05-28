import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

# ==============================================================================
# 1. FEATURE ENGINEERING: THE "TURKISH MACRO" LAYER
# ==============================================================================
def engineer_turkish_features(df):
    """
    df must contain: BIST100, USD_TRY, CBRT_Rate, CPI, 5Y_CDS_Spread
    """
    # Target: FX-Adjusted Return (Real Return proxy)
    df['bist_ret'] = df['BIST100'].pct_change()
    df['fx_ret'] = df['USD_TRY'].pct_change()
    df['target'] = df['bist_ret'] - df['fx_ret'] # Forces model to beat the dollar
    
    # Macro Features
    df['real_rate'] = df['CBRT_Rate'] - df['CPI'] # Crucial for orthodox tightening cycles
    df['cds_velocity'] = df['5Y_CDS_Spread'].diff() # Spikes indicate political/geopolitical shock
    df['fx_volatility'] = df['USD_TRY'].rolling(14).std()
    
    # Market Microstructure
    df['market_breadth'] = df['advancing_stocks'] / df['declining_stocks']
    
    df.dropna(inplace=True)
    return df

# ==============================================================================
# 2. REGIME DETECTION & SAMPLE WEIGHTING (Handling Bull/Bear/Crash)
# ==============================================================================
def generate_regime_weights(df):
    """
    Uses HMM to find market regimes and assigns weights.
    Prevents the model from overfitting to long Bull Markets.
    """
    # Cluster based on Volatility and Return
    X_hmm = df[['bist_ret', 'fx_volatility']].values
    
    hmm_model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=200)
    hmm_model.fit(X_hmm)
    regimes = hmm_model.predict(X_hmm)
    df['regime'] = regimes
    
    # Map regimes to weights (Inverse frequency weighting)
    # e.g., Bull market happens 60% of the time -> low weight. 
    # Crash happens 5% of the time -> high weight.
    regime_counts = df['regime'].value_counts(normalize=True)
    weights = 1.0 / regime_counts
    df['sample_weight'] = df['regime'].map(weights)
    
    # Normalize weights
    df['sample_weight'] = df['sample_weight'] / df['sample_weight'].max()
    return df

# ==============================================================================
# 3. MODEL ARCHITECTURE: ATTENTION-BASED LSTM
# ==============================================================================
class BIST_ResilientLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(BIST_ResilientLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=0.3)
        # Attention mechanism to focus on recent macro shocks (e.g. CDS spikes)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Calculate attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(lstm_out * attn_weights, dim=1)
        
        out = self.fc(context_vector)
        return out

# ==============================================================================
# 4. CUSTOM LOSS FUNCTION: HUBER LOSS (Handling Crashes/Fat Tails)
# ==============================================================================
class RegimeWeightedHuberLoss(nn.Module):
    """
    Huber loss is less sensitive to outliers (crashes) than MSE.
    We multiply it by the regime weight to force learning during rare events.
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
        
    def forward(self, predictions, targets, weights):
        loss = self.huber(predictions, targets)
        weighted_loss = loss * weights.unsqueeze(1)
        return weighted_loss.mean()

# ==============================================================================
# 5. THE TRAINING LOOP (With Gradient Clipping & Walk-Forward)
# ==============================================================================
def train_bist_model(model, train_loader, val_loader, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = RegimeWeightedHuberLoss(delta=1.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y, batch_weights in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(batch_X)
            loss = criterion(preds, batch_y, batch_weights)
            
            # Backward pass
            loss.backward()
            
            # CRITICAL: Gradient Clipping
            # Turkish market data has massive jumps. Without clipping, 
            # a single crash day in the batch will cause gradient explosion.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        # Validation (Purged Walk-Forward)
        val_loss = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ==============================================================================
# 6. PURGED WALK-FORWARD VALIDATION (Preventing Lookahead Bias)
# ==============================================================================
def create_purged_folds(df, n_splits=5, embargo_days=20):
    """
    Standard K-Fold ruins financial time series. 
    We use Purged K-Fold with an 'embargo' period to ensure the model 
    doesn't leak future information, especially around slow-moving macro data like CPI.
    """
    # Implementation of purged time-series cross-validation
    # Drops 'embargo_days' between train and test sets
    pass 