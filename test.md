import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# ==============================================================================
# 1. THE PURGED TIME SERIES SPLIT CLASS
# ==============================================================================
class PurgedWalkForward:
    def __init__(self, n_splits=5, embargo_days=10):
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(self, X, y):
        """
        Generates train/test indices with an embargo (purge) period 
        to prevent macro data leakage.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate the size of each test fold
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Define Training End and Test Start
            test_start = i * test_size + test_size
            test_end = test_start + test_size
            
            # Training set goes from 0 up to (test_start - embargo)
            train_end = test_start - self.embargo_days
            
            if train_end <= 0:
                continue
                
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices

# ==============================================================================
# 2. THE WALK-FORWARD TESTING LOOP
# ==============================================================================
def run_walk_forward_test(df, model_class, features, target):
    X = df[features].values
    y = df[target].values # FX-Adjusted Returns
    
    # We want a rolling window, so we use an expanding or fixed-size rolling approach.
    # Here we simulate testing the model year-over-year.
    
    results = []
    wf = PurgedWalkForward(n_splits=10, embargo_days=15) # 15 day embargo for CPI/Rate leaks
    
    print("Starting Purged Walk-Forward Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(wf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # A. Instantiate a fresh model for this fold
        # This prevents the model from "remembering" 2021 when trading in 2024
        model = model_class() 
        
        # B. Train the model (with early stopping)
        model.fit(X_train, y_train, 
                  validation_split=0.1, 
                  epochs=50, 
                  callbacks=[EarlyStopping(patience=5)])
        
        # C. Predict on the out-of-sample test set
        predictions = model.predict(X_test)
        
        # D. Evaluate REAL performance
        # Calculate Annualized Sharpe Ratio on the test set
        test_returns = calculate_strategy_returns(predictions, y_test)
        sharpe = calculate_sharpe(test_returns)
        max_dd = calculate_max_drawdown(test_returns)
        
        results.append({
            'fold': fold,
            'test_sharpe': sharpe,
            'test_max_dd': max_dd,
            'rmse': np.sqrt(mean_squared_error(y_test, predictions))
        })
        
        print(f"Fold {fold} Complete | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}")
        
    return pd.DataFrame(results)

# ==============================================================================
# 3. CRITICAL METRIC: THE "NOMINAL ILLUSION" CHECK
# ==============================================================================
def calculate_strategy_returns(predictions, actuals):
    """
    In Turkey, a 5% daily gain might just be inflation. 
    We must measure the strategy's return vs holding USD/TRY.
    """
    # Directional accuracy check
    correct_direction = np.sign(predictions) == np.sign(actuals)
    
    # Simulate PnL (Long if prediction > 0, Short if prediction < 0)
    pnl = np.sign(predictions) * actuals 
    return pnl