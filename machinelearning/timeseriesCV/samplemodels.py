import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import feature_selection
import pmdarima as pm


class Model:
    def __init__(self):
        self.model = None
    
    def train(self, train_df):
        # Fit the model
        return None
    
    def predict(self, train_df, test_df, *args):
        """ Return the test predictions and insample predictions.
        Returns:
            - predictions: pd.Series of the forecasts.
            - train_predictions: pd.Series of the insample forecasts.
        """
        predictions, train_predictions = [None, None]
        return predictions, train_predictions
    
    
class ARIMAModel(Model):
    def __init__(self, **kwargs):
        self.model = pm.ARIMA(**kwargs)
        #self.model.__init__(**kwargs)
        print(self.model.order)
            
            
    def train(self, train_df):
        # For fitting pm.ARIMA
        y = train_df['y']
        if train_df.columns.drop(['y', 'ds']).shape[0] != 0:
            X = train_df.drop(['y', 'ds'], axis=1)
            self.model.fit(y, X)
            print(f"Estimated AR parameters: {self.model.arparams()}")
        else:
            self.model.fit(y)
           
        
    def predict(self, train_df, test_df, size):
        y = train_df['y']
        
        # Deal with the case when we have no regressors
        if len(train_df.columns.drop(['y', 'ds'])) != 0:
            X = train_df.drop(['y', 'ds'], axis=1)
            predictions = self.model.predict(X = test_df.drop(['y', 'ds'], axis=1), n_periods = size)
            train_predictions = self.model.predict_in_sample(X)
        else:
            predictions = self.model.predict(n_periods = size)
            train_predictions = self.model.predict_in_sample()
        return predictions, train_predictions

    
    
class ProphetModel(Model):
    def __init__(self, prophet_seasonality = False, **kwargs):
        self.model = Prophet(**kwargs)

        # add seasonality to Prophet
        if prophet_seasonality:
            self.model.add_seasonality(name='monthly', period=30.5, fourier_order=1, prior_scale = 0.1)
            
            
    def train(self, train_df):
        # add regressors to prophet 
        for col in train_df.columns.drop(['y', 'ds']):
            self.model.add_regressor(col)
                
        self.model.fit(train_df)
        
        
    def predict(self, train_df, test_df, size=None):
        # forecasts
        predictions = self.model.predict(test_df.drop(['y'], axis=1))['yhat']
        train_predictions = self.model.predict(train_df.drop(['y'], axis=1))['yhat']
        
        return predictions, train_predictions