import numpy as np
import pandas as pd
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import feature_selection


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

    
            
class TimeSeriesCrossValidator:
    def __init__(self, model, fold_size, cv_step=1, **kwargs):
        """
        Input:
            - model_type: str either "ARIMA", or "Prophet" is supported
            - fold_size: int indicating the size of each cross-validation fold
        """
        # Model object must follow above schema
        self.model = model
        self.size = fold_size
        self.cv_step = cv_step
        
        # Store models that are trained and tested
        self.model_list = []
        
        self.predict_naive = True if self.size == 1 else None
        
        # Initialize the model args to no arguments
        self.model_args = kwargs
        # Principal component analysis settings
        self.pca = False
        self.pca_n_components = None
        self.pca_subset = None
        
        self.feature_selection = False
        self.n_features = None
        
        self.feature_scaling = False
        
        # store performance metrics
        self.train_mse = []
        self.mse = {}
        self.mae = []
        self.mape = []
        self.mse_naive = []
    
    
    def set_model_args(self, **kwargs):
        """Sets the keyword arguments for the model
        Returns:
            - None
        """
        self.model_args = kwargs
        
    
    def principal_components(self, n_components:int=2, subset:list=None):
        """ Enables principal component analysis to be done during cross-validation.
        Input:
            -n_components int of number of components to keep for PCA
            -subset: list of variable names to be used in PCA
        Returns:
            - None
        """
        self.pca = True
        self.pca_n_components = n_components
        self.pca_subset = subset
    
    
    def feat_selection(self, n_features):
        """ Enables feature selection to be done during cross-validation.
        Input:
            -n_features: Number of features to select
        Returns:
            - None
        """
        self.feature_selection = True
        self.n_features = n_features
    
            
    def select_features(self, train, test):
        # Feature selection based on pearson correlation
        pearson = feature_selection.SelectKBest(feature_selection.mutual_info_regression, k=self.n_features)
        pearson.fit_transform(train.drop(['y','ds'], axis=1), train['y'])
        cols = pearson.get_feature_names_out()
        print(f"feature selection chose {cols}")
        
        train = train[list(cols)+ ['y','ds']]
        test = test[list(cols)+ ['y','ds']]
        return train, test
    
    
    def dimension_reduction(self, train, test=None):
        pca_cols = [f"pc{i}" for i in range(1, self.pca_n_components + 1)]
        pca_ = PCA(n_components=self.pca_n_components)
        if not self.pca_subset:
            cols = train.columns.drop(['y', 'ds'])
        else:
            cols = self.pca_subset

        train_dim_red = pd.DataFrame(pca_.fit_transform(train.loc[:,cols]), columns=pca_cols)
        train = pd.concat([train[train.columns.drop(cols)].reset_index(drop=True), train_dim_red], axis=1)

        if test is not None:
            test_dim_red  = pd.DataFrame(pca_.transform(test[cols]), columns=pca_cols)
            test = pd.concat([test[test.columns.drop(cols)].reset_index(drop=True), test_dim_red], axis=1)
            return train, test
        else:
            return train
        
    
    def scaling(self, train, test):
        """ Standardize variables
        """
        scaler_ = StandardScaler(copy=False)
        train.loc[:,train.columns.drop(['y','ds'])]= scaler_.fit_transform(train.drop(['y','ds'], axis=1))
        test.loc[:, test.columns.drop(['y','ds'])]= scaler_.transform(test.drop(['y','ds'], axis=1))
        return train, test
    
    
    def calculate_performance_metrics(self, actual_train, actual_test, test_predictions, train_predictions, dates):
        predict_naive = [actual_train.iloc[-1]] # Naive forecast. will only be used if fold_size = 1
        y_hat = test_predictions
        y_hat_train = train_predictions
        
        self.mse[str(dates[0])] = mean_squared_error(y_hat, actual_test).item()
        self.train_mse.append(mean_squared_error(y_hat_train, actual_train))
        self.mae.append(mean_absolute_error(y_hat, actual_test).item())
        self.mape.append(mean_absolute_percentage_error(y_hat, actual_test).item())
        
        if self.predict_naive:
            self.mse_naive.append(mean_squared_error(predict_naive, actual_test).item())
    
    
    def fit(self, data:pd.DataFrame, start:int, end:int, print_test_dates:bool = False):
        """  Back test the selected time series model using time series cross-validation
        Input:
            - data: pd.DataFrame with date column "ds", target column "y", and optional regressors
            - start: int index indicating the initial size of the train set
            - end: int index indicating what index to end cross-validation
        Returns:
            - 
        """
        import math
        n_folds = math.floor(((end-start) - self.size + self.cv_step) / self.cv_step)
        assert n_folds > 0, "Start index must be larger than end index"
        continue_ = True
        
        print("N_folds:", n_folds)
        print(f"Features: {data.columns.drop(['y', 'ds'])}")
        
        while continue_:
            for fold in range(n_folds+1):
                
                next_index = start + fold*self.cv_step # increase size of train
                data_copy = data.copy()
                # new train set
                train = data_copy[:next_index]
                    
                # Reset model each fold
                self.model.__init__(**self.model_args)
                
                if next_index + self.size <= end and fold <= n_folds: # make sure index doesn't go beyond the length of end
                    train, test = self.__cross_validation_step(train, data_copy, next_index)
                    self.__performance(train, test, fold, print_test_dates)

                else:
                    continue_ = False      
    
    
    def __cross_validation_step(self, train, df, next_index):
        test = df[next_index:next_index + self.size] # new test set
       
        if self.feature_selection:
            train, test = self.select_features(train, test)
        # standard scaling if enabled
        if self.feature_scaling:
            train, test = self.scaling(train, test)
        # Dimension reduction
        if self.pca:
            train, test = self.dimension_reduction(train, test)
        # Train and predict
        self.model.train(train)
        print(test.shape)
        self.model_list.append(self.model.model)
        return train, test
        
        
    def __performance(self, train:pd.DataFrame, test:pd.DataFrame, fold:int, print_test_dates:bool):
        """ Gets predictions and calculates performance. The model must return its predictions as a pd.Series
        Input:
            - train: pd.DataFrame of train data.
            - test: pd.DataFrame of test data.
            - fold: integer representing the fold that CV is currently on.
            - print_test_dates: bool if cv will print what dates it is evaluating on the test set for debugging.
        """
        predictions, train_predictions, = self.model.predict(train, test, self.size)

        # Check performance
        actual_test = test['y']
        actual_train = train['y']
        self.calculate_performance_metrics(actual_train, actual_test, predictions, train_predictions, test['ds'])

        if self.predict_naive and print_test_dates:
            print("Test date:", test['ds'].item())
        if print_test_dates:
            print("Test date(s):", test['ds'])
        print("Fold {} --- MSE: {} --- RMSE: {} --- MAE {} --- MAPE {}".format(fold+1, list(self.mse.values())[fold], np.sqrt(list(self.mse.values())[fold]).item(), self.mae[fold], self.mape[fold]))
        
    
    def print_metrics(self):
        """Prints mean squared error (MSE), root MSE, pooled RMSE, mean absolute error, and mean absolute percentage error for test and train sets.
        """
        mse = list(self.mse.values())
        print("MSE:", np.mean(mse))
        print("Standard Deviation MSE:", np.std(mse))
        
        rmse = np.mean(np.sqrt(mse))
        
        print("RMSE:", rmse)
        print("Standard Deviation RMSE:", np.std(np.sqrt(mse)))
        
        total_rmse = np.sqrt(np.mean(np.sqrt(mse)**2))
        
        print('TOTAL RMSE:', total_rmse) 
        print("MAE", np.mean(self.mae))
        print("MAPE",np.mean(self.mape))
        print("train mse:",np.mean(self.train_mse))
        print("train rmse:", np.mean(np.sqrt(self.train_mse)))
