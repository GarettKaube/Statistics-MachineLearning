class Model:
    def __init__(self):
        self.model = None
    
    def train(self, train_df):
        # Fit the model
        return None
    
    def predict(self, train_df, test_df, size):
        """ Return the test predictions and insample predictions
        """
        predictions, train_predictions = [None, None]
        return predictions, train_predictions
    
    
class ARIMAModel(Model):
    def __init__(self, **kwargs):
        print(kwargs)
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
        
        
    def predict(self, train_df, test_df):
        # forecasts
        predictions = self.model.predict(test_df.drop(['y'], axis=1)) 
        train_predictions = self.model.predict(train_df.drop(['y'], axis=1)) 
        return predictions, train_predictions

            
class TimeSeriesCrossValidator:
    def __init__(self, model, fold_size, **kwargs):
        """
        Input:
            - model: Model object 
            - fold_size: int indicating the size of each cross-validation fold
        """
        # Model object must follow above schema
        self.model = model
        self.size = fold_size
        
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
        self.mse = []
        self.mae = []
        self.mape = []
        self.mse_naive = []
    
    
    def set_model_args(self, **kwargs):
        self.model_args = kwargs
        
    
    def principal_components(self, n_components:int=2, subset:list=None):
        """ Enables principal component analysis to be done during cross-validation.
        """
        self.pca = True
        self.pca_n_components = n_components
        self.pca_subset = subset
    
    
    def feat_selection(self, n_features):
        """ Enables feature selection to be done during cross-validation.
        """
        self.feature_selection = True
        self.n_features = n_features
    
            
    def select_features(self, train, test):
        # Feature selection based on pearson correlation
        pearson = feature_selection.SelectKBest(feature_selection.r_regression, k=self.n_features)
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

        train_dim_red = pd.DataFrame(pca_.fit_transform(train.loc[:,cols]), columns = pca_cols)
        train = pd.concat([train[train.columns.drop(cols)].reset_index(drop=True), train_dim_red], axis=1)

        if test is not None:
            test_dim_red  = pd.DataFrame(pca_.transform(test[cols]), columns = pca_cols)
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
    
    
    def calculate_performance_metrics(self, actual_train, actual_test, test_predictions, train_predictions):
        predict_naive = [actual_train.iloc[-1]] # Naive forecast. will only be used if fold_size = 1
        # calculate performance metrics
        try:
            y_hat = test_predictions['yhat']
            y_hat_train = train_predictions['yhat']
        except Exception:
            y_hat = test_predictions
            y_hat_train = train_predictions

        self.mse.append(mean_squared_error(y_hat, actual_test))
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
        n_folds = int((end-start) / self.size)
        assert n_folds > 0, "Start index must be larger than end index"
        continue_ = True
        
        print("N_folds:", self.size)
        print(f"Features: {data.columns.drop(['y', 'ds'])}")
        
        while continue_:
            for fold in range(n_folds+1):
                
                next_index = start + fold*self.size # increase size of train
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

        return self.mse, self.mae, self.mape, self.mse_naive, self.train_mse
    
    
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
        return train, test
        
        
    def __performance(self, train, test, fold, print_test_dates):
        try:
            predictions, train_predictions, = self.model.predict(train, test, self.size)
        except TypeError:
            # Prophet does not need the size of the forecasts
            predictions, train_predictions, = self.model.predict(train, test)
            
        # Check performance
        actual_test = test['y']
        actual_train = train['y']
        self.calculate_performance_metrics(actual_train, actual_test, predictions, train_predictions)

        if self.predict_naive and print_test_dates:
            print("Test date:", test['ds'].item())
        if print_test_dates:
            print("Test date(s):", test['ds'])
        print("Fold {} --- MSE: {} --- RMSE: {} --- MAE {} --- MAPE {}".format(fold+1, self.mse[fold], np.sqrt(self.mse[fold]).item(), self.mae[fold], self.mape[fold]))
        
    
    def print_metrics(self):
        """Prints mean squared error (MSE), root MSE, pooled RMSE, mean absolute error, and mean absolute percentage error for test and train sets.
        """
        print("MSE:", np.mean(self.mse))
        print("Standard Deviation MSE:", np.std(self.mse))
        
        rmse = np.mean(np.sqrt(self.mse))
        
        print("RMSE:", rmse)
        print("Standard Deviation RMSE:", np.std(np.sqrt(self.mse)))
        
        total_rmse = np.sqrt(np.mean(np.sqrt(self.mse)**2))
        
        print('TOTAL RMSE:', total_rmse) 
        print("MAE", np.mean(self.mae))
        print("MAPE",np.mean(self.mape))
        print("train mse:",np.mean(self.train_mse))
        print("train rmse:", np.mean(np.sqrt(self.train_mse)))
