import pandas as pd
from utils import one_hot
from data import get_external_data
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class Get_Macro_Data(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dates = None
        self.external = None

    def create_date_windows(self, date:str, window:int=6):
        """
        Gathers dates around date in window
        Input: Date string
        Output: List of date strings
        """
        dates, forward, backward = [], [], []
        y,m,d = date.split('-')
        int_y, int_m = int(y), int(m)
        
        curr_y, curr_m = int_y, int_m
        for _ in range(window):
            curr_m += 1
            if curr_m > 12:
                curr_m = 1
                curr_y += 1
            forward.append(f"{curr_y:04d}-{curr_m:02d}-01")
        curr_y, curr_m = int_y, int_m
        for _ in range(window):
            curr_m -= 1
            if curr_m < 1:
                curr_m = 12
                curr_y -= 1
            forward.append(f"{curr_y:04d}-{curr_m:02d}-01")
        dates = sorted(forward + [f"{int_y:04d}-{int_m:02d}-01"] + backward)
        return dates
        

    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """
        Creates averages of external data for each datapoint in a window
        """
        external = get_external_data()
        X['date_originated'] = X['date_originated'].astype(str)
        new_columns_names = external.columns
        temp = pd.DataFrame(np.zeros(shape = (X.shape[0], len(new_columns_names))), columns=new_columns_names, index = X.index)
        X = pd.concat([X, temp], axis=1)

        for date in X.date_originated.unique():
            circa = self.create_date_windows(date)
            means = external.loc[external.index.isin(circa)].mean()
            X.loc[X.date_originated == date, new_columns_names] = means.to_list()

        X.drop("date_originated",axis=1, inplace=True)
        return X  
    


class Clean_Train(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le = LabelEncoder()
        #categorical columns
        self.categorical = "province" 
        #numeric columns                                         
        self.num = ['six_score', 'monthly_net_income', 'months_of_employment']

    def fit(self, X, y=None):
            # drop irrelavent columns
            X = X.drop(['id', 'postal_code', 'standard_occupational_classification_code','standard_occupational_classification_title','job_title'], axis=1)
            data_for_le = X.reset_index(drop=True)
            
            # transform province into number categories
            cat = data_for_le[self.categorical].astype('category')
            # only encode non-NaN values
            fit_by = cat[~(data_for_le[self.categorical].isna())]
            self.le.fit(fit_by)
            
            return self


    def transform(self, X):
        """
            Cleans and imputes NaNs and creates one-hot vectors
            Returns: pd.DataFrame 
        """
        X = X.drop(['id', 'postal_code', 'standard_occupational_classification_code','standard_occupational_classification_title','job_title'], axis=1)
        X = X.reset_index(drop=True)
        cat = X[self.categorical].astype('category')
        fit_by = cat[~(X[self.categorical].isna())]

        # only encode non-NaN values
        X[self.categorical] = fit_by.apply(lambda x: self.le.transform([x])[0] if type(x) == str else x) 


        return X
    

class Custom_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical = ["province"]                                         #categorical columns
        self.num = ['six_score', 'monthly_net_income', 'months_of_employment']  #numeric columns
        self.cat_imputer = IterativeImputer(estimator=RandomForestClassifier(),
                                       initial_strategy='most_frequent',
                                       max_iter=20,
                                       random_state=0
                                       )
        self.num_imputer = IterativeImputer(estimator=RandomForestRegressor(),
                                       initial_strategy='mean',
                                       max_iter=10,
                                       random_state=0
                                       )    



    def fit(self, X):
            data_for_imputing = X.drop("date_originated", axis=1)
            self.cat_imputer.fit(data_for_imputing[self.categorical])
            self.num_imputer.fit(data_for_imputing[self.num])
            return self

    def transform(self, X):
        data_for_imputing = X.drop("date_originated", axis=1)

        data_for_imputing[self.categorical] = self.cat_imputer.transform(data_for_imputing[self.categorical])
        data_for_imputing[self.num] = self.num_imputer.transform(data_for_imputing[self.num])

        data_for_imputing = one_hot(data_for_imputing, self.categorical, labels = [['AB','BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'PE', 'QC', 'SK']])
        X = pd.concat([X['date_originated'], data_for_imputing ] ,axis=1)

        return X        




 
        
        