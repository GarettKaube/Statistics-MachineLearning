import urllib
import urllib.request
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import os
import pandas as pd
import numpy as np
from utils import download_and_unzip, create_paths, fetch_external_data

# ----------VERY UGLY CODE-----------


def get_external_data():
    # VERY VERY BAD CODE
    DOWNLOAD_ROOT_CPI = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=CPALCY01CAM661N&scale=left&cosd=1961-01-01&coed=2022-12-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-06&revision_date=2023-03-06&nd=1961-01-01"
    DOWNLOAD_ROOT_YIELD_SPREAD = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=T10Y2YM&scale=left&cosd=1976-06-01&coed=2023-02-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-06&revision_date=2023-03-06&nd=1976-06-01"
    DOWNLOAD_ROOT_UNEMPLOYMENT = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=LRUNTTTTCAM156N&scale=left&cosd=1960-01-01&coed=2022-12-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-06&revision_date=2023-03-06&nd=1960-01-01"
    DOWNLOAD_ROOT_FIVE_YEAR_TREASURY_YIELDS = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GS5&scale=left&cosd=1953-04-01&coed=2023-02-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-06&revision_date=2023-03-06&nd=1953-04-01"
    DOWNLOAD_ROOT_TWO_YEAR_TREASURY = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GS2&scale=left&cosd=1976-06-01&coed=2023-02-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-10&revision_date=2023-03-10&nd=1976-06-01"
    DOWNLOAD_ROOT_POLICY_UNCERTAINTY = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GEPUCURRENT&scale=left&cosd=1997-01-01&coed=2022-12-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-12&revision_date=2023-03-12&nd=1997-01-01" 
    DOWNLOAD_ROOT_GS10 = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GS10&scale=left&cosd=1953-04-01&coed=2023-02-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-12&revision_date=2023-03-12&nd=1953-04-01"
    DOWNLOAD_ROOT_OTHER_DATA = "https://www.stevanovic.uqam.ca/LCDMA_February_2023.zip"
    DOWNLOAD_ROOT_MONEY_INTERBANK = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=off&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=IRSTCI01CAM156N&scale=left&cosd=1975-01-01&coed=2022-12-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-13&revision_date=2023-03-13&nd=1975-01-01"
    DOWNLOAD_ROOT_VOLATILITY = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=EMVOVERALLEMV&scale=left&cosd=1985-01-01&coed=2023-02-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-16&revision_date=2023-03-16&nd=1985-01-01"
    DOWNLOAD_ROOT_BANK_PRIME = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=MPRIME&scale=left&cosd=1949-01-01&coed=2023-02-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-16&revision_date=2023-03-16&nd=1949-01-01"
    DOWNLOAD_ROOT_HPI = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=CSUSHPINSA&scale=left&cosd=1987-01-01&coed=2022-12-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-16&revision_date=2023-03-16&nd=1987-01-01"
    DOWNLOAD_ROOT_SAVINGS_RATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=PSAVERT&scale=left&cosd=1959-01-01&coed=2023-01-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-03-16&revision_date=2023-03-16&nd=1959-01-01"

    DOWNLOAD_ROOTS = [DOWNLOAD_ROOT_CPI, DOWNLOAD_ROOT_YIELD_SPREAD, DOWNLOAD_ROOT_UNEMPLOYMENT, DOWNLOAD_ROOT_FIVE_YEAR_TREASURY_YIELDS,DOWNLOAD_ROOT_TWO_YEAR_TREASURY,DOWNLOAD_ROOT_POLICY_UNCERTAINTY,DOWNLOAD_ROOT_MONEY_INTERBANK, DOWNLOAD_ROOT_VOLATILITY, DOWNLOAD_ROOT_BANK_PRIME, DOWNLOAD_ROOT_HPI, DOWNLOAD_ROOT_SAVINGS_RATE]
    download = ["CPI", "YIELD_SPREAD", "UNEMPLOYMENT", "FIVE_YEAR_TREASURY", "TWO_YEAR_TREASURY", "POLICY_UNCERTAINTY","InterestRatesImmediateRates" ,"EquityMarketVolatilityTracker:Overall", "BankPrimeLoanRate", "HomePriceIndex", "PersonalSavingsRate"]

    paths = create_paths(download)
    fetch_external_data(paths, DOWNLOAD_ROOTS)
    download_and_unzip(DOWNLOAD_ROOT_OTHER_DATA, "./data/general_macro_data")

    # balanced_can_md
    other_path = "./data/general_macro_data/LCDMA_February_2023/balanced_can_md.csv"
    other = pd.read_csv(other_path)

    # cpi data for individal provinces 
    cpis = ["CPI_SHEL_ALB", "CPI_DUR_ALB", "CPI_DUR_ONT", "CPI_SHEL_ONT", "CPI_SHEL_BC","CPI_MINUS_FEN_ALB","CPI_DUR_NS"]

    # columns to keep from balanced_can_md
    other = other[["Date", "MANU_INV_RAT_new", "TBILL_6M", "GOOD_HRS_CAN","UNEMP_DURAvg_CAN_new","CPI_ALL_CAN","CRED_HOUS_MORT", "CLAIMS_CAN", "MORTG_5Y", "CRED_HOUS", "BANK_RATE_L","CRED_BUS", "GOV_AVG_1_3Y","CPI_SHEL_CAN","CPI_MINUS_FOO_CAN", "GDP_new","CRED_HOUS_cb","TBILL_3M","G_AVG_1.3.Bank_rate", "G_AVG_10p.TBILL_3M","GOV_AVG_3_5Y", "CRED_MORT_HOUSE_cb", "CRED_T_cb","G_AVG_5.10.Bank_rate","CPI_GOO_CAN","CPI_DUR_CAN","CPI_SERV_CAN","CPI_ALL_ALB", "CPI_ALL_ONT","CPI_ALL_NS"]+cpis]
    other = other.rename({'Date':"DATE"}, axis=1)
    
    external = pd.read_csv(paths[0])
    for path in paths[1:]:
        temp = pd.read_csv(path)
        external = pd.merge(external, temp, on='DATE')
    
    # calcualte percentage change
    external_to_percent = external.drop(['CPALCY01CAM661N'], axis=1)
    external_to_percent['DATE'] = pd.to_datetime(external_to_percent['DATE'])
    external_to_percent.set_index("DATE", inplace=True,drop=True)
    external_to_percent = external_to_percent.pct_change()
    external_to_percent.dropna(inplace=True)
    external_to_percent.reset_index(drop=False,inplace=True)
    external.rename({"":"DATE"}, inplace=True, axis=1)
    external_to_percent['DATE'] = external_to_percent['DATE'].astype(str)
    external[external_to_percent.columns.to_list()] = external_to_percent
    external.dropna(inplace=True)

    external = pd.merge(external, other, on='DATE')

    dbt = get_statcan_data()
    external = external.merge(dbt, left_on="DATE", right_on= "REF_DATE")
    external.drop('REF_DATE',axis=1,inplace=True)
    
    external.set_index("DATE", drop=True, inplace=True)
    
    external = external.shift(1) # use previous observed values to avoid data leakage
    
    return external



def remove_chars(df ,chars:list=[' ', '-', ',']):
    """Helper for get_statcan_data()
    """
    for char in chars:
        df.columns = df.columns.str.replace(char, '_')



def get_statcan_data():
    """Makes statscan data actually usable
    Returns: pd.DataFrame time series of statscan data
    """
    data = pd.read_csv("./data/1010010901_databaseLoadingData.csv")
    data.drop(["GEO", "DGUID", "VECTOR", "SCALAR_ID", "SCALAR_FACTOR", "UOM_ID", "UOM", "STATUS", "SYMBOL", "TERMINATED", "DECIMALS", "COORDINATE"],axis=1, inplace=True)
    data_p = data.pivot_table(["VALUE"], ["REF_DATE"], "Assets and liabilities")
    data_p.index = pd.to_datetime(data_p.index)
    data_p.reset_index(drop=False, inplace=True)
    data_p = data_p.droplevel(level=0, axis=1)

    data_p.rename({"":"REF_DATE"}, inplace=True, axis=1)
    data_p['REF_DATE'] = data_p['REF_DATE'].astype(str)

    data_p.columns = remove_chars(data_p.columns)

    keep = ["Mortgages_insured", "Personal_loan_plan_loans","Total__holdings_of_selected_short_term_assets","Total__other_Canadian_securities","Total__major_assets","Non_mortgage_loans__total","Other_personal_loans","Subordinated_debt_payable_in_Canadian_dollars"]
    data_p = data_p[['REF_DATE'] + keep]
    data_p.dropna(axis=1,inplace=True)

    # convert to percent change
    data_p.set_index(pd.to_datetime(data_p['REF_DATE']), inplace=True, drop=True)
    data_p.dropna(inplace=True)
    data_p = data_p.drop("REF_DATE", axis=1)
    data_p = data_p.pct_change()
    data_p.reset_index(drop=False, inplace=True)
    data_p.rename({"":"REF_DATE"}, inplace=True, axis=1)
    data_p['REF_DATE'] = data_p['REF_DATE'].astype(str)
    return data_p



from sklearn.model_selection import train_test_split


class Train_test:
    def __init__(self, data, train=None, test = None):
        self.data = data
        self.X_train = self.data.drop('default', axis=1)
        self.y_train = self.data['default']
        self.X_test = None
        self.y_test = None
        self.train = train
        self.test = test


    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=0,stratify=self.y_train)
        self.train = pd.concat([self.X_train, self.y_train], axis=1)
        self.test = pd.concat([self.X_test, self.y_test], axis=1)
        

    def get_train_test_split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    

    def get_train_full_test_full(self):
        return self.train, self.test
    

    def separate_labels(self):
        self.X_train = self.train.drop('default', axis=1)
        self.y_train = self.train['default']
        self.X_test = self.test.drop('default', axis=1)
        self.y_test = self.test['default']


    def save(self,name):
        self.train.to_csv(f"./data/{name}_train.csv")
        self.test.to_csv(f"./data/{name}_test.csv")


