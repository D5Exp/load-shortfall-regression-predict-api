"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
# Settings to produce nice plots in a Jupyter notebook
import seaborn as sns
# Libraries for data preparation and model building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import math
from sklearn.model_selection import cross_val_score
from statsmodels.graphics.correlation import plot_corr
import statsmodels.formula.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import random
import warnings
warnings.filterwarnings('ignore')
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------
    df_new=data
    df_new['Valencia_pressure'] = df_new['Valencia_pressure'].fillna(df_new['Valencia_pressure'].mode()[0])
    df_new = df_new.drop(['Unnamed: 0'], axis = 1)
    df_new['time']=pd.to_datetime(data['time'])
    df_new["day"] = df_new["time"].dt.day
    df_new["month"] = df_new["time"].dt.month
    df_new["year"] = df_new["time"].dt.year
    df_new["weekday"] = df_new["time"].dt.weekday
    df_new["hour"] = df_new["time"].dt.hour
    df_new['Valencia_wind_deg'] = df_new['Valencia_wind_deg'].str.extract('(\d+)')
    df_new['Seville_pressure'] = df_new['Seville_pressure'].str.extract('(\d+)')
    df_new[['Valencia_wind_deg','Seville_pressure']] = df_new[['Valencia_wind_deg','Seville_pressure']].apply(pd.to_numeric)
    def g(x):
        min=0
        incre=36
        for i in range (1,11):
            if x==i : x=random.uniform(min+(i-1)*incre,min+incre*i)
        return x
    df_new['Valencia_wind_deg']=df_new['Valencia_wind_deg'].apply(lambda row : g(row))
    max_17=30.262*33.9639
    min_17=29.613*33.9639
    max_16=30.695*33.9639
    min_16=29.431*33.9639
    max_15=30.670*33.9639
    min_15=29.472*33.9639
    avg_min=(min_17+min_16+min_15)/3
    avg_max=(max_17+max_16+max_15)/3
    incre=(avg_max-avg_min)/25

    def f(x):
        for i in range(0,26):
            if x==i: x=random.uniform(avg_min+(i-1)*incre, avg_min+i*incre)
        return round(x)

    df_new["Seville_pressure"] = df_new["Seville_pressure"].apply(lambda row: f(row))
    df_new = df_new.drop(['Bilbao_rain_1h', 
                                    'Bilbao_clouds_all', 
                                    'Seville_clouds_all', 
                                    'Madrid_clouds_all', 
                                    'Barcelona_rain_1h', 
                                    'Seville_rain_1h', 
                                    'Bilbao_snow_3h', 
                                    'Seville_rain_3h', 
                                    'Madrid_rain_1h', 
                                    'Barcelona_rain_3h', 
                                    'Valencia_snow_3h', 
                                    'Madrid_weather_id', 
                                    'Barcelona_weather_id', 
                                    'Seville_weather_id', 
                                    'Bilbao_weather_id', 'Barcelona_temp_max',
                      'Barcelona_temp_min',
                      'Bilbao_temp_max',
                      'Bilbao_temp_min',
                      'Madrid_temp_max',
                      'Madrid_temp_min', 
                      'Seville_temp_min', 
                      'Valencia_temp_min'], axis=1)
    def n_season(x):
        Seasons=["Winter","Spring","Summer","Autumn"]
        if x in [12, 1, 2] : x=Seasons[0]
        else: 
            if x in [3, 4, 5] : x=Seasons[1]
            else: 
                if x in [6, 7, 8] : x=Seasons[2]
                else: x=Seasons[3]
        return str(x)
    df_new["Season"]=df_new["month"].apply(lambda x: n_season(x))
    column_titles = [col for col in df_new.columns if col!= 'load_shortfall_3h'] + ['load_shortfall_3h']
    df_new = df_new.reindex(columns=column_titles)
    df_new=df_new.drop(['time'],axis=1)
    df_new=pd.get_dummies(df_new)
    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = feature_vector_df[['Madrid_wind_speed', 'Valencia_wind_deg', 'Valencia_wind_speed',
       'Seville_humidity', 'Madrid_humidity', 'Bilbao_wind_speed',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Seville_wind_speed', 'Seville_pressure', 'Barcelona_pressure',
       'Bilbao_pressure', 'Valencia_pressure', 'Seville_temp_max',
       'Madrid_pressure', 'Valencia_temp_max', 'Valencia_temp', 'Seville_temp',
       'Valencia_humidity', 'Barcelona_temp', 'Bilbao_temp', 'Madrid_temp',
       'day', 'month', 'year', 'weekday', 'hour',
       'Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter']]
    
    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
