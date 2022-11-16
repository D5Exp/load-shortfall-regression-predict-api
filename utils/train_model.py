"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Fetch training data and preprocess for modeling
#train = pd.read_csv('./data/df_train.csv')
train = pd.read_csv('data/df_train.csv')

y_train = train[['load_shortfall_3h']]
#X_train = train[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
X_train = train.drop(['load_shortfall_3h'],axis=1)

# Fit model
#lm_regression = LinearRegression(normalize=True)
RF_reg =RandomForestRegressor(n_estimators=100, random_state=0)
print ("Training Model...")
#lm_regression.fit(X_train, y_train)
RF_reg.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/load_shortfall_RF_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
#pickle.dump(lm_regression, open(save_path,'wb'))
pickle.dump(RF_reg, open(save_path,'wb'))
