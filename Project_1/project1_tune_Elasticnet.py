#%%

import warnings
warnings.filterwarnings("ignore")

# ----------------
# Evaluating
def evaluating(y_pred):
    y_test = pd.read_csv("test_y.csv")
    y_test_array = y_test.iloc[:, 1].to_numpy(dtype='float32')
    rmse = np.sqrt(np.mean((np.log(y_pred) - np.log(y_test_array))**2))
    print("RMSE: %f" % (rmse))

###########################################
###########################################
###########################################
# Main start below
# Step 0: Load necessary libraries
###########################################
# Step 0: Load necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
# import glmnet_python
# import matplotlib.pyplot as plt

np.random.seed(9455)
def train_pre_process(train):
    x = train.drop(['PID', 'Sale_Price'], axis=1, inplace=False)
    y = train.loc[:, ['Sale_Price']]
    x = x.fillna(0)
    return x, y


def lasso_try():
    train = pd.read_csv("train.csv")

    # ---------------------- #
    # Try the model of Lasso
    x_train, y_train = train_pre_process(train) # fresh data
    # x_drop = ['MS_SubClass', 'Street', 'Condition_2', 
    #         'Roof_Matl', 'Utilities', 'Heating', 'Pool_QC', 'Pool_Area', 
    #         'Misc_Feature', 'Electrical', 'Bsmt_Half_Bath', 'Low_Qual_Fin_SF']
    x_drop = ['Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 
            'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 
            'Longitude','Latitude']
    x_winsor = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", 
            "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', 
            "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", 
            "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]
    x_train = x_train.drop(columns = x_drop)
    y_train = np.log(y_train)

    # Select variables for not included
    x_coltype = dict(x_train.dtypes)
    x_cat_list = [col_name for col_name, t in x_coltype.items() if t == 'O']

    for col in x_train:
        if col not in x_winsor:
            x_train.loc[:, col] = winsorize(x_train.loc[:, col], limits=[0.05, 0.05])

    # Encode for categorical variables
    x_encoder = OneHotEncoder(handle_unknown='ignore')
    x_encoder.fit(x_train[x_cat_list])

    d_frame = pd.DataFrame(x_encoder.transform(x_train[x_cat_list]).toarray())
    x_train = pd.concat([x_train, d_frame], axis=1)
    x_train = x_train.drop(columns = x_cat_list)

    # ratios = np.arange(0, 1, 0.1)
    # alphas = np.arange(0, 1, 0.1)
    # eln_model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=10, n_jobs=-1)
    # eln_model.fit(x_train, y_train)

    ridge_cv = RidgeCV(normalize=True, cv=10)
    fit_ridge = ridge_cv.fit(x_train, y_train)

    ###########################################
    # Step 2: Preprocess test data
    #         and output predictions into two files
    test = pd.read_csv("test.csv")

    def test_pre_process(test):
        x = test.drop(['PID'], axis=1, inplace=False)
        x = x.fillna(0)
        return x

    def submission_gen(test, y_pred, file_name):
        result = pd.DataFrame(y_pred, columns = ['Sale_Price'])
        result = pd.concat([test.loc[:, 'PID'], result], axis=1)
        result.to_csv(file_name, index=False)

    # ---------------------- #
    # Test for Lasso
    x_test = test_pre_process(test) # Fresh Data
    x_test = x_test.drop(columns = x_drop)

    # for col in x_test:
    #     if col not in x_winsor:
    #         x_test.loc[:, col] = winsorize(x_test.loc[:, col], limits=[0.05, 0.05])

    # Encode for category variables
    d_frame = pd.DataFrame(x_encoder.transform(x_test[x_cat_list]).toarray())
    x_test = pd.concat([x_test, d_frame], axis=1)
    x_test = x_test.drop(columns = x_cat_list)

    # Pred by 
    y_pred = fit_ridge.predict(x_test)
    y_pred = np.exp(y_pred)
    submission_gen(test, y_pred, "mysubmission2.txt")

    evaluating(y_pred)

#%%
# Import libraries
import pandas as pd
from pandas.core.arrays.sparse import dtype
import numpy as np

# Data Generating

## Ames Housing Data
### with 2930 rows (i.e., houses) and 83 columns.

## Test IDs: project1_testIDs.dat
### contains 879 rows and 10 columns --- 10 sets for train / test

# ------------------ 
# Try the first split first - j = 0

# j = 1
for j in range(0, 10):
    data = pd.read_csv("Ames_data.csv") # index start from zero as default
    testIDs = pd.read_table("project1_testIDs.dat", sep=' ', header=None)
    test_index = [x - 1 for x in testIDs.iloc[:,j]] # index start from 1

    train = data.loc[~data.index.isin(test_index)]
    test = data.loc[data.index.isin(test_index)]
    test_y = test.iloc[:, [0, 82]]
    test = test.iloc[:, 0:82]

    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
    test_y.to_csv("test_y.csv", index=False)

    lasso_try()

#%%
j = 1

data = pd.read_csv("Ames_data.csv") # index start from zero as default
testIDs = pd.read_table("project1_testIDs.dat", sep=' ', header=None)
test_index = [x - 1 for x in testIDs.iloc[:,j]] # index start from 1

train = data.loc[~data.index.isin(test_index)]
test = data.loc[data.index.isin(test_index)]
test_y = test.iloc[:, [0, 82]]
test = test.iloc[:, 0:82]

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
test_y.to_csv("test_y.csv", index=False)

lasso_try()