# import required libraries
import numpy as np 
import pandas as pd 
import pickle
from functions import *

# import dependencies

# scaler and imputer
scaler = pickle.load(open('../dpendencies/scaler.sav', 'rb'))
imputer = pickle.load(open('../dpendencies/imputer.sav', 'rb'))

# 24 hour mortality prediction models
model_24h_LR_uncalibrated = pickle.load(open('../dpendencies/clf_uncali_24_LR.sav', 'rb')) # logistic regression, fitted on full dataset
model_24h_LR_calibrated = pickle.load(open('../dpendencies/clf_cali_24_LR.sav', 'rb')) # logistic regression, fitted on 2/3 of data, recalibrated with remaining 1/3

model_24h_RF_uncalibrated = pickle.load(open('../dpendencies/clf_uncali_24_RF.sav', 'rb')) # random forest, fitted on full dataset
model_24h_RF_calibrated = pickle.load(open('../dpendencies/clf_cali_24_RF.sav', 'rb')) # random forest, fitted on 2/3 of data, recalibrated with remaining 1/3

# in-ICU mortality prediction models
model_ICU_LR_uncalibrated = pickle.load(open('../dpendencies/clf_uncali_in_ICU_LR.sav', 'rb')) # logistic regression, fitted on full dataset
model_ICU_LR_calibrated = pickle.load(open('../dpendencies/clf_cali_in_ICU_LR.sav', 'rb')) # logistic regression, fitted on 2/3 of data, recalibrated with remaining 1/3

model_ICU_RF_uncalibrated = pickle.load(open('../dpendencies/clf_uncali_in_ICU_RF.sav', 'rb')) # random forest, fitted on full dataset
model_ICU_RF_calibrated = pickle.load(open('../dpendencies/clf_cali_in_ICU_RF.sav', 'rb')) # random forest, fitted on 2/3 of data, recalibrated with remaining 1/3



# import new dataset 

# example code to validate 'model_24h_LR_uncalibrated' on imported new dataset 'data' (pd.DataFrame) and corresponding ground truth labels 'labels'

# generate predictions for new datapoints using model
preds = model_24h_LR_uncalibrated.predict_proba(data)[:,1]

# evaluate weak and moderate calibration using the function 'calibration_plot'
calibration_plot(preds,labels)

# define parameters of weak calibration
a, a_low, a_high, b, b_low, b_high, a_prime, a_prime_low, a_prime_high = get_cox_pred(preds,labels)

# recalibrate model by adjusting the calibration intercept 
model_update_intercept = model_updating(model_24h_LR_uncalibrated,a,b,a_prime,'intercept_only')
#generate new predictions with updated model
preds = model_update_intercept.predict_proba(data)[:,1]

# evaluate updated model
calibration_plot(preds,labels)

# recalibrate model by adjusting the calibration intercept and slope
model_update_intercept_slope = model_updating(model_24h_LR_uncalibrated,a,b,a_prime,'intercept_and_slope')
#generate new predictions with updated model
preds = model_update_intercept_slope.predict_proba(data)[:,1]

# evaluate updated model
calibration_plot(preds,labels)