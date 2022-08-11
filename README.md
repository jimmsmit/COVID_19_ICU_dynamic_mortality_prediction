# COVID_19_ICU_dynamic_mortality_prediction
Code to validate trained model for dynamic mortality prediction for COVID-19 patients admitted to the ICU


## Brief Introduction

## Data
These instructions go through the loading of the fitted models, pre-processing of the raw data and evaluation of our model on your own dataset.

You need to upload 2 datasets, both a set used to fit a calibrator (calibration dataset) and a set used to validate the model (validation dataset). 
Both the datasets need to be imported in the main file, where the path where the data is saved needs to be specified. These datasets need to be [N x M] sized tables, where N (nuber of rows) depicts the individual patient samples and M (number of columns) the variables collected for each sample. Also the corresponding [N x 1] label vectors (0=negative, 1=positive) need to be loaded. 
Label the samples as positive if unplanned ICU admission or death occurred within 24 hours from the moment of sampling, and negative otherwise.

Furthermore, make sure the columns of the datasets are in the correct order and the values are in the correct units (see the table below).





column | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11| #12 | #13 | #14 | #15 | #16 | #17 | #18 | #19 | #20 | #21 | #22 | #23 | #24 | #25 | #26 | #27 | #28 | #29
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---|--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |---
Variable | Age | Sex | Current length-of-stay in ICU | SpO2 | Heart rate | Systolic blood pressure | Respiratory rate | Temperature | FiO2 | SpO2/FiO2 | paO2 | paCO2 | pH| paO2/FiO2 | Base excess | CRP | Haemoglobin | White Cell Count | Urea | Magnesium | Sodium | Creatinine | Ionised calcium | Potassium | Glucose | Urea-Creatinine ratio | Chloride | Hematocrit | Platelet count
Unit | years | 0=female, 1=male | hours | % | bpm | mmHg | /min | °C | % | - | mmHg | mmHg | -| - | mmol/L | mg/L | mmol/L | 10^9/L | mmol/L | mmol/L | mmol/L | μmol/L | mmol/L | mmol/L | mmol/L | - | mmol/L | fraction | 10^9/L
