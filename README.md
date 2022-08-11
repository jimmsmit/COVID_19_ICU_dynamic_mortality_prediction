# COVID_19_ICU_dynamic_mortality_prediction
Code to validate trained model for dynamic mortality prediction for COVID-19 patients admitted to the ICU


## Brief Introduction

## Data
These instructions go through the loading of the fitted models, pre-processing of the raw data and evaluation of our model on your own dataset.

You need to upload 2 datasets, both a set used to fit a calibrator (calibration dataset) and a set used to validate the model (validation dataset). 
Both the datasets need to be imported in the main file, where the path where the data is saved needs to be specified. These datasets need to be [N x M] sized tables, where N (nuber of rows) depicts the individual patient samples and M (number of columns) the variables collected for each sample. Also the corresponding [N x 1] label vectors (0=negative, 1=positive) need to be loaded. 
Label the samples as positive if unplanned ICU admission or death occurred within 24 hours from the moment of sampling, and negative otherwise.

Furthermore, make sure the columns of the datasets are in the correct order and the values are in the correct units (see the table below).





column | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11| #12 | #13 | #14 | #15 | #16 | #17 | #18 | #19 | #20 | #21 | #22 | #23 | #24 | #25 | #26 | #27
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---|--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |---
Variable | Age | Sex | Current length-of-stay in ICU | Systolic blood pressure | Respiratory rate | Temperature | FiO_{2} | Heart rate | Systolic blood pressure | Respiratory rate | Temperature| AVPU | ΔSpO2 | ΔHeart rate | ΔSystolic blood pressure | ΔRespiratory rate | ΔTemperature | ΔSpO2/O2 | ΔRespiratory rate | ΔTemperature | ΔSpO2/O2 | ΔRespiratory rate | ΔTemperature | ΔSpO2/O2 | ΔRespiratory rate | ΔTemperature | ΔSpO2/O2
Unit | years | 0=female, 1=male | hours | mmHg | /min | °C | % | bpm | mmHg | /min | °C| A=0,V=1,P=2,U=3 | % | bpm | mmHg | /min | °C | %/(L/min) | /min | °C | %/(L/min) | /min | °C | %/(L/min) | /min | °C | %/(L/min)
