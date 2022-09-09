# COVID_19_ICU_dynamic_mortality_prediction
Code to validate trained model for dynamic mortality prediction for COVID-19 patients admitted to the ICU

For details of this model, see the corresponding article: 
https://www.sciencedirect.com/science/article/pii/S2666521222000242


## Brief Introduction

## Data
These instructions go through the loading of the fitted models, pre-processing of the raw data and evaluation of our model on your own dataset.

The required dataset contains N (nuber of rows) patient samples and 36 columns, ie the variables collected for each sample. 

There is a model for mortality risk within the coming 24 hours and for in-ICU mortality risk. 
Samples used to validate the different models should be labeled accordingly. 
For 24 hour mortality prediction, this means that samples which are sampled within 24 hours of in-ICU mortality are labeled as '1', and '0' otherwise.
For in-ICU mortality prediction, this means that samples which are sampled from patients who died in the ICU are labeled as '1', and '0' otherwise.
(see supplementary figure 2, supplementary material --> https://www.sciencedirect.com/science/article/pii/S2666521222000242#appsec1)


Make sure the columns of the datasets contains the correct variables, are in the correct order and are in the correct units (see the table below).


column | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11| #12 | #13 | #14 | #15 | #16 | #17 | #18 | #19 | #20 | #21 | #22 | #23 | #24 | #25 | #26 | #27 | #28 | #29 | #30 | #31 | #32 | #33 | #34 | #35 | #36 
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---|--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |---
Variable | Age | Sex | Current length-of-stay in ICU | SpO2 | Heart rate | Systolic blood pressure | Respiratory rate | Temperature | FiO2 | SpO2/FiO2 | paO2 | paCO2 | pH| paO2/FiO2 | Base excess | CRP | Haemoglobin | White Cell Count | Urea | Magnesium | Sodium | Creatinine | Ionised calcium | Potassium | Glucose | Urea-Creatinine ratio | Chloride | Hematocrit | Platelet count | ASAT | Lactate dehydrogenase | Alkaline phosphatase | Albumin | ALAT | Glascow coma scale-score (eye) | Glascow coma scale-score (motor)
Unit | years | 0=female, 1=male | hours | % | bpm | mmHg | /min | °C | % | - | mmHg | mmHg | -| - | mmol/L | mg/L | mmol/L | 10^9/L | mmol/L | mmol/L | mmol/L | μmol/L | mmol/L | mmol/L | mmol/L | - | mmol/L | fraction | 10^9/L | U/L | U/L | U/L | g/L | U/L | - | -
