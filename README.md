# Data-Driven Analysis of Traffic Crashes in Chicago

This is a repository for Data Science capstone project at the George Washington University.  
In this project, we developed classification models to predict the severity of traffic crashes and generalized linear models (GLM) to estimated the distribution of the count of traffic crashes within 2-hour intervals. This repository contains data, code, presentation slides, and report.  

## Data
Our data was retrieved from Chicago Data Portal (https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/about_data). The data is stored in the "Data" folder.  

## Code
We developed our codes using jupyter notebook.
### Classification models
For classification models, we developed four models: Decision Tree, Random Forest, CatBoost, and Feedforward Neural Network (FNN). Since our target variable was imbalanced, we utilized oversampling and undersampling methods. Each model was trained on both undersampled and oversampled. The notebooks related to the classification models have names starting with "Classification".
### GLM
For GLMs, we developed three models: Poisson, Negative Binomial, and Zero-inflated models. The notebook is "GLM.ipynb"

## Presentation Slides
### Mid-term Presentation
"Midterm_Presention.pdf" contains introduction, literature review, and methodology.
### Final Presentation
"Final_Presentaion.pptx" contains introduction, literature review, methodology, results & analysis, and conclusion. "Capstone_Final_Presentaion.mp4" is the video recoding our presentaion.

## Report
Our project is summarized in "Capstone_Report.pdf".
