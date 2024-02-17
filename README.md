# AutoML App for Supervised Learning Tasks using H20, ydata-profiling and Streamlit

<div align="center">
<img src="https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/225/744/datas/original.png" width="350">
</div>


This platform presents an interface to automates machine learning tasks for tabular data using [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html), [ydata-profiling](https://github.com/ydataai/ydata-profiling) and [Streamlit](https://docs.streamlit.io/get-started/installation/command-line) libraries. It provides a seamless building of ML models with little or no coding. The platform has 4 main parts:

**Data Upload:** This is where you upload your dataset. 
   
<img src="https://github.com/IbukunGracey/automlh20-project/blob/master/images/Upload_data.png">

**Data Profilling:** A profile report is generated, explaining the correlation among attributes, and detailed analysis of each feature.

<img src="https://github.com/IbukunGracey/automlh20-project/blob/master/images/Profiling.png">
<img src="https://github.com/IbukunGracey/automlh20-project/blob/master/images/Profile_report.png">
<img src="https://github.com/IbukunGracey/automlh20-project/blob/master/images/Attribute%20Heatmap%20.png" width="350">

**Model Development:** Here you specify the target variable and choose the type of supervised learning task (classification or regression). A model leaderboard is presented based on specific evaluation metrics.
   
<img src="https://github.com/IbukunGracey/automlh20-project/blob/master/images/Modelling.png" width="350">

**Download winning model:** After this you can download your winning model and make predictions.

# 💾 Usage - Demo

Launch the web app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://automlh2oplatform.streamlit.app/)


# Reproducing this web app
To recreate this web app on your own computer, do the following.

### Create virtual environment
First, you will create a  environment called *test_env* using python 3.10.4

```
py -3.10.4 -m venv test_env
```
Secondly, we will login to the *test_env* environment on your command prompt
```
test_env\Scripts\activate.bat
```
### Install prerequisite libraries

Download requirements.txt file

```
wget https://github.com/IbukunGracey/automlh20-project/blob/master/requirements.txt

```

Pip install libraries
```
pip install -r requirements.txt
```
###  Download and unzip contents from GitHub repo

Download and unzip contents from this repo https://github.com/IbukunGracey/automlh20-project/

###  Launch the app

```
streamlit run app.py
```
### Required dependencies to be installed on your system

[Coding pack for java](https://code.visualstudio.com/docs/java/java-tutorial )

[Developed by Grace](https://github.com/IbukunGracey)

