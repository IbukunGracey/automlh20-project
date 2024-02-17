from operator import index
import streamlit as st
import plotly.express as px
import ydata_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import h2o
from h2o.automl import H2OAutoML
import os 


if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=[0])

with st.sidebar: 
    st.image("https://littleml.files.wordpress.com/2019/09/steps-1.png?w=584&h=321")
    st.title("AutoH2OML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you explore your data and build ML model automatically.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=[0])
        df.to_csv('dataset.csv', index=False)
        st.dataframe(df)
    

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

        

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)

    model_type = st.selectbox('Choose the Model Type', ['Classification', 'Regression'])

    if st.button('Run Modelling'): 
        # init
        h2o.init()
        # aml = H2OAutoML(max_models =25,
        #         balance_classes=True,
        #         seed = 1)
        
        # modeling and training

        # Identify predictors and response

        train = h2o.H2OFrame(df)
        x = train.columns
        y = chosen_target
        # st.write(pd.factorize(x))

        if model_type == 'Classification':
            # # For binary classification, response should be a factor
            train[y] = train[y].asfactor()
            # # test[y] = test[y].asfactor()

            #Classification
             # Run AutoML for 20 base models
            aml = H2OAutoML(max_models=20, max_runtime_secs = 900,
                             seed=1, project_name='classification',
                             sort_metric = "AUC")
            aml.train(x=x, y=y, training_frame=train)
                
        else:  
            train[y] = train[y]
            # Run AutoML for 20 base models

            #Regression
            aml = H2OAutoML(max_models=20, max_runtime_secs = 900,
                                seed=1, project_name='regression',
                                stopping_metric='RMSE',
                                sort_metric = "RMSE")
            aml.train(x=x, y=y, training_frame=train)

        # View the AutoML Leaderboard
        lb = aml.leaderboard
        lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
        st.info('AutoML Leaderboard')
        lb = lb.as_data_frame()
        st.dataframe(lb)


        # # generate best model
        best_model = aml.get_best_model()
        h2o.download_model(best_model, 'best_model')
        # st.write(model_path)

       
if choice == "Download":
    # best_model = aml.get_best_model()
    # my_local_model = h2o.download_model(best_model, path="")
    st.download_button('Download the Model', 'best_model' )

    # h2o.load_model(model =best_model, path=model_path)
    # model_path = 'best_model'
    # with open(model_path, 'rb') as f: 
    #     st.download_button('Download Model', f, filename = 'best_model')
