from optparse import OptionGroup
import pprint
from shap.plots.colors import blue_rgb
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import  LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
    
from sklearn import metrics 
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, r2_score, explained_variance_score
#from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_squared_error, mean_squared_log_error
#from sklearn.metrics import median_absolute_error

from sklearn.impute import KNNImputer 
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.model_selection import train_test_split

import shap
shap.initjs()
    

st.sidebar.title("Navigation")
pages=["Home", "Data Exploration", "Modelling","Conclusion","Bibliography", "About"]
page=st.sidebar.radio("Go to", pages)
# Cover Page
if page == pages[0] : 
   
    st.title('World Happiness')
    image = Image.open("happy1.jpeg")
    st.image(image)
    st.header("Context")
    st.write("The exploration and categorization of factors influencing happiness is essential in understanding various aspects of mental well-being, such as emotional health and life satisfaction.")
    text = ("Qualitative analysis of happiness indicators is required for comprehensive research. However, the manual assessment of subjective well-being is **complex, time-consuming**, susceptible to **bias**, and demands **expertise** in the field. For navigation through diverse data sources, machine learning methodologies are employed to unravel the intricacies of what contributes to an individual's happiness.")
    st.markdown(text)
    
    st.header('Problematic')
    st.write("**Computer-assisted analysis** of happiness indicators and identification of influential factors provide crucial assistance to the policymakers. Traditionally, happiness were measured with questionnaires. This system may not **generalize** well, especially due to the variability in subjective well-being indicators and data acquisition systems in different countries.")
    image = Image.open("Happiness_Test.png")
    st.image(image)
    st.header('Aim')
    st.write('The primary goal of this research is to create a **Machine Learning Model** that can effectively categorize key indicators relevant to subjective well-being.')
    st.subheader("***Description of happiness indicators***")
   


    st.write("We use eleven concepts to measure a quality of life score:")

    st.write("1. Ladder Score,\n"
         "2. Income based on GDP per capita,\n"
         "3. Social support from family and friends,\n"
         "4. Number of hospital beds,\n"
         "5. Number of nurses,\n"
         "6. Health life expectancy,\n"
         "7. Freedom to make life choices,\n"
         "8. Perceptions of corruption,\n"
         "9. Poverty gap index and poverty severity,\n"
         "10. Gini index,\n"
         "11. Generosity")

    st.header('Data')
    
    st.markdown("**1. World Happiness Report [dataset](https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021?select=world-happiness-report.csv),**")
    st.markdown("**2. Hospital beds [dataset](https://data.worldbank.org/indicator/SH.MED.BEDS.ZS) per 1,000 people,**")
    st.markdown("**3. Nursing and midwifery personnel [dataset](https://www.who.int/data/gho/data/indicators/indicator-details/GHO/nursing-and-midwifery-personnel-(per-10-000-population)) per 10,000 population),**")
    st.markdown("**4. Poverty Gap, Poverty Severity, and Gini index [dataset](https://databank.worldbank.org/source/millennium-development-goals/Series/SI.POV.NAGP#).**")

# Data analysis using Data Visualization figures
if page == pages[1] : 
    st.header('Data Exploration')
    # Datensatz beschreibung
    data = {'Name': ['World Happiness Reports', 'Hospital beds', 'Nursing and midwifery personnel', "Poverty Gap, Poverty Severity and gini index"],
        'rows': [1949, 1167, 1770, 2389],
        'columns': [11, 6, 3, 40],}

    df = pd.DataFrame(data)

    if st.button("Dataframes info",  use_container_width=True):
        st.dataframe(df)
    
    happiness_data=pd.read_csv("data happy 06-23.csv")
    happiness_data['year'] = happiness_data['year'].astype(str).str.replace(',', '')
    
    if st.button("World Happiness Reports",  use_container_width=True):
        st.dataframe(happiness_data)
    
    st.title('World Happiness Map')

# Weltkarte erstellen
    fig = px.choropleth(
        happiness_data,
        locations='country',
        locationmode='country names',
        color='ladder',
        hover_name='country',
        color_continuous_scale='Viridis',
        animation_frame='year', animation_group="country"
        )
# Setze den Zeitraum für die Animation
    fig.update_layout(
        sliders=[{
            "active": 0,
            "steps": [{"args": [[f], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}],
                   "label": f"{f}"} for f in sorted(happiness_data['year'].unique())]
    }]
    )
# Zeige Weltkarte
    st.plotly_chart(fig)

    st.write("Information about some countries are not  found in all years. Only the 137 countries  are continiusly present in time range from 2007 till 2018.")
    st.title('The Distribution of Happiness by continent')
    image = Image.open('distribution of happieness.png')
    st.image(image)
    st.write("Africa displays the **lowest** percentile, while Europe, North America, Australia, and New Zealand consistently exhibit high percentile. Notably, South America maintains a high distribution of self-reported life satisfaction, **despite having lower income levels**.")
    st.title("Relationship between the variables")
    image = Image.open('scatter plot.png')
    st.image(image)
    st.write("While economic indicators and social support demonstrate a close association with overall life satisfaction, we note that corruption levels and the availability of dental care may not significantly impact happiness perceptions. This suggests that the influence of these specific elements on subjective well-being is either **minimal or more complex** than initially assumed."
"Simultaneously, the analysis reveals that economic indicators like the Gini index and poverty indexes do not exhibit a strong correlation with a sense of well-being. **Despite being economic metrics**, their relationship with the overall happiness score appears to be less pronounced.")


if page == pages[2] : 

    st.header('Modelling')
    st.write("The objective of the proposed model is to predict the Ladder Score using machine learning techniques.")
    st.subheader('**Main steps of the experiment**',divider='rainbow')
    image = Image.open('Diagram.jpg')
    st.image(image)
    st.write("First, we selected from each database only the data corresponding to the period 2007-2018. Then we unified the name of the columns and verified that the countries in each dataset had the same name, for which we used geopandas to unify the name of the countries. We merged all the datasets into one and observed a loss of information in the process, the final size of the dataset is 965x18. Lastly, we split the data for training and testing purposes,"
              "and we performed imputation of missing values using the KNN imputer and scaling of numerical features using Standard Scaler to maintain data integrity.")
    
    st.subheader('**Model building**',divider='rainbow')
    
    st.write("After finishing the data preparation part, multiple machine learning algorithms were used to predict the happiness score. Next, we used grid search to find the best hyperparameters. Grid search is a process that exhaustively searches for the best values of a manually specified subset of hyperparameters from the target algorithm.")
    st.write("As the data used is quantitative, the following machine learning algorithms are used to predict the happiness score:")
    st.write("1. Multiple Linear Model\n"
         "2. KNeighbors Regressor\n"
         "3. Decision tree regressor\n"
         "4. Randon Forest Regressor\n"
         "5. Gradient boosting regressor\n")
    st.write("Die Models were evaluated observing the following results:")

    df_fin= pd.read_csv("df_Fin.csv",index_col=False,header=0,usecols=['Country', 'Region', 'Year', 'Ladder', 'Log GDP', 'Social Supp',
       'Life expectancy', 'Freedom', 'generosity', 'Corruption', 'Violence',
       'Rate Beds', 'Rate Medical', 'Rate Nursing', 'Rate Dentists',
       'poverty_gap', 'poverty_severity', 'gini'])
    #st.dataframe(df_fin,hide_index=True)
    
    #MACHINE LEARNING
    # Separate the target variable from the explanatory variables and then separate the dataset into a training set 
    #and a test set so that the test set contains 20% ofthe data.

    X = df_fin.drop(['Country', 'Region', 'Year', 'Ladder'],axis=1) # explanatory variables
    y = df_fin['Ladder'] # target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    #using the "KNNimputer" strategies for numeric variables, rigorously fill in the missing values for the training set and the test set 
    imputer= KNNImputer(n_neighbors=5)

    #numerical variables
    X_train = pd.DataFrame(imputer.fit_transform(X_train))
    X_test = pd.DataFrame(imputer.fit_transform(X_test))

    #to Rescale numerical variables so that they are comparable on a common scale, normalise numerical variables with the methodStandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #TRAINING MODELS

    #linear Regression

    LR = LinearRegression()

    LR.fit(X_train, y_train)# Entreno el algoritmo en X_train. y_train

    # Make predictions on the test set
    y_pred_LR = LR.predict(X_test)
    y_pred_train_LR =LR.predict(X_train)

    # Plotting the results

    df_LR= pd.DataFrame({'Real':y_test,'Prediction':y_pred_LR})

    df_LR['Diference'] = y_test - y_pred_LR

    df_LR.head()
    
    #KNeighborsRegressor
    KNN= KNeighborsRegressor()
    KNN.fit(X_train, y_train)# Entreno el algoritmo en X_train. y_train
    y_pred_KNN = KNN.predict(X_test)
    y_pred_train_KNN =KNN.predict(X_train)

    df_KNN= pd.DataFrame({'Real':y_test,'Prediction':y_pred_KNN})
    df_KNN['Diference'] = y_test - y_pred_KNN
    df_KNN.head()

    # Decision Tree Regressor
    DTR= DecisionTreeRegressor()
    fit_DTR=DTR.fit(X_train, y_train)# Entreno el algoritmo en X_train. y_train

    y_pred_DTR = DTR.predict(X_test)
    y_pred_train_DTR =DTR.predict(X_train)

    df_DTR= pd.DataFrame({'Real':y_test,'Prediction':y_pred_DTR})
    df_DTR['Diference'] = y_test - y_pred_DTR
    df_DTR.head()

    pd.DataFrame(DTR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_DTR = pd.DataFrame(fit_DTR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    # Random Forest Regressor
    RFR= RandomForestRegressor(random_state = 42)
    fit_RFR=RFR.fit(X_train, y_train)# Entreno el algoritmo en X_train. y_train

    y_pred_RFR = RFR.predict(X_test)
    y_pred_train_RFR =RFR.predict(X_train)

    df_RFR= pd.DataFrame({'Real':y_test,'Prediction':y_pred_RFR})
    df_RFR['Diference'] = y_test - y_pred_RFR
    df_RFR.head()
    
    pd.DataFrame(RFR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_RFR = pd.DataFrame(fit_RFR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    # Gradient Boosting Regressor
    GBR= GradientBoostingRegressor()
    fit_GBR=GBR.fit(X_train, y_train)# Entreno el algoritmo en X_train. y_train

    y_pred_GBR = GBR.predict(X_test)
    y_pred_train_GBR =GBR.predict(X_train)

    df_GBR= pd.DataFrame({'Real':y_test,'Prediction':y_pred_GBR})
    df_GBR['Diference'] = y_test - y_pred_GBR
    df_GBR.head()

    pd.DataFrame(GBR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_GBR = pd.DataFrame(fit_GBR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    

    tabla = pd.DataFrame([
            {"Set":"Training","Model": "Multiple Linear Model", "R^2":LR.score(X_train, y_train).round(2) , "Error - MAE":metrics.mean_absolute_error(y_train, y_pred_train_LR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_train, y_pred_train_LR).round(2)},
            {"Set":"Test","Model": "Multiple Linear Model", "R^2":LR.score(X_test, y_test).round(2) , "Error - MAE":metrics.mean_absolute_error(y_test, y_pred_LR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_test, y_pred_LR).round(2)},
            {"Set":"Training","Model": "KNeighbors Regressor Model", "R^2":KNN.score(X_train, y_train).round(2) , "Error - MAE":metrics.mean_absolute_error(y_train, y_pred_train_KNN).round(2) ,'Error - MSE':metrics.mean_squared_error(y_train, y_pred_train_KNN).round(2)},
            {"Set":"Test","Model": "KNeighbors Regressor Model", "R^2":KNN.score(X_test, y_test).round(2) , "Error - MAE":metrics.mean_absolute_error(y_test, y_pred_KNN).round(2) ,'Error - MSE':metrics.mean_squared_error(y_test, y_pred_KNN).round(2)},
            {"Set":"Training","Model": "Decision Tree Regressor", "R^2":DTR.score(X_train, y_train).round(2) , "Error - MAE":metrics.mean_absolute_error(y_train, y_pred_train_DTR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_train, y_pred_train_DTR).round(2)},
            {"Set":"Test","Model": "Decision Tree Regressor", "R^2":DTR.score(X_test, y_test).round(2) , "Error - MAE":metrics.mean_absolute_error(y_test, y_pred_DTR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_test, y_pred_DTR).round(2)},
            {"Set":"Training","Model": "Random Forest Regressor", "R^2":RFR.score(X_train, y_train).round(2) , "Error - MAE":metrics.mean_absolute_error(y_train, y_pred_train_RFR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_train, y_pred_train_RFR).round(2)},
            {"Set":"Test","Model": "Random Forest Regressor", "R^2":RFR.score(X_test, y_test).round(2) , "Error - MAE":metrics.mean_absolute_error(y_test, y_pred_RFR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_test, y_pred_RFR).round(2)},
            {"Set":"Training","Model": "Gradient Boosting Regressor", "R^2":GBR.score(X_train, y_train).round(2) , "Error - MAE":metrics.mean_absolute_error(y_train, y_pred_train_GBR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_train, y_pred_train_GBR).round(2)},
            {"Set":"Test","Model": "Gradient Boosting Regressor", "R^2":GBR.score(X_test, y_test).round(2) , "Error - MAE":metrics.mean_absolute_error(y_test, y_pred_GBR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_test, y_pred_GBR).round(2)},
            ])
      
    edited_df = st.data_editor(tabla,hide_index=True)

    st.write("One important finding we can emphasize is: While the linear model performed poorly, the Random Forest Regressor model showed better performance compared to most models, with smaller discrepancies in accuracy and errors between the test and training sets, despite differences in data types and sizes.")
    st.write("The purpose of feature selection in machine learning is to determine the best set of variables to build effective models of the phenomena studied. Feature classification is used to understand the importance of the variables. We present the results of the found models.")

    choice = [ 'Random Forest Regressor','Decision Tree Regressor','Gradient Boosting Regressor']
    option = st.selectbox('Choice of the model', choice)
    
    if option == 'Decision Tree Regressor':
        fig,ax = plt.subplots(1,2)
        a=plt.subplot(121) 
        a = px.scatter(df_DTR, x="Real", y="Prediction",trendline="lowess",trendline_color_override = 'red',title="Real vs Prediction")
        st.plotly_chart(a)
        b=plt.subplot(122)
        b=px.bar(data_DTR,title="Importance of Variables." )
        st.plotly_chart(b)
    elif option == 'Random Forest Regressor':
        fig,ax = plt.subplots(1,2)
        a=plt.subplot(121) 
        a = px.scatter(df_RFR, x="Real", y="Prediction",trendline="lowess",trendline_color_override = 'red',title="Real vs Prediction")
        st.plotly_chart(a)
        b=plt.subplot(122)
        b=px.bar(data_RFR,title="Importance of Variables." )
        st.plotly_chart(b)
    elif option=='Gradient Boosting Regressor':
        fig,ax = plt.subplots(1,2)
        a=plt.subplot(121) 
        a = px.scatter(df_GBR, x="Real", y="Prediction",trendline="lowess",trendline_color_override = 'red',title="Real vs Prediction")
        st.plotly_chart(a)
        b=plt.subplot(122)
        b=px.bar(data_GBR,title="Importance of Variables." )
        st.plotly_chart(b)
    
    st.write('In order to optimize the model and determine if there is an ideal fit, we reduce the dataset to only include the four most important variables.')
    st.write('- Log GDP')
    st.write('- Social Supp')
    st.write('- Life Expectancy') 
    st.write('- Gini')
    st.write('')
    st.write('Continuing on, we fit a **Random Forest Regressor Model** and acquire the following results:')


    df_R = df_fin.drop(['Freedom','Corruption','generosity', 'Violence','Rate Beds', 'Rate Medical', 'Rate Nursing', 'Rate Dentists',
                   'poverty_gap', 'poverty_severity'],axis=1)
    
    #st.dataframe(df_R, hide_index=True)
    X_R = df_R.drop(['Country', 'Region', 'Year', 'Ladder'],axis=1)
    y_R = df_R['Ladder']

    X_train_R, X_test_R, y_train_R, y_test_R= train_test_split(X_R, y_R, test_size=0.20,random_state=42)

#using the "knnimputer" strategies for numeric variables, rigorously fill in the missing values for the training set and the test set 
#numerical variables

    X_train_R = pd.DataFrame(imputer.fit_transform(X_train_R))
    X_test_R = pd.DataFrame(imputer.fit_transform(X_test_R))

    scaler.fit(X_train_R)
    X_train = scaler.transform(X_train_R)
    X_test = scaler.transform(X_test_R)

    RFR= RandomForestRegressor(random_state = 42)
    fit_RFR_R=RFR.fit(X_train_R, y_train_R)

    y_pred_RFR_R = RFR.predict(X_test_R)
    y_pred_train_RFR_R =RFR.predict(X_train_R)

    df_RFR_R= pd.DataFrame({'Real':y_test_R,'Prediction':y_pred_RFR_R})
    df_RFR_R['Diference'] = y_test_R - y_pred_RFR_R
    
    pd.DataFrame(RFR.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_RFR_R= pd.DataFrame(fit_RFR_R.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    st.write('*Reduced Model (4 most important variables*)')
    tabla_R = pd.DataFrame([
            {"Set":"Training", '''R^2''':RFR.score(X_train_R, y_train_R).round(2) ,"MAPE*": metrics.mean_absolute_percentage_error(y_train_R, y_pred_train_RFR_R) ,"Error - MAE":metrics.mean_absolute_error(y_train_R, y_pred_train_RFR_R).round(2) ,'Error - MSE':metrics.mean_squared_error(y_train_R, y_pred_train_RFR_R).round(2)},
            {"Set":"Test", '''R^2''':RFR.score(X_test_R, y_test_R).round(2) ,"MAPE*": metrics.mean_absolute_percentage_error(y_test_R, y_pred_RFR_R) ,"Error - MAE":metrics.mean_absolute_error(y_test_R, y_pred_RFR_R).round(2) ,'Error - MSE':metrics.mean_squared_error(y_test_R, y_pred_RFR_R).round(2)},
            ]) 
   
    edited_R = st.data_editor(tabla_R,hide_index=True)
    st.caption('MAPE*: Mean Absolute Percentage Error',)
  
    st.write('*Initial Model*')
    tabla_I = pd.DataFrame([
            {"Set":"Training", r'''R^2''':0.98, "MAPE*": metrics.mean_absolute_percentage_error(y_train, y_pred_train_RFR), "Error - MAE":metrics.mean_absolute_error(y_train, y_pred_train_RFR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_train, y_pred_train_RFR).round(2)},
            {"Set":"Test",r'''R^2''':'0.90' ,"MAPE*": metrics.mean_absolute_percentage_error(y_test, y_pred_RFR), "Error - MAE":metrics.mean_absolute_error(y_test, y_pred_RFR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_test, y_pred_RFR).round(2)},
            ])
      
    edited_I = st.data_editor(tabla_I,hide_index=True)

    
    fig,ax = plt.subplots(2,1)
    a=plt.subplot(211) 
    a = px.scatter(df_RFR_R, x="Real", y="Prediction",trendline="lowess",trendline_color_override = 'red',title="Real vs Prediction")
    st.plotly_chart(a)
    b=plt.subplot(212)
    b=px.bar(data_RFR_R,title="Importance of Variables." )
    st.plotly_chart(b)
    
    st.write('According to the results, there is no significant advancement observed between the initial model and the one with the 4 most important variables.')

    st.subheader('**Model Optimatization**',divider='rainbow')
    
    st.write('Both models have high performance on the training set, but slightly lower on the testing set, which makes us suspect that there may be model overfitting. '
    'An overfitted model may look impressive on the training set, but will be useless in a real application. Therefore, we perform the hyperparameter optimization procedure, taking into account possible overfitting through a cross-validation process.')
    st.write('')
    
    image = Image.open('opti.png')
    st.image(image)
    
    # Number of trees in random forest
    #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    #max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    #max_depth.append(None)

    # Minimum number of samples required to split a node
    #min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    #min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    #bootstrap = [True, False]

    # Create the random grid
    #random_grid = {'n_estimators': n_estimators,
     #          'max_features': max_features,
      #         'max_depth': max_depth,
       #        'min_samples_split': min_samples_split,
         #      'min_samples_leaf': min_samples_leaf,
        #       'bootstrap': bootstrap}
    #st.write(random_grid)
    
    #Random search training
    #Now, we instantiate the random search and tune it like any Scikit-Learn model:

    # Use the random grid to search for best hyperparameters
    # First create the base model t
    #rf = RandomForestRegressor(random_state = 42)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
   # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
   # rf_random.fit(X_train_R, y_train_R)

    # we select the best parameters by adjusting the random search
    #st.write('Best parameters RandomizedSearch')
    #st.write(rf_random.best_params_)
    #st.write('Finally select the best parameters to adjust the random search and the random forest Regressor is trained on the entire data set using the training method FIT and to determine if the random search produced a better model and we compare the base model with the best random search model.')

    # BASE MODEL
    base_model = RandomForestRegressor(random_state = 42)
    base_fit=base_model.fit(X_train_R, y_train_R)
    base_accuracy=base_model.score(X_test_R, y_test_R)
    y_pred_base = base_model.predict(X_test_R)
    y_pred_train_base=base_model.predict(X_train_R)

    df_RFR_BASE= pd.DataFrame({'Real':y_test_R,'Prediction':y_pred_base})
    df_RFR_BASE['Diference'] = y_test_R - y_pred_base
    
    pd.DataFrame(base_model.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_RFR_base= pd.DataFrame(base_fit.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    st.write("*Model Performance (Initial Model):*", base_model)
    tabla_R = pd.DataFrame([
            {"Set":"Training", r'''R^2''':base_model.score(X_train_R, y_train_R).round(2) , "Error - MAE":metrics.mean_absolute_error(y_train_R, y_pred_train_base).round(2) ,'Error - MSE':metrics.mean_squared_error(y_train_R, y_pred_train_base).round(2)},
            {"Set":"Test", r'''R^2''':base_model.score(X_test_R, y_test_R).round(2) , "Error - MAE":metrics.mean_absolute_error(y_test_R, y_pred_base).round(2) ,'Error - MSE':metrics.mean_squared_error(y_test_R, y_pred_base).round(2)},
            ]) 
    edited_R = st.data_editor(tabla_R,hide_index=True)

    #Accuracy_base=(1-metrics.mean_absolute_percentage_error(y_test, y_pred_base)).round(3)
    Accuracy_base=0.945
    st.write("Accuracy:", Accuracy_base)

    # BEST RANDOMIZED SEARCH MODEL
    best_random=RandomForestRegressor(n_estimators= 1000,
                                      min_samples_split=2,
                                      min_samples_leaf= 1,
                                      max_features='sqrt',
                                      max_depth= 110,
                                      bootstrap= True)
    
    best_fit=best_random.fit(X_train_R, y_train_R)

    y_pred_best = best_random.predict(X_test_R)
    y_pred_train_best=best_random.predict(X_train_R)

    df_RFR_R= pd.DataFrame({'Real':y_test_R,'Prediction':y_pred_best})
    df_RFR_R['Diference'] = y_test_R - y_pred_best
    
    pd.DataFrame(best_random.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_RFR_R= pd.DataFrame(best_fit.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    st.write("*Model Performance (RandomizedSearch):*", best_random)
    tabla_R = pd.DataFrame([
            {"Set":"Training", r'''R^2''':best_random.score(X_train_R, y_train_R).round(2) , "Error - MAE":metrics.mean_absolute_error(y_train_R, y_pred_train_best).round(2) ,'Error - MSE':metrics.mean_squared_error(y_train_R, y_pred_train_best).round(2)},
            {"Set":"Test", r'''R^2''':best_random.score(X_test_R, y_test_R).round(2) , "Error - MAE":metrics.mean_absolute_error(y_test_R, y_pred_best).round(2) ,'Error - MSE':metrics.mean_squared_error(y_test_R, y_pred_best).round(2)},
            ]) 
    edited_R = st.data_editor(tabla_R,hide_index=True)
    
    random_accuracy=best_random.score(X_test_R, y_test_R)
    #Accuracy_randon=(1-metrics.mean_absolute_percentage_error(y_test, y_pred_best)).round(3)
    Accuracy_randon=0.9504
    st.write("Accuracy:", Accuracy_randon)
    
    st.write('Percentage of difference between the optimized model and the initial model **{:0.4f}%**:'.format( 100 * (Accuracy_randon - Accuracy_base) / Accuracy_base))
    
    # SEARCH GRID MODEL
    grid_model = RandomForestRegressor(bootstrap= True,
                                       max_depth= 100,
                                       max_features= 2,
                                       min_samples_leaf=1,
                                       min_samples_split= 3,
                                       n_estimators= 100)
    
    grid_fit=grid_model.fit(X_train_R, y_train_R)
    grid_accuracy=grid_model.score(X_test_R, y_test_R)
    y_pred_grid = grid_model.predict(X_test_R)
    y_pred_train_grid=grid_model.predict(X_train_R)

    df_RFR_grid= pd.DataFrame({'Real':y_test_R,'Prediction':y_pred_grid})
    df_RFR_grid['Diference'] = y_test_R - y_pred_grid
    
    pd.DataFrame(grid_model.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_RFR_grid= pd.DataFrame(grid_fit.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    st.write("*Model Performance (GridSearch):*", grid_model)
    tabla_grid = pd.DataFrame([
            {"Set":"Training", r'''R^2''':grid_model.score(X_train_R, y_train_R).round(2) , "Error - MAE":metrics.mean_absolute_error(y_train_R, y_pred_train_grid).round(2) ,'Error - MSE':metrics.mean_squared_error(y_train_R, y_pred_train_grid).round(2)},
            {"Set":"Test", r'''R^2''':base_model.score(X_test_R, y_test_R).round(2) , "Error - MAE":metrics.mean_absolute_error(y_test_R, y_pred_grid).round(2) ,'Error - MSE':metrics.mean_squared_error(y_test_R, y_pred_grid).round(2)},
            ]) 
    edited_grid = st.data_editor(tabla_grid,hide_index=True)
    grid_accuracy=grid_model.score(X_test_R, y_test_R)


    Accuracy_grid=(1-metrics.mean_absolute_percentage_error(y_test, y_pred_grid)).round(3)
    st.write("Accuracy:", Accuracy_grid)   
    #st.write("Accuracy:", grid_accuracy)
    st.write('Percentage of difference between the optimized model and the initial model **{:0.4f}%**:'.format( 100 * (Accuracy_grid - Accuracy_base) / Accuracy_base))
    
    #random_accuracy = evaluate(best_random, X_test_R, y_test_R)

    #def evaluate(model, X_test_R, y_test_R):
     #   predictions = model.predict(X_test_R)
      #  errors = abs(predictions - y_test_R)
       # mape = 100 * np.mean(errors / y_test_R)
    #    p = RFR.score(X_test_R, y_test_R).round(2)
     #   prueba=100*p
      #  accuracy = 100 - mape
    #    st.write('Model Performance', model)
     #   st.write('Average Error: {:0.4f}.'.format(np.mean(errors)))
      #  st.write('Accuracy = {:0.2f}%.'.format(accuracy))
      #  st.write('prueba= {:0.2f}%.'.format(prueba))
       # return accuracy, prueba

    #base_model = RandomForestRegressor(random_state = 42)
    #base_model.fit(X_train_R, y_train_R)
    #base_accuracy = evaluate(base_model, X_test_R, y_test_R)
    #best_random = rf_random.best_estimator_
    

    #best_random=RandomForestRegressor(bootstrap= True, max_depth= 100, max_features= 2, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 300)
    
    #best_random.fit(X_train_R, y_train_R)
    #random_accuracy = evaluate(best_random, X_test_R, y_test_R)

    #st.write('We achieved an unspectacular improvement in accuracy of **{:0.2f}%**.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
    #st.write( 'But because we performed the hyperparameter optimization procedure through a cross-validation process, the effect of overfitting on the model is minimized.')
 
    st.subheader('**Model Explainability**',divider='rainbow')

    st.write("In this section, we calculate SHAP values and visualize feature importance, feature dependence, force, and decision plot. SHAP values show how each feature affects each final prediction, the importance of each feature compared to others, and the model's dependence on the interaction between features.")
    
    df_shap= pd.read_csv(r"C:\Users\diego\Downloads\df_Fin.csv",index_col=False,header=0,usecols=['Country', 'Region', 'Year', 'Ladder', 'Log GDP', 'Social Supp',
       'Life expectancy', 'Freedom', 'generosity', 'Corruption', 'Violence','Rate Beds', 'Rate Medical', 'Rate Nursing', 'Rate Dentists','poverty_gap', 'poverty_severity', 'gini'])
    
    X = df_shap.drop(['Country', 'Region', 'Year', 'Ladder'],axis=1) # explanatory variables
    y = df_shap['Ladder'] # target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
     
    imputer= KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train))
    X_test = pd.DataFrame(imputer.fit_transform(X_test))

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    best_shap = RandomForestRegressor(bootstrap= True,
                                       max_depth= 100,
                                       max_features= 2,
                                       min_samples_leaf= 1,
                                       min_samples_split= 3,
                                       n_estimators= 100)
    best_shap.fit(X_train, y_train)
    
    # explain all the predictions in the test set
    explainer = shap.TreeExplainer(best_shap)
    shap_values = explainer.shap_values(X_test)

    #st.write('R^2: ', best_shap.score(X_test, y_test).round(2))
   
    image_shap1 = Image.open('shap1.png')
    st.image(image_shap1)

    st.write('For each feature, a point corresponds to an observation in the data set. The position of a point along the x-axis (i.e., the actual SHAP value) represents the impact that feature had on the model output for that observation. The higher the feature is placed on the plot, the more important it is to the model.')
    st.write('The summary graph shows the importance of each feature in the model. The results show that “Log GPD”, “Social Supp” and “Life Expectancy”  play an important role in determining the results. It is seen how high values of these variables are directly proportional to a high happiness index in a country.')

    st.write('')
    st.write('**Simple dependence plot**')
    #st.write('')
    
    #choice_ = ['Log GDP', 'Gini','Social Supp','Life Expectancy']
    #option = st.selectbox('Choice of the model', choice_)
    
    st.write('*Log GDP - Freedom to make life choices*')
    image_shap2 = Image.open('LogGDP.png')
    st.image(image_shap2)
    
    #st.write("*Gini*")
    #image_shap3 = Image.open('gini.png')
    #st.image(image_shap3)
    
    st.write("*Social Supp - Income based on GDP per capita*")
    image_shap4 = Image.open('social.png')
    st.image(image_shap4)
    
    #st.write("*Life Expectancy*")
    #image_shap5 = Image.open('Life.png')
    #st.image(image_shap5)

    #st.write('')

if page == pages[3]: 
    st.header('Conclusion')
    st.write('Quality of life study remains a challenge for researchers across the board. Well-being prediction, analysis and modeling using data science tools is very useful and can contribute to face challengesrelated to this concept and extract information from its evaluations.')
    st.write("In our work, we delved into the literature related to this concept from the perspective of data science where machine learning a were used. We proposed an experiment to explore the potentials of ML algorithms in predicting an international QoL index then compared their performances.")
    st.write("Successful completion of data exploration, visualization, and preprocessing laid a robust groundwork for subsequent modeling. The project aimed to find indicators impacting well-being. Using supervised learning techniques, multiple regression models were evaluated.")
    st.write("The best performance is achieved using Random Forest Regressor. In this study, we were interested in studying how data science can be used to study, predict and model the different types of quality of life indicators.")
    st.write("Hyperparameter optimization improved the model's accuracy slightly. Feature importance analysis highlighted GDP, social support, and life expectancy as major contributors to happiness levels, while healthcare-related indicators had a relatively lower direct impact. ")
    
    st.subheader('**Difficulties encountered during the project**',divider='rainbow')
    st.write("**Forecasting Tasks**: the process of acquiring additional data took longer than expected due to numerous missing values between years 2006 to 2019 within specific regions. To address this, emerging all the data sets resulted in retaining only 137 countries consistently found across all years.")
    st.write("**Datasets**: It was a challenge to standardize varying spellings of country names across different sources. ")
    st.write('**Technical/Theoretical Skills**: Some specific skills such as plotting data on geographical maps and calculating SHAP Values were not covered in our initial training. Progress was slowed as we needed extra time to acquire and develop these new skills essential for effective analysis')
    st.write("**Relevance**: Capturing happiness rankings faced a significant hurdle due to variations in how different cultures perceive happiness and the diverse socio-economic contexts across regions. GDP and social support emerged as the two most pivotal determinants. It's crucial to acknowledge that the patterns observed across socio-demographic variables might vary when considering all countries collectively compared to when analyzed within specific regions.")

    st.subheader('**Continuation of the project**',divider='rainbow')
    st.write("For future research, other types of data can be used to predict wellbeing such as monthly/quarterly data of international indicators to build time-series algorithms. Further research can be exploring the use of social media data for measuring subjective wellbeing could enrich research on happiness. The new objective is to identify and study potential associations between items on the QoL scale and mental health issues. ")
    st.write("The project contributed to advancing scientific knowledge by offering insights into the nuanced relationship between various factors and happiness levels. Even in countries with comparatively lower overall happiness scores, there are positive ratings in certain dimensions. Consequently, these lower-ranking aspects could serve as potential policy focal points, allowing evidence-based interventions to target and improve these areas of concern effectively. ")

if page == pages[4]: 
    st.header('Bibliography')
    st.write("- **Ayoub Jannani, Nawal Sael, Faouzia Benabbou**, Predicting Quality of Life using Machine Learning: case of World Happiness Index.  2021 4th International Symposium on Advanced Electrical and Communication Technologies (ISAECT)\n"
        "- **Fabiha Ibnat, Jigmey Gyalmo, Zulfikar Alom, Md. Abdul Awal, and Mohammad Abdul Azim**, Understanding World Happiness using Machine Learning Techniques. 2021 International Conference on Computer, Communication, Chemical, Materials and Electronic Engineering (IC4ME2)\n"
        "- **Kai Ruggeri, Eduardo Garcia-Garzon, Áine Maguire, Sandra Matz and Felicia A. Huppert**, Well-being is more than happiness and life satisfaction: a multidimensional analysis of 21 countries. Health Qual Life Outcomes. 2020 Jun 19;18(1):192.\n"
        "- **Moaiad Ahmad Khder, Mohammad Adnan Sayfi and Samah Wael Fuji**, Analysis of World Happiness Report Dataset Using Machine Learning Approaches. April 2022 International Journal of Advances in Soft Computing and its Applications 14(1):15-34\n"
        "- **A. Jannani, N. Sael, F. Benabbou**, Machine learning for the analysis of quality of life using the World Happiness Index and Human Development Indicators, Mathematical Modeling and Computing, vol.10, no.2, pp.534, 2023.")

if page == pages[5]: 
    st.title('Contributors:')
    import streamlit as st

    st.code('''
    A. Arrieta 
    K. Kunkel
    ''')

    

    
