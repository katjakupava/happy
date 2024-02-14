import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import plotly.express as px


st.sidebar.title("Navigation")
pages=["Home", "Exploration", "Modelling","Conclusion", "About"]
page=st.sidebar.radio("Go to", pages)
# Cover Page
if page == pages[0] : 
   
    st.title('World Happiness')
    image = Image.open(r"C:\Users\diego\OneDrive\Documents\streamlit - presentacion\Happy1.jpeg")
    st.image(image)
    st.header("Context")
    st.write("The exploration and categorization of factors influencing happiness is essential in understanding various aspects of mental well-being, such as emotional health and life satisfaction.")
    text = ("Qualitative analysis of happiness indicators is required for comprehensive research. However, the manual assessment of subjective well-being is **complex, time-consuming**, susceptible to **bias**, and demands **expertise** in the field. For navigation through diverse data sources, machine learning methodologies are employed to unravel the intricacies of what contributes to an individual's happiness.")
    st.markdown(text)
    

    st.header('Problematic')
    st.write("**Computer-assisted analysis** of happiness indicators and identification of influential factors provide crucial assistance to the policymakers. Traditionally, happiness were measured with questionnaires. This system may not **generalize** well, especially due to the variability in subjective well-being indicators and data acquisition systems in different countries.")
    image = Image.open(r"C:\Users\diego\OneDrive\Documents\streamlit - presentacion\Happiness_Test.png")
    st.image(image)
    st.header('Aim')
    st.write('The primary goal of this research is to create a **machine learning regressor** that can effectively categorize key indicators relevant to subjective well-being.')
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
    st.write("Africa displays the **lowest** average happiness scores, while Europe, North America, Australia, and New Zealand consistently exhibit high average scores. Notably, South America maintains a high distribution of self-reported life satisfaction, **despite having lower income levels**.")
    st.title("Relationship between the variables")
    image = Image.open('scatter plot.png')
    st.image(image)
    st.write("While economic indicators and social support demonstrate a close association with overall life satisfaction, we note that corruption levels and the availability of dental care may not significantly impact happiness perceptions. This suggests that the influence of these specific elements on subjective well-being is either **minimal or more complex** than initially assumed."
"Simultaneously, the analysis reveals that economic indicators like the Gini index and poverty indexes do not exhibit a strong correlation with a sense of well-being. **Despite being economic metrics**, their relationship with the overall happiness score appears to be less pronounced.")

if page == pages[2] : 
    st.header('Modelling')
if page == pages[3] : 
    st.header('Conclusion')
if page == pages[4] : 
    st.title('Contributors:')
    st.code('''A. Arrieta
K. Kunkel''')
    

    