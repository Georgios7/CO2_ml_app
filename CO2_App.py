# How to run: Open command propmpt and run the command: streamlit run app.py

from pycaret.regression import load_model, predict_model
from sqlalchemy import false
import streamlit as st
import pandas as pd
import numpy as np


# Load Model
model = load_model('deployment_CO2')

def run():

    # Use the full page instead of a narrow central column
    st.set_page_config( page_title='C02 car emmision app' , page_icon="🌳", layout="centered", 
    initial_sidebar_state="auto", menu_items=None)



    from PIL import Image
    image_CO2 = Image.open('CO2.jpg')
    image_pc = Image.open('pycaret.png')
    image_s = Image.open('streamlit.png')

    st.sidebar.info('This App is created using PyCaret and Streamlit')
    st.sidebar.image(image_s)
    st.sidebar.image(image_pc)

    st.image(image_CO2)
    st.title('CO2 Car Emmissions Application')

    # create dataframe for Transmission, Fuel Type
    df =pd.DataFrame(
    {'Transmission': ['A = Automatic', 'AM = Automated Manual', 'AS = Automated with select shift', 
    'AV = Continuously variable','M = Mannual', '3 - 10 = Number of gears'],
    'Fuel type': ['X = Regular gasoline', 'Z = Premium gasoline', 'D = Diesel',
    'E = Ethanol (E85)' ,'N = Natural gas','']})

    st.write(df)



    Vehicle_class = st.selectbox('Vehicle Class',np.sort(['COMPACT', 'SUV - SMALL', 'MID-SIZE', 'TWO-SEATER', 'MINICOMPACT',
       'SUBCOMPACT', 'FULL-SIZE', 'STATION WAGON - SMALL',
       'SUV - STANDARD', 'VAN - CARGO', 'VAN - PASSENGER',
       'PICKUP TRUCK - STANDARD', 'MINIVAN', 'SPECIAL PURPOSE VEHICLE',
       'STATION WAGON - MID-SIZE', 'PICKUP TRUCK - SMALL']))

    engine_size = st.slider('Engine Size(L)', 0.9,8.5,step=0.1, value=1.4 )


    Transmition = st.selectbox('Transmission',np.sort(['AS5', 'M6', 'AV7', 'AS6', 'AM6', 'A6', 'AM7', 'AV8', 'AS8', 'A7',
       'A8', 'M7', 'A4', 'M5', 'AV', 'A5', 'AS7', 'A9', 'AS9', 'AV6',
       'AS4', 'AM5', 'AM8', 'AM9', 'AS10', 'A10', 'AV10']) )

    #cylinder = st.selectbox('Cylinders', np.sort([ 4,  6, 12,  8, 10,  3,  5, 16]))
    cylinder = st.slider('Cylinders', 3,16,step=1, value=4 )

    Fuel_Type = st.selectbox('Fuel Type', np.sort(['Z', 'D', 'X', 'E', 'N']) )

    Fuel_city = st.number_input('Fuel Consumption City (L/100 km)', min_value = 4, max_value = 40, value = 8)
    
    
    Fuel_hw = st.number_input('Fuel Consumption Hwy (L/100 km)', min_value = 2, max_value = 30, value = 6)
    Fuel_comp = st.number_input('Fuel Consumption Comb (L/100 km)', min_value = 3, max_value = 35, value = 7)
    output = ""


# Input Dict
    input_dict = {'Vehicle Class': Vehicle_class, 'Engine Size(L)':engine_size ,'Transmission': Transmition, 'Cylinders': cylinder, 
    'Fuel Type': Fuel_Type, 'Fuel Consumption City (L/100 km)': Fuel_city, 'Fuel Consumption Hwy (L/100 km)': Fuel_hw , 'Fuel Consumption Comb (L/100 km)':Fuel_comp}
    input_df = pd.DataFrame([input_dict])

    # Predict
    # it runs without clicking the button
    #if st.button('Predict'):
    output = predict_model(model, data = input_df)
    output =  str(round(float(output['Label'][0]),2)) +  ' (g/km)'
        #str(output['Label'][0])

    # Display
    st.success('The CO2 amount is {}'.format(output))


if __name__ == '__main__':
    run()