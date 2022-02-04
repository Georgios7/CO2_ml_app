# How to run: Open command propmpt and run the command: streamlit runn app.py

from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

# Load Model
model = load_model('deployment_CO2')

def run():

    from PIL import Image
    image_CO2 = Image.open('CO2.jpg')

    st.sidebar.info('This app is created using PyCaret and Streamlit')
    st.image(image_CO2)

    st.title('CO2 Car Emmissions Application')


    Vehicle_class = st.selectbox('Vehicle Class', ['COMPACT', 'SUV - SMALL', 'MID-SIZE', 'TWO-SEATER', 'MINICOMPACT',
       'SUBCOMPACT', 'FULL-SIZE', 'STATION WAGON - SMALL',
       'SUV - STANDARD', 'VAN - CARGO', 'VAN - PASSENGER',
       'PICKUP TRUCK - STANDARD', 'MINIVAN', 'SPECIAL PURPOSE VEHICLE',
       'STATION WAGON - MID-SIZE', 'PICKUP TRUCK - SMALL'])

    Transmition = st.selectbox('Transmission',['AS5', 'M6', 'AV7', 'AS6', 'AM6', 'A6', 'AM7', 'AV8', 'AS8', 'A7',
       'A8', 'M7', 'A4', 'M5', 'AV', 'A5', 'AS7', 'A9', 'AS9', 'AV6',
       'AS4', 'AM5', 'AM8', 'AM9', 'AS10', 'A10', 'AV10'])

    cylinder = st.selectbox('Cylinders', np.sort([ 4,  6, 12,  8, 10,  3,  5, 16]))

    Fuel_Type = st.selectbox('Fuel Type', ['Z', 'D', 'X', 'E', 'N'] )

    Fuel_city = st.number_input('Fuel Consumption City (L/100 km)', min_value = 4, max_value = 40, value = 10)
    Fuel_hw = st.number_input('Fuel Consumption Hwy (L/100 km)', min_value = 2, max_value = 30, value = 8)

    output = ""


# Input Dict
    input_dict = {'Vehicle Class': Vehicle_class, 'Transmission': Transmition, 'Cylinders': cylinder, 
    'Fuel Type': Fuel_Type, 'Fuel Consumption City (L/100 km)': Fuel_city, 'Fuel Consumption Hwy (L/100 km)': Fuel_hw}
    input_df = pd.DataFrame([input_dict])

    # Predict
    if st.button('Predict'):
        output = predict_model(model, data = input_df)
        output =  str(round(float(output['Label'][0]),2)) +  ' (g/km)'
        #str(output['Label'][0])

    # Display
    st.success('The CO2 amount is {}'.format(output))


if __name__ == '__main__':
    run()