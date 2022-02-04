
import pandas as pd
from pycaret.regression import *

data = pd.read_csv('CO2 Emissions_Canada.csv')
data = data[["Vehicle Class", "Cylinders","Transmission","Fuel Type",
"Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)", "CO2 Emissions(g/km)"]]

models_setup = setup(data, target = "CO2 Emissions(g/km)", session_id = 123)

# Gradient boosting reggressor
gbr = create_model('gbr')
tuned_gbr =  tune_model(gbr)

save_model(tuned_gbr,model_name = 'deployment_CO2')