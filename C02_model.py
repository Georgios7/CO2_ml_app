
import pandas as pd
from pycaret.regression import *

data = pd.read_csv('CO2 Emissions_Canada.csv')
data = data[["Vehicle Class","Engine Size(L)", "Cylinders","Transmission","Fuel Type",
"Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)","Fuel Consumption Comb (L/100 km)" ,"CO2 Emissions(g/km)"]]

models_setup = setup(data, target = "CO2 Emissions(g/km)", session_id = 123)

# Random forrest reggressor
rf = create_model('rf')
#tuned_rf =  tune_model(rf)

save_model(rf,model_name = 'deployment_CO2')