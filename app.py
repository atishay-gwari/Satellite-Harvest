import streamlit as st
import pandas as pd
import joblib
from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
from functools import reduce
import tempfile
import sklearn
from sklearn.preprocessing import StandardScaler

def unpack(path, lat_location, lon_location):
    # data = Dataset(path, 'r')
    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as temp_file:
            temp_file.write(path.read())

        # Open the temporary file using netCDF4
        data = Dataset(temp_file.name, 'r')
    except Exception as e:
        print(f"An error occurred while opening the file: {str(e)}")
        return None  # Return None to indicate failure
    variable_name = list(data.variables.keys())[-1]
    # Storing the lat and lon data into the variables 
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]


    # Squared difference of lat and lon 
    sq_diff_lat = (lat - lat_location)**2
    sq_diff_lon = (lon - lon_location)**2

    # Identifying the index of the minimum value for lat and lon 
    min_index_lat = sq_diff_lat.argmin()
    min_index_lon = sq_diff_lon.argmin()

    feature = data.variables[variable_name]

    days = data.variables['day']
    start_date = datetime(1900, 1, 1)  # Start date in the 1900 system
    dates = [start_date + timedelta(days=int(day)) for day in days]


    df = pd.DataFrame(columns=['Date', variable_name])
    df['Date'] = dates
  

    dt = np.arange(0, data.variables['day'].size)
    for time_index in dt:
        # Use numpy.ma.getdata to get unmasked values
        feature_values = feature[time_index, min_index_lat, min_index_lon]
        
        # Now, you can assign the unmasked values to the 'Temperature' column
        df.at[time_index, variable_name] = feature_values

    return df

def preprocess_and_predict(user_state, path_dict, model_choice):

    result = states[states['state'] == user_state]
    lat = result['Latitude'].values[0]
    lon = result['Longitude'].values[0]

    dfs =[]

    for variable,filepath in path_dict.items():
        df = unpack(filepath,lat,lon)
        # print(df)
        df.rename(columns={df.columns[1]: variable}, inplace=True)
        dfs.append(df)
    def merge_dataframes(df1, df2):
        return pd.merge(df1, df2, on="Date", how="outer") 
    combined_df = reduce(merge_dataframes, dfs)

    combined_df['Year'] = combined_df['Date'].dt.year
    combined_df['Week'] = (combined_df['Date'].dt.strftime('%W').astype(int) + 1).astype(str)
    df_grouped =combined_df.groupby(['Year', 'Week']).agg({'min_humidity': 'mean', 'max_humidity': 'mean','min_temp': 'mean', 'max_temp': 'mean','vapor_pressure_deficit': 'mean', 'near_surface_specific_humidity': 'mean','precipitation': 'mean', 'solar_radiation': 'mean','wind_speed':'mean'}).reset_index()

    pivot_table = df_grouped.pivot_table(index=['Year'], columns='Week', values=['min_humidity', 'max_humidity','min_temp','max_temp','vapor_pressure_deficit','near_surface_specific_humidity','precipitation','solar_radiation','wind_speed']).reset_index()
    pivot_table.columns = [' '.join(col).strip() for col in pivot_table.columns.values]


    variables = ['min_humidity', 'max_humidity', 'min_temp', 'max_temp', 'vapor_pressure_deficit', 
             'near_surface_specific_humidity', 'precipitation', 'solar_radiation', 'wind_speed']

    columns_to_drop = []

    for var in variables:
        columns_to_drop.extend(['{} {}'.format(var, i) for i in range(1, 22)])
        columns_to_drop.append(var+' 53')
    # Drop the columns from the dataframe
    final_df = pivot_table.drop(columns=columns_to_drop)
    features = final_df.drop(columns=['Year'],axis=1)

    # scaler_soy = StandardScaler()
    # X_soy_scaled = scaler_soy.fit_transform(final_df)
    # X_soy_scaled = pd.DataFrame(X_soy_scaled, columns=final_df.columns)
    # # features = final_df.drop(columns=['Year'],axis=1)
    # features = X_soy_scaled.values
    
    features = features.values
    if model_choice=='Corn':
        model = joblib.load('./models/CatBoostRegressorCorn_best_model.joblib')
    if model_choice=='Soya':
        model = joblib.load('./models/CatBoostRegressorsoy_best_model.joblib')
    
    predicted_yield = model.predict(features)
    print(f"Predicted corn yield: {predicted_yield[0]}")

    return predicted_yield


st.set_page_config(layout="wide")



st.markdown("<h1 style='text-align: center;'>Crop Yield Prediction	🧑‍🌾</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Welcome to our Crop Yield Prediction Tool! 🌾</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Our project is designed to help you, whether you're a farmer or a crop enthusiast, make informed decisions about your crops. It's all about predicting how much you can expect to harvest based on the weather conditions in your area.</p>", unsafe_allow_html=True)

# Assuming your preprocess_and_predict function is in a separate file and is imported as shown above

# List of available states
states = pd.read_csv("./csv/state_coordinates.csv")

# Crop choices
crop_choices = ['Corn', 'Soya']

col1, col2 = st.columns(2) 


# Upload 9 NetCDF files
def left_column():
    st.header("Upload NetCDF Files")
    
    max_temp = st.file_uploader("Max Temperature (Ex: tmmx_2020.nc)", type=["nc"])
    min_temp = st.file_uploader("Min Temperature (Ex: tmmn_2020.nc)", type=["nc"])
    precipitation = st.file_uploader("Precipitation (Ex: pr_2020.nc)", type=["nc"])
    max_humidity = st.file_uploader("Max Humidity (Ex: rmax_2020.nc)", type=["nc"])
    min_humidity = st.file_uploader("Min Humidity (Ex: rmin_2020.nc)", type=["nc"])
    near_surface_specific_humidity = st.file_uploader("Near Surface Specific Humidity (Ex: sph_2020.nc)", type=["nc"])
    vapor_pressure_deficit = st.file_uploader("Vapor Pressure Deficit (Ex: vpd_2020.nc)", type=["nc"])
    solar_radiation = st.file_uploader("Solar Radiation (Ex: srad_2020.nc)", type=["nc"])
    wind_speed = st.file_uploader("Wind Speed (Ex: vs_2020.nc)", type=["nc"])

    
    

    

    # Select state
    selected_state = st.selectbox("Select State", states['state'])
    
    # Select crop
    selected_crop = st.selectbox("Select Crop", crop_choices)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
        # Preprocess and predict crop yield
        
                # Use the uploaded files
            path_dict = {
                "max_temp": max_temp,
                "min_temp": min_temp,
                "precipitation": precipitation,
                "max_humidity": max_humidity,
                "min_humidity": min_humidity,
                "near_surface_specific_humidity": near_surface_specific_humidity,
                "vapor_pressure_deficit": vapor_pressure_deficit,
                "solar_radiation": solar_radiation,
                "wind_speed": wind_speed
            }
                # Call the prediction function
            if max_temp and min_temp and precipitation and max_humidity and min_humidity and near_surface_specific_humidity and vapor_pressure_deficit and solar_radiation and wind_speed:
                prediction = preprocess_and_predict(selected_state, path_dict, selected_crop)
                st.markdown(f'<p style="font-size:20px; font-weight:light">Predicted <strong>{selected_crop}</strong> yield for {selected_state}: <strong>{prediction[0]}</strong> BU/Acres </p>', unsafe_allow_html=True)



# Right column to write about your project
with col1:
    st.markdown(
        """
## Here's How It Works:

##### Upload Weather Data: 
You start by uploading weather data files (NetCDF) that contain historical information about things like temperature, humidity, rainfall, and more. Think of these as treasure troves of weather details!

##### Pick Your State: 
Next, you select your state. Why? Because the weather can be quite different depending on where you are. We need to know where you're planting your crops.

##### Choose Your Crop: 
Are you growing corn or soybeans? Different crops react differently to the weather, so tell us what you're cultivating.

##### We Do the Magic: 
Once you've provided the data, state, and crop, we get to work. We crunch the numbers and use smart computer models to predict how much you're likely to harvest. It's like having a weather-savy crystal ball!

##### Get Your Prediction: 
After a few moments, we'll reveal your crop yield prediction. You'll know what to expect based on the historical weather patterns.

It's that simple! This tool can be a game-changer for farmers, helping you plan your planting and resources more effectively. And for crop enthusiasts, it's a fascinating way to explore the relationship between weather and agriculture.

So, go ahead, upload your data, make your selections, and let's predict your crop yield! 🌱🌞🌧️"""
    )
    st.image('./image/corn.jpg',use_column_width=True)

with col2:
    left_column()