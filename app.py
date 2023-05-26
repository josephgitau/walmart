# import libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# add page title
st.title("Walmart Weekly Sales Prediction App")

# add app description
st.markdown("This app predicts the **Weekly Sales**!")

# read the data
df = pd.read_csv('Walmart.csv')


# display the top 5 rows of the data
st.header("Basic Data Analysis")
st.subheader('Data Information:')
st.dataframe(df.head())

# show the summary statistics
st.subheader('Data Statistics:')
st.dataframe(df.describe())

# create a timeseries plot for Date vs Weekly_Sales
st.subheader('Weekly Sales Trend:')
st.line_chart(df['Weekly_Sales'])

# Machine Learning Prediction Section
st.header("Ml Model Prediction")

# data preprocessing
df['Date'][1].split('-')
# Extract day, month and year from the Date column using split function
df['Day'] = df['Date'].apply(lambda x : x.split('-')[0])
df['Month'] = df['Date'].apply(lambda x : x.split('-')[1])
df['Year'] = df['Date'].apply(lambda x : x.split('-')[2])
# save the columns as integer type
df['Day'] = df['Day'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Year'] = df['Year'].astype(int)
# drop the Date column
df.drop('Date', axis = 1, inplace = True)

# display the top 5 rows of the data again
st.subheader('ML Data Information:')
st.dataframe(df.head())

# create X, y 
X = df.drop('Weekly_Sales', axis = 1)
y = df['Weekly_Sales']

# fit the model
rf = RandomForestRegressor()
rf.fit(X, y)

# take input from user
def user_inputs(Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Day,Month, Year):
    Store = int(Store) 
    Holiday_Flag = int(Holiday_Flag)
    Temperature = float(Temperature)
    Fuel_Price = float(Fuel_Price)
    CPI = float(CPI)
    Unemployment = float(Unemployment)
    Day = int(Day)
    Month = int(Month)
    Year = int(Year)

    # store the inputs
    features = [Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Day,Month, Year]

    # Create a dataframe from the above features and predict the sales
    user_df = pd.DataFrame([features], columns = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Day','Month', 'Year'])
    return user_df

# take input from user
Store = st.sidebar.slider('Store', 1, 45)
Holiday_Flag = st.sidebar.slider('Holiday_Flag', 0, 1)
Temperature = st.sidebar.number_input('Temperature')
Fuel_Price = st.sidebar.number_input('Fuel_Price')
CPI = st.sidebar.number_input('CPI')
Unemployment = st.sidebar.number_input('Unemployment', 3, 15)
Day = st.sidebar.slider('Day', 1, 31)
Month = st.sidebar.slider('Month', 1, 12)
Year = st.sidebar.slider('Year', 2010, 2012)

# store the inputs
user_df = user_inputs(Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Day,Month, Year)

# display the user inputs
st.subheader('User Input Features:')
st.dataframe(user_df)

# make predictions and display the results
prediction = rf.predict(user_df)

st.subheader('Predicted Weekly Sales:')
st.write(prediction)