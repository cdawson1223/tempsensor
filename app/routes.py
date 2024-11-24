from flask import Flask,render_template
import pandas as pd
from app import app
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error


@app.route('/')
@app.route('/index')
def index():
    actual = 72
    measured = 66
    humidity = 36
    ambient_light = 15000
    pressure = 30.43
    data_cols = handle_data('app/WeatherAppData.csv')
    other,a,b,c = linear_regression(data_cols[0], data_cols[1], data_cols[2], data_cols[3], data_cols[4])
    actual = other*measured + a*humidity + b*ambient_light + c*pressure
    return render_template('index.html', title='Home', actual = actual, measured = measured, humidity = humidity, ambient_light = ambient_light, pressure = pressure)
    #return "Hello, World!"

#read in csv, return numpy arrays for measured_data, humidity_data, light_data, pressure_data, actual_data
def handle_data(filename):
    df = pd.read_csv(filename)
    print("df is: ")
    #print(df)
    print(df.columns)
    actual_data = df['Actual Temperature ']
    measured_data = df['Feels Like Temperature ']
    humidity_data = df['Humidity? - %']
    pressure_data = df['Pressure ']
    light_data = df['ambient estimation using visibility (lux)']
    return measured_data, humidity_data, light_data, pressure_data, actual_data 

#input numpy arrays of measured_temp, humidity, ambient light, pressure, and **truth values as your y**
#create lin model and get params for 
# actual = measure + alpha*humidity + b * light + c*pressure
def linear_regression(measured_data, humidity_data, light_data, pressure_data, actual_data): #thisis currently shit 
    n_samples = 100
    x1 = measured_data
    x2 = humidity_data
    x3 = light_data
    x4 = pressure_data
    #x4 = np.random.rand(n_samples)

    # True parameters -> these we need to fine tune 
    a, b, c = 2.0, -1.5, 0.5

    # Calculate y based on the equation
    #y = x1 + a * x2 + b * x3 + c * x4 + np.random.randn(n_samples) * 0.1  # Add noise
    Y = actual_data
    # Combine features into a single array
    X = np.column_stack((x1, x2, x3, x4))

    # Split the data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Create and train the model
    model = LinearRegression()
    model.fit(X, Y)
    # def model(x, a, b, c):
    #     x1, x2, x3, x4 = x
    #     return x1 + a * x2 + b * x3 + c * x4
    # Make predictions
    #y_pred = model.predict(X_test)
    params = model.coef_
    print("params are: ", params)
    #params, covariance = curve_fit(model, X, y)

    # Evaluate the model
    #mse = mean_squared_error(y_test, y_pred)
    #print(f'Mean Squared Error: {mse}')

    # Inspect coefficients
    # print('Estimated coefficients:', model.coef_)
    # print('Intercept:', model.intercept_)
    return params 