from flask import Flask,render_template
import pandas as pd
from app import app
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import numpy as np
import os 
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error


@app.route('/')
@app.route('/index')
def index():
    #actual = 72
    measured = 66
    humidity = 36
    ambient_light = 15000
    pressure = 30.43
    data_cols = handle_data('app/WeatherAppData.csv')
    names_array = ["Brooklyn Apartment Window", "Brooklyn Apartment Radiator", "Brooklyn Apartment Hallway", "Brooklyn Apartment Living Room",
                   "Manhattan Apartment Freezer", "Manhattan Apartment Heater", "Manhattan Apartment Bedroom"]
    info_array = read_test_data()
    print("thru test data")
    temp1, intercept = linear_regression(data_cols[0], data_cols[1], data_cols[2], data_cols[3], data_cols[4])
    other,a,b,c = temp1
    outputed_array = []
    for index,item in enumerate(info_array):
        location = names_array[index]
        measured, humidity, pressure, ambient_light = item
        measured = measured*(9/5) + 32 
        pressure *= 0.0295301
        actual = other*measured + a*humidity + b*ambient_light + c*pressure + intercept
        outputed_array.append([location, round(measured,2), round(humidity,2), round(pressure,2), 
                                round(ambient_light,2), round(actual,2)])
    #print("before calc")
    #actual = other*measured + a*humidity + b*ambient_light + c*pressure + intercept
    #print("after calc")
    #return render_template('index.html', title='Home', actual = actual, measured = measured, humidity = humidity, ambient_light = ambient_light, pressure = pressure)
    return render_template('index.html', title='Home', output = outputed_array)
    #return "Hello, World!"

@app.route('/empty')
def empty():
    return render_template('empty.html')

def read_test_data():
    files = os.listdir('app/temp_readings')
    print()
    apt_metrics = []
    print("the files are ", files)
    for file_name in files: 
        if os.path.splitext(file_name)[-1].lower() == '.csv':
            file = 'app/temp_readings/' + file_name
            print("file is: ", file)
            df = pd.read_csv(file)
            #print("test dataframe: ", file_name)
            #print(df)
            column_averages = df.mean(axis=0).to_list()[1:]
            print ("col avgs: ", column_averages)
            print("type of col is: ", type(column_averages))
            apt_metrics.append(column_averages) #added in order of temp, humidity, pressure, and light levels 
            print("round successful")
    print("thru test data")
    return apt_metrics


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
    intercept = model.intercept_
    print("params are: ", params)
    print("intercept is: ", intercept)
    #params, covariance = curve_fit(model, X, y)

    # Evaluate the model
    #mse = mean_squared_error(y_test, y_pred)
    #print(f'Mean Squared Error: {mse}')

    # Inspect coefficients
    # print('Estimated coefficients:', model.coef_)
    # print('Intercept:', model.intercept_)
    return params, intercept