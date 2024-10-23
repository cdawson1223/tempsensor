from flask import render_template
from app import app

@app.route('/')
@app.route('/index')
def index():
    actual = 72
    measured = 66
    humidity = 36
    ambient_light = 300
    pressure = 30.43
    return render_template('index.html', title='Home', actual = actual, measured = measured, humidity = humidity, ambient_light = ambient_light, pressure = pressure)
    #return "Hello, World!"