from flask import Flask, render_template, redirect, url_for, jsonify, json, request
from flask_pymongo import PyMongo
import pymongo
import requests
import sys
import os


# Data Variables --------------------------------------------------
app=Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config["MONGO_URI"]=os.environ['MONGO_URI']
mongo=PyMongo(app)
vg_data = mongo.db.vg_data



# Frontend App Routes --------------------------------------------------
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


################## Main ##################
@app.route("/")
def index():
    return render_template(
        "index.html",
    )

################## Games ##################
@app.route('/game')
def games():
    return render_template('game.html')

################## Predictions ##################
@app.route('/predictions')
def predictions():
    return render_template('predictions.html')

################## Data ##################
@app.route('/data')
def data():
    return render_template('data.html')

    
# Temporary Dev Routes
@app.route("/daren")
def darenDev():
    return render_template(
        "daren.html",
    )
@app.route("/erin")
def erinDev():
    return render_template(
        "erin.html",
    )
@app.route("/johnny")
def johnnyDev():
    return render_template(
        "johnny.html",
    )
@app.route("/michelle")
def michelleDev():
    return render_template(
        "michelle.html",
    )
@app.route("/nicole")
def nicoleDev():
    return render_template(
        "nicole.html",
    )


# Backend App Routes --------------------------------------------------
@app.route("/vg_data")
def serveData():
    return jsonify(list(vg_data.find({ },
   { '_id': 0})))


# Run App --------------------------------------------------
if __name__=="__main__":
    app.run(debug=True)