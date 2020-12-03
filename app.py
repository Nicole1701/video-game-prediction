from flask import Flask, render_template, redirect, url_for, jsonify, json, request
from flask_pymongo import PyMongo
import pymongo
import requests
import sys


# Data Variables --------------------------------------------------
app=Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config["MONGO_URI"]="mongodb://localhost:27017/DBNAMEHERE"
mongo=PyMongo(app)
variable = mongo.db.collection


# Frontend App Routes --------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
    )


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
@app.route("/data")
def serveData():
    return jsonify(list(variable.find({ },
   { '_id': 0})))


# Run App --------------------------------------------------
if __name__=="__main__":
    app.run(debug=True)