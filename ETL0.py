# Import Dependencies
import pandas as pd
import pymongo
# Read CSV
games_data = pd.read_csv("data/vgsales.csv", encoding='utf-8')
# Main dictionary to push to MongoDB
vg_project = {}
# Convert to list/array and push to main dictionary
wip_dict = games_data.to_dict("records")
vg_project["raw_data"] = (wip_dict)
# Push main dictionary to MongoDB
conn = "mongodb://localhost:27017"
client = pymongo.MongoClient(conn)
db = client.vgpredict
raw_data = db.raw_data
raw_data.drop()
raw_data.insert_one(vg_project)
