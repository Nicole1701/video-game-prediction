# Dependencies
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import numpy as np
import scipy.stats as st
import pymongo

# URLs to scrape
base_url = "https://gamevaluenow.com/"
console = ["atari-2600",
            "nintendo-nes",
            "sega-genesis",
            "super-nintendo",
            "nintendo-64",
            "sega-cd",
            "sega-saturn",
            "playstation-1-ps1"]
console_col = ["2600",
                "NES",
                "GEN",
                "SNES",
                "N64",
                "SCD",
                "SAT",
                "PS"]

# Put the all the console complete prices data in a list
complete_list = []

for name in range(len(console)):
    all_prices = []
    
    # Retrieve page with the requests module
    response = requests.get(base_url + console[name])
    # Create a Beautiful Soup object
    soup = bs(response.text, 'html.parser')
    
    prices_table = soup.find("table")
    prices_data = prices_table.find_all("tr")
    
    # Get all the price data
    for item in range(len(prices_data)):
        for td in prices_data[item].find_all("td"):
            # Remove all the markup from the text
            all_prices.append(td.text.strip())
        
        all_prices.append(console_col[name])
        # Make a list of the item names from every fifth index eg 1,6,10 et
        game_title = all_prices[1::5]             
        # Make a list of the complete price from starting at the fourth index
        price_complete = all_prices[3::5]
        # Make a list of the console types from every fifth index eg 0,5,9 etc
        console_name = all_prices[5::5] 
        # Make the lists in to a datframe
        game_prices_df = pd.DataFrame({'Console' : console_name, 'Game Title' : game_title, 'Price' : price_complete})
    
    # Create a list of data frames
    complete_list.append(game_prices_df)
    
# Concatenate the list of data frames in to one
game_price_list = pd.concat(complete_list)
game_price_list['Price'] = game_price_list['Price'].str.replace(',','')
game_price_list['Price'] = game_price_list['Price'].astype(float)

# Read in game sales data
games_data = pd.read_csv("data/vgsales.csv", encoding='utf-8')
# Remove extra platforms
games_clean = (games_data[(games_data['Platform'] == '2600') | (games_data['Platform'] == 'NES')
                                     | (games_data['Platform'] == 'GEN') | (games_data['Platform'] == 'SNES')
                                     | (games_data['Platform'] == 'N64') | (games_data['Platform'] == 'SCD')
                                     | (games_data['Platform'] == 'SAT') | (games_data['Platform'] == 'PS')]).reset_index(drop=True)
# Remove Rank column and drop blank years
games_clean_df = games_clean[['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].sort_values(by=['Platform', 'Name']).reset_index(drop=True)
games_clean_df['Year'].replace('', np.nan)
games_clean_df = games_clean_df.dropna()
games_clean_df['Year'] = games_clean_df['Year'].astype(int)
# Convert sales to currency
games_clean_df['NA_Sales']  = games_clean_df['NA_Sales'] .multiply(1000000).astype(int).replace(0, np.NaN)
games_clean_df['EU_Sales']  = games_clean_df['EU_Sales'] .multiply(1000000).astype(int).replace(0, np.NaN)
games_clean_df['JP_Sales']  = games_clean_df['JP_Sales'] .multiply(1000000).astype(int).replace(0, np.NaN)
games_clean_df['Other_Sales']  = games_clean_df['Other_Sales'] .multiply(1000000).astype(int).replace(0, np.NaN)
games_clean_df['Global_Sales']  = games_clean_df['Global_Sales'] .multiply(1000000).astype(int).replace(0, np.NaN)
# Make game names uppercase and remove punctuation
games_clean_df['Name'] = games_clean_df['Name'].str.upper() 
games_clean_df['Name'] = games_clean_df['Name'].str.replace(r'[^\w\s]+', '')

# Sort prices dataframe
price_data_df = game_price_list[['Console', 'Game Title', 'Price']].sort_values(by=['Console', 'Game Title']).reset_index(drop=True)
# Make game names uppercase and remove punctuation
price_data_df['Game Title'] = price_data_df['Game Title'].str.upper() 
price_data_df['Game Title'] = price_data_df['Game Title'].str.replace(r'[^\w\s]+', '')
# Remove null prices
price_data_df.drop(price_data_df[price_data_df['Price'] == 0].index, inplace = True)
# Calculate quartiles and remove outliers
quartiles = price_data_df['Price'].quantile([.25,.5,.75])
lowerq = quartiles[0.25]
upperq = quartiles[0.75]
iqr = upperq-lowerq
lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
price_data_df.drop(price_data_df[price_data_df['Price'] < lower_bound].index, inplace = True) 
price_data_df.drop(price_data_df[price_data_df['Price'] > upper_bound].index, inplace = True)
# Find average, and median price and add binary columns
mean = price_data_df[["Price"]].mean()
median = price_data_df[["Price"]].median()
price_data_df['Mean'] = np.where(price_data_df[['Price']] > mean, True, False)
price_data_df['Median'] = np.where(price_data_df[['Price']] > median, True, False)
# Merge data
merged_df = pd.merge(games_clean_df, price_data_df,  how='inner', left_on=['Name','Platform'], right_on = ['Game Title','Console'])
merged_df = merged_df.fillna(0)
merged_df = merged_df.drop(columns=["Console","Game Title"])
# Create List/Array of Genres
genres_obj = merged_df["Genre"].unique()
genres = []
for i in genres_obj:
    genres.append(i)

# Combine into single dict for push to MongoDB
wip_dict = merged_df.to_dict("records")
vgpredict_data = {}
vgpredict_data["consoles"] = (console_col)
vgpredict_data["genres"] = (genres)
vgpredict_data["vg_data"] = (wip_dict)

# Push merged dataframe to MongoDB
conn = "mongodb://localhost:27017"
client = pymongo.MongoClient(conn)
db = client.vgpredict
vg_data = db.vg_data
vg_data.drop()
vg_data.insert_one(vgpredict_data)