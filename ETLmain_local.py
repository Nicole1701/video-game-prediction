# Import Dependencies
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import json
import numpy as np
import scipy.stats as st
import pymongo
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.style.palettes import PALETTES, SEQUENCES, color_palette
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import plotly.express as px
import plotly.graph_objects as go

# Main dictionary to push to MongoDB
vg_project = {}

#Embed all in function
def scrape_ETL():
    ### WEB SCRAPE ###
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
    price_data = pd.concat(complete_list)
    price_data['Price'] = price_data['Price'].str.replace(',','')
    price_data['Price'] = price_data['Price'].astype(float)
    # Convert to list/array and push to main dictionary
    prices_dict = price_data.to_dict("records")

    ### ETL ###
    # Load CSV for video game sales data
    games_data = pd.read_csv("data/vgsales.csv", encoding='utf-8')

    # Create cleaned DF for all video games sales data
    games_all_df = games_data.copy()
    # Remove Rank column and drop blank years
    games_all_df = games_all_df[['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].sort_values(by=['Platform', 'Name']).reset_index(drop=True)
    games_all_df['Year'].replace(np.nan,'')
    games_all_df = games_all_df.dropna()
    games_all_df['Year'] = games_all_df['Year'].astype(int)
    # Convert sales to currency
    games_all_df['NA_Sales']  = games_all_df['NA_Sales'] .multiply(1000000).astype(int).replace(np.nan,0)
    games_all_df['EU_Sales']  = games_all_df['EU_Sales'] .multiply(1000000).astype(int).replace(np.nan,0)
    games_all_df['JP_Sales']  = games_all_df['JP_Sales'] .multiply(1000000).astype(int).replace(np.nan,0)
    games_all_df['Other_Sales']  = games_all_df['Other_Sales'] .multiply(1000000).astype(int).replace(np.nan,0)
    games_all_df['Global_Sales']  = games_all_df['Global_Sales'] .multiply(1000000).astype(int).replace(np.nan,0)
    # Make game names uppercase and remove punctuation
    games_all_df['Name'] = games_all_df['Name'].str.upper() 
    games_all_df['Name'] = games_all_df['Name'].str.replace(r'[^\w\s]+', '')
    # Convert to list/array and push to main dictionary
    all_sales_dict = games_all_df.to_dict("records")

    # Create cleaned DF for console filtered sales data
    # Remove extra platforms
    games_clean_df = (games_all_df[(games_all_df['Platform'] == '2600') | (games_all_df['Platform'] == 'NES')
                                        | (games_all_df['Platform'] == 'GEN') | (games_all_df['Platform'] == 'SNES')
                                        | (games_all_df['Platform'] == 'N64') | (games_all_df['Platform'] == 'SCD')
                                        | (games_all_df['Platform'] == 'SAT') | (games_all_df['Platform'] == 'PS')]).reset_index(drop=True)
    filtered_sales_dict = games_clean_df.to_dict("records")

    # Create merged DF of Sales and Price
    # Sort prices dataframe
    price_data_df = price_data[['Console', 'Game Title', 'Price']].sort_values(by=['Console', 'Game Title']).reset_index(drop=True)
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
    # merged_df = merged_df.drop(columns=["Console","Game Title"])
    # Convert to list/array and push to main dictionary
    merged_dict = merged_df.to_dict("records")

    # Create List/Array of Genres
    genres_obj = merged_df["Genre"].unique()
    genres = []
    for i in genres_obj:
        genres.append(i)

    #Setup Encoded Data for Modelling
    fill_zero = merged_df
    # Change the floats to integers
    fill_zero['NA_Sales'] = fill_zero['NA_Sales'].apply(np.int64)
    fill_zero['EU_Sales'] = fill_zero['EU_Sales'].apply(np.int64)
    fill_zero['JP_Sales'] = fill_zero['JP_Sales'].apply(np.int64)
    fill_zero['Global_Sales'] = fill_zero['Global_Sales'].apply(np.int64)
    fill_zero['Other_Sales'] = fill_zero['Other_Sales'].apply(np.int64)
    fill_zero['Price'] = fill_zero['Price'].apply(np.int64)
    # Label Encode the Platform and Genre columns
    le = preprocessing.LabelEncoder()
    le.fit(fill_zero["Platform"])
    fill_zero["Platform"]= le.transform(fill_zero["Platform"])
    le.fit(fill_zero["Genre"])
    fill_zero["Genre"]= le.transform(fill_zero["Genre"])
    encoded_df = fill_zero.rename(columns={"Platform": "Platform LabelCode", "Genre": "Genre LabelCode"})
    platform_ohe = pd.get_dummies(encoded_df["Platform LabelCode"], prefix="Platform")
    genre_ohe = pd.get_dummies(encoded_df["Genre LabelCode"], prefix="Genre")
    pricesWithEncode = encoded_df.join(platform_ohe, how='left', sort=False)
    pricesWithEncode2 = pricesWithEncode.join(genre_ohe, how='left', sort=False)
    # Convert to list/array and push to main dictionary
    encoded_dict = pricesWithEncode2.to_dict("records")

    # Assemble main dictionary
    vg_project["consoles"] = (console_col)
    vg_project["genres"] = (genres)
    vg_project["games_prices"] = (prices_dict)
    vg_project["games_all_sales"] = (all_sales_dict)
    vg_project["games_filtered_sales"] = (filtered_sales_dict)
    vg_project["merged_data"] = (merged_dict)
    vg_project["encoded_data"] = (encoded_dict)
    

    # Push main dictionary to MongoDB
    conn = "mongodb://localhost:27017"
    client = pymongo.MongoClient(conn)
    db = client.vgpredict
    vg_data = db.vg_data
    vg_data.drop()
    vg_data.insert_one(vg_project)


def log_regression():
    response = json.loads(requests.get("http://127.0.0.1:5000/vg_data").text)
    data = response[0]["encoded_data"]
    encoded_data = pd.DataFrame.from_dict(data)
    encoded_data = encoded_data.drop(columns =['Platform_0', 'Platform_1',
       'Platform_2', 'Platform_3', 'Platform_4', 'Platform_5', 'Platform_6',
       'Platform_7', 'Genre_0', 'Genre_1', 'Genre_2', 'Genre_3', 'Genre_4',
       'Genre_5', 'Genre_6', 'Genre_7', 'Genre_8', 'Genre_9', 'Genre_10',
       'Genre_11',"Price", "Console", "Name", "Game Title", "Platform LabelCode", "Genre LabelCode", "Publisher"])
    X_train, X_test, y_train, y_test = train_test_split(encoded_data.drop(["Mean"],axis=1), 
                                                    encoded_data["Mean"], test_size=0.35, 
                                                    random_state=42)
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    predictions = logmodel.predict(X_test)
    cl_report = classification_report(y_test,predictions)
    matrix_array = confusion_matrix(y_test,predictions)
    matrix_array_df = pd.DataFrame(matrix_array)
    matrix_array_df = matrix_array_df.rename(index={0 : "False", 1 : "True"})
    matrix_array_df = matrix_array_df.rename(columns={0 : "Correct \n Predictions", 1 : "Incorrect \n Predictions"})
    matrix_array_df = matrix_array_df.reset_index()
    matrix_array_df = matrix_array_df.rename(columns={"index" : "Above Avg Price"})
    sns.set()
    stack_plot = matrix_array_df.set_index('Above Avg Price').T.plot(kind='bar', 
                stacked=True, color=["#0033FF", "#FF0099"], figsize=(16,9))
    ylabel = "Total Predictions"
    plt.ylabel(ylabel,fontsize="16")
    plt.xticks(rotation=360, size="16")
    plt.title("Game Price Prediction Accuracy", size="24")
    plt.savefig("static/img/matrix_plot.png")
    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(logmodel, support=True, title="Logistic Regression Learning Model",
                                    cmap='YlGnBu', size=(900, 500))
    visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show(outpath="static/img/report_plot.png") # Draw/show the data


def vintagedataviz():
    response = json.loads(requests.get("http://127.0.0.1:5000/vg_data").text)
    data = response[0]["merged_data"]
    sales_df = pd.DataFrame.from_dict(data)
    # create genre group by for genre viz
    genres =sales_df.groupby("Genre", as_index=False).count()
    genre_count = genres[["Genre", "Name"]].rename(columns={"Name": "Game Count"})
    sorted_genre_count = genre_count.sort_values(by=["Game Count"], ascending=False)
    # load in color palette, created in AllDataViz notebook
    palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#ff8001', '#00fcfe', '#FE00FA', '#16FF32', '#FF97FF', 
           '#FFFF00','#2dc997','#B10DA1']
    # Bar Graph of top Genres
    fig = px.bar(sorted_genre_count, title="Top Game Genres - vintage games", x="Genre", y="Game Count")
    fig.show()
    fig.write_html("static/img/vintagegenres.html")
    # create genre by year table for viz
    genre_year = sales_df.groupby(["Genre", "Year"], as_index=False).count()
    genre_by_year = genre_year[["Genre", "Year", "Platform"]].rename(columns={"Platform": "Games per Genre"})
    # stacked bar graph of genre by year
    fig = px.bar(genre_by_year, title="Top Genres by Year", x="Year", y="Games per Genre", 
             color="Genre", range_y=[0,250], color_discrete_sequence=palette)
    # creating cumulative count of games per genre per year for bar chart race
    genre_year =sales_df.groupby(["Genre","Year"]).agg({"Platform":"count"}).reset_index()
    # genre_year
    genre_by_year = genre_year[["Genre", "Year", "Platform"]].rename(columns={"Platform": "Games per Genre"})
    # genre_by_year
    idx = pd.MultiIndex.from_product([genre_by_year.Year.unique(), 
                                    genre_by_year.Genre.unique()], names=['Year', 'Genre'])
    genre_by_year2 = genre_by_year.set_index(['Year', 'Genre']).reindex(idx).fillna(0).sort_values(ascending=True,by=["Genre","Year"])
    cumulative_genre_by_year = pd.concat([genre_by_year2, genre_by_year2.groupby(level=1).cumsum().add_prefix('Cumulative_')], 1).sort_index(level=1).reset_index()
    cumulative_genre_by_year.sort_values(by=["Year","Genre"],inplace=True)
    cumulative_genre_by_year.rename(columns={"Cumulative_Games per Genre":"Cumulative Games per Genre"},inplace=True)
    # top genre over time bar chart race
    fig = px.bar(cumulative_genre_by_year, title="Top Genres over Time-vintage", x="Genre", y="Cumulative Games per Genre", 
             color="Genre", animation_frame="Year", animation_group="Genre", range_y=[0,260],
             color_discrete_sequence=palette)
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300
    # Create Sales info table for viz
    sales_year = sales_df.groupby(["Year"], as_index=False).sum()
    sales_by_year = sales_year[["Year","NA_Sales","EU_Sales", "JP_Sales", "Other_Sales","Global_Sales"]]
    # global sales by year - export to HTML
    fig = px.line(sales_by_year, title="Global Sales - 1980-2003", x="Year", y="Global_Sales")
    fig.write_html("static/img/vintageglobalsales.html")
    # comparison of sales by country by year
    sales_by_year.plot(x="Year", y=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"], kind="line")
    # Groupby platform and year for platform viz
    platform_and_year = sales_df.groupby(["Platform","Year"], as_index=False).count()
    games_by_year = platform_and_year[["Platform", "Year", "Name"]].rename(columns={"Name": "Games per Platform"})
    # games by platform by year - export to HTML
    fig = px.bar(games_by_year, title="Top Platforms by Year", x="Year", y="Games per Platform", 
                color="Platform", range_y=[0,250], color_discrete_sequence=palette)
    fig.write_html("static/img/vintageplatforms.html")
    # faceted platform bubble chart
    fig = px.scatter(games_by_year, x="Year", y="Games per Platform",
                size="Games per Platform", color="Year", facet_col="Platform")
    # games per genre bubble chart
    fig = px.scatter(genre_by_year, x="Genre", y="Games per Genre",
                size="Games per Genre", color="Year")


def alldataviz():
    response = json.loads(requests.get("http://127.0.0.1:5000/vg_data").text)
    data = response[0]["games_all_sales"]
    all_sales_df = pd.DataFrame.from_dict(data)
    clean_all_sales = all_sales_df.drop(all_sales_df[all_sales_df.Year > 2016].index)
    all_sales_year = clean_all_sales.groupby(["Year"], as_index=False).sum()
    all_sales_by_year = all_sales_year[["Year","NA_Sales","EU_Sales", "JP_Sales", "Other_Sales","Global_Sales"]]
    fig = px.colors.qualitative.swatches()
    # create color palette
    palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#ff8001', '#00fcfe', '#FE00FA', '#16FF32', '#FF97FF', 
            '#FFFF00','#2dc997','#B10DA1']
    # global sales by year - export to HTML
    fig = px.line(all_sales_by_year, title="Global Sales - all data", x="Year", y="Global_Sales")
    fig.write_html("static/img/allglobalsales.html")
    # comparison of sales by country by year
    all_sales_by_year.plot(x="Year", y=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"], kind="line")
    # create genre group by for genre viz
    genres =clean_all_sales.groupby("Genre", as_index=False).count()
    genre_count = genres[["Genre", "Name"]].rename(columns={"Name": "Game Count"})
    sorted_genre_count = genre_count.sort_values(by=["Game Count"], ascending=False)
    # Bar Graph of top Genres
    fig = px.bar(sorted_genre_count, title="Top Game Genres - all data", x="Genre", y="Game Count")
    fig.write_html("static/img/alltopgenres.html")
    # create genre by year table for viz
    genre_year = clean_all_sales.groupby(["Genre", "Year"], as_index=False).count()
    genre_by_year = genre_year[["Genre", "Year", "Platform"]].rename(columns={"Platform": "Games per Genre"})
    # stacked bar graph of genre by year
    fig = px.bar(genre_by_year, title="Top Genres by Year", x="Year", y="Games per Genre", 
                color="Genre", range_y=[0,1600], color_discrete_sequence=palette)
    # creating cumulative count of games per genre per year for bar chart race
    genre_year =clean_all_sales.groupby(["Genre","Year"]).agg({"Platform":"count"}).reset_index()
    # genre_year
    genre_by_year = genre_year[["Genre", "Year", "Platform"]].rename(columns={"Platform": "Games per Genre"})
    # genre_by_year
    idx = pd.MultiIndex.from_product([genre_by_year.Year.unique(), 
                                    genre_by_year.Genre.unique()], names=['Year', 'Genre'])
    genre_by_year2 = genre_by_year.set_index(['Year', 'Genre']).reindex(idx).fillna(0).sort_values(ascending=True,by=["Genre","Year"])
    cumulative_genre_by_year = pd.concat([genre_by_year2, genre_by_year2.groupby(level=1).cumsum().add_prefix('Cumulative_')], 1).sort_index(level=1).reset_index()
    cumulative_genre_by_year.sort_values(by=["Year","Genre"],inplace=True)
    cumulative_genre_by_year.rename(columns={"Cumulative_Games per Genre":"Cumulative Games per Genre"},inplace=True)
    # top genre over time bar chart race
    fig = px.bar(cumulative_genre_by_year, title="Top Genres over Time", x="Genre", y="Cumulative Games per Genre", 
                color="Genre", animation_frame="Year", animation_group="Genre", range_y=[0,3500],
                color_discrete_sequence=palette)
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300
    fig.show()
    fig.write_html("static/img/allgenresovertime.html")
    # Bubble chart of genre by year
    fig = px.scatter(genre_by_year, title="Top Genres by Year", x="Year", y="Games per Genre", 
                size="Games per Genre", color="Genre", range_y=[0,300],
                color_discrete_sequence=palette)
    # group data by platform and year for platform viz
    platform_and_year = clean_all_sales.groupby(["Platform","Year"], as_index=False).count()
    games_by_year = platform_and_year[["Platform", "Year", "Name"]].rename(columns={"Name": "Games per Platform"})
    # games by platform by year - export to HTML
    fig = px.bar(games_by_year, title="Top Platforms by Year", x="Year", y="Games per Platform", 
                color="Platform", range_y=[0,1500],color_discrete_sequence=palette)
    # a different look at games per genre
    fig = px.scatter(genre_by_year, x="Games per Genre", y="Year",
                    size="Games per Genre", color="Genre", color_discrete_sequence=palette)


# Call all functions on script run
#scrape_ETL()
#log_regression()
vintagedataviz()
alldataviz()