import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def provinceVsPrice():
    wineData = pd.read_csv("data/winemag-data-130k-v2.csv")
    plottingData = wineData.groupby(['province'])['price'].mean().sort_values(ascending=False)[:10]
    plt.bar(plottingData.index, plottingData, color="#F8005C")
    plt.xticks(rotation=90)
    plt.ylabel('Avg Price')
    plt.xlabel('Provinces')
    plt.title('Highest Avg Wine Price By Province')
    plt.show()

def countryVsPrice():
    wineData = pd.read_csv("data/winemag-data-130k-v2.csv")
    plottingData = wineData.groupby(['country'])['price'].mean().sort_values(ascending=False)[:10]
    plt.bar(plottingData.index, plottingData, color="#F8005C")
    plt.xticks(rotation=90)
    plt.ylabel('Avg Price')
    plt.xlabel('Country')
    plt.title('Highest Avg Wine Price By Country')
    plt.show()

def varietyVsPrice():
    wineData = pd.read_csv("data/winemag-data-130k-v2.csv")
    plottingData = wineData.groupby(['variety'])['price'].mean().sort_values(ascending=False)[:10]
    plt.bar(plottingData.index, plottingData, color="#F8005C")
    plt.xticks(rotation=90)
    plt.ylabel('Avg Price')
    plt.xlabel('Variety')
    plt.title('Highest Avg Wine Price By Variety')
    plt.show()

def wineryVsPrice():
    wineData = pd.read_csv("data/winemag-data-130k-v2.csv")
    plottingData = wineData.groupby(['winery'])['price'].mean().sort_values(ascending=False)[:10]
    plt.bar(plottingData.index, plottingData, color="#F8005C")
    plt.xticks(rotation=90, fontsize = 9)
    plt.ylabel('Avg Price')
    plt.xlabel('Winery')
    plt.title('Highest Avg Wine Price By Winery')
    plt.show()


def pointsVsPrice():
    wineData = pd.read_csv("data/winemag-data-130k-v2.csv")
    plt.plot(wineData['points'],wineData['price'], 'o', color = '#F8005C', markersize= 3)
    plt.ylabel('Price')
    plt.xlabel('Points')
    plt.title('Wine Score vs Price')
    plt.show()

def main():
    wineryVsPrice()
if __name__ == "__main__":
    main()