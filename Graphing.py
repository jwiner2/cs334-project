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

def main():
    provinceVsPrice()
if __name__ == "__main__":
    main()