import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    # vectorizedData = pd.read_csv("data/vectorized_data.csv")
    # uniqueNames = vectorizedData.province.unique()
    # nameMeans = []
    # for name in uniqueNames:
    #     dataAttribute = vectorizedData[vectorizedData["province"]==name]
    #     avgPrice = dataAttribute["price"]
    #     nameMeans.append(avgPrice.mean())
    # y_pos = np.arange(5)
    # tuple_list = [(p1, p2) for idx1, p1 in enumerate(uniqueNames)  
    # for idx2, p2 in enumerate(nameMeans) if idx1 == idx2] 
    # top5 = sorted(tuple_list, key=lambda t: t[1], reverse=True)[]
    # print([e[1] for e in top5])
    # print([e[0] for e in top5])
    # plt.bar(y_pos, [e[1] for e in top5], align='center', alpha=1, color = '#F8005C')
    # plt.xticks(y_pos, [e[0] for e in top5])
    # plt.ylabel('Avg Price')
    # plt.xlabel('Provinces')
    # plt.title('Lowest Avg Wine Price By Province')
    # plt.show()
    vectorizedData = pd.read_csv("data/vectorized_data.csv")
    ranges = [(81,85), (86,90), (91,95), (96,100)]
    rangeStrings = ["81-85", "86-90", "91-95", "96-100"]
    rangeMeans = []
    for r in ranges:
        dataAttribute = vectorizedData[vectorizedData["points"]>=r[0]]
        dataAttribute = dataAttribute[dataAttribute["points"]<=r[0]]
        avgPrice = dataAttribute["price"]
        rangeMeans.append(avgPrice.mean())
    y_pos = np.arange(4)
    tuple_list = [(p1, p2) for idx1, p1 in enumerate(rangeStrings)  
    for idx2, p2 in enumerate(rangeMeans) if idx1 == idx2] 
    top5 = sorted(tuple_list, key=lambda t: t[1], reverse=True)
    print([e[1] for e in top5])
    print([e[0] for e in top5])
    plt.bar(y_pos, [e[1] for e in top5], align='center', alpha=1, color = '#F8005C')
    plt.xticks(y_pos, [e[0] for e in top5])
    plt.ylabel('Avg Price')
    plt.xlabel('Points range')
    plt.title('Avg Price Based on Points')
    plt.show()
if __name__ == "__main__":
    main()