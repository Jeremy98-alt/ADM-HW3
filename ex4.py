import re
import operator
import matplotlib.pyplot as plt
import seaborn as sns
def newdata(df):
    ### input = dataset obtain in 1.3
    ### output = new dataset with some changes in bookSeries columns and Published columns
    newdf = df[df.bookSeries != ' '].reset_index(drop=True)
    newdf = df.dropna(subset = ["bookSeries"]).reset_index(drop=True)
    newdf["bookSeries"]=newdf["bookSeries"].str.replace("(", "").str.replace(")", "")
    newdf["bookSeries"]=newdf["bookSeries"].str.split("#")
    for i, rows in newdf.iterrows():
        if len(newdf.at[i,"bookSeries"]) < 2:
            newdf = newdf.drop(index=i)
    newdf= newdf.reset_index(drop=True)
    for i in range(len(newdf)):
        try:
            newdf.at[i,"Published"] = re.findall('([0-9]{4})', newdf.at[i,"Published"])
        except:
            pass
    return newdf

def new_dictionary(df):
    
    ### input = dataset created from newdata function
    ### output = dictionary with 10 keys (the first 10 bookSeries obtain by dataset). Every keys contain a list of list
    ### every list contains the year passed from the first book of the series and cumulative pages
    TopSeries = []
    for i in range(0,10):
        TopSeries.append(df.at[i,"bookSeries"][0])
    dictionary=dict.fromkeys(TopSeries)
    for keys in dictionary.keys():
        dictionary[keys] = []
        for i in range(len(df)):
            try:
                if len(df.at[i,"bookSeries"][1]) == 1 and df.at[i,"bookSeries"][0] == keys:
                    item = [int(df.at[i,"Published"][0]),int(df.at[i,"NumberofPages"])]
                    dictionary[keys].append(item)
                    dictionary[keys].sort(key = operator.itemgetter(0))
            except:
                pass
    for keys in dictionary:
        try:
            min_value = dictionary[keys][0][0]
            count_y = 0
            for item in dictionary[keys]:
                count_y += item[1]
                item[0] = (item[0]-min_value)
                item[1] = count_y
        except:
            continue
    
    return dictionary

def plot_dictionary(dictionary):
    
    ### input == Take like input the dictionary create with new_dictionary function
    ### output == Create plot for each key in the dictionary. Each plot contain in x-axis 
    for keys in dictionary:
        x=[]
        y=[]
        for item in dictionary[keys]:
            x.append(item[0])
            y.append(item[1])
        sns.set(font_scale=1)
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        ax = sns.barplot(x=x, y=y)
        ax.set_title(keys)
        ax.set(xlabel = 'Years from publication 1st book', ylabel = 'Cumulative series page',)
    return plt.show()
    
    
