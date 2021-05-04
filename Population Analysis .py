#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# visualization
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import folium

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().run_line_magic('time', "population = pd.read_csv(r'C:\\Users\\shubham.kj\\Desktop\\Countries Population from 1995 to 2020.csv')")


# In[3]:


# let
population.head(20)


# In[4]:


population.describe()


# In[5]:


# checking NULL value

population.isnull().sum()


# In[6]:


population['Density (P/Km²)'] = population['Density (P/Km²)'].str.replace(',','')


# In[7]:


population['Density (P/Km²)'] = population['Density (P/Km²)'].astype(int)
population['Country'] = population['Country'].astype(str)


# In[8]:


unique_countries = population['Country'].unique()
plt.style.use("seaborn-talk")


# set year
year = 2020
df_last_year = population[population['Year'] == year]
series_last_year = df_last_year.groupby('Country')['Population'].sum().sort_values(ascending=False)
print(series_last_year)

labels = []
values = []
country_count = 10
other_total = 0
for country in series_last_year.index:
    if country_count > 0:
        labels.append(country)
        values.append(series_last_year[country])
        country_count -= 1
    else:
        other_total += series_last_year[country]
labels.append("Other")
values.append(other_total)

wedge_dict = {
    'edgecolor': 'black',
    'linewidth': 2        
}

explode = (0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)

plt.title(f"Total Share of in World's Population the top 10 countries in {year}")
plt.pie(values, labels=labels, explode=explode, autopct='%1.2f%%', wedgeprops=wedge_dict)
plt.show()


# In[9]:


def country_wise_population(country):
    return population[population['Country'] == country]


# In[10]:


india_population = country_wise_population('India')


# In[11]:


fig = plt.figure(figsize=(10,5))
plt.plot(india_population['Year'], india_population['Yearly Change'])
plt.title('Yearly Population Change in India')
plt.xlabel('Year')
plt.ylabel('Population in 10 Million')
plt.show()


# In[12]:


india_population[india_population['Yearly Change']==india_population['Yearly Change'].max()][['Year', 'Population', 'Yearly % Change',
                                                                                             'Yearly Change']]


# In[13]:


india_population[india_population['Yearly Change']==india_population['Yearly Change'].min()][['Year', 'Population', 'Yearly % Change',
                                                                                             'Yearly Change']]


# In[14]:


population_top5_2020 = population[population['Year'] == 2020][:5]
top_5_countries = population_top5_2020['Country'].unique()


# In[15]:


top5_popultion = population[population['Country'].isin(top_5_countries)][['Year', 'Country', 'Population']]
top5_popultion_pivot = top5_popultion.pivot(index='Year', columns='Country', values='Population')
top5_popultion_pivot.style.background_gradient(cmap='PuBu')


# In[16]:


population_2020 = population[population['Year'] == 2020]


# In[17]:


fig = px.choropleth(population_2020, locations="Country", 
                    locationmode='country names', color="Density (P/Km²)", 
                    hover_name="Country", 
                    color_continuous_scale="blues", 
                    title='Density of Countries in 2020')
fig.update(layout_coloraxis_showscale=True)
fig.show()


# In[18]:


# highest dense country by population
population_2020[population_2020['Density (P/Km²)']==population_2020['Density (P/Km²)'].max()][['Country','Density (P/Km²)']]


# In[19]:


# lowest dense country by population
population_2020[population_2020['Density (P/Km²)']==population_2020['Density (P/Km²)'].min()][['Country','Density (P/Km²)']]


# In[20]:


fig = px.choropleth(population_2020, locations="Country", 
                    locationmode='country names', color="Population", 
                    hover_name="Country",
                    color_continuous_scale="dense", 
                    title='Population of Countries in 2020')
fig.update(layout_coloraxis_showscale=True)
fig.show()


# In[28]:


ps = get_ipython().run_line_magic('time', 'population')


# In[29]:


# Storing the value of India in new Dataframe
Ind=pd.DataFrame()
Ind=ps.loc[ps['Country']=='India']


# In[30]:


Ind.head()


# In[31]:


plt.plot(Ind['Year'], Ind['Population'], color='g')
plt.plot(Ind['Year'], Ind['World Population'], color='orange')
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population Change')
plt.show()


# In[32]:


plt.plot(Ind['Year'], Ind['Density (P/Km²)'])
plt.gca().invert_yaxis()
plt.show()//Change in Population Density with respect to years


# In[33]:


# creating new dataframe
ds1=pd.DataFrame()
ds1['Year']=Ind['Year']
ds1['Fertility Rate']=Ind['Fertility Rate']
ds1['Migrants (net)']=Ind['Migrants (net)']
ds1['Population']=Ind['Population']


# In[34]:


ds1.head()


# In[35]:


# our features
X = ds1[['Fertility Rate', 'Migrants (net)']]
y = ds1['Population']


# In[36]:


# Testing and training dataset split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


# In[37]:


# Building model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[38]:


# Beta coefficients of our model
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df


# In[39]:


# predicting the value
y_pred = regressor.predict(X_test)


# In[40]:


# Actual and Predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1=df
df


# In[41]:


# Diffference in actual and predicted value
df1.plot(kind='bar')


# In[43]:


import numpy as np


# In[44]:


# accuracy check
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
rmsd = np.sqrt(mean_squared_error(y_test, y_pred))      
r2_value = r2_score(y_test, y_pred)                     

print("Root Mean Square Error :", rmsd)
print("R^2 Value :", r2_value)


# # Forecasting

# In[54]:


ds1.head(100)


# In[46]:


# Dropping irrelevant features
ds1.drop(['Fertility Rate','Migrants (net)'],axis=1,inplace=True)


# In[47]:


# making year as index
ds1.set_index('Year',inplace=True)


# In[55]:


ds1.head(10)


# In[51]:


Test=ds1[:8] 
Train=ds1[8:]


# Naive Forecasting :
# Estimating technique in which the last period's actuals are used as this period's forecast, without adjusting them or attempting to establish causal factors. It is used only for comparison with the forecasts generated by the better (sophisticated) techniques.

# In[52]:


# Naive forecast - It gives our forecast value seeing our past few values
dd= np.asarray(Test.Population)
y_hat = Test.copy()
y_hat['naive'] = dd[len(dd)-1]
plt.plot(Train.index, Train['Population'], label='Train')
plt.plot(Test.index,Test['Population'], label='Test')
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()


# In[ ]:




