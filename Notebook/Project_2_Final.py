#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
import sys
import scipy as sp
import matplotlib.pyplot as plt
# import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
get_ipython().run_line_magic('matplotlib', 'inline')
import hvplot.pandas
import panel as pn
pn.extension()


# In[4]:


sales_df=pd.read_csv("Project_2/Data/sales_promo_data_19_20.csv", parse_dates=True)


# In[5]:


sales_df.tail()


# In[6]:


sales_df=sales_df.rename(columns={"Category":"Date"})


# In[7]:


sales_df.index = pd.to_datetime(sales_df.index)


# In[8]:


sales_df=sales_df.T


# In[9]:


sales_df.columns= sales_df.iloc[0]


# In[10]:


sales_df=sales_df.iloc[1:,:]
sales_df.head()


# In[11]:


sales_df.index = pd.to_datetime(sales_df.index)


# In[12]:


acne_df= sales_df[["Acne","acne_promo"]]
baby_df= sales_df[["Baby","baby_promo"]]
bath_df= sales_df[["Bath","bath_promo"]]
body_df= sales_df[["Body Wash","body_promo"]]
hair_df= sales_df[["Hair","hair_promo"]]
lotion_df= sales_df[["Lotion","lotion_promo"]]
makeup_df= sales_df[["Makeup","makeup_promo"]]
shaving_df= sales_df[["Shaving","shaving_promo"]]
sunscreen_df= sales_df[["Sunscreen","sunscreen_promo"]]
tanner_df= sales_df[["Tanner","tanner_promo"]]


# In[13]:


acne_df.dtypes


# In[14]:


acne_df=acne_df.astype(int)
acne_df=acne_df.dropna()


# In[15]:


makuep_df=makeup_df.astype(int)
makeup_df=makeup_df.dropna()


# In[16]:


lotion_df=lotion_df.astype(int)
lotion_df=lotion_df.dropna()


# In[17]:


acne_plot = acne_df["Acne"].hvplot.line(xlabel = "Date", ylabel = "Sales", title = "Acne Sales")
hvplot.save(acne_plot, "Project_2/Images/Acne_Sales.png")
acne_plot


# In[18]:


makeup_plot = makeup_df["Makeup"].hvplot.line(xlabel = "Date", ylabel = "Sales", title = "Makeup Sales")
hvplot.save(makeup_plot, "Project_2/Images/Makeup_Sales.png")
makeup_plot


# In[19]:


lotion_plot = lotion_df["Lotion"].hvplot.line(xlabel = "Date", ylabel = "Sales", title = "Lotion Sales")
hvplot.save(lotion_plot, "Project_2/Images/Lotion_Sales.png")
lotion_plot


# In[ ]:





# In[20]:


#Decomposing Acne Data
decomp = sm.tsa.seasonal_decompose(acne_df['Acne'],freq=40) 
acne_decomp_data = pd.concat([acne_df, decomp.trend, decomp.seasonal, decomp.resid], axis=1)
acne_decomp_data.columns = ['Acne','acne_promo', 'trend', 'seasonal', 'resid']

acne_decomp_data.head()


# In[21]:


# Seasonality of Acne
acne_sales_decomposed = acne_decomp_data['seasonal'].hvplot.line(xlabel = "Date", title = "Acne Sales Decomposed")
hvplot.save(acne_sales_decomposed, "Project_2/Images/Acne_Sales_Decomposed.png")
acne_sales_decomposed


# In[22]:


# Trend of Acne Sales
acne_sales_trend = acne_decomp_data['trend'].hvplot.line(xlabel = "Date", title = "Acne Sales Trend")
hvplot.save(acne_sales_trend, "Project_2/Images/Acne_Sales_Trend.png")
acne_sales_trend


# In[23]:


# Residuals of Acne Sales
acne_residual = acne_decomp_data['resid'].hvplot.line(xlabel = "Date", title = "Acne Sales Residual")
hvplot.save(acne_residual, "Project_2/Images/Acne_Sales_Residual.png")
acne_residual


# In[24]:


#ADFuller Test -Seasonal Data is Stationary Acne
result = adfuller(acne_decomp_data["seasonal"], autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}') 


# In[25]:


#Acne Transform to stationary
acne_df['returns']= acne_df['Acne'].pct_change()
acne_df=acne_df.dropna()
acne_df.head()


# In[26]:


acne_noise, acne_trend = sm.tsa.filters.hpfilter(acne_df['Acne']) 


# In[27]:


# Acne Noise
acne_noise_graph = acne_df['returns'].hvplot.line(xlabel = "Date", title = "Acne Noise")
hvplot.save(acne_noise_graph, "Project_2/Images/Acne_Noise.png")
acne_noise_graph


# In[28]:


# Lotion Trend
acne_trend_graph = acne_trend.hvplot.line(xlabel = "Date", title = "Acne Trend")
hvplot.save(acne_trend_graph, "Project_2/Images/Acne_Trend.png")
acne_trend_graph


# In[29]:


#Acne Autocorrelation
plot_acf(acne_df['Acne'], lags=40)


# In[30]:


#Acne Partial Autocorrelation
plot_pacf(acne_df['Acne'], zero=False, lags= 40)


# In[31]:


#Acne GARCH Model
returns = acne_df['Acne'].pct_change() * 100
returns = returns.dropna()
returns.tail()


# In[32]:


model = arch_model(returns, mean='Zero', vol='GARCH')


# In[33]:


res = model.fit(disp='off')


# In[34]:


res.summary()


# In[35]:


fig = res.plot(annualize='W')


# In[36]:


#Acne Future forecast 252 days
forecast_horizon = 252
forecasts = res.forecast(start='29-AUG-20', horizon=forecast_horizon)
forecasts


# In[37]:


intermediate = np.sqrt(forecasts.variance.dropna() * 252)
intermediate.head()


# In[38]:


final = intermediate.dropna().T
final


# In[39]:


#ACNE SARIMA Model
model=SARIMAX(endog=acne_df['Acne'], order=(2,1,2),
              seasonal_order=(2,1,2,12),
              trend='c',enforce_invertibility=False
             )
acne_results=model.fit()
print(acne_results.summary())


# In[40]:


acne_results.plot_diagnostics(figsize=(20,10))


# In[41]:


#Acne Arima Model
acne_arima_mod = ARIMA(acne_df["Acne"], order=(2,1,2))


# In[42]:


acne_arima_results = acne_arima_mod.fit()


# In[43]:


acne_arima_results.summary()


# In[44]:


acne_arima_results.plot_predict()
plt.savefig('Project_2/Images/Acne_ARIMA.png')


# In[45]:


#Lotion Product Sales
lotion_df.head()


# In[46]:


#Decomposing Lotion Data
decomp = sm.tsa.seasonal_decompose(lotion_df['Lotion'],freq=40)
lotion_decomp_data = pd.concat([lotion_df, decomp.trend, decomp.seasonal, decomp.resid], axis=1)
lotion_decomp_data.columns = ['Lotion','lotion_promo', 'trend', 'seasonal', 'resid']

lotion_decomp_data.head()


# In[47]:


# Seasonality of Lotion
lotion_sales_decomposed = lotion_decomp_data['seasonal'].hvplot.line(xlabel = "Date", title = "Lotion Sales Decomposed")
hvplot.save(lotion_sales_decomposed, "Project_2/Images/Lotion_Sales_Decomposed.png")
lotion_sales_decomposed


# In[49]:


# Trend of Lotion Sales
lotion_sales_trend = lotion_decomp_data['trend'].hvplot.line(xlabel = "Date", title = "Lotion Sales Trend")
hvplot.save(lotion_sales_trend, "Project_2/Images/Lotion_Sales_Trend.png")
lotion_sales_trend


# In[50]:


# Residuals of Lotion Sales
lotion_residual = lotion_decomp_data['resid'].hvplot.line(xlabel = "Date", title = "Lotion Sales Residual")
hvplot.save(lotion_residual, "Project_2/Images/Lotion_Sales_Residual.png")
lotion_residual


# In[54]:


lotion_noise, lotion_trend = sm.tsa.filters.hpfilter(lotion_df['Lotion']) 


# In[55]:


lotion_noise_graph = lotion_noise.hvplot.line(title = "Lotion Noise")
hvplot.save(lotion_noise_graph, 'Project_2/Images/Lotion_Noise.png')
lotion_noise_graph


# In[57]:


# Lotion Trend
lotion_trend_graph = lotion_trend.hvplot.line(xlabel = "Date", title = "Lotion Trend")
hvplot.save(lotion_trend_graph, "Project_2/Images/Lotion_Trend.png")
lotion_trend_graph


# In[ ]:





# In[59]:


#ADFuller Test -Seasonal Data is Stationary
result = adfuller(lotion_decomp_data["seasonal"], autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}') 


# In[60]:


#Acne Transform to stationary
acne_df['returns']= acne_df['Acne'].pct_change()
acne_df=acne_df.dropna()
acne_df.head()


# In[61]:


acne_noise


# In[63]:


acne_noise_graph


# In[64]:


#Acne Autocorrelation
plot_acf(acne_df['Acne'], lags=40)


# In[65]:


#Acne Partial Autocorrelation
plot_pacf(acne_df['Acne'], zero=False, lags= 40)


# In[66]:


#Acne GARCH Model
returns = acne_df['Acne'].pct_change() * 100
returns = returns.dropna()
returns.tail()


# In[67]:


model = arch_model(returns, mean='Zero', vol='GARCH')


# In[68]:


res = model.fit(disp='off')


# In[69]:


res.summary()


# In[70]:


fig = res.plot(annualize='W')


# In[71]:


#Acne Future forecast 52 weeks
forecast_horizon = 10
forecasts = res.forecast(start='29-AUG-20', horizon=forecast_horizon)
forecasts


# In[72]:


intermediate = np.sqrt(forecasts.variance.dropna() * 252)
intermediate.head()


# In[73]:


final = intermediate.dropna().T
final


# In[74]:


final.plot()


# In[75]:


#ACNE SARIMA Model
model=SARIMAX(endog=acne_df['Acne'], order=(2,1,2),
              seasonal_order=(2,1,2,12),
              trend='c',enforce_invertibility=False
             )
acne_results=model.fit()
print(acne_results.summary())


# In[76]:


acne_results.plot_diagnostics(figsize=(20,10))


# In[77]:


#Acne Arima Model
acne_arima_mod = ARIMA(acne_df["Acne"], order=(2,1,2))


# In[78]:


acne_arima_results = acne_arima_mod.fit()


# In[79]:


acne_arima_results.summary()


# In[80]:


acne_arima_results.plot_predict()


# In[81]:


#Lotion Product Sales


# In[82]:


#Decomposing Lotion Data
decomp = sm.tsa.seasonal_decompose(lotion_df['Lotion'],freq=40)
lotion_decomp_data = pd.concat([lotion_df, decomp.trend, decomp.seasonal, decomp.resid], axis=1)
lotion_decomp_data.columns = ['Lotion','lotion_promo', 'trend', 'seasonal', 'resid']

lotion_decomp_data.head()


# In[83]:


lotion_sales_decomposed


# In[86]:


lotion_sales_trend


# In[88]:


lotion_residual


# In[90]:


lotion_noise, lotion_trend = sm.tsa.filters.hpfilter(lotion_df['Lotion'])


# In[91]:


lotion_noise_graph


# In[93]:


lotion_trend_graph


# In[96]:


#ADFuller Test -Seasonal Data is Stationary
result = adfuller(lotion_decomp_data["seasonal"], autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}') 


# In[97]:


# Transform Lotion data to stationary
lotion_df['returns']= lotion_df['Lotion'].pct_change()
lotion_df=lotion_df.dropna()
lotion_df.head()


# In[100]:


lotion_df['returns'].plot()


# In[101]:


plot_acf(lotion_df['Lotion'], lags=40)


# In[102]:


plot_pacf(lotion_df['Lotion'], zero=False, lags= 40)


# In[103]:


#Lotion Garch Model
lotion_returns = lotion_df['Lotion'].pct_change() * 100
lotion_returns = lotion_returns.dropna()
lotion_returns.tail()


# In[104]:


lotion_model = arch_model(lotion_returns, mean='Zero', vol='GARCH')


# In[105]:


lotion_res = lotion_model.fit(disp='off')


# In[106]:


lotion_res.summary()


# In[107]:


#Lotion Sales Volatility
lotion_fig = lotion_res.plot(annualize='W')


# In[108]:


forecast_horizon = 10
lotion_forecasts = lotion_res.forecast(start='29-AUG-20', horizon=forecast_horizon)
lotion_forecasts


# In[109]:


lotion_intermediate = np.sqrt(lotion_forecasts.variance.dropna() * 252)
lotion_intermediate.head()


# In[110]:


lotion_final = lotion_intermediate.dropna().T
lotion_final


# In[111]:


lotion_final.plot()


# In[112]:


#Lotion SARIMA Model
lotion_sar_model=SARIMAX(endog=lotion_df['Lotion'], order=(2,1,2),
              seasonal_order=(2,1,2,12),
              trend='c',enforce_invertibility=False
             )
lotion_results=lotion_sar_model.fit()
print(lotion_results.summary())


# In[113]:


lotion_results.plot_diagnostics(figsize=(20,10))


# In[114]:


#Lotion ARIMA Model
lotion_arima_mod = ARIMA(lotion_df["Lotion"], order=(2,1,2))


# In[115]:


lotion_arima_results = lotion_arima_mod.fit()


# In[116]:


lotion_arima_results.summary()


# In[117]:


#Lotion Future Forecast Predict ARIMA
lotion_arima_results.plot_predict()
plt.savefig('Project_2/Images/Lotion_ARIMA')


# In[ ]:





# In[118]:


#Makeup Product Sales
makeup_df


# In[119]:


#Decomposing Makeup Data
decomp = sm.tsa.seasonal_decompose(makeup_df['Makeup'],freq=40)
makeup_decomp_data = pd.concat([makeup_df, decomp.trend, decomp.seasonal, decomp.resid], axis=1)
makeup_decomp_data.columns = ['Makeup','makeup_promo', 'trend', 'seasonal', 'resid']


# In[120]:


makeup_decomp_data.head()


# In[121]:


# Seasonality of Makeup
makeup_sales_decomposed = makeup_decomp_data['seasonal'].hvplot.line(xlabel = "Date", title = "Makeup Sales Decomposed")
hvplot.save(makeup_sales_decomposed, "Project_2/Images/Makeup_Sales_Decomposed.png")
makeup_sales_decomposed


# In[122]:


makeup_decomp_data['seasonal'].plot()


# In[123]:


# Trend of Lotion Sales
makeup_sales_trend = makeup_decomp_data['trend'].hvplot.line(xlabel = "Date", title = "Makeup Sales Trend")
hvplot.save(makeup_sales_trend, "Project_2/Images/Makeup_Sales_Trend.png")
makeup_sales_trend


# In[124]:


makeup_decomp_data['trend'].plot()


# In[125]:


# Residuals of Lotion Sales
makeup_residual = makeup_decomp_data['resid'].hvplot.line(xlabel = "Date", title = "Makeup Sales Residual")
hvplot.save(makeup_residual, "Project_2/Images/Makeup_Sales_Residual.png")
makeup_residual


# In[126]:


makeup_decomp_data['resid'].plot()


# In[127]:


#Makeup Noise and Trend
makeup_noise, makeup_trend = sm.tsa.filters.hpfilter(makeup_df['Makeup'])


# In[128]:


makeup_noise_graph = makeup_noise.hvplot.line(title = "Makeup Noise")
hvplot.save(makeup_noise_graph, 'Project_2/Images/Makeup_Noise.png')
makeup_noise_graph


# In[129]:


makeup_noise.plot()


# In[130]:


# Lotion Trend
makeup_trend_graph = makeup_trend.hvplot.line(xlabel = "Date", title = "Makeup Trend")
hvplot.save(makeup_trend_graph, "Project_2/Images/Makeup_Trend.png")
makeup_trend_graph


# In[131]:


makeup_trend.plot()


# In[132]:


#ADFuller Test -Seasonal Data is Stationary Makeup Seasonal Data
result = adfuller(makeup_decomp_data["seasonal"], autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}') 


# In[133]:


#Transform Makeup Data into Stationary Data
makeup_df['returns']= makeup_df['Makeup'].pct_change()
makeup_df=makeup_df.dropna()
makeup_df.head()


# In[134]:


makeup_df['returns'].plot()


# In[135]:


#Makeup Autocorrelation
plot_acf(makeup_df['Makeup'], lags=40)


# In[136]:


plot_pacf(makeup_df['Makeup'], zero=False, lags=40)


# In[137]:


#Makeup Garch Results
makeup_garch_returns = makeup_df['Makeup'].pct_change() * 100
makeup_garch_returns = makeup_garch_returns.dropna()
makeup_garch_returns.tail()


# In[138]:


model = arch_model(makeup_garch_returns, mean='Zero', vol='GARCH', p=1, q=1)


# In[139]:


res = model.fit(disp='off')


# In[140]:


res.summary()


# In[141]:


#Makeup Sales Volatility
fig = res.plot(annualize='W')


# In[142]:


forecast_horizon = 10
forecasts = res.forecast(start='29-AUG-20', horizon=forecast_horizon)
forecasts


# In[143]:


intermediate = np.sqrt(forecasts.variance.dropna() * 52)
intermediate.head()


# In[144]:


final = intermediate.dropna().T
final


# In[145]:


final.plot()


# In[146]:


#SARIMA Analysis
makeup_model=SARIMAX(endog=makeup_df['Makeup'].astype('float'), order=(2,1,2),
              seasonal_order=(2,1,2,12),
              trend='c',enforce_invertibility=False
             )
makeup_results=makeup_model.fit()
print(makeup_results.summary())


# In[147]:


#Makeup SARIMA Diagnostic
makeup_results.plot_diagnostics(figsize=(20,10))


# In[148]:


#Makeup ARIMA model
makeup_model = ARIMA(makeup_df['Makeup'].values, order=(2, 1, 2))
makeup_results = makeup_model.fit()
makeup_results.summary()


# In[149]:


#Makeup Future Predict
makeup_results.plot_predict()
plt.savefig('Project_2/Images/Makeup_ARIMA.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[150]:


# Determing if promos affect sales.


# In[151]:


X = acne_df['Acne'].to_frame()


# In[152]:


# Generate a column with the week of the year
X['Week_of_Year'] = X.index.weekofyear
X.head()


# In[153]:


acne_week=X.copy()


# In[154]:


acne_scatter = X["Acne"].hvplot.scatter(x = "Week_of_Year", y = "Acne", title = "Acne Sales Scatterplot")
hvplot.save(acne_scatter, "Project_2/Images/Acne_Scatter.png")
acne_scatter


# In[156]:


# Binary encode the weekly column to use as new input features for the model
X_binary_encoded = pd.get_dummies(X, columns=['Week_of_Year'])
X_binary_encoded.head()


# In[157]:


# The Acne Sales column must be deleted from the input features
X_binary_encoded = X_binary_encoded.drop('Acne', axis=1)
X_binary_encoded.head()


# In[158]:


# The target for our model is to predict the Acne Sales
y = acne_df['Acne'].copy()


# In[159]:


# Create the Linear Regression model
model = LinearRegression()

# Fit the model to the data (Train the model)
model.fit(X_binary_encoded, y)

# Predict the sales using the week of the year as inputs
predictions = model.predict(X_binary_encoded)


# In[160]:



from sklearn.metrics import mean_squared_error, r2_score

# The default scoring method is the R2 score
score = model.score(X_binary_encoded, y)
r2 = r2_score(y, predictions)

print(f"Score: {score}, r2: {r2}")


# In[161]:


# Mean Squared Error
mse = mean_squared_error(y, predictions)
mse


# In[162]:


# Root Mean Squared Error
rmse = np.sqrt(mse)
rmse


# In[163]:


# Standard deviation
np.std(y)


# In[164]:


plt.scatter(X['Week_of_Year'], y)
plt.plot(X['Week_of_Year'], predictions, color='red')
plt.savefig('Project_2/Images/Acne_Scatter_MSE.png')


# In[165]:


### This graph and R2 shows that we can only predict the sales based on what week of the year with a 12 % accuracy. 


# In[166]:


## Re Running for Makeup as it is the second largest to 


# In[167]:


X = makeup_df['Makeup'].to_frame()


# In[168]:


# Generate a column with the week of the year
X['Week_of_Year'] = X.index.weekofyear
X.head()


# In[169]:


makeup_week=X.copy()


# In[170]:


makeup_scatter = X["Makeup"].hvplot.scatter(x = "Week_of_Year", y = "Makeup", title = "Makeup Sales Scatterplot")
hvplot.save(makeup_scatter, "Project_2/Images/Makeup_Scatter.png")
makeup_scatter


# In[172]:


X_binary_encoded = pd.get_dummies(X, columns=['Week_of_Year'])
X_binary_encoded = X_binary_encoded.drop('Makeup', axis=1)


# In[173]:


y = makeup_df['Makeup'].copy()


# In[174]:


# Create the Linear Regression model
model = LinearRegression()

# Fit the model to the data (Train the model)
model.fit(X_binary_encoded, y)

# Predict the sales using the week of the year as inputs
predictions = model.predict(X_binary_encoded)


# In[175]:


from sklearn.metrics import mean_squared_error, r2_score

# The default scoring method is the R2 score
score = model.score(X_binary_encoded, y)
r2 = r2_score(y, predictions)

print(f"Score: {score}, r2: {r2}")


# In[176]:


# Mean Squared Error
mse = mean_squared_error(y, predictions)
mse


# In[177]:


plt.scatter(X['Week_of_Year'], y)
plt.plot(X['Week_of_Year'], predictions, color='red')
plt.savefig('Project_2/Images/Makeup_Scatter_MSE.png')


# In[178]:


## Now trying Lotion data


# In[186]:


X = lotion_df['Lotion'].to_frame()


# In[187]:


# Generate a column with the week of the year
X['Week_of_Year'] = X.index.weekofyear
X.head()


# In[188]:


lotion_scatter = X["Lotion"].hvplot.scatter(x = "Week_of_Year", y = "Lotion", title = "Lotion Sales Scatterplot")
hvplot.save(lotion_scatter, "Project_2/Images/Lotion_Scatter.png")
lotion_scatter


# In[189]:


# Binary encode the weekly column to use as new input features for the model
X_binary_encoded = pd.get_dummies(X, columns=['Week_of_Year'])
X_binary_encoded.head()


# In[190]:


# The Acne Sales column must be deleted from the input features
X_binary_encoded = X_binary_encoded.drop("Lotion", axis=1)


# In[191]:


y = lotion_df['Lotion'].copy()


# In[192]:


# Create the Linear Regression model
model = LinearRegression()

# Fit the model to the data (Train the model)
model.fit(X_binary_encoded, y)

# Predict the sales using the week of the year as inputs
predictions = model.predict(X_binary_encoded)


# In[193]:


from sklearn.metrics import mean_squared_error, r2_score

# The default scoring method is the R2 score
score = model.score(X_binary_encoded, y)
r2 = r2_score(y, predictions)

print(f"Score: {score}, r2: {r2}")


# In[194]:


# Mean Squared Error
mse = mean_squared_error(y, predictions)
mse


# In[195]:


plt.scatter(X['Week_of_Year'], y)
plt.plot(X['Week_of_Year'], predictions, color='red')
plt.savefig('Project_2/Images/Lotion_Scatter_MSE.png')


# In[196]:


## The Lotion was the strongest model with 84% model fit for having the week predict the sales outcome


# In[ ]:





# In[ ]:


### Now trying to determine if promos effect sales using A/B Testing ###


# In[ ]:


### Code below implementeed from https://harvard-iacs.github.io/2018-CS109A/lectures/lecture-23/demo/ ###


# In[ ]:


# H(NULL)=The Promos have no effect on sales
# H1== Promos have an effect on sales price


# In[ ]:


## ACNE TEST


# In[197]:


acne_df=acne_df.astype(int)
acne_df1=acne_df.dropna()


# In[198]:


#creating the sales and promo measures for each group
controlgroup = acne_df1['Acne'][acne_df1['acne_promo']==0]
trt1group = acne_df1['Acne'][acne_df1['acne_promo']==1]


# In[200]:


# The two-sample t-test is used to test whether the unknown means of two groups are equal or not.


# In[201]:


#two sample t-test testing promo and non promo group
sp.stats.ttest_ind(controlgroup,trt1group)


# In[202]:


#ANOVA F-test
sp.stats.f_oneway(controlgroup,trt1group)


# In[203]:


## P <.05 so we reject the NULL Hyptohesis and conclude that promos are affecting sales performance. 


# In[204]:


##insert graph here


# In[205]:


fig = plt.figure(figsize= (20, 10))
ax = fig.add_subplot(111)


p_bp_male = plt.hist(acne_df1['Acne'][acne_df1['acne_promo']==0], label= "Non-Promos",color="red",
                     density= True,
                     alpha=0.75)
p_bp_female = plt.hist(acne_df1['Acne'][acne_df1['acne_promo']==1], label= "Promos",color="blue",
                       density= True,
                       alpha=0.75)


plt.suptitle("Promo vs. Non- Promo Sales of Acne", fontsize= 20)
plt.xlabel("Sales", fontsize= 16)
plt.ylabel("Probability density", fontsize= 16)



plt.show()


# In[206]:


##Using Makeup


# In[207]:


#creating the sales and promo measures for each group
controlgroup = makeup_df['Makeup'][makeup_df['makeup_promo']==0]
trt1group = makeup_df['Makeup'][makeup_df['makeup_promo']==1]


# In[208]:


#two sample t-test testing promo and non promo group
sp.stats.ttest_ind(controlgroup,trt1group)


# In[209]:


#ANOVA F-test
sp.stats.f_oneway(controlgroup,trt1group)


# In[210]:


fig = plt.figure(figsize= (20, 10))
ax = fig.add_subplot(111)


p_bp_male = plt.hist(makeup_df['Makeup'][makeup_df['makeup_promo']==0], label= "Non-Promos",color="red",
                     density= True,
                     alpha=0.75)
p_bp_female = plt.hist(makeup_df['Makeup'][makeup_df['makeup_promo']==1], label= "Promos",color="blue",
                       density= True,
                       alpha=0.75)


plt.suptitle("Promo vs. Non- Promo Sales of Makeup", fontsize= 20)
plt.xlabel("Sales", fontsize= 16)
plt.ylabel("Probability density", fontsize= 16)



plt.show()


# In[211]:


## Using Lotion


# In[212]:


#creating the sales and promo measures for each group
controlgroup = lotion_df['Lotion'][lotion_df['lotion_promo']==0]
trt1group = lotion_df['Lotion'][lotion_df['lotion_promo']==1]


# In[213]:


#two sample t-test testing promo and non promo group
sp.stats.ttest_ind(controlgroup,trt1group)


# In[214]:


#ANOVA F-test
sp.stats.f_oneway(controlgroup,trt1group)


# In[215]:


fig = plt.figure(figsize= (20, 10))
ax = fig.add_subplot(111)


p_bp_male = plt.hist(lotion_df['Lotion'][lotion_df['lotion_promo']==0], label= "Non-Promos",color="red",
                     density= True,
                     alpha=0.75)
p_bp_female = plt.hist(lotion_df['Lotion'][lotion_df['lotion_promo']==1], label= "Promos",color="blue",
                       density= True,
                       alpha=0.75)


plt.suptitle("Promo vs. Non- Promo Sales of Lotion", fontsize= 20)
plt.xlabel("Sales", fontsize= 16)
plt.ylabel("Probability density", fontsize= 16)



plt.show()


# In[ ]:





# In[216]:


## Running SVR Prediction on Acne Sales prices


# In[217]:


acne_df


# In[218]:


acne_df['Week'] = acne_df.index.weekofyear


# In[219]:


acne_df.head()


# In[220]:


X = acne_df.iloc[:,0].values     
y = acne_df.iloc[:,3].values        


# In[221]:


# split the dataset into test set and train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)


# In[222]:


X_train[:3]
# some features are big in scale, so adjust the scale


# In[223]:


X_train= X_train.reshape(-1,1)
y_train= y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)


# In[224]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[225]:


# PAY ATTENTION, WE ARE IMPORTING SVR, WHICH IS FOR REGRESSION
from sklearn.svm import SVR


# In[226]:


regressor = SVR(kernel = 'rbf') # using rbf kernel
# also, we here call it a regressor, since it is a regression

# in SVC = support vector classification, many name it as a classifier


# In[227]:


regressor.fit(X_train, y_train)


# In[228]:


predictions = regressor.predict(X_test)


# In[229]:


from sklearn.metrics import r2_score,mean_squared_error
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
rmse


# In[230]:


r2_score(y_test,predictions)


# In[231]:


# Visualising the Regression results
plt.scatter(y_test, predictions, color = 'red')
plt.plot(y_test, predictions, color = 'blue')
plt.title('Predictions by Week of Year')
plt.xlabel('Test Week')
plt.ylabel('Predicted Week')
plt.show()


# In[232]:


## Outsput shows that the model is only 8.2% accurate in prediction. 


# In[233]:


##Using Makeup


# In[234]:


makeup_df.head()


# In[235]:


X = makeup_df.iloc[:,0].values     
y = makeup_df.iloc[:,2].values        


# In[236]:


# split the dataset into test set and train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)


# In[237]:


X_train[:3]
# some features are big in scale, so adjust the scale


# In[238]:


X_train= X_train.reshape(-1,1)
y_train= y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)


# In[239]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[240]:


# PAY ATTENTION, WE ARE IMPORTING SVR, WHICH IS FOR REGRESSION
from sklearn.svm import SVR


# In[241]:


regressor = SVR(kernel = 'rbf') # using rbf kernel
# also, we here call it a regressor, since it is a regression

# in SVC = support vector classification, many name it as a classifier


# In[242]:


regressor.fit(X_train, y_train)


# In[243]:


predictions = regressor.predict(X_test)


# In[244]:


from sklearn.metrics import r2_score,mean_squared_error
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
rmse


# In[245]:


r2_score(y_test,predictions)


# In[246]:


# Visualising the Regression results
plt.scatter(y_test, predictions, color = 'red')
plt.plot(y_test, predictions, color = 'blue')
plt.title('Predictions by Week of Year')
plt.xlabel('Test Week')
plt.ylabel('Predicted Week')
plt.show()


# In[247]:


## Using Lotion


# In[248]:


lotion_df['Week'] = makeup_df.index.weekofyear


# In[249]:


lotion_df.head()


# In[250]:


X = lotion_df.iloc[:,0].values     
y = lotion_df.iloc[:,2].values        


# In[251]:


# split the dataset into test set and train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)


# In[252]:


X_train[:3]


# In[253]:


X_train= X_train.reshape(-1,1)
y_train= y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)


# In[254]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[255]:


from sklearn.svm import SVR


# In[256]:


regressor = SVR(kernel = 'rbf')


# In[257]:


regressor.fit(X_train, y_train)


# In[258]:


predictions = regressor.predict(X_test)


# In[259]:


from sklearn.metrics import r2_score,mean_squared_error
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
rmse


# In[260]:


r2_score(y_test,predictions)


# In[261]:


# Visualising the Regression results
plt.scatter(y_test, predictions, color = 'red')
plt.plot(y_test, predictions, color = 'blue')
plt.title('Predictions by Week of Year')
plt.xlabel('Test Week')
plt.ylabel('Predicted Week')
plt.show()


# In[262]:


###LSTM Predictions  Acne


# In[263]:


df=acne_df.copy()


# In[264]:


# This function accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y
def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)


# In[265]:


# Predict Closing Prices using a 10 day window of previous closing prices

window_size = 6
# Column index 0 is the 'fng_value' column
# Column index 1 is the `Close` column
feature_column = 1
target_column = 0
X, y = window_data(df, window_size, feature_column, target_column)


# In[266]:


# Use 70% of the data for training and the remaineder for testing
from sklearn.model_selection import train_test_split

split = int(0.7 * len(X))
X_train = X[: split]
X_test = X[split:]
y_train = y[: split]
y_test = y[split:]


# In[267]:


from sklearn.preprocessing import MinMaxScaler
# Use the MinMaxScaler to scale data between 0 and 1.
scaler = MinMaxScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler.fit(y)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)


# In[268]:


# Reshape the features for the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# In[269]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# In[270]:


# Build the LSTM model. 


# In[271]:


model = Sequential()

number_units = 5 # The number of units in each LSTM layer, is equal to the size of the time window
dropout_fraction = 0.2 # fraction of nodes that will be dropped on each epoch. randomly drop 20% of the units.

# Model set-up
number_input_features = 20
hidden_nodes_layer1 = 5
hidden_nodes_layer2 = 5


# Layer 1
model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )
model.add(Dropout(dropout_fraction))
# Layer 2
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))
# Layer 3
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))
# Output layer
model.add(Dense(1))


# In[272]:


# Compile the model
model.compile(optimizer="adam", loss = "mean_squared_error")


# In[273]:


# Summarize the model
model.summary()


# In[274]:


# Train the model

model.fit(X_train, y_train, epochs =10, shuffle = False, batch_size=1, verbose=1 ) 


# In[275]:


# Evaluate the model
model.evaluate(X_test, y_test)


# In[276]:


# Make some predictions
predicted= model.predict(X_test)


# In[277]:


# Recover the original prices instead of the scaled version
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[278]:


# Create a DataFrame of Real and Predicted values
acne_pred_vs_real = pd.DataFrame({
    "Real Sales": real_prices.ravel(),
    "Predicted Sales": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ]) 
acne_pred_vs_real.head()


# In[279]:


acne_lstm = acne_pred_vs_real.hvplot(title = "Acne Predicted Sales vs Real Sales")
hvplot.save(acne_lstm, 'Project_2/Images/Acne_LSTM.png')
acne_lstm


# In[280]:


acne_pred_vs_real.plot()  ## Acne


# In[281]:


##Testing with Lotion


# In[282]:


df=lotion_df


# In[283]:


df.head()


# In[284]:


window_size = 6
# Column index 0 is the 'fng_value' column
# Column index 1 is the `Close` column
feature_column = 1
target_column = 0
X, y = window_data(df, window_size, feature_column, target_column)


# In[285]:


from sklearn.model_selection import train_test_split

split = int(0.7 * len(X))
X_train = X[: split]
X_test = X[split:]
y_train = y[: split]
y_test = y[split:]


# In[286]:


scaler = MinMaxScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler.fit(y)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)


# In[287]:


# Reshape the features for the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# In[288]:


model = Sequential()

number_units = 5 # The number of units in each LSTM layer, is equal to the size of the time window
dropout_fraction = 0.2 # fraction of nodes that will be dropped on each epoch. randomly drop 20% of the units.

# Model set-up
number_input_features = 20
hidden_nodes_layer1 = 10
hidden_nodes_layer2 = 10


# Layer 1
model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )
model.add(Dropout(dropout_fraction))
# Layer 2
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))
# Layer 3
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))
# Output layer
model.add(Dense(1))


# In[289]:


# Compile the model
model.compile(optimizer="adam", loss = "mean_squared_error")


# In[290]:


# Summarize the model
model.summary()


# In[291]:


model.fit(X_train, y_train, epochs =10, shuffle = False, batch_size=1, verbose=1 ) 


# In[292]:


model.evaluate(X_test, y_test)


# In[293]:


predicted= model.predict(X_test)


# In[294]:


predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[295]:


lotion_pred_vs_real = pd.DataFrame({
    "Real Sales": real_prices.ravel(),
    "Predicted Sales": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ]) 
lotion_pred_vs_real.head()


# In[296]:


lotion_lstm = lotion_pred_vs_real.hvplot(title = "Lotion Predicted Sales vs Real Sales")
hvplot.save(lotion_lstm, 'Project_2/Images/Lotion_LSTM.png')
lotion_lstm


# In[297]:


lotion_pred_vs_real.plot()  ## Lotion Pred vs Actual


# In[298]:


# Testing Makeup


# In[299]:


df=makeup_df
df.head()


# In[300]:


window_size = 6
# Column index 0 is the 'fng_value' column
# Column index 1 is the `Close` column
feature_column = 1
target_column = 0
X, y = window_data(df, window_size, feature_column, target_column)


# In[301]:


from sklearn.model_selection import train_test_split

split = int(0.7 * len(X))
X_train = X[: split]
X_test = X[split:]
y_train = y[: split]
y_test = y[split:]


# In[302]:


scaler = MinMaxScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler.fit(y)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)


# In[303]:


# Reshape the features for the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# In[304]:


model = Sequential()

number_units = 5 # The number of units in each LSTM layer, is equal to the size of the time window
dropout_fraction = 0.2 # fraction of nodes that will be dropped on each epoch. randomly drop 20% of the units.

# Model set-up
number_input_features = 20
hidden_nodes_layer1 = 10
hidden_nodes_layer2 = 10


# Layer 1
model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )
model.add(Dropout(dropout_fraction))
# Layer 2
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))
# Layer 3
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))
# Output layer
model.add(Dense(1))


# In[305]:


# Compile the model
model.compile(optimizer="adam", loss = "mean_squared_error")


# In[306]:


# Summarize the model
model.summary()


# In[307]:


model.fit(X_train, y_train, epochs =10, shuffle = False, batch_size=1, verbose=1 ) 


# In[308]:


model.evaluate(X_test, y_test)


# In[309]:


predicted= model.predict(X_test)


# In[310]:


predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[311]:


makeup_pred_vs_real = pd.DataFrame({
    "Real Sales": real_prices.ravel(),
    "Predicted Sales": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ]) 
makeup_pred_vs_real.head()


# In[312]:


makeup_lstm = makeup_pred_vs_real.hvplot(title = "Makeup Predicted Sales vs Real Sales")
hvplot.save(makeup_lstm, 'Project_2/Images/Makeup_LSTM.png')
makeup_lstm


# In[ ]:





# In[ ]:





# In[315]:


# Dashboard
dashboard = pn.Tabs(
        ("Welcome",
        pn.Column("Project_2/Images/Tobias_Fire_Sale.gif")),
        ("DATA",
        pn.Column(acne_plot, "Project_2/Images/Acne_Sales_Decomposed.png", "Project_2/Images/Acne_Sales_Trend.png", "Project_2/Images/Acne_Sales_Residual.png",
                  lotion_plot, "Project_2/Images/Lotion_Sales_Decomposed.png", "Project_2/Images/Lotion_Sales_Trend.png", "Project_2/Images/Lotion_Sales_Residual.png",
                  makeup_plot, "Project_2/Images/Makeup_Sales_Decomposed.png", "Project_2/Images/Makeup_Sales_Trend.png", "Project_2/Images/Makeup_Sales_Residual.png")),
        ("ARIMA Method",
        pn.Column("Project_2/Images/Acne_ARIMA.png", "Project_2/Images/Acne_Scatter.png", "Acne Scatterplot of Sales as Predictor", "Project_2/Images/Acne_Scatter_MSE.png", "Project_2/Images/ProvNoAcne.png",
                  "Project_2/Images/Lotion_ARIMA.png","Project_2/Images/Lotion_Scatter.png", "Lotion Scatterplot of Sales as Predictor", "Project_2/Images/Lotion_Scatter_MSE.png", "Project_2/Images/ProvNoLotion.png",
                  "Project_2/Images/Makeup_ARIMA.png", "Project_2/Images/Makeup_Scatter.png", "Makeup Scatterplot of Sales as Predictor", "Project_2/Images/Makeup_Scatter_MSE.png", "Project_2/Images/ProvNoMakeup.png")),
        ("SVR",
        pn.Column()),
        ("LSTM Method",
        pn.Column("Project_2/Images/Acne_LSTM.png", "Project_2/Images/Lotion_LSTM.png", "Project_2/Images/Makeup_LSTM.png")),
         ("Conclusions",
         pn.Column("Project_2/Images/Tobias_Fire_Sale.jpg"))
    )


# In[316]:


# Serves Panel
dashboard.servable()


# In[ ]:




