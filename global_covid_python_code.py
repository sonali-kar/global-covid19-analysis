import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot


import plotly.io as pio
pio.renderers.default='browser'


url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

url_total = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv'

confirmed_cases = pd.read_csv(url_confirmed)
print(confirmed_cases.shape)

deaths_cases = pd.read_csv(url_deaths)
print(deaths_cases.shape)

recovered_cases = pd.read_csv(url_recovered)
print(recovered_cases.shape)

#total_cases = pd.read_csv('confirmed_cases.csv')
total_cases = pd.read_csv(url_total)
print(total_cases.shape)


global_cases = total_cases
global_cases = global_cases.drop(['Last_Update','Lat','Long_','Incident_Rate','People_Tested','People_Hospitalized',
                                  'Mortality_Rate','UID','ISO3'],axis=1)

global_cases_sum = pd.DataFrame(global_cases.sum()).transpose()
#global_cases_sum.("print "{:f}".format(float("1.70000043572e-05"))")

global_cases_sum['Confirmed'] = global_cases_sum['Confirmed'].astype('int64') 
global_cases_sum['Active'] = global_cases_sum['Active'].astype('int64') 
global_cases_sum['Deaths'] = global_cases_sum['Deaths'].astype('int64') 
global_cases_sum['Recovered'] = global_cases_sum['Recovered'].astype('int64') 
confirmed_cases = confirmed_cases.drop(['Lat','Long'],axis=1)
deaths_cases = deaths_cases.drop(['Lat','Long'],axis=1)
recovered_cases = recovered_cases.drop(['Lat','Long'],axis=1)



# *************************************************************************** 
confirmed_cases_copy = confirmed_cases
confirmed_cases_copy = confirmed_cases_copy.drop(['Province/State','Country/Region'],axis=1)
confirmed_cases_copy_sum = pd.DataFrame(confirmed_cases_copy.sum()) 




#**********************************************************************
confirmed_cases_sum_index =  confirmed_cases_copy_sum.index
confirmed_cases_sum_values = confirmed_cases_copy_sum.values
confirmed_cases_dates_df = pd.DataFrame(confirmed_cases_sum_index)


confirmed_cases_sum_values = pd.DataFrame(confirmed_cases_sum_values)
confirmed_cases_dates_df.columns = ['date']
 
# ****************************************************************************



total_death_cases = deaths_cases
total_death_cases = total_death_cases.drop(['Province/State','Country/Region'],axis=1)
total_death_cases_sum = pd.DataFrame(total_death_cases.sum()) 

death_cases_sum_index =  total_death_cases_sum.index
death_cases_sum_values = total_death_cases_sum.values
death_cases_sum_values = pd.DataFrame(death_cases_sum_values)

#***********************************
death_cases_dates_df = pd.DataFrame(death_cases_sum_index)
death_cases_dates_df.columns = ['date']
#*********************************

# -------------------------------------------------------------------------------


total_recovered_cases = recovered_cases
total_recovered_cases = total_recovered_cases.drop(['Province/State','Country/Region'],axis=1)
total_recovered_cases_sum = pd.DataFrame(total_recovered_cases.sum()) 

recovered_cases_sum_index =  total_recovered_cases_sum.index
recovered_cases_sum_values = total_recovered_cases_sum.values

recovered_cases_sum_values = pd.DataFrame(recovered_cases_sum_values)
#******new********************************************************************

recovered_cases_dates_df = pd.DataFrame(recovered_cases_sum_index)
recovered_cases_dates_df.columns = ['date']   
    
#**********************************************************************************    
    
active_cases_sum_values = np.abs(confirmed_cases_sum_values[0]-death_cases_sum_values[0]-recovered_cases_sum_values[0])
active_cases_sum_values= pd.DataFrame(active_cases_sum_values)



#**********************************************
fig = go.Figure()
for col in confirmed_cases_sum_values.columns:
    fig.add_trace(go.Scatter(x=confirmed_cases_copy_sum.index , y = confirmed_cases_sum_values[col], name='Confirmed',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=total_death_cases_sum.index , y = death_cases_sum_values[col],name='Deaths',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=total_recovered_cases_sum.index , y = recovered_cases_sum_values[col],name='Recovered',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=total_death_cases_sum.index , y = active_cases_sum_values[col],name='Active',mode='lines+markers'))
fig.show()
 
# *********************************************** 


global_cases.copy().sort_values('Confirmed',ascending = False).reset_index(drop=True).style.bar(align='left',width= 98, color= '#d65f5f')




#************************************************************************************************

import datetime as dt
deaths_dates = death_cases_dates_df['date']
confirmed_dates = confirmed_cases_dates_df['date']
recovered_dates = recovered_cases_dates_df['date']
active_dates = confirmed_cases_dates_df['date']


date_format_deaths = [pd.to_datetime(d) for d in deaths_dates]
date_format_confirmed = [pd.to_datetime(d) for d in confirmed_dates]

date_format_recovered = [pd.to_datetime(d) for d in recovered_dates]
date_format_active = [pd.to_datetime(d) for d in active_dates]


import numpy as np
from sklearn import linear_model

X_train_deaths = date_format_deaths
X_train_confirmed = date_format_confirmed
X_train_recovered = date_format_recovered
X_train_active = date_format_active


y_train_confirmed = confirmed_cases_sum_values
y_train_deaths = death_cases_sum_values
y_train_recovered = recovered_cases_sum_values
y_train_active = active_cases_sum_values


starting_peak_date = 58
#starting_date_confirmed = 58

day_numbers = []

for i in range(1, (len(X_train_deaths)+1)):
    day_numbers.append([i])
X_train_deaths = day_numbers
X_train_confirmed = day_numbers
X_train_recovered = day_numbers
X_train_active = day_numbers

X_train_confirmed = X_train_confirmed[starting_peak_date:]
X_train_deaths = X_train_deaths[starting_peak_date:]
X_train_recovered = X_train_recovered[starting_peak_date:]
X_train_active = X_train_active[starting_peak_date:]


y_train_deaths = y_train_deaths[starting_peak_date:]
y_train_confirmed = y_train_confirmed[starting_peak_date:]
y_train_recovered = y_train_recovered[starting_peak_date:]
y_train_active = y_train_active[starting_peak_date:]


linear_regr_deaths = linear_model.LinearRegression()
linear_regr_confirmed = linear_model.LinearRegression()
linear_regr_recovered = linear_model.LinearRegression()
linear_regr_active = linear_model.LinearRegression()


linear_regr_deaths.fit(X_train_deaths, y_train_deaths)
linear_regr_confirmed.fit(X_train_confirmed, y_train_confirmed)
linear_regr_recovered.fit(X_train_recovered,y_train_recovered)
linear_regr_active.fit(X_train_active,y_train_active)


print ("Linear Regression Model Score for Confirmed cases: %s" % (linear_regr_confirmed.score(X_train_confirmed, y_train_confirmed)))
print ("Linear Regression Model Score for Deaths cases: %s" % (linear_regr_deaths.score(X_train_deaths, y_train_deaths)))
print ("Linear Regression Model Score for Recovered cases: %s" % (linear_regr_recovered.score(X_train_recovered, y_train_recovered)))
print ("Linear Regression Model Score for Active cases: %s" % (linear_regr_active.score(X_train_active, y_train_active)))


# Predict future trend
from sklearn.metrics import max_error
import math
y_pred_confirmed = linear_regr_confirmed.predict(X_train_confirmed)
y_pred_deaths = linear_regr_deaths.predict(X_train_deaths)
y_pred_recovered = linear_regr_recovered.predict(X_train_recovered)
y_pred_active = linear_regr_active.predict(X_train_active)



error_deaths = max_error(y_train_deaths, y_pred_deaths)
error_confirmed = max_error(y_train_confirmed, y_pred_confirmed)
error_recovered= max_error(y_train_recovered, y_pred_recovered)
error_active= max_error(y_train_active, y_pred_active)



X_test = []
future_days = 100
for i in range(starting_peak_date, starting_peak_date + future_days):
    X_test.append([i])
y_pred_linear_deaths = linear_regr_deaths.predict(X_test)
y_pred_linear_confirmed = linear_regr_confirmed.predict(X_test)
y_pred_linear_recovered = linear_regr_recovered.predict(X_test)
y_pred_linear_active = linear_regr_active.predict(X_test)


y_pred_max_deaths = []
y_pred_min_deaths = []
for i in range(0, len(y_pred_linear_deaths)):
    y_pred_max_deaths.append(y_pred_linear_deaths[i] + error_deaths)
    y_pred_min_deaths.append(y_pred_linear_deaths[i] - error_deaths)
    

y_pred_max_confirmed = []
y_pred_min_confirmed = []
for i in range(0, len(y_pred_linear_confirmed)):
    y_pred_max_confirmed.append(y_pred_linear_confirmed[i] + error_confirmed)
    y_pred_min_confirmed.append(y_pred_linear_confirmed[i] - error_confirmed)
    


y_pred_max_recovered = []
y_pred_min_recovered = []
for i in range(0, len(y_pred_linear_recovered)):
    y_pred_max_recovered.append(y_pred_linear_recovered[i] + error_recovered)
    y_pred_min_recovered.append(y_pred_linear_recovered[i] - error_recovered)


y_pred_max_active = []
y_pred_min_active = []
for i in range(0, len(y_pred_linear_active)):
    y_pred_max_active.append(y_pred_linear_active[i] + error_active)
    y_pred_min_active.append(y_pred_linear_active[i] - error_active)



y_pred_linear_deaths = pd.DataFrame(y_pred_linear_deaths)
y_pred_linear_confirmed = pd.DataFrame(y_pred_linear_confirmed)
y_pred_linear_recovered = pd.DataFrame(y_pred_linear_recovered)
y_pred_linear_active = pd.DataFrame(y_pred_linear_active)


y_train_confirmed = pd.DataFrame(y_train_confirmed)
y_train_deaths = pd.DataFrame(y_train_deaths)
y_train_recovered = pd.DataFrame(y_train_recovered)
y_train_active = pd.DataFrame(y_train_active)


X_test = pd.DataFrame(X_test)
X_train_confirmed = pd.DataFrame(X_train_confirmed)
X_train_deaths = pd.DataFrame(X_train_deaths)
X_train_recovered = pd.DataFrame(X_train_recovered)  
X_train_active = pd.DataFrame(X_train_active)  




fig = go.Figure()
for col in y_train_deaths.columns:
    fig.add_trace(go.Scatter(x=X_test[col] , y = y_pred_linear_confirmed[col],name='Predicted CONFIRMED',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=X_train_confirmed[col] , y = y_train_confirmed[col],name='Confirmed',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=X_train_recovered[col] , y = y_train_recovered[col],name='Recovered',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=X_test[col] , y = y_pred_linear_recovered[col], name='Predicted RECOVERED',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=X_train_deaths[col] , y = y_train_deaths[col],name='Deaths',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=X_test[col] , y = y_pred_linear_deaths[col], name='Predicted DEATHS',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=X_train_active[col] , y = y_train_active[col],name='Active',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=X_test[col] , y = y_pred_linear_active[col], name='Predicted ACTIVE',mode='lines+markers'))
fig.show()





# # plot maximum error
# plt.plot(X_test, y_pred_max, color='red', linewidth=1, linestyle='dashed')
# #plot minimum error
# plt.plot(X_test, y_pred_min, color='black', linewidth=1, linestyle='dashed')
# plt.xlabel('Days')
# plt.xlim(starting_date, starting_date + future_days)
# plt.xticks(x_ticks, date_prev)
# plt.ylabel('gravi_deceduti')
# plt.yscale("log")
# plt.savefig("prediction.png")
# plt.show() 





#*******************************************************************************
#           NOT WORKING LINEAR REGRESSION PART

# #********************************************************************************************


# import datetime as dt
# confirmed_cases_dates_df['date'] = pd.to_datetime(confirmed_cases_dates_df['date'])
# confirmed_cases_dates_df['Date']=confirmed_cases_dates_df['date'].map(dt.datetime.toordinal)

# del confirmed_cases_dates_df['date']


# # **********************************************
# death_cases_dates_df['date'] = pd.to_datetime(death_cases_dates_df['date'])
# death_cases_dates_df['Date']=death_cases_dates_df['date'].map(dt.datetime.toordinal)

# del death_cases_dates_df['date']
# # ************************************************


# from sklearn.model_selection import train_test_split


# x_lr = confirmed_cases_dates_df.values.reshape(-1,1)
# y_lr=confirmed_cases_sum_values.values.reshape(-1,1)


# # *********************************************



# x2_lr = death_cases_dates_df.values.reshape(-1,1)
# death_cases_sum_values_lr=death_cases_sum_values.values.reshape(-1,1)




# #******************************************************************************
# X_train, X_test, y_train, y_test = train_test_split(x2_lr, death_cases_sum_values_lr, test_size=0.1, random_state= 42)

# from sklearn.linear_model import LinearRegression
# linear_model = LinearRegression()
# linear_model.fit(X_train,y_train)
# test_linear_pred = linear_model.predict(X_test)
# test_linear_pred = test_linear_pred.astype('int64')
# #linear_pred = linear_model.predict(future_forcast)


# #****************************************************************



