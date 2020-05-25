import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot


import plotly.io as pio
pio.renderers.default='browser'


confirmed_cases = pd.read_csv('time_series_covid19_confirmed_global.csv')
print(confirmed_cases.shape)
deaths_cases = pd.read_csv('time_series_covid19_deaths_global.csv')
print(deaths_cases.shape)
recovered_cases = pd.read_csv('time_series_covid19_recovered_global.csv')
print(recovered_cases.shape)

#total_cases = pd.read_csv('confirmed_cases.csv')
total_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
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
total_confirmed_cases = confirmed_cases
total_confirmed_cases = total_confirmed_cases.drop(['Province/State','Country/Region'],axis=1)
total_confirmed_cases_sum = pd.DataFrame(total_confirmed_cases.sum()) 

x =  total_confirmed_cases_sum.index
y1 = total_confirmed_cases_sum.values

#x = pd.DataFrame(x)
y1 = pd.DataFrame(y1)

#x.columns = ['date']
#y.columns = ['cases']
 
# ****************************************************************************


total_death_cases = deaths_cases
total_death_cases = total_death_cases.drop(['Province/State','Country/Region'],axis=1)
total_death_cases_sum = pd.DataFrame(total_death_cases.sum()) 

x2 =  total_death_cases_sum.index
y2 = total_death_cases_sum.values

y2 = pd.DataFrame(y2)

# -------------------------------------------------------------------------------


total_recovered_cases = recovered_cases
total_recovered_cases = total_recovered_cases.drop(['Province/State','Country/Region'],axis=1)
total_recovered_cases_sum = pd.DataFrame(total_recovered_cases.sum()) 

x3 =  total_recovered_cases_sum.index
y3 = total_recovered_cases_sum.values

y3 = pd.DataFrame(y3)
    
    
#**********************************************************************************    
    
y4= np.abs(y1[0]-y2[0]-y3[0])
y4= pd.DataFrame(y4)

#label = ['Confirmed','Deaths','Recovered','Active']




#**********************************************
fig = go.Figure()
for col in y1.columns:
    fig.add_trace(go.Line(x=total_confirmed_cases_sum.index , y = y1[col], name='Confirmed'))
    fig.add_trace(go.Line(x=total_death_cases_sum.index , y = y2[col],name='Deaths'))
    fig.add_trace(go.Line(x=total_recovered_cases_sum.index , y = y3[col],name='Recovered'))
    fig.add_trace(go.Line(x=total_death_cases_sum.index , y = y4[col],name='Active'))
fig.show()
 

