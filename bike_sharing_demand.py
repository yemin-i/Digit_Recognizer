#!/usr/bin/env python
# coding: utf-8

# # 라이브러리 설정

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# # 데이터 로드

# In[4]:


train = pd.read_csv('train.csv')

print(train.shape)
train.head()


# In[5]:


test = pd.read_csv('test.csv')

print(test.shape)
test.head()


# # 결측치 확인

# In[6]:


train.isnull().sum()


# In[7]:


test.isnull().sum()


# # 전처리

# ## datetime

# In[8]:


# datetime 컬럼을 연,월,일,시,분,초 로 나누어 각각의 컬럼을 생성하겠습니다.

train['datetime'] = pd.to_datetime(train['datetime'])

train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second

train[['datetime','year','month','day','hour','minute','second']].head()


# In[9]:


# test 파일도 위와 동일한 작업을 하겠습니다.

test['datetime'] = pd.to_datetime(test['datetime'])

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test['minute'] = test['datetime'].dt.minute
test['second'] = test['datetime'].dt.second

test[['datetime','year','month','day','hour','minute','second']].head()


# # EDA

# ## datetime

# In[10]:


plt.rc('font', family='Malgun Gothic')


# In[11]:


f, ax = plt.subplots(2,3, figsize=(18,10))

sns.barplot(data=train, x='year', y='count', ax=ax[0][0])
sns.barplot(data=train, x='month', y='count', ax=ax[0][1])
sns.barplot(data=train, x='day', y='count', ax=ax[0][2])
sns.barplot(data=train, x='hour', y='count', ax=ax[1][0])
sns.barplot(data=train, x='minute', y='count', ax=ax[1][1])
sns.barplot(data=train, x='second', y='count', ax=ax[1][2])

ax[0][0].set_title('연도별 대여량')
ax[0][1].set_title('월별 대여량')
ax[0][2].set_title('일별 대여량')
ax[1][0].set_title('시간별 대여량')
ax[1][1].set_title('분별 대여량')
ax[1][2].set_title('초별 대여량')

plt.subplots_adjust(hspace=0.3)


# **시각화 결과, 연도, 월, 시간 컬럼에서의 패턴이 분명해 모델 성능을 높일 수 있을 것입니다.**

# ***year_month 컬럼***

# In[12]:


# year 컬럼과 month 컬럼을 합친 year_month 컬럼을 생성하겠습니다.

train['year_month'] = train['year'].astype('str') + '-' + train['month'].astype('str')

train[['year','month','year_month']]


# In[13]:


plt.figure(figsize=(15,4))
ax = sns.barplot(data=train, x='year_month', y='count')
ax.set_title('연-월별 대여량')


# **연-월별 대여량에서 차이가 있는 것으로 나타났습니다.**

# In[14]:


# 분석에 활용할 수 있게 One-Hot Encoding을 하겠습니다.

train['ym_1101'] = train['year_month'] == '2011-1'
train['ym_1102'] = train['year_month'] == '2011-2'
train['ym_1103'] = train['year_month'] == '2011-3'
train['ym_1104'] = train['year_month'] == '2011-4'
train['ym_1105'] = train['year_month'] == '2011-5'
train['ym_1106'] = train['year_month'] == '2011-6'
train['ym_1107'] = train['year_month'] == '2011-7'
train['ym_1108'] = train['year_month'] == '2011-8'
train['ym_1109'] = train['year_month'] == '2011-9'
train['ym_1110'] = train['year_month'] == '2011-10'
train['ym_1111'] = train['year_month'] == '2011-11'
train['ym_1112'] = train['year_month'] == '2011-12'
train['ym_1201'] = train['year_month'] == '2012-1'
train['ym_1202'] = train['year_month'] == '2012-2'
train['ym_1203'] = train['year_month'] == '2012-3'
train['ym_1204'] = train['year_month'] == '2012-4'
train['ym_1205'] = train['year_month'] == '2012-5'
train['ym_1206'] = train['year_month'] == '2012-6'
train['ym_1207'] = train['year_month'] == '2012-7'
train['ym_1208'] = train['year_month'] == '2012-8'
train['ym_1209'] = train['year_month'] == '2012-9'
train['ym_1210'] = train['year_month'] == '2012-10'
train['ym_1211'] = train['year_month'] == '2012-11'
train['ym_1212'] = train['year_month'] == '2012-12'


# In[15]:


# test 파일도 마찬가지로 year_month 컬럼을 생성하겠습니다.

test['year_month'] = test['year'].astype('str') + '-' + test['month'].astype('str')

test[['year','month','year_month']]


# In[16]:


# 분석에 활용할 수 있게 One-Hot Encoding을 하겠습니다.

test['ym_1101'] = test['year_month'] == '2011-1'
test['ym_1102'] = test['year_month'] == '2011-2'
test['ym_1103'] = test['year_month'] == '2011-3'
test['ym_1104'] = test['year_month'] == '2011-4'
test['ym_1105'] = test['year_month'] == '2011-5'
test['ym_1106'] = test['year_month'] == '2011-6'
test['ym_1107'] = test['year_month'] == '2011-7'
test['ym_1108'] = test['year_month'] == '2011-8'
test['ym_1109'] = test['year_month'] == '2011-9'
test['ym_1110'] = test['year_month'] == '2011-10'
test['ym_1111'] = test['year_month'] == '2011-11'
test['ym_1112'] = test['year_month'] == '2011-12'
test['ym_1201'] = test['year_month'] == '2012-1'
test['ym_1202'] = test['year_month'] == '2012-2'
test['ym_1203'] = test['year_month'] == '2012-3'
test['ym_1204'] = test['year_month'] == '2012-4'
test['ym_1205'] = test['year_month'] == '2012-5'
test['ym_1206'] = test['year_month'] == '2012-6'
test['ym_1207'] = test['year_month'] == '2012-7'
test['ym_1208'] = test['year_month'] == '2012-8'
test['ym_1209'] = test['year_month'] == '2012-9'
test['ym_1210'] = test['year_month'] == '2012-10'
test['ym_1211'] = test['year_month'] == '2012-11'
test['ym_1212'] = test['year_month'] == '2012-12'


# ## year+season

# In[17]:


# 연도와 계절을 합하여 컬럼을 생성하겠습니다.

train['year_season'] = train['year'].astype('str') + '-' + train['season'].astype('str')

train[['year','season','year_season']]


# In[18]:


ax = sns.barplot(data=train, x='year_season', y='count')
ax.set_title('연도에 따른 계절별 자전거 대여량')


# **그래프를 살펴보면, 크게 보면 연도에 관계없이 계절별 대여량 패턴이 매우 흡사합니다. 2년간의 계절을 쭉 이어서 보자면 계속 증가하다가 2011년 겨울과 2012년 봄에 대여량이 감소하고 여름부터 다시 증가합니다. 이는 겨울과 봄 사이에는 다소 쌀쌀한 날씨일 수도 있기 때문인 것으로 간주됩니다. 반면, 기온이 높아지는 여름~가을에는 대여량이 증가했습니다.**

# In[19]:


# One-Hot encoding을 실시하겠습니다.

train['spring_2011'] = train['year_season'] == '2011-1'
train['summer_2011'] = train['year_season'] == '2011-2'
train['fall_2011'] = train['year_season'] == '2011-3'
train['winter_2011'] = train['year_season'] == '2011-4'
train['spring_2012'] = train['year_season'] == '2012-1'
train['summer_2012'] = train['year_season'] == '2012-2'
train['fall_2012'] = train['year_season'] == '2012-3'
train['winter_2012'] = train['year_season'] == '2012-4'


# In[20]:


# test 파일도 year_season 컬럼을 생성하겠습니다.

test['year_season'] = test['year'].astype('str') + '-' + test['season'].astype('str')

test[['year','season','year_season']]


# In[21]:


# One-Hot encoding을 실시하겠습니다.

test['spring_2011'] = test['year_season'] == '2011-1'
test['summer_2011'] = test['year_season'] == '2011-2'
test['fall_2011'] = test['year_season'] == '2011-3'
test['winter_2011'] = test['year_season'] == '2011-4'
test['spring_2012'] = test['year_season'] == '2012-1'
test['summer_2012'] = test['year_season'] == '2012-2'
test['fall_2012'] = test['year_season'] == '2012-3'
test['winter_2012'] = test['year_season'] == '2012-4'


# ## season

# In[22]:


ax = sns.barplot(data=train, x='season', y='count')
ax.set_title('계절별 대여량')


# **시각화 결과, 날씨가 따뜻해질수록 대여량이 많아지고 있었지만, 날씨가 추운 겨울에도 대여량이 많은 것으로 나타났습니다.**

# **계절은 범주형 데이터이므로 One Hot Encoding을 실시하겠습니다**

# In[23]:


train['season_sp'] = train['season'] == 1
train['season_su'] = train['season'] == 2
train['season_fa'] = train['season'] == 3
train['season_wi'] = train['season'] == 4

train[['season','season_sp','season_su','season_fa','season_wi']].head()


# In[24]:


# test 파일도 One Hot Encoding을 실시하겠습니다

test['season_sp'] = test['season'] == 1
test['season_su'] = test['season'] == 2
test['season_fa'] = test['season'] == 3
test['season_wi'] = test['season'] == 4

test[['season','season_sp','season_su','season_fa','season_wi']].head()


# ## holiday

# In[25]:


ax = sns.barplot(data=train, x='holiday', y='count')

ax.set_title('공휴일여부별 대여량')


# **시각화 결과, 공휴일이 아닐 때에 자전거 대여량이 조금 더 많은 것으로 나타났습니다. 그런데 공휴일일 때는 편차가 큰 것으로 봐서는 자전거 대여 외에 다양한 활동을 할 수 있기 때문인 것으로 보여집니다.**

# ## workingday

# In[26]:


ax = sns.barplot(data=train, x='workingday', y='count')
ax.set_title('근무일여부별 대여량')


# **큰 차이는 없었지만 근무일에 자전거 대여량이 더 많은 것으로 나타났습니다.**

# ## weather

# In[27]:


ax = sns.barplot(data=train, x='weather',y='count')
ax.set_title('날씨별 대여량')


# **봄->여름->가을로 갈수록 자전거 대여량이 줄어들고 있지만, 겨울에는 갑자기 많아집니다. 또한, 겨울에는 데이터가 한 개여서 신뢰구간 바가 표시되지 않았습니다. 그래서 겨울 데이터는 최빈치인 1에 포함시키겠습니다.**

# In[28]:


train['weather_not4'] = train['weather'].replace(4,1)

train['weather_not4'].value_counts()


# In[29]:


ax = sns.barplot(data=train, x='weather_not4', y='count')
ax.set_title('데이터 수정 후 날씨별 대여량')


# In[30]:


# test 파일에서도 4의 빈도가 2개 밖에 되지 않아 train과 동일하게 4를 1로 변환하겠습니다.

test['weather'].value_counts()


# In[31]:


test['weather_not4'] = test['weather'].replace(4,1)

test['weather_not4'].value_counts()


# In[32]:


# 또한 weather_not4 컬럼도 범주형 데이터이므로 One Hot Encoding을 실시하겠습니다.

train['weather_1'] = train['weather_not4'] == 1
train['weather_2'] = train['weather_not4'] == 2
train['weather_3'] = train['weather_not4'] == 3

train[['weather_not4', 'weather_1', 'weather_2', 'weather_3']].head()


# In[33]:


test['weather_1'] = test['weather_not4'] == 1
test['weather_2'] = test['weather_not4'] == 2
test['weather_3'] = test['weather_not4'] == 3

test[['weather_not4', 'weather_1', 'weather_2', 'weather_3']].head()


# In[34]:


train.head()


# ## temp

# In[35]:


corrlist = train[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered','count']]
corrMatrix = corrlist.corr(method='pearson')
corrMatrix


# **각 컬럼 간의 상관분석을 실시한 결과, temp 와 atemp는 상관계수가 약 .99인 것으로 다중공선성 문제로 temp만 EDA를 진행하여 학습 데이터로 사용하겠습니다.**

# In[36]:


plt.figure(figsize=(20,4))
ax = sns.distplot(train['temp'])
ax.set_title('기온별 대여량 distplot')


# **그림을 보면 온도가 낮거나 높을 때 자전거 대여량도 적고 10~30도 사이에 대여량이 많은 것으로 나타났습니다.**

# In[37]:


train.head()


# In[38]:


# 온도(temp) - 체감온도(atemp)

train['temp(int)'] = train['temp'].round()
train['atemp(int)'] = train['atemp'].round()

train['temp(difference)'] = train['temp(int)'] - train['atemp(int)']
train[["temp(int)", "atemp(int)", "temp(difference)"]].head()


# In[39]:


plt.figure(figsize=(18,4))
ax = sns.pointplot(data=train, x='temp(difference)',y='count')
ax.set_title('실제 기온과 체감 기온 간의 차이별 대여량(temp - atemp)')


# In[40]:


train['temp(difference)'].value_counts()


# In[41]:


train.loc[train['temp(difference)'] < -6, 'temp(difference)'] = -6
train.loc[train['temp(difference)'] > 0, 'temp(difference)'] = 0


# In[42]:


plt.figure(figsize=(18,4))
ax = sns.pointplot(data=train, x='temp(difference)',y='count')
ax.set_title('실제 온도와 체감 온도 간의 차이별 대여량(수정 후)')


# **실제온도와 체감온도 간의 차이가 적을수록 대여량이 적어지고 있었습니다.**

# In[43]:


test['temp(int)'] = test['temp'].round()
test['atemp(int)'] = test['atemp'].round()

test['temp(difference)'] = test['temp(int)'] - test['atemp(int)']
test[["temp(int)", "atemp(int)", "temp(difference)"]].head()


# In[44]:


test['temp(difference)'].value_counts()


# In[45]:


test.loc[test['temp(difference)'] < -6, 'temp(difference)'] = -6
test.loc[test['temp(difference)'] > 0, 'temp(difference)'] = 0


# ## humidity

# In[46]:


plt.figure(figsize=(20,4))
ax = sns.distplot(train['humidity'])
ax.set_title('습도별 대여량 distplot')


# **자전거 대여량이 전반적으로 40~80 사이에 많습니다.**

# ## windspeed

# In[47]:


plt.figure(figsize=(20,4))
ax = sns.distplot(train['windspeed'])
ax.set_title('풍속별 대여량 distplot')


# In[48]:


plt.figure(figsize=(20,4))
ax = sns.pointplot(data=train, x='windspeed', y='count')
ax.set_title('풍속별 대여량 pointplot')


# In[49]:


# windspeed 에서 정수에 해당하는 부분을 뺀 값을 그래프로 그리면 다음과 같습니다.

train['windspeed(point)'] = train['windspeed'] - train['windspeed'].astype('int')
print(round(train['windspeed(point)'],3).unique())
ax = sns.distplot(train['windspeed(point)'])
ax.set_title('수정후 풍속별 대여량')


# **windspeed - windspeed의 정수부분을 뺀 값을 시각화하면, 0에 가까울 때에 대여량이 더 많았습니다.**

# In[50]:


# test 파일도 위와 같은 작업을 하겠습니다.

plt.rc('axes', unicode_minus=False)
test['windspeed(point)'] = test['windspeed'] - test['windspeed'].astype('int')
print(round(test['windspeed(point)'],3).unique())

ax = sns.distplot(test['windspeed(point)'])
ax.set_title('수정 후 풍속별 대여량')


# ## dayofweek(요일)

# In[51]:


train['dayofweek'] = train['datetime'].dt.weekday

train['dayofweek'].head()


# In[52]:


test['dayofweek'] = test['datetime'].dt.weekday
test['dayofweek'].head()


# In[53]:


# 월:0 ~ 일:6

ax = sns.barplot(data=train, x='dayofweek', y='count')
ax.set_title('요일별 대여량')


# **시각화 그림을 살펴보면, 목~토요일은 자전거 대여량이 많았습니다**

# In[54]:


plt.figure(figsize=(18,4))
ax = sns.pointplot(data=train, x='hour', y='count', hue='dayofweek') # hue_order는 hue 값과 일치해야 함.

ax.set_title('요일 및 시간별 대여량')


# **평일과 주말간의 시간별 대여량 패턴이 달랐습니다. 평일 중에선 금요일, 주말 중에선 일요일이 자전거 대여량이 적은 것으로 나타났습니다.**

# In[55]:


# 요일 컬럼은 범주형 데이터이므로 One-Hot encoding을 실시하겠습니다.

train['week_mon'] = train['dayofweek'] == 0
train['week_tue'] = train['dayofweek'] == 1
train['week_wed'] = train['dayofweek'] == 2
train['week_thu'] = train['dayofweek'] == 3
train['week_fri'] = train['dayofweek'] == 4
train['week_sat'] = train['dayofweek'] == 5
train['week_sun'] = train['dayofweek'] == 6

print(train.columns)
train.head()


# In[56]:


# test 파일도 One-Hot encoding을 실시하겠습니다.


test['week_mon'] = test['dayofweek'] == 0
test['week_tue'] = test['dayofweek'] == 1
test['week_wed'] = test['dayofweek'] == 2
test['week_thu'] = test['dayofweek'] == 3
test['week_fri'] = test['dayofweek'] == 4
test['week_sat'] = test['dayofweek'] == 5
test['week_sun'] = test['dayofweek'] == 6

print(test.columns)
test.head()


# ## discomfort index

# * 기상청 홈페이지 내 불쾌지수 산출 공식을 활용하여 불쾌지수 컬럼을 만들어 불쾌지수별 대여량을 보고자 합니다.
# 
# 
# 단계별 해석
#  * 80이상 - 매우 높음
#  * 75~80미만 - 높음
#  * 68~75미만 - 보통
#  * 68미만 - 낮음

# In[57]:


def discomfort(temp, humidity):
    return 9/5 * temp - 0.55 * (1 - (humidity/100)) * (9/5*temp - 26) +32


# In[58]:


train['discomfort'] = discomfort(train['temp'], train['humidity'])
train['discomfort']


# In[59]:


test['discomfort'] = discomfort(test['temp'], test['humidity'])
test['discomfort']


# In[60]:


ax = sns.distplot(train['discomfort'])
ax.set_title('불쾌지수별 대여량 distplot')


# In[61]:


ax = sns.scatterplot(data=train, x='discomfort', y='count')
ax.set_title('불쾌지수별 대여량 산포도')


# **불쾌지수가 높을수록 대여량도 많다는 것을 알 수 있습니다.**

# # 랜덤 포레스트 모델

# ## train

# In[62]:


train.columns


# In[66]:


# level7 과제 피드백을 바탕으로 year_season(One-Hot Encoding) 컬럼을 새로 추가하고 windspeed 컬럼을 다시 추가했습니다.

feature_names = ['holiday','workingday','humidity','temp','atemp' ,'year', 'hour', 'season_sp','season_su','season_fa','season_wi',
                 'weather_1','weather_2','weather_3', 'week_mon', 'week_tue','week_wed','week_thu','week_fri','week_sat','week_sun',
                 'discomfort','spring_2011', 'summer_2011', 'fall_2011','winter_2011', 'spring_2012', 'summer_2012', 'fall_2012',
                 'winter_2012','windspeed']

feature_names


# In[67]:


x_train = train[feature_names]

print(x_train.shape)
x_train.head()


# In[68]:


x_test = test[feature_names]

print(x_test.shape)
x_test.head()


# ## predict

# **label을 count, casual, registered별로 모두 살펴보겠습니다.**

# In[69]:


# label 을 count로 지정하겠습니다.

label = 'count'
y_train = train[label]

print(y_train.shape)
y_train.head()


# In[70]:


# RMSLE 공식에 기초해 count 컬럼에 log를 씌우겠습니다.

log_y_train = np.log(y_train + 1)

print(log_y_train.shape)
log_y_train.head()


# In[71]:


# label 을 casual로 지정하겠습니다.

label = 'casual'
y_train_casual = train[label]

print(y_train_casual.shape)
y_train_casual.head()


# In[72]:


# RMSLE 공식에 기초해 casual 컬럼에 log를 씌우겠습니다.

log_y_train_casual = np.log(y_train_casual + 1)

print(log_y_train_casual.shape)
log_y_train_casual.head()


# In[73]:


# label 을 registered로 지정하겠습니다.

label = 'registered'
y_train_registered = train[label]

print(y_train_registered.shape)
y_train_registered.head()


# In[74]:


# RMSLE 공식에 기초해 registered 컬럼에 log를 씌우겠습니다.

log_y_train_registered = np.log(y_train_registered + 1)

print(log_y_train_registered.shape)
log_y_train_registered.head()


# ## 모델 적용

# In[70]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1,
                              random_state=33)
model


# ## Evaluation

# In[71]:


# casual, registered 을 각 한 번씩 평가해 각 값을 합하여 y_predict(count) 변수를 생성하겠습니다.

from sklearn.model_selection import cross_val_predict

log_y_predict_count = cross_val_predict(model, x_train, log_y_train, cv=20)
log_y_predict_casual = cross_val_predict(model, x_train, log_y_train_casual, cv=20)
log_y_predict_registered = cross_val_predict(model, x_train, log_y_train_registered, cv=20)

# 지수(np.exp)를 사용하여 log를 풀어주겠습니다.
y_predict_count = np.exp(log_y_predict_count) - 1
y_predict_casual = np.exp(log_y_predict_casual) - 1
y_predict_registered = np.exp(log_y_predict_registered) - 1

y_predict = np.sqrt((y_predict_count + y_predict_casual)* y_predict_registered)

print(y_predict.shape)
y_predict


# In[72]:


# RMSLE 방식으로 평가 (log 씌우기 전 score = 0.372 -> log 씌운 후 score = 2.951 -> log 풀어 준 후 score = 0.352
# -> 새로운 feature 추가 후 score = 0.352)

from sklearn.metrics import mean_squared_log_error

score = mean_squared_log_error(y_train, y_predict)
score = np.sqrt(score)
print(f'score : {score:.3f}')


# ## Hyperparameter Tuning

# ### Random Search (coarse)

# In[73]:


# 하이퍼 파라미터 튜닝을 하겠습니다.(시간제약으로 인해 10번 반복하겠습니다)

n_estimators=300

for i in range(10):
    max_depth = np.random.randint(2,100)
    max_features = np.random.uniform(0.1,1.0)
    
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth = max_depth,
                                  max_features = max_features,
                                  n_jobs=-1,
                                  random_state=33)
    
    log_y_predict_count = cross_val_predict(model, x_train, log_y_train, cv=20)
    log_y_predict_casual = cross_val_predict(model, x_train, log_y_train_casual, cv=20)
    log_y_predict_registered = cross_val_predict(model, x_train, log_y_train_registered, cv=20)
    
    y_predict_count = np.exp(log_y_predict_count) - 1
    y_predict_casual = np.exp(log_y_predict_casual) - 1
    y_predict_registered = np.exp(log_y_predict_registered) - 1
    
    y_predict = np.sqrt((y_predict_count + y_predict_casual)* y_predict_registered)
    
    score = mean_squared_log_error(y_train, y_predict)
    score = np.sqrt(score)
    print(f'n_estimators : {n_estimators}, max_depth : {max_depth}, max_features : {max_features}, score : {score:.4f}')


# ### Random Search (fine)

# In[74]:


# 위에서 실시한 결과 중 상위 5개의 파라미터를 기준으로 다시 랜덤 서치하겠습니다.

n_estimators=300

for i in range(10):
    max_depth = np.random.randint(40,100)
    max_features = np.random.uniform(0.7,1.0)
    
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth = max_depth,
                                  max_features = max_features,
                                  n_jobs=-1,
                                  random_state=33)
    
    log_y_predict_count = cross_val_predict(model, x_train, log_y_train, cv=20)
    log_y_predict_casual = cross_val_predict(model, x_train, log_y_train_casual, cv=20)
    log_y_predict_registered = cross_val_predict(model, x_train, log_y_train_registered, cv=20)
    
    y_predict_count = np.exp(log_y_predict_count) - 1
    y_predict_casual = np.exp(log_y_predict_casual) - 1
    y_predict_registered = np.exp(log_y_predict_registered) - 1
    
    y_predict = np.sqrt((y_predict_count + y_predict_casual)* y_predict_registered)
    
    score = mean_squared_log_error(y_train, y_predict)
    score = np.sqrt(score)
    print(f'n_estimators : {n_estimators}, max_depth : {max_depth}, max_features : {max_features}, score : {score:.4f}')


# ## 하이퍼 파라미터를 활용한 모델 적용

# In[87]:


# fine 방식으로 얻은 결과 중 score가 .3488 중에서 max_features 가 높은 값을 우선적으로 사용하겠습니다.
# 하이퍼 파라미터 수행시 n_estimators = 300으로 했지만, 모델 적용 시에는 3000으로 넣고 나머지 값은 하이퍼 파라미터 값을 넣겠습니다.

# n_estimators : 3000, max_depth : 56, max_features : 0.7952672746852549, score : 0.3488 -> 0.38408
# n_estimators : 3000, max_depth : 94, max_features : 0.7701863859564818, score : 0.3488 -> 0.38425
# (추천 파라미터 값) n_estimators : 3000, max_depth : 97, max_features : 0.897703  ->  0.38353

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=3000,
                              max_depth = 97,
                              max_features = 0.897703,
                              n_jobs=-1,
                              random_state=33)
model


# ## 학습 및 예측

# In[88]:


# 학습을 casual 컬럼, registered 컬럼 한 번씩 진행하여 학습결과를 합한 prediction 변수를 생성하겠습니다.
# 학습 시, label은 log를 씌우겠습니다.

model.fit(x_train, log_y_train)
log_count_prediction = model.predict(x_test)

model.fit(x_train, log_y_train_casual)
log_casual_prediction = model.predict(x_test)

model.fit(x_train, log_y_train_registered)
log_registered_prediction = model.predict(x_test)


# In[89]:


# log 값은 지수(np.exp-1) 방법을 활용하여 다시 log를 지우겠습니다.

count_prediction = np.exp(log_count_prediction) - 1
casual_prediction = np.exp(log_casual_prediction) - 1
registered_prediction = np.exp(log_registered_prediction) - 1

count_prediction, casual_prediction, registered_prediction


# In[90]:


prediction = np.sqrt((count_prediction + casual_prediction) * registered_prediction)

print(prediction.shape)
prediction


# # 제출

# In[91]:


submit = pd.read_csv('sampleSubmission.csv')
submit['count'] = prediction

print(submit.shape)
submit.head()


# In[92]:


submit.to_csv('bike_predict_level8_hyper.csv', index=False)


# In[1]:


130/3242 * 100


# In[ ]:




