#!/usr/bin/env python
# coding: utf-8

# # 딥러닝 온라인 1:1 과외반 <Level 7> 과제
# 
# (이 쥬피터 노트북은 다음의 링크 <b>https://bit.ly/dsd-0702</b> 에서 다운받으실 수 있습니다)

# <b><Level 7></b>의 주제는 <b>Model Evaluation</b> 입니다. 앞선 레벨들에서 배웠던 퍼셉트론, 다층 퍼셉트론 등 여러 알고리즘을 다뤘는데, 현업에서, 그리고 실제 연구에서 이 알고리즘을 사용하기 위해서 알고리즘의 성능을 평가하는 방법을 배워보았습니다. 지금까지는 scikit-learn(sklearn) 에서 제공하는 보스턴 집 값 데이터셋, 붓꽃(iris) 데이터셋, MNIST 필기체 데이터셋을 활용하여 알고리즘을 만들고 학습하고 예측해보았습니다. 이 때, 사용했던 데이터셋에는 모든 데이터에 대해 정답지가 있었기 때문에, train 데이터로 학습하고 test 데이터에 대해 예측한 후 정답지와 비교하여 얼마나 예측이 잘 되었는지 확인했습니다. 하지만, 실제 상황에서는 정답지가 있는 train 데이터를 활용하여 <b>정답지가 없는 test 데이터</b>를 잘 예측할 수 있는 모델을 만들어야하는 경우가 많습니다. 

# 따라서, <b>이번 <Level 7> 과제</b>에서는 좀 더 실제 상황에 가깝게, 정답지를 알지 못하는 새로운 데이터셋에 대해 정말 예측을 잘하는 좋은 모델을 만드는 것을 목표로 합니다. 이런 실전 상황들을 연습할 수 있는 좋은 방법은 바로 <b> 데이터 사이언스 경진대회 캐글 (Kaggle, https://kaggle.com )에 참여</b>해 보는 것입니다. 캐글(Kaggle)의 시스템은 각종 기업들에서 상금(Prize)를 걸고  데이터셋과 해결 과제를 등록하면, 전세계에 있는 데이터 사이언티스트들이 이 문제를 해결하기 위해 머신러닝 모델을 개발하고 정답(예측값)을 제출하여 모델의 성능에 따라 순위 경쟁을 하는 시스템입니다. 제공되는 데이터셋은 크게 train과 test로 나눠져 있는데, train 데이터의 경우 정답지가 있지만 test 데이터에는 정답이 없어 이 test 데이터의 정답을 가장 잘 맞추는 사람이 우승을 하게 됩니다. 사이트에 들어가 보시면 현재 진행중인 경진대회부터 이미 종료된 경진대회들까지 굉장히 다양한 분야의 많은 경진대회가 있는 것을 확인하실 수 있습니다. 

# 이들 중 <b>이번 <Level 7> 과제</b>는 지난 레벨들에서 계속해서 다뤘기때문에 조금은 익숙한 <b>MNIST 필기체를 인식</b>하는 Digit Recognizer (<b> https://www.kaggle.com/c/digit-recognizer </b>) 라는 입문용 경진대회에 참여해볼 것입니다. 우선 Kaggle에 계정이 없으신 분들은 우측 상단의 파란색 "Register" 버튼을 눌러 회원가입을 해주시기 바랍니다.

# <img src="http://drive.google.com/uc?export=view&id=1U4GIX58wJgUQ81lJ7RoT8LtbxW84He5Q" width="800">
# 

# 이 곳 https://www.kaggle.com/c/digit-recognizer/data 링크로 들어가시면 이번 경진대회에서 사용할 데이터셋을 다운받을 수 있습니다. (오른쪽의 파란색 "Join Competition" 버튼을 누르시고 몇 가지 동의과정을 거치면 제공되는 데이터셋을 다운 받을 수 있습니다.) Data는 train.csv, test.csv, sample_submission.csv 이렇게 총 3개의 데이터셋을 다운 받고 train.csv 데이터를 이용하여 모델을 학습하고 test.csv 데이터로 정답을 예측한 뒤, 정답 제출 양식인 sample_submission.csv 에 예측한 값을 담아 "Submit Predictions" 에서 제출하면 예측모델의 점수(Score)를 계산해줍니다. (digit-recognizer 경진 대회의 경우, 정확도(Accuracy)가 Score가 됩니다.) 

# 이번 과제의 목표는 이 곳에 test 데이터의 예측값을 제출하여 <b>Score 0.9(90%의 정확도) 이상을 달성</b>하는 것입니다.  

# 과제를 풀기 위해 몇 가지 주의사항 및 힌트를 드리겠습니다.
# 
# 1. 아래 적혀있는 코드는 가장 기본적인 Single-layer Neural Network 모델에 대한 코드입니다. **Hidden layer를 추가**하면 좀 더 높은 정확도를 달성할 수 있습니다. 
# 2. Test 데이터에 대한 예측값을 Kaggle에 제출하여 Score를 확인해 볼 수 있는 횟수는 **24시간 동안 5회로 제한**되어 있습니다. 따라서, 이번 레벨에서 배웠던 **Model Evaluation**을 통해 최대한 모델의 성능을 끌어올린 후, Test데이터에 대해 예측을 진행하고 정답을 제출해보아야 할 것입니다. (기본 코드는 Hold-Out validation 방법으로 짜여져있습니다.)
# 3. Overfitting 문제로 Model Evaluation 결과가 train set과 validation set에 대해 다소 차이가 생기게 됩니다. 이를 감안하여 최종적으로 Test 데이터에 대한 예측 성능(Score)이 0.9를 달성할 수 있도록 전략을 짜야합니다.
# 4. 학습 속도 및 정확도 향상을 위해 여러가지 **Hyperparameter를 튜닝**할 수 있습니다. num_epoch, Hidden Layer의 노드 갯수(w 및 b의 size), learning_rate, 초기값(w 및 b의 low/high 값) 등을 조정해 줄 수 있습니다. 
# 

# ## Load Data

# 먼저, Kaggle에서 다운받은 train, test 데이터를 불러옵니다.

# **1. 데이터셋 구성**
# > 가로 28px, 세로 28px의 필기체 이미지가 주어지며, 필기체는 숫자 0부터 9까지 총 10개의 Label로 구성되어 있습니다. 이미지는 컬러가 없는 흑백 데이터이며, 한 픽셀의 값은 0 ~ 255입니다. (0일수록 어둡고, 255일수록 밝습니다.)
# 
# **2. Train/Test Set**
# > 데이터는 42,000개의 train 데이터와 28,000개의 test 데이터가 주어지는데, Train 데이터로 학습한 뒤 Test 데이터를 예측을 합니다. 각 변수의 세부 정보는 다음과 같습니다.
# 
#    * **train**: **train 데이터의 Feature + Label**입니다. 784px(가로 28px * 세로 28px)의 각각의 값을 저장한 column들과 정답값인 "label" column을 합쳐 총 785개의 columns, 그리고 총 42,000개의 데이터로 구성되어 있습니다. 픽셀 하나의 값은 0 ~ 255이고. label에는 이미지가 어떤 숫자를 나타내는지에 대해 0부터 9까지의 숫자가 적혀 있습니다.   
#    * **test**: **test 데이터의 Feature**입니다. 784px (가로 28px * 세로 28px)의 각각의 값을 저장한 784개의 columns, 총 28,000개의 데이터로 구성되어 있습니다. 픽셀 하나의 값은 0 ~ 255이고. label에는 이미지가 어떤 숫자를 나타내는지에 대해 0부터 9까지의 숫자가 적혀 있습니다.   

# In[1]:


import pandas as pd
# train.csv 파일을 불러와 train에 저장합니다. 
train = pd.read_csv('train.csv')
train.head(1)


# In[2]:


# test.csv 파일을 불러와 test에 저장합니다. 
test = pd.read_csv('test.csv')
test.head(1)


# In[3]:


print(train.shape, test.shape)


# train 데이터는 총 42,000개, test 데이터는 28,000개가 있음을 볼 수 있습니다. 그리고 train 데이터에는 test 데이터엔 없는 label 컬럼(정답지)이 포함되어 있어 총 785개의 columns, test 데이터에는 하나 적은 총 784개의 columns가 있음을 할 수 있습니다.  

# ## Preprocessing
# 
# 학습을 위해 train 데이터를 Feature와 Label로 나눠줍니다.

# In[4]:


# train 데이터의 label 컬럼을 제외한 나머지 데이터를 X_train, label 컬럼값은 y_train 에 저장해줍니다. 
X_train = train.drop(['label'], axis=1).copy()
y_train = train['label']

# 나눠준 X_train과 y_train 의 형태를 출력해봅니다.
print(X_train.shape, y_train.shape)
# X_train 은 (42000, 784), y_train 은 (42000,0) 의 모양을 가지고 있습니다.
X_train.head(1)


# In[5]:


y_train.head(3)


# ### Create Validation Set (train-valid Split)
# 
# 모델의 성능을 측정하기 위해 사용할 Validation Set을 만듭니다. Training Data 중 일부를 Validation Set으로 뗴어내어 모델의 예측력을 테스트하는데 사용합니다. 이번 과제에서는 Hold-Out-Validation을 사용합니다.(Cross Validation을 사용할 수도 있습니다.) 적당한 비율을 선택하여 **X_train을 X_train_f, X_valid_f로** 나누고, **y_train은 y_train_f 와 y_valid_f로** 나눕니다.

# In[6]:


# hold-out validation용으로 쓰이는 train_test_split를 가져옵니다.
from sklearn.model_selection import train_test_split

# Write your code here!
X_train_f, X_valid_f, y_train_f, y_valid_f = train_test_split(X_train, y_train, test_size = 0.2, random_state = 123)

# X_train_f, X_valid_f, y_train_f, y_valid_f 의 형태를 출력하여 잘 나눠졌는지 각각의 데이터 갯수를 확인합니다. 
print(X_train_f.shape, X_valid_f.shape)
print(y_train_f.shape, y_valid_f.shape)


# label 값인 y_train_f, y_valid_f 에는 0부터 9까지의 숫자가 저장되어 있습니다. 이들을 One-hot encoding을 해준 뒤 각각 y_train_hot, y_valid_hot에 저장하도록 하겠습니다.

# In[7]:


# One-hot encoding을 위해 Keras의 to_categorical을 가져옵니다. 
from keras.utils import to_categorical

# Write your code here!
y_train_hot = to_categorical(y_train_f)
y_valid_hot = to_categorical(y_valid_f)

# y_train_hot, y_valid_hot의 형태를 출력하여 잘 나눠졌는지 각각의 데이터 갯수를 확인합니다. 
print(y_train_hot.shape, y_valid_hot.shape)
# (???, 10), (???, 10) 가 나옵니다.


# In[8]:


# 퍼셉트론 알고리즘에 투입하는 X_train_f, X_valid_f, y_train_hot, y_valid_hot을 transpose해줍니다.
X_train_f = X_train_f.T
X_valid_f = X_valid_f.T
y_train_hot = y_train_hot.T
y_valid_hot = y_valid_hot.T

print(X_train_f.shape, X_valid_f.shape)
# feature set인 X_train_f, X_valid_f의 경우 (784, ???), (784, ???)입니다.

print(y_train_hot.shape, y_valid_hot.shape)
# label set인 y_train_hot, y_valid_hot의 경우 (10, ???), (10, ???)입니다.


# ### Define sigmoid

# In[9]:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ### Define Cross-Entropy
# 

# In[10]:


# loss function으로써 cross entropy를 정의해줍니다.
def cross_entropy(actual, predict, eps=1e-15):
    
    # 실제 값과 예측 값을 Numpy 배열로 바꿔줍니다.
    actual = np.array(actual)
    predict = np.array(predict)
    
    # 0이 log에 들어가게 되면 무한대로 발산해버릴 수 있으니 아주 작은 값을 넣어 이를 방지합니다.
    clipped_predict = np.minimum(np.maximum(predict, eps), 1 - eps)
    
    # 실질적인 Loss를 계산합니다
    loss = actual * np.log(clipped_predict) + (1 - actual) * np.log(1 - clipped_predict)
    
    return -1.0 * loss.mean()


# ## Solving MNIST problem

# train 데이터를 이용하여 퍼셉트론 알고리즘을 학습합니다.

# In[11]:


num_epoch = 500 # 300번 반복
learning_rate = 3.5
# 우리가 학습해야하는 값들을 먼저 정의해줍니다.
# 1000개의 노드가 있는 한 개의 hidden layer를 쌓겠습니다.

w1 = np.random.uniform(low=-1.0, high=1.0, size=(1000, 784)) # (num_nodes, num_features)
b1 = np.random.uniform(low=-1.0, high=1.0, size=(1000, 1)) # (num_nodes, 1)

#hidden layer를 한 층 더 쌓았습니다. hidden neuron 갯수는 1000개로 설정했습니다.
w2 = np.random.uniform(low=-1.0, high=1.0, size=(10, 1000)) # (num_labels, num_nodes)
b2 = np.random.uniform(low=-1.0, high=1.0, size=(10, 1)) # (num_labels, 1)

# 샘플 수도 저장해줍니다.
num_data = X_train_f.shape[1]

# 학습 시작!
for epoch in range(num_epoch):
    # 먼저 합성곱을 해준 다음, 시그모이드 함수에 넣어줍니다.
    z1 = np.dot(w1, X_train_f) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    # a2를 가지고 라벨의 예측값을 만들어줍니다.
    y_predict_hot = a2
    y_predict = np.argmax(y_predict_hot, axis=0)
    accuracy = (y_predict == y_train_f).mean()
    
    # 정확도가 0.9에 도달할 때까지 학습합니다.
    if accuracy > 0.97:
        break

    # loss 함수는 cross entropy를 사용하였습니다.
    loss = cross_entropy(y_train_hot, y_predict_hot)

    # 일정 시간이 지나면 학습의 경과를 출력합니다.
    if epoch % 25 == 0:
        print("{0:2} accuracy = {1:.5f}, loss = {2:.5f}".format(epoch, accuracy, loss))

    # 경사하강법에 따라 비용함수를 최소화하도록 각 값들을 갱신해줍니다.
    d2 = y_predict_hot - y_train_hot
    d1 = np.dot(w2.T, d2) * a1 * (1-a1) # dsigmoid(a1) = a1 * (1-a1)

    w2 = w2 - learning_rate * np.dot(d2, a1.T) / num_data
    w1 = w1 - learning_rate * np.dot(d1, X_train_f.T) / num_data

    # b를 학습시킬 때 mean은 행연산(axis=1)이고, shape을 유지할 것입니다.
    b2 = b2 - learning_rate * d2.mean(axis=1, keepdims=True)
    b1 = b1 - learning_rate * d1.mean(axis=1, keepdims=True)
print("----" * 10)
print("{0:2} accuracy = {1:.5f}, loss = {2:.5f}".format(epoch, accuracy, loss))


# ### Model Evaluation(train set)
# 
# 학습이 완료된 w, b를 이용하여 train set에 대한 실제값과 예측값을 비교하여 모델의 성능을 평가해봅시다.

# In[12]:


# 학습한 w, b를 가지고 실제값과 예측값을 계산해봅시다.
# 먼저 train set에 대해서 결괏값을 만들어줍니다.
z1 = np.dot(w1, X_train_f) + b1
a1 = sigmoid(z1)
z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)

y_predict_hot = a2
y_predict = np.argmax(y_predict_hot, axis=0)

# actual vs. predict
train_result = pd.DataFrame({'actual': y_train_f, 'predict': y_predict})

# accuracy는 다음과 같이 계산됩니다.
train_accuracy = (train_result["actual"] == train_result["predict"]).mean()
print("Accuracy(train) = {0:.5f}".format(train_accuracy))

print(train_result.shape)
train_result.head(10)


# ### Model Evaluation(validation set)
# 
# 학습이 완료된 w, b를 이용하여 validation set에 대한 실제값과 예측값을 비교하여 모델의 성능을 평가해봅시다.
# validation set의 데이터는 학습에 전혀 사용되지 않은 데이터이기 때문에, 머신러닝 모델의 실제적인 성능을 측정할 수 있습니다. 

# In[13]:


# 학습한 w, b를 가지고 실제값과 예측값을 계산해봅시다.
# 먼저 train set에 대해서 결괏값을 만들어줍니다.
z1 = np.dot(w1, X_valid_f) + b1
a1 = sigmoid(z1)
z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)

y_predict_hot = a2
y_predict = np.argmax(y_predict_hot, axis=0)

# actual vs. predict
valid_result = pd.DataFrame({'actual': y_valid_f, 'predict': y_predict})

# accuracy는 다음과 같이 계산됩니다.
valid_accuracy = (valid_result["actual"] == valid_result["predict"]).mean()
print("Accuracy(valid) = {0:.5f}".format(valid_accuracy))

print(valid_result.shape)
valid_result.head(10)


# validation set으로 Model Evaluation한 결과가 train set으로 Model Evaluation한 결과보다 다소 낮게 나오는 이유는 **Overfitting 되었기 때문**입니다. 
# 즉, 이 모델은 train set에 대해 과대적합이 되어 학습해준 데이터가 아닌 새로운 데이터, validation set에 대해서는 예측성능이 다소 떨어질 수 있으며, **이 모델의 진짜 성능은 validation set으로 Model Evaluation한 결괏값**이 됩니다. (test 데이터를 학습하여 Kaggle 경진대회에 제출하게 된다면 바로 이 점수가 예상되는 점수입니다.)

# validation set을 이용한 모델의 성능지표가 만족스럽지 못하다면 Test data를 예측하기 전인, 이 단계에서 모델을 수정하고 다시 모델의 성능을 평가해 볼 수 있습니다. 

# ### Predict (Test data set)

# 이제 모델이 완성되었다면, Test data set을 불러와 예측값을 만들어줍니다.

# In[14]:


# test.csv 파일을 불러와 test에 저장합니다. 
test = pd.read_csv('test.csv')
test.head(1)


# In[15]:


# 퍼셉트론 알고리즘에 투입하는 test 데이터를 transpose해줍니다.
X_test = test.T

print(X_test.shape)
X_test.head()


# In[16]:


# 다음으로 test set에 대해서 결괏값을 만들어줍니다.
z1 = np.dot(w1, X_test) + b1
a1 = sigmoid(z1)
z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)

y_predict_hot = a2
y_predict = np.argmax(y_predict_hot, axis=0)
# 예측값을 출력해 봅니다.
y_predict


# 예측한 값을 Kaggle에 제출하기 위해 Kaggle에서 제공하는 제출폼(sample_submission.csv)을 불러와 예측값을 넣어줍니다. 

# In[17]:


# sample_submission.csv 파일을 읽어오고 ImageId 컬럼을 index로 지정하여 submission 에 저장합니다.. 
submission = pd.read_csv('sample_submission.csv', index_col="ImageId")
submission.head()


# In[18]:


# 예측값을 submission의 Label 컬럼에 저장해줍니다. 
submission["Label"]= y_predict
submission.head()


# In[19]:


# 예측값이 저장된 submission을 다시 test01.csv이라는 이름으로 저장합니다.
submission.to_csv("test01.csv")


# 이 주피터노트북이 있는 폴더경로에 test01.csv 라는 이름의 파일이 생성되었을 것입니다. 그 파일을 https://www.kaggle.com/c/digit-recognizer/submit 에 업로드한 후 아래쪽의 파란색 "Make Submission" 버튼을 누르시면 제출이 완료가 되고, Score가 계산됩니다.

# 제출하여 Score가 0.9 이상 달성하셨다면 목표 달성입니다! 목표달성 후 실행결과가 그대로 담긴 주피터 노트북 파일을 과제제출폼에 제출하시면 됩니다.
# 고생 많으셨습니다!!
