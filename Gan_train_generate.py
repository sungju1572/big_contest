# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 19:48:01 2022

@author: user
"""


import pandas as pd

loan_result  = pd.read_csv('C:/Users/user/Desktop/대학원수업/dacon/빅콘테스트/loan_result.csv')

log_data =pd.read_csv('C:/Users/user/Desktop/대학원수업/dacon/빅콘테스트/log_data.csv')

user_spec =pd.read_csv('C:/Users/user/Desktop/대학원수업/dacon/빅콘테스트/user_spec.csv')



loan_result 

#6월 기준 train / test 나누기
test = loan_result[loan_result["loanapply_insert_time"].str.contains("2022-06")]
train = loan_result[loan_result["loanapply_insert_time"].str.contains("2022-06")==False]


#데이터 구조 확인
len(loan_result["application_id"].value_counts())
len(user_spec["application_id"].value_counts())






#두개 공통변수 application_id 가지고 merge 하기전 loan_result에 있는 id중 user_spec에 없는 id 찾기
a = loan_result["application_id"].unique()
b = user_spec["application_id"].unique()

str_a = list(map(str,a))
str_b = list(map(str,b))

intersection = list(set(str_b) & set(str_a))

len(intersection )
len(str_a)

complement = list(set(str_a) - set(intersection))

int_complement = list(map(int, complement ))

#찾아낸것 몇개 보기
loan_result[loan_result["application_id"] == 771476]
loan_result[loan_result["application_id"] == 1470833]
loan_result[loan_result["application_id"] == 1562843]

#user_spec에 없는 행 id만 뽑기
df_loss = pd.DataFrame()

for i in int_complement:
    df_loss = pd.concat([df_loss, loan_result[loan_result["application_id"] == i]],axis = 0)
    print(i)


#113개 확인  (train 부분에만 포함되어있는것 확인)
len(df_loss)

df_loss

#train에서 113개 데이터 삭제
len(train)

df_loss.index

train = train.drop(df_loss.index)

#행갯수 10270011
len(train)



#train + user_spec merge
merge_df = pd.merge(train, user_spec, on = 'application_id', how = 'inner')

merge_df.columns

merge_df_head = merge_df.head(5000)

merge_df.info()


#na 있는 컬럼들 찾기
na_columns_list = []
for i in range(len(merge_df.columns)):
    if merge_df.iloc[:,i].isna().sum() != 0 :
        na_columns_list.append(merge_df.columns[i])
"""
na있는 컬럼들
['loan_limit',
 'loan_rate',
 'birth_year',
 'gender',
 'credit_score',
 'company_enter_month',
 'personal_rehabilitation_yn',
 'personal_rehabilitation_complete_yn',
 'existing_loan_cnt',
 'existing_loan_amt']
"""


#loan_limit : 승인한도

#5625
merge_df["loan_limit"].isna().sum()
merge_df["loan_rate"].isna().sum()

#91626
merge_df["birth_year"].isna().sum()
merge_df["gender"].isna().sum()

#1243812
merge_df["credit_score"].isna().sum()

#303568
merge_df["company_enter_month"].isna().sum()

#5873229
merge_df["personal_rehabilitation_yn"].isna().sum()
#9232232
merge_df["personal_rehabilitation_complete_yn"].isna().sum()


merge_df["existing_loan_cnt"].isna().sum()
merge_df["existing_loan_amt"].isna().sum()

len(merge_df)


merge_df = merge_df[merge_df["loan_limit"].notna()]

#loan_rate, loan_limit 제거 (평가x)
merge_df = merge_df[merge_df["loan_limit"].notna()]
merge_df = merge_df[merge_df["loan_rate"].notna()]

#birth_year, gender 제거 
merge_df = merge_df[merge_df["birth_year"].notna()]
merge_df = merge_df[merge_df["gender"].notna()]

#company_enter_month 제거
merge_df = merge_df[merge_df["company_enter_month"].notna()]

#'personal_rehabilitation_yn', : 개인회생자 여부 ,'personal_rehabilitation_complete_yn', : 개인회생자 납입완료 여부 / 0으로 변환
#'existing_loan_cnt', : 기대출수 , 'existing_loan_amt' : 기대출금액 / 0으로 변환

merge_df["personal_rehabilitation_yn"].fillna(0, inplace=True)
merge_df["personal_rehabilitation_complete_yn"].fillna(0, inplace=True)
merge_df["existing_loan_cnt"].fillna(0, inplace=True)
merge_df["existing_loan_amt"].fillna(0, inplace=True)


import swifter

#신용점수 등급으로 변경
def change_credit_rate(x):
        if x >=942:
            return 1
        elif x  >= 891 and x <942:
            return 2
        elif x  >= 832 and x <891:
            return 3
        elif x  >= 768 and x <832:
            return 4
        elif x  >= 698 and x <768:
            return 5
        elif x  >= 630 and x <698:
            return 6
        elif x  >= 530 and x <630:
            return 7
        elif x  >= 454 and x <530:
            return 8
        elif x  >= 335 and x <454:
            return 9
        elif x <334 and x>0 :
            return 10
 

merge_df["credit_score"] = merge_df["credit_score"].swifter.apply(change_credit_rate)


#Unbalanced data 처리


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score, auc

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

import tensorflow as tf
from tensorflow import keras

import optuna

# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSample


merge_df.info()

feature = merge_df[['loan_limit','loan_rate','credit_score','yearly_income','income_type','employment_type','houseown_type','desired_amount','purpose','personal_rehabilitation_yn','personal_rehabilitation_complete_yn',
                    'existing_loan_cnt','existing_loan_amt']]

target = merge_df['is_applied']




#label encoder
from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()

feature_cat_feature = feature[['income_type','employment_type','houseown_type','purpose']]

#income_type
encoder.fit(feature['income_type'])


feature['income_type'] = encoder.transform(feature['income_type'])


#employment_type
encoder.fit(feature['employment_type'])


feature['employment_type'] = encoder.transform(feature['employment_type'])

#houseown_type
encoder.fit(feature['houseown_type'])


feature['houseown_type'] = encoder.transform(feature['houseown_type'])

#purpose
encoder.fit(feature['purpose'])


feature['purpose'] = encoder.transform(feature['purpose'])


# #범주형 변수 처리
# cat_feature = ['income_type','employment_type','houseown_type','purpose','personal_rehabilitation_complete_yn','personal_rehabilitation_yn','credit_score']

# for i in enumerate(cat_feature) :
#     ca = i[1]
#     feature[ca] = feature[ca].astype('category')


# feature.info()



#정규화 진행
numeric_feature = ['loan_limit','loan_rate','yearly_income','desired_amount','existing_loan_cnt','existing_loan_amt']


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


feature[numeric_feature] = scaler.fit_transform(feature[numeric_feature])



#train_test split
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, stratify = target, random_state=123)




#GAN Numerical feature Generate 
#생성 대상 : 대출 승인 : 1인 사람


prac = merge_df[merge_df['is_applied']==1]

num = ['loan_limit','loan_rate','yearly_income','desired_amount','existing_loan_cnt','existing_loan_amt']

numerical_feature  = prac[['loan_limit','loan_rate','yearly_income','desired_amount','existing_loan_cnt','existing_loan_amt']]

min_max_numerical_feature = scaler.fit_transform(numerical_feature)

numerical_feature[num] = min_max_numerical_feature

nr_features = numerical_feature.shape[1]


# Select a batch of random samples, returns features and target
def generate_real_samples(numerical_feature, n_samples):
	# choose random instances of indices
	ix = randint(0, numerical_feature.shape[0], n_samples)
	# retrieve samples from indices
	X = numerical_feature.iloc[ix].values
	# generate 'real' class labels (1)
	y = ones((n_samples,1))
	return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
  # generate points in latent space
  x_input = generate_latent_points(latent_dim, n)
  # predict outputs
  X = generator.predict(x_input)
  # create class labels
  y = zeros((n, 1))
  return X, y







from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import pandas as pd
from numpy import hstack,zeros,ones,set_printoptions
from numpy.random import rand, randn, randint
from sklearn.preprocessing import MinMaxScaler
# Common MinMaxScaler for all features.
scaler = MinMaxScaler()


# define the standalone discriminator model
def define_discriminator(n_inputs=nr_features):
  model = Sequential()
  model.add(Dense(n_inputs, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
  model.add(Dense(32,activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
 
# define the standalone generator model
def define_generator(latent_dim, n_outputs=nr_features):
  model = Sequential()
  model.add(Dense(latent_dim, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
  model.add(Dense(64,activation = 'relu'))
  model.add(Dense(n_outputs, activation='linear'))
  return model
 
# Composite Model - define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

nr_samples = 300000 # Select smaller value when using the LTR dataframe with only 34 data points as 'real_data'

# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, numerical_feature, generator, discriminator, latent_dim, n=nr_samples):
  # prepare real samples
  x_real, y_real = generate_real_samples(numerical_feature, n)
  # evaluate discriminator on real examples
  _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
  # prepare fake examples
  x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
  # evaluate discriminator on fake examples
  _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
  # summarize discriminator performance
  print('Epoch:',epoch,'Accuracy(RealData):', round(acc_real,3),'Accuracy(FakeData):', round(acc_fake,3))
#   original_data = scaler.inverse_transform(x_real)
#   synthetic_data = scaler.inverse_transform(x_fake)
# #   print("Original_data  = %s" % original_data)
# #   print("Synthetic_data  = %s" % synthetic_data)
#   # Scatter plot of real and fake data points
#   plt.scatter(original_data[:, 0], original_data[:, -1], color='red',label = 'Real Data')
#   plt.scatter(synthetic_data[:, 0], synthetic_data[:, -1], color='blue', label = 'Synthesized Data')
#   plt.ylabel("Xr_y")
#   plt.xlabel("Xf")
#   plt.legend(loc="upper left")
#   plt.show()

batch_size = 12

  # train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=batch_size, n_eval=500):
  # determine half the size of one batch, for updating the discriminator
  half_batch = int(n_batch / 2)
  # prepare lists for storing stats each iteration
  disc_rd_hist, disc_fd_hist, g_hist, acc1_hist, acc2_hist = list(), list(), list(), list(), list()
	# manually enumerate epoch
  for i in range(n_epochs):
    # prepare real samples
    x_real, y_real = generate_real_samples(numerical_feature,half_batch)
    # Update discriminator
    d_loss1, d_acc1 = d_model.train_on_batch(x_real, y_real)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
    # update discriminator
    d_loss2, d_acc2 = d_model.train_on_batch(x_fake, y_fake)
    # prepare points in latent space as input for the generator
    x_gan = generate_latent_points(latent_dim, n_batch)
    # create inverted labels for the fake samples
    y_gan = ones((n_batch, 1))
    # update the generator via the discriminator's error
    g_loss = gan_model.train_on_batch(x_gan, y_gan)
    # summarize loss on this batch for every epoch
#     print('>%d, disc_loss_real=%.3f, disc_loss_fake=%.3f gen_loss=%.3f, disc_acc_real=%d, disc_acc_fake=%d'
#           %(i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
    # record history
    disc_rd_hist.append(d_loss1)
    disc_fd_hist.append(d_loss2)
    g_hist.append(g_loss)
    acc1_hist.append(d_acc1)
    acc2_hist.append(d_acc2)
    # evaluate the model every n_eval epochs
    if (i+1) % n_eval == 0:      
      # Summarize Loss every n_eval epochs
      print('Epoch:%d, disc_loss_real=%.3f, disc_loss_fake=%.3f gen_loss=%.3f, disc_acc_real=%d, disc_acc_fake=%d'
            %(i, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
      summarize_performance(i,numerical_feature, g_model, d_model, latent_dim) 
  return disc_rd_hist, disc_fd_hist, g_hist, acc1_hist, acc2_hist






# size of the latent space
latent_dim = 20
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
disc_rd_hist, disc_fd_hist, g_hist, acc1_hist, acc2_hist = train(generator, discriminator, gan_model, latent_dim)



#n =생성개수
fake_num_scale,_ = generate_fake_samples(generator, latent_dim, n=500000)




fake_num_feature = pd.DataFrame(data = fake_num_scale, columns = num)


fake_num_feature['is_applied'] = 1



#수치형 변수로만 예측 + fake_data 추가

fake_total_data = merge_df[['loan_limit','loan_rate','yearly_income','desired_amount','existing_loan_cnt','existing_loan_amt','is_applied']]

min_max_numerical_feature = scaler.fit_transform(fake_total_data[num])

fake_total_data[num] = min_max_numerical_feature


fake_train_test = pd.concat([fake_total_data,fake_num_feature])




#train_test split

feature = fake_total_data[num]

target = merge_df['is_applied']



X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, stratify = target, random_state=123)


X_train = pd.concat([X_train,fake_num_feature[num]])

y_train = pd.concat([y_train,fake_num_feature['is_applied']])




#lightGBM
import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)


y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))



from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print('f1 score :', f1)

from sklearn.metrics import classification_report


print(classification_report(y_pred,y_test))




