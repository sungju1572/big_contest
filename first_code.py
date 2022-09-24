

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

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSample


merge_df.info()

feature = merge_df[['loan_limit','loan_rate','credit_score','yearly_income','income_type','employment_type','houseown_type','desired_amount','purpose','personal_rehabilitation_yn','personal_rehabilitation_complete_yn',
                    'existing_loan_cnt','existing_loan_amt']]

target = merge_df['is_applied']



#범주형 변수 처리
cat_feature = ['income_type','employment_type','houseown_type','purpose','personal_rehabilitation_complete_yn','personal_rehabilitation_yn','credit_score']

for i in enumerate(cat_feature) :
    ca = i[1]
    feature[ca] = feature[ca].astype('category')


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



#train_test split
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, stratify = target, random_state=123)


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



#under sampling
from imblearn.under_sampling import RandomUnderSampler



x_under, y_under = RandomUnderSampler(random_state=0).fit_resample(X_train, y_train)


#lightGBM
import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(x_under, y_under)


y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))



from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print('f1 score :', f1)

from sklearn.metrics import classification_report


print(classification_report(y_pred,y_test))




#over sampling
