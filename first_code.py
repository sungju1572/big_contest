import pandas as pd


loan_result = pd.read_csv("loan_result.csv")
log_data = pd.read_csv("log_data.csv")
user_spec = pd.read_csv("user_spec.csv")


loan_result 

#6월 기준 train / test 나누기
test = loan_result[loan_result["loanapply_insert_time"].str.contains("2022-06")]
train = loan_result[loan_result["loanapply_insert_time"].str.contains("2022-06")==False]


#데이터 구조 확인
len(loan_result["application_id"].value_counts())
len(user_spec["application_id"].value_counts())



loan_result["application_id"].value_counts().index()


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

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#loan_limit : 승인한도
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_minmax = scaler.fit_transform(merge_df)

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


#loan_rate, loan_limit 제거 (평가x)
merge_df["loan_limit"] =  merge_df["loan_limit"].dropna()
merge_df["loan_rate"] =  merge_df["loan_rate"].dropna()




