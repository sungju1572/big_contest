import pandas as pd


loan_result = pd.read_csv("loan_result.csv")
log_data = pd.read_csv("log_data.csv")
user_spec = pd.read_csv("user_spec.csv")


loan_result 

test = loan_result[loan_result["loanapply_insert_time"].str.contains("2022-06")]
train = loan_result[loan_result["loanapply_insert_time"].str.contains("2022-06")==False]


len(loan_result["application_id"].value_counts())
len(user_spec["application_id"].value_counts())



loan_result["application_id"].value_counts().index()


a = loan_result["application_id"].unique()
b = user_spec["application_id"].unique()

str_a = list(map(str,a))
str_b = list(map(str,b))


intersection = list(set(str_b) & set(str_a))

len(intersection )
len(str_a)

complement = list(set(str_a) - set(intersection))

int_complement = list(map(int, complement ))


'771476' in str_a 

loan_result[loan_result["application_id"] == 771476]
loan_result[loan_result["application_id"] == 1470833]
loan_result[loan_result["application_id"] == 1562843]

df = pd.DataFrame()

for i in int_complement:
    df = pd.concat([df, loan_result[loan_result["application_id"] == i]],axis = 0)
    print(i)

    
len(df)




train.columns

train["is_applied"].value_counts()

loan_result.columns





#loan_data + user_spec


len(loan_result['application_id'].unique())


len(user_spec['application_id'].unique())

key_value = loan_result['application_id'].unique()

unique_user_sepc = user_spec[user_spec['application_id'].isin(key_value)]



total_data = pd.merge(loan_result,unique_user_sepc,left_on=loan_result['application_id'],right_on=unique_user_sepc['application_id'],how='outer')


total_data = total_data.drop(['key_0'],axis=1)

#필요없는 컬럼 제거

total_data = total_data.drop(['application_id_x','loanapply_insert_time','bank_id','product_id','application_id_y','user_id','birth_year','insert_time'],axis=1)

total_data_head = total_data.head(100)



test = loan_result[loan_result["loanapply_insert_time"].str.contains("2022-06")]
train = loan_result[loan_result["loanapply_insert_time"].str.contains("2022-06")==False]


user_id

