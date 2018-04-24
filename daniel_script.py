import numpy as np
import pandas as pd
import glob
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

path = '/users/danielchavez/Documents/EconProject'
allFiles = glob.glob(path + "/*.csv")
data = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, index_col = None, header = 1)
    list_.append(df)
data = pd.concat(list_)



l = []
for set in (data.columns):
    if data[set].isnull().sum().sum() > 100:
        l.append(set)
l.remove('emp_length')





df = data.drop(l, axis = 1)




#changing interest rate
df['int_rate'] = df['int_rate'].str.rstrip('%').astype('float')/100




#dropping NA rows
df = df.dropna(axis=0, how='any')




df = df[df.home_ownership.str.contains("ANY") == False]
df = df[df.home_ownership.str.contains("OTHER") == False]
df = df[df.home_ownership.str.contains("NONE") == False]





#splitting dataframe into types in order to clean further
num = []
string = []
for set in (df.columns):
    if type(df[set].loc[0]) == np.int64:
        num.append(set)
    elif type(df[set].loc[0]) == np.float64:
        num.append(set)
    else:
        string.append(set)

df_num = df[num]
df_string = df[string]
#removing variables we dont need
df_string = df_string.drop(['url', 'issue_d', 'zip_code',
                            'addr_state', 'earliest_cr_line',
                            'initial_list_status',
                           'hardship_flag',
                            'disbursement_method',
                            'debt_settlement_flag',
                            'loan_status', 'sub_grade',
                            'pymnt_plan'], axis = 1)
df_num = df_num.drop(['collection_recovery_fee',
                    'total_rec_late_fee', 'acc_now_delinq', 'delinq_amnt',
                     'open_acc', 'revol_bal', 'total_acc',
                     'out_prncp', 'out_prncp_inv', 'recoveries',
                     'inq_last_6mths',
                     'total_rec_prncp', 'total_pymnt', 'total_pymnt_inv',
                     'total_rec_int','last_pymnt_amnt',
                     'policy_code'], axis = 1)



#cleaning and categorizing string variables
le = preprocessing.LabelEncoder()
for set in (df_string):
    df_string[set] = le.fit_transform(df_string[set])





df = pd.concat([df_num, df_string], axis = 1)





df = df.drop(['Unnamed: 0', 'id'], axis = 1)





from sklearn.model_selection import train_test_split
X_train, X_test, y_train,
y_test = train_test_split(df[df.columns.difference(['grade',
                                                    'funded_amnt',
                                                    'funded_amnt_inv',
                                                    'int_rate',
                                                    'installment'])],
                                                    df.grade, test_size=0.2,
                                                    stratify=df.grade,
                                                    random_state=123456)





from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification




rf = RandomForestClassifier(n_estimators=10, oob_score=True,
                            random_state=123456)
rf.fit(X_train, y_train)




from sklearn.metrics import accuracy_score
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print('Out-of-bag score estimate:', rf.oob_score)
print('Mean accuracy score:', accuracy)
