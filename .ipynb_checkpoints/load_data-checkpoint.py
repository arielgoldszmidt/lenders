# Import libraries

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Load data
path = r'data'
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, index_col = None, header = 1)
    list_.append(df)
frame = pd.concat(list_)

# Change interest rate to float
frame['int_rate_num'] = frame['int_rate'].str.rstrip('%').astype('float') / 100.0

# Cast emp_length to int
def emp_length_to_int(s):
    return max([int(n) for n in str(s).split() if n.isdigit()] + [0])
frame['emp_length_num'] = frame['emp_length'].apply(emp_length_to_int)

# Add late and default dummies
is_late_16_30 = lambda s: int(s == "Late (16-30 days)")
is_late_31_120 = lambda s: int(s == "Late (31-120 days)")
is_default = lambda s: int(s == "Default")
is_bad = lambda s: int(s == "Default" or s == 'Does not meet the credit policy. Status:Charged Off' or s == "Charged Off")

frame['late_16_30'] = frame.loan_status.apply(is_late_16_30)
frame['late_31_120'] = frame.loan_status.apply(is_late_31_120)
frame['default'] = frame.loan_status.apply(is_default)
frame['bad'] = frame.loan_status.apply(is_bad)

# Add interest-to-total-received and late-fees-to-total-received ratios
frame['int_to_total'] = frame['total_rec_int'] / frame['total_pymnt']
frame['late_fees_to_total'] = frame['total_rec_late_fee'] / frame['total_pymnt']

# Add length in years
frame["term_years"] = frame.term.str.rstrip(" months").astype("float") / 12

# Add returns
frame['return'] = (frame.total_pymnt / frame.funded_amnt - 1) / frame.term_years + 1
frame['return2'] = frame["return"] ** 2

# Add year issued
frame['issued_year'] = frame.issue_d.apply(lambda s: str(s)[-4:])


# Construct data frame of just pre-application characteristics
frame_completed = frame[frame.loan_status != 'Current']

post_variables = ["all_util", "term",
                "collection_recovery_fee", 
                "funded_amnt", "funded_amnt_inv", "id", "initial_list_status",
                "installment", "int_rate", "issue_d", "last_credit_pull_d", 
                "last_fico_range_high", "last_fico_range_low", "last_pymnt_amnt",
                "last_pymnt_d", "member_id", "next_pymnt_d", "out_prncp", 
                "out_prncp_inv", "pymnt_plan", "sub_grade", 
                "total_pymnt", "total_pymnt_inv", "total_rec_int", "total_rec_late_fee",
                "total_rec_prncp", "hardship_flag", "hardship_type", "hardship_reason",
                "hardship_status", "deferral_term", "hardship_amount",
                "hardship_start_date", "hardship_end_date", "payment_plan_start_date",
                "hardship_length", "hardship_dpd", "hardship_resaon", "hardship_loan_status",
                "earliest_cr_line", "id", "loan_status", "debt_settlement_flag", 
                "zip_code", "title", "desc", "url", "emp_length", "settlement_status",
                "emp_title", 'settlement_date', "debt_settlement_flag_date", 
                "sec_app_earliest_cr_line", "collection_recovery_fee", "revol_util",
                "late_16_30", "late_31_120", "default", "int_to_total", "late_fees_to_total"]
post_variables = list(set(post_variables).intersection(frame.columns))

pre_data = frame_completed.drop(post_variables, axis=1)
pre_data_dummies = pd.get_dummies(pre_data)

# Drop nas
pre_data_dummies_no_na = pre_data_dummies[pre_data_dummies.columns[pre_data_dummies.isnull().sum() < 200]].dropna()


# Split data into traning and test
from sklearn.model_selection import train_test_split

regressands = ['bad', 'return', 'return2', 'recoveries', 
               "grade_A", "grade_B", "grade_C", "grade_D", "grade_E", "grade_F", "grade_G"]
features = pre_data_dummies_no_na.drop(regressands, axis=1)
labels = pre_data_dummies_no_na[regressands]
_train_features, _test_features, _train_labels, _test_labels = train_test_split(features, labels, test_size = 0.5, 
                                                                            random_state = 42)
train_features, test_features, train_labels, test_labels = _train_features.copy(), _test_features.copy(), _train_labels.copy(), _test_labels.copy()