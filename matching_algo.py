import pandas as pd
import numpy as np

#create dataframe of random preferences based on parameters
np.random.seed(0)
N = 1*(10**6)
id = list(range(N))
risk_score = np.random.uniform(0,1, N)
char = np.random.uniform(0,1, N)
error = np.random.normal(size = 7)
alpha = np.random.uniform(0,1, size = 7)
gamma = np.random.uniform(0,1, size = 7)
util = []

for i in range(7):
    util.append(alpha[i]*risk_score+gamma[i]*char + error[i])

data = pd.DataFrame({'A':util[0],'B':util[1],'C':util[2],'D':util[3],
    'E':util[4], 'F':util[5],'G':util[6]})

data['member_id'] = list(range(N))

capacities = pd.DataFrame()
capacities['A'] = [125000]
capacities['B'] = [250000]
capacities['C'] = [125000]
capacities['D'] = [52500]
capacities['E'] = [1000]
capacities['F'] = [69320]
capacities['G'] = [377180]

def matching_algorithm(data, capacities):
    df = pd.DataFrame()
    data_length = len(data)
    for grade in grades:
        data = data.sort_values(by = [grade], ascending = False)
        temp = data[:int(capacities[grade])].member_id.tolist()
        diff = data_length-len(temp)
        na = ['NaN']*diff
        df[grade] = temp + na
        data = data[int(capacities[grade]):]
    return(df)

final_matching = matching_algorithm(data, capacities)



#LogisticRegression(multi_class='multinomial',solver ='newton-cg').fit(dff[['a','b']], util[0]).summary()
