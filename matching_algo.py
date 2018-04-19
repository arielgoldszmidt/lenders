import numpy as np
import pandas as pd

#generating random utilities normalized to the unit inverval
A = np.random.uniform(low = 0.0, high = 1.0, size = 1*(10**6))
B = np.random.uniform(low = 0.0, high = 1.0, size = 1*(10**6))
C = np.random.uniform(low = 0.0, high = 1.0, size = 1*(10**6))
D = np.random.uniform(low = 0.0, high = 1.0, size = 1*(10**6))
E = np.random.uniform(low = 0.0, high = 1.0, size = 1*(10**6))
F = np.random.uniform(low = 0.0, high = 1.0, size = 1*(10**6))
G = np.random.uniform(low = 0.0, high = 1.0, size = 1*(10**6))

grades = {'A':A,'B':B,'C':C,'D':D,'E':E,'F':F,'G':G}

data = pd.DataFrame(grades)

data['member_id'] = list(range(1*(10**6)))

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
