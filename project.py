#Import libraries
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt



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
frame['int_rate'] = frame['int_rate'].str.rstrip('%').astype('float') / 100.0





# Plot distribution of interest rates for each loan grade

for grade in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    subset = frame['grade'] == grade
    frame['int_rate'][subset].hist(label = grade, normed = True, alpha = 0.75)

plt.legend()
plt.show()

# Plot distribution of interest rates for each loan subgrade

for sub_grade in {sg for sg in set(frame['sub_grade']) if sg == sg}:
    subset = frame['sub_grade'] == sub_grade
    frame['int_rate'][subset].hist(label = sub_grade, normed = True, alpha = 0.75)

plt.legend()
plt.show()
