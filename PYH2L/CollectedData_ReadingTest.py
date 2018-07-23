import numpy as np
import pandas as pd
import os

Ting_path = os.getcwd()+'\Data\MergeDat\Ting.csv'

test = pd.read_csv(Ting_path, engine='c')
test = test.drop(['Unnamed: 2'], axis=1).set_index(['Unnamed: 0', 'Unnamed: 1']).rename_axis(('Source',
                                                                                              'Type'))
print(test)
print(test.loc[(slice(None), 'Img'), :])
test3 = test.loc[(slice(None), 'Img'), :]
test2 = test3.groupby(level=0).apply(lambda x: x.values.tolist()).values
test2 = [np.extract(np.logical_not(np.isnan(test2[i])[0]), test2[i][0]) for i in range(0, len(test2))]
print(test2)
