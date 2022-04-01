e2_1p = [139.4,
148.3,
141.8,
143.4,
141.4]


e2_2p = [83.7,
85.9,
80.8,
84.2,
82.3]

e2_3p = [59.9,
56.2,
56.5,
57.6,
57.7]

import matplotlib.pyplot as plt
import seaborn as sb

import pandas as pd

df = pd.read_csv('e2_data.csv', header=[0])

sb.boxplot(y="Type", x="Time(s)", data=df)

plt.show()