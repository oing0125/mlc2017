# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["dir", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn import tree

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

# Count the male person who survived
print(df_train["Survived"][df_train["Sex"] == 'male'].value_counts(normalize = True))

# 18세 이상과 이하 구분
df_train["Child"] = float('NaN')

df_train["Child"][df_train["Age"]<18] = 1
df_train["Child"][df_train["Age"]>=18] = 0

print(df_train["Survived"][df_train["Child"] == 1].value_counts())
print(df_train["Survived"][df_train["Child"] == 0].value_counts())