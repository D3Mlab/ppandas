#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:07:47 2020

@author: kwanale1
"""
import pandas as pd
import ppandas
from ppandas import PDataFrame

df1 = pd.read_csv("testing/ab.csv")
df5 = pd.read_csv("testing/a-1.csv")

pd1 = PDataFrame(["A"],df1)
pd1.visualise(show_tables=True)
print(pd1.num_of_records)


pd5 = PDataFrame(["A"],df5)
#pd5.visualise(show_tables=True)
#print(pd5.num_of_records)

ppd_join = pd5.pjoin(pd1)
print('---Joined PDataFrame---')
ppd_join.visualise(show_tables=True)
#queryResult = ppd_join.query(["B"])
#print(queryResult)
print(ppd_join.num_of_records) 