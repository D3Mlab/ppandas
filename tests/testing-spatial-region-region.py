#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 00:53:17 2020

@author: kwanale1
"""
import pandas as pd
from ppandas import PDataFrame

df1 = pd.read_csv("data/spatial-ref-region-region.csv")
df2 = pd.read_csv("data/spatial-sec-region-region.csv")
pd1 = PDataFrame(["Region"],df1)
pd2 = PDataFrame(["Region"],df2)
pd_join = pd1.pjoin(pd2, mismatches={"Region":"spatial"})
#pd_join = pd2.pjoin(pd1, mismatches={"Region":"spatial"})

print(pd1.query(["Region"]))
print(pd1.query(["A"]))
print(pd2.query(["Region"]))
print(pd2.query(["B"]))
#pd1.visualise(show_tables=True)
#pd2.visualise(show_tables=True)
pd_join.visualise(show_tables=True)
#queryResult = pd_join.query(['A'],{"Region":"POLYGON ((0 0, 1 0, 1 1.5, 0 1.5, 0 0))"})
#queryResult = pd_join.query(['A'],{"Region":"POLYGON ((0 1.5, 1 1.5, 1 3, 0 3, 0 1.5))"})
queryResult = pd_join.query(['B'],{"Region":"POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"},entries='second')

print('conditional query')
print(queryResult)