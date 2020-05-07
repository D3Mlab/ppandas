#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 01:12:40 2020

@author: kwanale1
"""

import pandas as pd
from ppandas import PDataFrame

df1 = pd.read_csv("data/spatial-ref-region-point.csv")
df1.rename(columns={"Region":"Location"},inplace=True)
#print(df1.columns)
df2_point = pd.read_csv("data/spatial-sec-region-point.csv",usecols=["Point","A"])
df2_point.rename(columns={"Point":"Location"},inplace=True)
#print(df2_point.columns)
df2_x_y = pd.read_csv("data/spatial-sec-region-point.csv",usecols=["X","Y","A"])
#zip together X and Y to make one Location column
df2_x_y['Location'] = list(zip(df2_x_y.X, df2_x_y.Y))
df2_x_y.drop(columns=["X","Y"],inplace=True)

pd1 = PDataFrame(["Location"],df1)
print(pd1.query(["Location"]))
print(pd1.query(["A"]))
#pd1.visualise(show_tables=True)
pd2_point = PDataFrame(["Location"],df2_point)
pd2_x_y = PDataFrame(["Location"],df2_x_y)
#pd2_point.visualise(show_tables=True)
#pd2_x_y.visualise(show_tables=False)
#print(pd1.bayes_net.get_cpds(node = "A"))
#print(pd2_x_y.bayes_net.get_cpds(node = "A"))
print(pd2_point.query(["A"]))
print(pd2_x_y.query(["A"]))
#pd2_x_y.visualise(show_tables=True)

pd_join = pd1.pjoin(pd2_point, mismatches={"Location":"spatial"})
#print(pd_join.query(["Location"]))
#pd_join = pd1.pjoin(pd2_x_y, mismatches={"Location":"spatial"})
#print(pd_join.bayes_net.get_cpds(node = "A"))
#pd_join.visualise(show_tables = True)

#queryResult = pd_join.query(['A'],{"Location":"POINT (0.5 0.5)"}) # ToDo: Rewrite query of Point to Regions? How about brand new points?
#queryResult = pd_join.query(['A'],{"Location":"POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"}) #
queryResult = pd_join.query(['A'],{"Location":"POLYGON ((0 1, 1 1, 1 2, 0 2, 0 1))"}) #
print('conditional query')
print(queryResult)

#ToDo: Future Query by bounds on points: Eg South of this longitude X?
#Fails since reference distribution can't use Point data
pd_join = pd2_x_y.pjoin(pd1, mismatches={"Location":"spatial"})