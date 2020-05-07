#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:49:18 2020

@author: kwanale1
"""
import pandas as pd
from shapely.geometry import Multipolygon, Polygon, Point
#voting_shp = gpd.read_file("VOTING_SUBDIVISION_2014_WGS84.shp")
#wards_shp = gpd.read_file("icitw_wgs84.shp")
#print(wards_shp)
#shapefile.plot()
#print(wards_shp.columns)
#print(wards_shp.head)
#Spatial Dataset Generator
#Two multi-polgyons being two wards in Toronto
#wards_shp_two = wards_shp.loc[0:1]
#print(wards_shp_two.columns)

#reference cell: first row, geometry column
poly0a = Polygon([(0,0),(1,0),(1,1.5),(0,1.5)])
poly0b = Polygon([(0,1.5),(1,1.5),(1,3),(0,3)])

poly1a = Polygon([(0,0),(1,0),(1,1),(0,1)])
poly1b = Polygon([(0,1),(1,1),(1,2),(0,2)])
poly1c = Polygon([(0,2),(1,2),(1,3),(0,3)])

#1. Generate points within a multipolygon for second dist
#Generate 2 spatial csv files
# First csv is reference distribution with location column using Regions, 80/20 split 
# Second csv is secondary distribution that uses points, 50/50 split 
poly_ref = []
A_ref = []
A_ref += [0]*20
A_ref += [1]*50
A_ref += [0]*30

poly_sec = []
A_sec = []
A_sec += [0]*30
A_sec += [1]*60
A_sec += [0]*10
i = 0
while i < 50:
    poly_ref.append(poly0a)
    poly_sec.append(poly1a)
    ##polypoints.append(poly0.representative_point()) #generate a point within poly0
    i += 1
while i < 80:
    poly_ref.append(poly0a)
    poly_sec.append(poly1b)
    i += 1
while i < 100:
    poly_ref.append(poly0b)
    poly_sec.append(poly1c)
    i += 1
df_ref = pd.DataFrame(data=poly_ref,columns=['Region'])
df_ref['A']=pd.Series(A_ref)

df_sec = pd.DataFrame(data=poly_sec,columns=['Region'])
df_sec['A']=pd.Series(A_sec)
#print(df_ref.head())
#print(df_ref.head())
df_ref.to_csv('spatial_ref_region.csv',index=False,header=True)
df_sec.to_csv('spatial_sec_region.csv',index=False,header=True)



