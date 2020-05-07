#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:49:18 2020

@author: kwanale1
"""
import pandas as pd
from shapely.geometry import Polygon, Point
import geopandas as gpd
voting_shp = gpd.read_file("VOTING_SUBDIVISION_2014_WGS84.shp")
wards_shp = gpd.read_file("icitw_wgs84.shp")
#print(wards_shp)
#shapefile.plot()
print(wards_shp.columns)
print(wards_shp.head)
#Spatial Dataset Generator
#Two multi-polgyons being two wards in Toronto
wards_shp_two = wards_shp.loc[0:1]
#print(wards_shp_two.columns)

#reference cell: first row, geometry column
poly0 = wards_shp_two.at[0,'geometry']
poly1 = wards_shp_two.at[1,'geometry']


#1. Generate points within a multipolygon for second dist
#Generate 2 spatial csv files
# First csv is reference distribution with location column using Regions, 80/20 split 
# Second csv is secondary distribution that uses points, 50/50 split 
poly_ref = []
A_ref = []
A_ref += [0]*20
A_ref += [1]*50
A_ref += [0]*30

point_sec = []
point_sec_lat_Y = []
point_sec_long_X = []
A_sec = []
A_sec += [0]*30
A_sec += [1]*60
A_sec += [0]*10
print(type(poly1.representative_point()))
i = 0
while i < 50:
    poly_ref.append(poly0)
    point_sec.append(poly0.representative_point())
    point_sec_long_X.append(poly0.representative_point().x)
    point_sec_lat_Y.append(poly0.representative_point().y)
    ##olypoints.append(poly0.representative_point()) #generate a point within poly0
    i += 1
while i < 80:
    poly_ref.append(poly0)
    point_sec.append(poly1.representative_point())
    point_sec_long_X.append(poly1.representative_point().x)
    point_sec_lat_Y.append(poly1.representative_point().y)
    i += 1
while i < 100:
    poly_ref.append(poly1)
    point_sec.append(poly1.representative_point())
    point_sec_long_X.append(poly1.representative_point().x)
    point_sec_lat_Y.append(poly1.representative_point().y)
    i += 1
df_ref = pd.DataFrame(data=poly_ref,columns=['Region'])
df_ref['A']=pd.Series(A_ref)

df_sec = pd.DataFrame(data=point_sec,columns=['Point'])
df_sec['X']=pd.Series(point_sec_long_X)
df_sec['Y']=pd.Series(point_sec_lat_Y)
df_sec['A']=pd.Series(A_sec)
#print(df_ref.head())
#print(df_ref.head())
df_ref.to_csv('spatial_ref.csv',index=False,header=True)
df_sec.to_csv('spatial_sec.csv',index=False,header=True)



