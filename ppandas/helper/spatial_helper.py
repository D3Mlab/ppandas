#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 04:05:44 2020

@author: kwanale1
"""
from shapely.geometry import Point, LineString, shape
import shapely.wkt
import numpy as np
import geopandas
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import combinations
from shapely.ops import unary_union
from descartes import PolygonPatch
import csv

class SpatialHelper():
    @staticmethod
    # Input: List of strings, where each string is a geometric object value
    #       (MultPolygon,Polygon,Point) or a tuple of X and Y coordinates,
    #       of an old Bayes Net Location node
    # Ouput: Dictionary of strings, where key is string from input
    #        and value is a list of strings,
    #        mapping to new Bayes Net Location node
    def getMappingAsString(old_geometric_strings, new_geometric_strings):
        #Each new object should map to only 1 old object?
        mapping = defaultdict(list)
        for geo_old in old_geometric_strings:
            geo_old_obj = SpatialHelper.stringToGeoObject(geo_old)
            mapping[geo_old] = []
            for geo_new in new_geometric_strings:
                geo_new_obj = SpatialHelper.stringToGeoObject(geo_new)
                if geo_old_obj.geom_type == 'Point':
                    if geo_old_obj.within(geo_new_obj):
                        mapping[geo_old].append(geo_new)
                elif geo_old_obj.geom_type == 'Polygon' or \
                        geo_old_obj.geom_type == 'MultiPolygon':
                    if geo_old_obj.contains(geo_new_obj.buffer(-1e-10)):
                    #if geo_new_obj.within(geo_old_obj):
                        mapping[geo_old].append(geo_new)
                        #x,y = geo_old_obj.exterior.xy
                        #plt.plot(x,y)
                    #plt.show()
                    #if geo_old_obj.overlaps(geo_new_obj):
                        

#        for geo_new in new_geometric_strings:
#            geo_new_obj = SpatialHelper.stringToGeoObject(geo_new)
#            for geo_old in old_geometric_strings:
#                geo_old_obj = SpatialHelper.stringToGeoObject(geo_old)
#                if geo_old_obj.geom_type == 'Point':
#                    # If Region contains Point
#                    if geo_new_obj.contains(geo_old_obj):
#                        mapping[geo_old].append(geo_new)
#                elif geo_old_obj.geom_type == 'Polygon' or \
#                        geo_old_obj.geom_type == 'MultiPolygon':
#                    #if geo_old_obj.intersects(geo_new_obj):
#                    # Can't use .intersects because two polygons' boundaries
#                    #   overlapping means they intersect which is unwanted
#                    if geo_new_obj.overlaps(geo_old_obj) or \
#                        geo_old_obj.contains(geo_new_obj):
#                        mapping[geo_new].append(geo_old)
#                        #mapping[geo_old].append(geo_new)
##                        break
        #print(mapping)
        #with open('mycsvfile.csv', 'w') as f:  # Just use 'w' mode in 3.x
        #    w = csv.DictWriter(f, mapping.keys())
        #    w.writeheader()
        #    w.writerow(mapping)
        
        #print("inside getMappingAsString")
        numValues = 0
        OldRegionsNoMapping = 0
        for key, value in mapping.items():
            numValues += len(value)
            if len(value) == 0:
                OldRegionsNoMapping +=1
                #print("---")
                #print(value)
                #print("NumValues is " + str(numValues))
            
        #print('Old Region No Mapping')
        #print(OldRegionsNoMapping)
        #print(numValues)
        
        #print("leaving getMappingAsString")
        return mapping

    @staticmethod
    # convert string to shapely geometric object
    def stringToGeoObject(tuple_xy_or_geo_string):
        if isinstance(tuple_xy_or_geo_string, tuple):
            return Point(tuple_xy_or_geo_string)
        else:
            return shapely.wkt.loads(tuple_xy_or_geo_string)

    @staticmethod
    # output list of geo objects
    def getGeoObjectsFromString(string_geos):
        geos = []
        for geo in string_geos:
            geos.append(SpatialHelper.stringToGeoObject(geo))
        return geos

    @staticmethod
    # input:  Dictionary of strings where key is geometric object or
    #         tuple string and value is a list of strings,
    #         mapping to new Bayes Net Location node
    # output: dictionary where key are geometric object string
    #         and value is a list of geometric objects
    def convertMappingFromString(mapping):
        new_mapping = {}
        # geo_news can be a list of geometric objects
        for geo_old, geo_news in mapping.items():
            new_mapping[geo_old] = \
                SpatialHelper.getGeoObjectsFromString(geo_news)
        return new_mapping

    @staticmethod
    # output list of strings
    def convertGeoObjectsToString(geo_objects):
        string_geo_objects = []
        for geo in geo_objects:
            string_geo_objects.append(str(geo))
        return string_geo_objects

    @staticmethod
    # input: string, list of geometric objects, and old cpd value
    # Distribute cpd value for geo_old based on amount of overlap with
    # each of the geometric objects in geo_news
    def getUniformDistribution(geo_old, geo_news, value):
        # Convert string to geometric object
        geo_old_obj = SpatialHelper.stringToGeoObject(geo_old)
        size_portion = []
        for geo_new_obj in geo_news:
            geo_new_obj_buffer = geo_new_obj.buffer(0)
            if geo_old_obj.intersects(geo_new_obj):
                if geo_old_obj.area > 0:
                    size_portion.append(geo_old_obj.
                                        intersection(geo_new_obj_buffer).area
                                        / geo_old_obj.area)
                else:
                    size_portion.append(geo_old_obj.
                                        intersection(geo_new_obj).area)
            else:
                raise ValueError("{} does not overlap with {} "
                                 .format(geo_old, str(geo_new_obj)))
        size_portion = np.array(size_portion)
        # print(size_portion)
        # print(type(size_portion))
        # print(value)
        return size_portion*value

    @staticmethod
    # input: list of strings of regions (Polygons or MultiPolygons)
    def computeNewRegions(ref_regions, sec_regions):
        #https://gis.stackexchange.com/questions/326316/getting-unique-intersections-of-multiple-polygons-efficiently-in-python
        #https://gis.stackexchange.com/questions/187402/how-to-find-the-intersection-areas-of-overlapping-buffer-zones-in-single-shapefi/187499#187499
        #Issue when two sets of polygons don't have same boundary areas.
        ref_regions_obj = SpatialHelper.getGeoObjectsFromString(ref_regions)
        sec_regions_obj = SpatialHelper.getGeoObjectsFromString(sec_regions)
        #intersections = [pair[0].overlaps(pair[1]) for pair in combinations(list(ref_regions_obj), 2)]
        #intersections = [pair[0].overlaps(pair[1]) for pair in combinations(non_overlap, 2)]
        #if any(intersections):
        #    raise ValueError("%d of %d combinations overlap" % (sum(intersections), len(intersections)))
        #else:
        #    print("no overlaps in ref")
        
        #for polygon1 in sec_regions_obj:
        #    x,y = polygon1.exterior.xy
        #    plt.plot(x,y)
        
        # Intersection of all polgyons in both reference and secondary regions
        # Unique intersection of polygons 
        region_cross_product = []
        polys1 = geopandas.GeoSeries(ref_regions_obj)
        #polys1.plot()
        #plt.axis('off')
        #plt.show()
        polys2 = geopandas.GeoSeries(sec_regions_obj)
        #print(polys2.head())
        #polys2.plot()
        #plt.axis('off')
        #plt.savefig('wn_comm_areas.pdf')  
        #plt.show()
        dfA = geopandas.GeoDataFrame({'geometry': polys1})
        dfB = geopandas.GeoDataFrame({'geometry': polys2})
        #df1_explode = df1.geometry.explode()
        #print(type(df1_explode))
        #df2_explode = df2.geometry.explode()
        #df1_explode.to_csv('df1_explode.csv')
        #df2_explode.to_csv('df2_explode.csv')
        
        res_union = geopandas.overlay(dfA, dfB, how='intersection')
        #non_overlap = unary_union(list(res_union.geometry.values))
        #print(type(non_overlap))
        #intersections = [pair[0].overlaps(pair[1]) for pair in combinations(list(res_union.geometry.values), 2)]
        #intersections = [pair[0].overlaps(pair[1]) for pair in combinations(non_overlap, 2)]
       # if any(intersections):
       #     print("yes, %d of %d combinations overlap" % (sum(intersections), len(intersections)))
       # else:
       #     print("no overlaps")
        
        
        #print(res_union)
        res_union = res_union.geometry.explode()
        #print(res_union)
        region_cross_product  = list(res_union.geometry.values)
        polys_cp = geopandas.GeoSeries(region_cross_product)
        
        #
        #print(type(polys2))
        #print(type(polys_cp))
        #dict_sjoin = geopandas.sjoin(polys2, polys_cp, how='inner',op='contains')
        #dict_sjoin.head()
        #
        
        polys_cp.boundary.plot()
        #for polygon1 in region_cross_product:
        #    x,y = polygon1.exterior.xy
        #    plt.plot(x,y)
        plt.axis('off')
        plt.savefig('wn_voronoi_comm_union.pdf')
        plt.show()

        #https://gis.stackexchange.com/questions/60017/how-to-computational-efficiently-check-if-there-are-any-overlapping-polygons-in

        #intersections = [pair[0].overlaps(pair[1]) for pair in combinations(region_cross_product, 2)]
        #if any(intersections):
        #    print("yes, %d of %d combinations overlap" % (sum(intersections), len(intersections)))
        #else:
        #    print("no overlaps")
        non_overlapping = []
        for p in region_cross_product:
            overlaps = []
            for g in filter(lambda g: not g.equals(p), region_cross_product):
                overlaps.append(g.overlaps(p))
            if not any(overlaps):
                non_overlapping.append(p)
        
        #intersections = [pair[0].overlaps(pair[1]) for pair in combinations(non_overlapping, 2)]
        #if any(intersections):
        #    print("yes, %d of %d combinations overlap" % (sum(intersections), len(intersections)))
        #else:
        #    print("no overlaps")
        
        #print(len(region_cross_product))
        #print("**")
        #print(len(non_overlapping))
        #print(len(set(region_cross_product)))
        #res_union.to_csv('list.csv',index=False)
        #ax = res_union.plot(alpha=0.5, cmap='tab10')
        #dfA.plot(ax=ax, facecolor='none', edgecolor='k');
        #dfB.plot(ax=ax, facecolor='none', edgecolor='k');
        
        #for ref_region in ref_regions_obj:
        #    for sec_region in sec_regions_obj:
                # If intersection of two polygons is not null
                # then append the polygon derived from the intersection
        #        if sec_region.overlaps(ref_region) or \
        #                    ref_region.contains(sec_region):
                #if ref_region.intersects(sec_region):
        #            region_cross_product.append(ref_region.
        #                                        intersection(sec_region))
        # print(SpatialHelper.convertGeoObjectsToString(region_cross_product))
        
        return SpatialHelper.convertGeoObjectsToString(region_cross_product)
        #return SpatialHelper.convertGeoObjectsToString(non_overlapping)