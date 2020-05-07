from ppandas.helper.bayes_net_helper import BayesNetHelper
from ppandas.helper.interval_helper import IntervalHelper
from ppandas.helper.spatial_helper import SpatialHelper
from pgmpy.factors.discrete import TabularCPD
from shapely.ops import unary_union
from geovoronoi import voronoi_regions_from_coords
from geovoronoi import points_to_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area

class MismatchHandler():

    def __init__(self, node):
        self.node = node

    def computeMapping(self, reference_bayes, second_bayes):
        reference_old_entries = reference_bayes.get_cpds(
            node=self.node).state_names[self.node]
        second_old_entries = second_bayes.get_cpds(
            node=self.node).state_names[self.node]
        ref_old_entries, second_old_entries, new_entries = \
            self.computeCrossProduct(reference_old_entries, second_old_entries)
        # here, mapping is reperesented in strings
        # for example, mapping = {'[20,50]':['[20,40]','[40,50]'],
        # '[50,80]':['[50,60]','[60,80]']}
        #print("inside computeMapping")
        reference_mapping = self.getMapping(ref_old_entries, new_entries)
        second_mapping = self.getMapping(second_old_entries, new_entries)
        return reference_mapping, second_mapping

    def replaceMismatchNode(self, bayes_net, mapping):
        # for intervals, convert string representation to intervals first
        bayes_net_copy = BayesNetHelper.removeRelatedCpds(bayes_net, self.node)
        bayes_net_copy.add_cpds(self.computeParentCpd(bayes_net, mapping))
        bayes_net_copy = BayesNetHelper.mapCondtionalCpd(
            bayes_net, bayes_net_copy, mapping, self.node)
        return bayes_net_copy


class categoricalHandler(MismatchHandler):

    def computeCrossProduct(self, reference_old_entries, second_old_entries):
        new_entries = []
        for ref_entry in reference_old_entries:
            for sec_entry in second_old_entries:
                new_entries.append("{},{}".format(ref_entry, sec_entry))
        return reference_old_entries, second_old_entries, new_entries

    def getMapping(self, old_entries, new_entries):
        mapping = {}
        for old_entry in old_entries:
            mapping[old_entry] = []
            for s in new_entries:
                if old_entry in s:
                    mapping[old_entry].append(s)
        return mapping

    def computeParentCpd(self, bayes_net, mapping):
        cpd_node = bayes_net.get_cpds(node=self.node)
        values = cpd_node.values
        new_values = []
        new_state_names = []
        new_card = 0
        i = 0
        for old_entry, new_entries in mapping.items():
            new_card += len(new_entries)
            new_state_names.extend(new_entries)
            new_values.extend(
                self.getUniformDistribution(len(new_entries), values[i]))
            i += 1
        new_state_names = {self.node: new_state_names}
        new_cpd = TabularCPD(self.node, new_card, [new_values],
                             state_names=new_state_names)
        return new_cpd

    def getUniformDistribution(self, cardinality, value):
        return[value/cardinality for i in range(0, cardinality)]


class NumericalHandler(MismatchHandler):

    def computeCrossProduct(self, reference_old_entries, second_old_entries):
        reference_intervals = IntervalHelper.getIntervalsFromString(
            reference_old_entries)
        second_intervals = IntervalHelper.getIntervalsFromString(
            second_old_entries)
        new_intervals = IntervalHelper.computeNewIntervals(
            reference_intervals, second_intervals)
        return reference_intervals, second_intervals, new_intervals

    def replaceMismatchNode(self, bayes_net, mapping):
        i_mapping = IntervalHelper.convertMappingFromString(mapping)
        return MismatchHandler.replaceMismatchNode(self, bayes_net, i_mapping)

    def computeParentCpd(self, bayes_net, mapping):
        cpd_node = bayes_net.get_cpds(node=self.node)
        values = cpd_node.values
        i = 0
        new_values = []
        new_state_names = []
        new_card = 0
        for iv_old in cpd_node.state_names[self.node]:
            iv_news = mapping[IntervalHelper.getIntervalFromString(iv_old)]
            new_card += len(iv_news)
            new_state_names.extend(
                IntervalHelper.convertIntervalsToString(iv_news))
            new_values.extend(
                IntervalHelper.getUniformDistribution(
                    iv_old, iv_news, values[i]))
            i += 1
        new_state_names = {self.node: new_state_names}
        new_cpd = TabularCPD(self.node, new_card, [new_values],
                             state_names=new_state_names)
        return new_cpd

    def getMapping(self, old_intervals, new_intervals):
        return IntervalHelper.getMappingAsString(old_intervals, new_intervals)


# Region -  input could be a .csv with a MultiPolygon
#           or .shp file with a MultiPolygon,
#        - user must convert into dataframe with single column for Region
#          containing Multipolygons
# Point - data could be a .csv with Latitude and Longitude columns
#         or a .shp file with a Point column
#       - user must convert into a dataframe with either two columns for
#         Lat and Long or a single column with Point in wkt format
class spatialHandler(MismatchHandler):
    # input: list of state names from two Bayes Nets
    # output same as input, plus list of new Bayes Net state names
    def computeCrossProduct(self, reference_old_entries, second_old_entries):
        # 1. Reference distribution is regions, second dist is regions
        #    - uniform distribute overlap of regions
        # 2. Reference distribution is regions, second dist is points
        #    - map points to regions and perform as usual
        # 3. Reference distribution is points, second dist is regions
        #    - create regions from reference dist. points with Voronoi diagram
        # 4. Reference distribution is points, second dist is points
        #    - " "
        if 'POLYGON' in reference_old_entries[0]:
            if 'POLYGON' in second_old_entries[0]:
                # For 1. compute cross product for new regions
                new_entries = SpatialHelper.computeNewRegions(
                        reference_old_entries, second_old_entries)
                #print('inside spatial handler')
                #print(len(reference_old_entries))
                #print(len(second_old_entries))
                #print(len(new_entries))
                #print('leaving spatial handler')
            elif 'POINT' in second_old_entries[0] or \
                 type(second_old_entries[0]) is tuple:
                # For 2., don't need to compute cross product
                # Just use Regions of reference distribution
                new_entries = reference_old_entries
        elif 'POINT' in reference_old_entries[0] or \
             type(reference_old_entries[0]) is tuple:
            # For 3. compute Voronoi
            ref_points_obj = SpatialHelper.getGeoObjectsFromString(list(set(reference_old_entries)))
            sec_regions_obj = SpatialHelper.getGeoObjectsFromString(second_old_entries)
            sec_regions_obj_union = unary_union(sec_regions_obj)
            for point in ref_points_obj:
                #Remove points from reference that are outside sec_region_union
                if not point.within(sec_regions_obj_union):
                    raise ValueError("Points from reference distribution lie outside union of regions from secondary distribution.")
                    #ref_points_obj.remove(point)
                    #train_df = train_df[train_df.Region != (point.x,point.y)]
            coords = points_to_coords(ref_points_obj)
            ref_vor_regions_obj, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, sec_regions_obj_union)
            #Convert back to string
            ref_vor_regions = SpatialHelper.convertGeoObjectsToString(ref_vor_regions_obj)
            #print(ref_vor_regions[0])
            new_entries = SpatialHelper.computeNewRegions(ref_vor_regions, second_old_entries)
            # For 3. and 4., return error for now
            #raise ValueError("Invalid input for the {} variable. Spatial\
            #                 mismatch variables of the reference distribution\
            #                 must be 'Multipolygon' or 'Polygon.'"
            #                 .format(str(self.node)))
            #print('inside spatial handler')
            #print(len(reference_old_entries))
            #print(len(second_old_entries))
            #print(len(new_entries))
            #print('leaving spatial handler')
        else:
            raise ValueError("Invalid input for {} variable. Spatial mismatch\
                             variables must be a 'Multipolygon', 'Polygon'\
                             or 'Point', or be a tuple of (X,Y) coordinates."
                             .format(str(self.node)))
        return reference_old_entries, second_old_entries, new_entries

    # Mapping from old bayes net state names to new bayes net state names
    def getMapping(self, old_entries, new_entries):
        #print("these are old entries")
        #print(old_entries)
        return SpatialHelper.getMappingAsString(old_entries, new_entries)

    def replaceMismatchNode(self, bayes_net, mapping):
        geo_mapping = SpatialHelper.convertMappingFromString(mapping)
        return MismatchHandler.replaceMismatchNode(self, bayes_net,
                                                   geo_mapping)

    def computeParentCpd(self, bayes_net, mapping):
        cpd_node = bayes_net.get_cpds(node=self.node)
        values = cpd_node.values
        i = 0
        new_values = []
        new_state_names = []
        new_card = 0
        for geo_old in cpd_node.state_names[self.node]:
            geo_news = mapping[geo_old]
            new_card += len(geo_news)
            new_state_names.extend(SpatialHelper.
                                   convertGeoObjectsToString(geo_news))
            new_values.extend(SpatialHelper.getUniformDistribution(geo_old,
                                                                   geo_news,
                                                                   values[i]))
            i += 1
        new_state_names = {self.node: new_state_names}
        
        #print(new_state_names)
        new_cpd = TabularCPD(self.node, new_card, [new_values],
                             state_names=new_state_names)
        return new_cpd
