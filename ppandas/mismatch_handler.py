import numpy as np
import ast
from .helper.bayes_net_helper import BayesNetHelper
from .helper.interval_helper import IntervalHelper
from .helper.spatial_helper import SpatialHelper
from pgmpy.factors.discrete import TabularCPD
from shapely.ops import unary_union
from geovoronoi import voronoi_regions_from_coords
from geovoronoi import points_to_coords
from shapely.geometry import Point
import re

# Well-Known Text (WKT) ensurer so that all the mapping works as expected

def ensure_wkt_point(s):
    if isinstance(s, str):
        s_strip = s.strip()
        if s_strip.startswith('POINT'):
            return s_strip  # already good WKT
        # Check if it looks like a tuple string "(x, y)"
        elif re.match(r'^\(\s*[\d\.\-]+,\s*[\d\.\-]+\s*\)$', s_strip):
            # convert string tuple to WKT
            x, y = s_strip.strip('()').split(',')
            return f"POINT ({x.strip()} {y.strip()})"
    # If not string or unknown format, return as is (or raise)
    return s

class MismatchHandler():

    def __init__(self, node):
        self.node = node

    def computeMapping(self, reference_bayes, second_bayes):
        reference_old_entries = reference_bayes.get_cpds(
            node=self.node).state_names[self.node]
        second_old_entries = second_bayes.get_cpds(
            node=self.node).state_names[self.node]
        ref_entries, sec_entries, new_entries = self.computeCrossProduct(
            reference_old_entries, second_old_entries
        )
        # here, mapping is represented in strings
        # for example, mapping = {'[20,50]':['[20,40]','[40,50]'],
        # '[50,80]':['[50,60]','[60,80]']}
        ref_entries_wkt = [ensure_wkt_point(e) if isinstance(e, str) else Point(e).wkt for e in ref_entries]
        sec_entries_wkt = [ensure_wkt_point(e) if isinstance(e, str) else Point(e).wkt for e in sec_entries]
        new_entries_wkt = [ensure_wkt_point(e) if isinstance(e, str) else Point(e).wkt for e in new_entries]

        reference_mapping = self.getMapping(ref_entries_wkt, new_entries_wkt)
        second_mapping = self.getMapping(sec_entries_wkt, new_entries_wkt)

        return reference_mapping, second_mapping

    def replaceMismatchNode(self, bayes_net, mapping):
        # Remove all related CPDs
        bayes_net_copy = BayesNetHelper.removeRelatedCpds(bayes_net, self.node)
        # Add new parent CPD
        new_parent_cpd = self.computeParentCpd(bayes_net, mapping)
        bayes_net_copy.add_cpds(new_parent_cpd)
        # Synchronize state names for the mismatch variable
        canonical_state_names = new_parent_cpd.state_names[self.node]
        # Always rebuild all child CPDs to match new cardinality
        bayes_net_copy = BayesNetHelper.rebuild_categorical_child_cpds(
            bayes_net, bayes_net_copy, mapping, self.node, canonical_state_names=canonical_state_names)
        return bayes_net_copy


class categoricalHandler(MismatchHandler):

    def computeCrossProduct(self, reference_old_entries, second_old_entries):
        # For categorical: new state space is the sorted union of both sets
        union_entries = sorted(set(reference_old_entries) | set(second_old_entries))
        return union_entries, union_entries, union_entries

    def getMapping(self, old_entries, new_entries):
        # For categorical: identity mapping for present categories
        mapping = {}
        for old_entry in old_entries:
            if old_entry in new_entries:
                mapping[old_entry] = [old_entry]
            else:
                mapping[old_entry] = []
        return mapping

    def computeParentCpd(self, bayes_net, mapping):
        cpd_node = bayes_net.get_cpds(node=self.node)
        old_states = list(cpd_node.state_names[self.node])
        new_states = sorted(set(old_states) | set(mapping.keys()))
        new_card = len(new_states)
        old_values = np.array(cpd_node.values).flatten()
        new_values = np.zeros(new_card)
        for i, s in enumerate(new_states):
            if s in old_states:
                idx = old_states.index(s)
                new_values[i] = old_values[idx]
            else:
                new_values[i] = 0.0
        total = new_values.sum()
        if total > 0:
            new_values = new_values / total
        new_state_names_dict = {self.node: new_states}
        new_values_array = new_values.reshape(new_card, 1)
        new_cpd = TabularCPD(self.node, new_card, new_values_array, state_names=new_state_names_dict)
        new_cpd.normalize()
        return new_cpd

    def getUniformDistribution(self, cardinality, value):
        return[value/cardinality for i in range(0, cardinality)]


class NumericalHandler(MismatchHandler):

    def computeCrossProduct(self, reference_old_entries, second_old_entries):
        # Convert string intervals to interval objects
        reference_intervals = IntervalHelper.getIntervalsFromString(reference_old_entries)
        second_intervals = IntervalHelper.getIntervalsFromString(second_old_entries)
        new_intervals = IntervalHelper.computeNewIntervals(reference_intervals, second_intervals)
        return (
            IntervalHelper.convertIntervalsToString(reference_intervals),
            IntervalHelper.convertIntervalsToString(second_intervals),
            IntervalHelper.convertIntervalsToString(new_intervals)
        )

    def replaceMismatchNode(self, bayes_net, mapping):
        i_mapping = IntervalHelper.convertMappingFromString(mapping)
        # Remove all related CPDs
        bayes_net_copy = BayesNetHelper.removeRelatedCpds(bayes_net, self.node)
        # Add new parent CPD
        new_parent_cpd = self.computeParentCpd(bayes_net, mapping)
        bayes_net_copy.add_cpds(new_parent_cpd)
        # Synchronize state names for the mismatch variable
        canonical_state_names = new_parent_cpd.state_names[self.node]
        # Always rebuild all child CPDs to match new cardinality (same as categorical)
        bayes_net_copy = BayesNetHelper.rebuild_categorical_child_cpds(
            bayes_net, bayes_net_copy, mapping, self.node, canonical_state_names=canonical_state_names)
        return bayes_net_copy

    def computeParentCpd(self, bayes_net, mapping):
        cpd_node = bayes_net.get_cpds(node=self.node)
        values = cpd_node.values
        i = 0
        new_values = []
        new_state_names = []
        new_card = 0
        for iv_old in cpd_node.state_names[self.node]:
            # mapping keys are strings, not interval objects
            iv_news = mapping[iv_old]
            # Convert iv_news to interval objects if needed
            iv_news_objs = IntervalHelper.getIntervalsFromString(iv_news)
            new_card += len(iv_news_objs)
            new_state_names.extend(
                IntervalHelper.convertIntervalsToString(iv_news_objs))
            new_values.extend(
                IntervalHelper.getUniformDistribution(
                    iv_old, iv_news_objs, values[i]))
            i += 1
        new_state_names = {self.node: new_state_names}
        new_values_array = np.array(new_values).reshape(new_card, 1)
        # Defensive: always normalize TabularCPD
        new_cpd = TabularCPD(self.node, new_card, new_values_array, state_names=new_state_names)
        new_cpd.normalize()
        return new_cpd

    def getMapping(self, old_intervals, new_intervals):
        old_intervals = IntervalHelper.getIntervalsFromString(old_intervals)
        new_intervals = IntervalHelper.getIntervalsFromString(new_intervals)
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
    new_entries = None
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
            sec = second_old_entries[0]
            if isinstance(sec, str) and 'POLYGON' in sec:
                new_entries = SpatialHelper.computeNewRegions(reference_old_entries, second_old_entries)
            elif isinstance(sec, str) and 'POINT' in sec:
                new_entries = reference_old_entries
            elif isinstance(sec, str) and sec.startswith("(") and sec.endswith(")"):
                second_old_entries[0] = ast.literal_eval(sec)
                new_entries = reference_old_entries
            elif isinstance(sec, tuple):
                new_entries = reference_old_entries
            else:
                raise ValueError(f"Unknown geometry format: {sec}")
        elif ('POINT' in second_old_entries[0]) or \
        (isinstance(second_old_entries[0], tuple)) or \
        (isinstance(second_old_entries[0], str) and second_old_entries[0].startswith('(')):
            new_entries = reference_old_entries
            # For 3. compute Voronoi
            ref_points_obj = SpatialHelper.getGeoObjectsFromString(list(set(reference_old_entries)))
            sec_regions_obj = SpatialHelper.getGeoObjectsFromString(second_old_entries)
            sec_regions_obj_union = unary_union(sec_regions_obj)
            for point in ref_points_obj:
                if not point.within(sec_regions_obj_union):
                    raise ValueError("Points from reference distribution lie outside union of regions from secondary distribution.")
            coords = points_to_coords(ref_points_obj)
            ref_vor_regions_obj, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, sec_regions_obj_union)
            ref_vor_regions = SpatialHelper.convertGeoObjectsToString(ref_vor_regions_obj)
            new_entries = SpatialHelper.computeNewRegions(ref_vor_regions, second_old_entries)
        else:
            raise ValueError("Invalid input for {} variable. Spatial mismatch\
                             variables must be a 'Multipolygon', 'Polygon'\
                             or 'Point', or be a tuple of (X,Y) coordinates."
                             .format(str(self.node)))
        
        if 'new_entries' not in locals():
            raise RuntimeError("new_entries was never assigned! Check input formats.")
        
        return reference_old_entries, second_old_entries, new_entries

    # Mapping from old bayes net state names to new bayes net state names
    def getMapping(self, old_entries, new_entries):
        return SpatialHelper.getMappingAsString(old_entries, new_entries)

    def replaceMismatchNode(self, bayes_net, mapping):
        fixed_mapping = {}
        for k, v in mapping.items():
            # Convert key to WKT string if it's a tuple
            k_str = Point(k).wkt if isinstance(k, tuple) else k

            # Convert values to WKT strings if needed
            v_str = [
                Point(t).wkt if isinstance(t, tuple) else t
                for t in v
            ]

            fixed_mapping[k_str] = v_str

        geo_mapping = SpatialHelper.convertMappingFromString(fixed_mapping)

        return MismatchHandler.replaceMismatchNode(self, bayes_net, geo_mapping)

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
        new_values_array = np.array(new_values).reshape(new_card, 1)
        new_cpd = TabularCPD(self.node, new_card, new_values_array, state_names=new_state_names)
        return new_cpd
