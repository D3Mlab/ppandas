import intervals as I
import numpy as np


class IntervalHelper():

    @staticmethod
    def computeNewIntervals(intervals1, intervals2):
        newIntervals = []
        for iv1 in intervals1:
            for iv2 in intervals2:
                iv_new = iv1 & iv2
                if not iv_new.is_empty():
                    newIntervals.append(iv_new)
        return newIntervals

    @staticmethod
    def getIntervalsFromString(string_intervals):
        intervals = []
        for s in string_intervals:
            intervals.append(I.from_string(str(s), conv=int))
        return intervals

    @staticmethod
    def getIntervalFromString(string):
        return I.from_string(str(string), conv=int)

    @staticmethod
    def convertIntervalsToString(intervals):
        string_intervals = []
        for iv in intervals:
            string_intervals.append(I.to_string(iv, conv=int))
        return string_intervals

    @staticmethod
    def getMapping(old_intervals, new_intervals):
        mapping = {}
        for iv_old in old_intervals:
            mapping[iv_old] = []
            for iv_new in newintervals:
                if iv_new in iv_old:
                    mapping[iv_old].append(iv_new)
        return mapping

    @staticmethod
    def getMappingAsString(old_intervals, new_intervals):
        mapping = {}
        for iv_old in old_intervals:
            mapping[I.to_string(iv_old, conv=int)] = []
            for iv_new in new_intervals:
                if iv_new in iv_old:
                    mapping[I.to_string(iv_old, conv=int)]\
                        .append(I.to_string(iv_new, conv=int))
        return mapping

    @staticmethod
    def convertMappingFromString(mapping):
        new_mapping = {}
        for iv_old, iv_news in mapping.items():
            new_mapping[I.from_string(iv_old, conv=int)] =\
                IntervalHelper.getIntervalsFromString(iv_news)
        return new_mapping

    @staticmethod
    def getUniformDistribution(iv_old, iv_news, value):
        size = [iv_new.upper - iv_new.lower for iv_new in iv_news]
        size_portion = size/np.sum(size)
        return size_portion*value
