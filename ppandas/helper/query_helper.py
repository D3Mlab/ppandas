import pandas as pd
import numpy as np
from .bayes_net_helper import BayesNetHelper

class QueryHelper():
    def __init__(self, mapping):
        self.mapping = mapping

    def query(self, bayes_net, query_vars, evidence_vars):
        # if evidence vars is mismatched and was assigned specific value,
        # need to expand the query
        need_to_expand = []
        if evidence_vars is not None:
            new_evidence_vars, need_to_expand = self.mapEvidenceVars(
                evidence_vars)
        if need_to_expand:
            list_of_new_evidence_vars = self.expandQueries(
                new_evidence_vars, need_to_expand)
            df_res = self.performExpandedQueries(
                bayes_net, query_vars, list_of_new_evidence_vars)
        else:
            df_res = BayesNetHelper.query(bayes_net, query_vars, evidence_vars)
        return self.combine(df_res)
    
    def map_query(self, bayes_net, query_vars, evidence_vars):
        # if evidence vars is mismatched and was assigned specific value,
        # need to expand the query
        need_to_expand = []
        if evidence_vars is not None:
            new_evidence_vars, need_to_expand = self.mapEvidenceVars(
                evidence_vars)
        if need_to_expand:
            list_of_new_evidence_vars = self.expandQueries(
                new_evidence_vars, need_to_expand)
            df_res = self.performExpandedQueries(
                bayes_net, query_vars, list_of_new_evidence_vars)
        else:
            df_res = BayesNetHelper.map_query(bayes_net, query_vars, evidence_vars)
        return df_res

    def mapEvidenceVars(self, evidence_vars):
        new_evidence_vars = {}
        need_to_expand = []
        for evidence, value in evidence_vars.items():
            if evidence in self.mapping.keys():
                new_evidence_vars[evidence] = self.mapping[evidence][value]
                need_to_expand.append(evidence)
            else:
                new_evidence_vars[evidence] = value
        return new_evidence_vars, need_to_expand

    def combine(self, df):
        vars = list(df.columns)[:-1]
        for var in vars:
            if var in self.mapping.keys():
                inverted_map = {}
                for k, v in self.mapping[var].items():
                    for value in v:
                        inverted_map[value] = k
                df[var] = df[var].map(inverted_map)
        return df.groupby(vars).sum().reset_index()

    def expandQueries(self, new_evidence_vars, need_to_expand):
        # input: {ev1:[1,2],ev2:[1,2,3],ev3:1.....},
        # need_to_expand contains all evidence var names
        # that has multiple entries
        expanded_evidenve_vars = []
        to_expand = []
        constant_evidence_vars = new_evidence_vars.copy()
        for ev in need_to_expand:
            to_expand.append({ev: constant_evidence_vars.pop(ev)})
        while to_expand:
            ev_dict = to_expand.pop()
            # should only contain a single entry
            ev, values = next(iter(ev_dict.items()))
            if not expanded_evidenve_vars:
                for value in values:
                    expanded_evidenve_vars.append({ev: value})
            else:
                for value in values:
                    for other_ev_dict in expanded_evidenve_vars:
                        other_ev_dict.update({ev: value})
        for evidence_dict in expanded_evidenve_vars:
            evidence_dict.update(constant_evidence_vars)
        return expanded_evidenve_vars

    def performExpandedQueries(self, bayes_net,
                               query_vars, list_of_new_evidence_vars):
        df_res = None
        # evidence1  |  evidence 2  |....  | P()
        #----------- | ------------ |....  | ---
        df_evidence_probability = BayesNetHelper.query(
            bayes_net, list_of_new_evidence_vars[0].keys(),
            evidence_vars=None)
        for evidence_vars in list_of_new_evidence_vars:
            if df_res is None:
                # P(query|evidence1,evidence2...)* P(evidence1,evidence2...)
                df_res = BayesNetHelper.query(
                    bayes_net, query_vars, evidence_vars)
                y = df_res.iloc[:, -1].values.astype(float)\
                    * self.get_probability_of_evidences(
                        df_evidence_probability, evidence_vars)
                df_res.iloc[:, -1] = y
            else:
                df_new = BayesNetHelper.query(
                    bayes_net, query_vars, evidence_vars)
                y = df_new.iloc[:, -1].values.astype(float)\
                    * self.get_probability_of_evidences(
                        df_evidence_probability, evidence_vars)
                df_new.iloc[:, -1] = y
                df_res = pd.concat([df_res, df_new], ignore_index=True)
        #normalize third column of df_res
        y = df_res.iloc[:, -1].values.astype(float)
        df_res.iloc[:, -1] = y/np.sum(y)
        return df_res

    def get_probability_of_evidences(self,
                                     df_evidence_probability, evidence_vars):
        i = None
        #'Difficulty == \'Easy\' & Intelligence == \'Smart\''
        #evidence_vars = {'Age': '[40,50)', 'Gender': 'female'}
        for evidence, value in evidence_vars.items():
            if i is None:
                i = '{} == \'{}\''.format(evidence, value)
            else:
                i += '& {} == \'{}\''.format(evidence, value)
        query_index = df_evidence_probability.query(i).index
        p = df_evidence_probability.iloc[query_index, -1].values[0]
        return float(p)
