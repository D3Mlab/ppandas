from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import pandas as pd


class BayesNetHelper():

    @staticmethod
    def join(reference_bayes, second_bayes, new_dependent_vars,
             new_independent_vars, ref_num_of_records, second_num_of_records):
        final_bayes = BayesianModel()
        #all independent variables should stay the same
        final_bayes.add_nodes_from(new_independent_vars)
        final_bayes.add_cpds(
            *[reference_bayes.get_cpds(node=node) if node in
                reference_bayes.nodes else second_bayes.get_cpds(node=node)
                for node in new_independent_vars])
        for node in new_dependent_vars:
            final_bayes.add_node(node)
            ref_parents = set()
            second_parents = set()
            if node in reference_bayes:
                ref_parents = set(reference_bayes.get_parents(node))
            if node in second_bayes:
                second_parents = set(second_bayes.get_parents(node))

            if(len(ref_parents) == 0):
                final_bayes.add_edges_from([(parent, node) for parent in
                                            second_parents])
                final_bayes.add_cpds(second_bayes.get_cpds(node=node))
            else:
                final_bayes.add_edges_from([(parent, node) for parent in
                                            ref_parents])
                if len(second_parents - ref_parents) > 0:
                    raise ValueError('This join can not be performed since the\
                         second distribution contains new independent variable\
                         (s) for node {}. Please consider dropping these new \
                         dependencies or switching reference distribution. '
                                     .format(str(node)))
                elif ref_parents == second_parents:
                    new_cpd = BayesNetHelper.calculate_weighted_cpds(
                        reference_bayes.get_cpds(node=node),
                        second_bayes.get_cpds(node=node),
                        ref_num_of_records, second_num_of_records)
                    final_bayes.add_cpds(new_cpd)
                else:
                    final_bayes.add_cpds(reference_bayes.get_cpds(node=node))
        return final_bayes

    @staticmethod
    def calculate_weighted_cpds(cpd1, cpd2, n1, n2):
        new_cpd = (n1 / (n1 + n2)) * cpd1 + (n2 / (n1 + n2)) * cpd2
        new_cpd.state_names = cpd1.state_names
        new_cpd.state_names.update(cpd2.state_names)
        return new_cpd

    @staticmethod
    def single_bayes_net(df, independent_vars, dependent_vars):
        model = BayesianModel()
        model.add_nodes_from(independent_vars)
        for independent_var in independent_vars:
            for dependent_var in dependent_vars:
                model.add_edge(independent_var, dependent_var)
        model.fit(df)
        return model

    @staticmethod
    def bayes_net_from_populational_data(data, independent_vars,
                                         dependent_vars):
        model = BayesianModel()
        model.add_nodes_from(independent_vars)
        for independent_var in independent_vars:
            for dependent_var in dependent_vars:
                model.add_edge(independent_var, dependent_var)
        cpd_list = []
        state_names = BayesNetHelper.get_state_names_from_df(
            data, independent_vars | dependent_vars)
        for node in independent_vars | dependent_vars:
            cpd = BayesNetHelper.compute_cpd(model, node, data, state_names)
            cpd_list.append(cpd)
        model.add_cpds(*cpd_list)
        return model

    @staticmethod
    def get_state_names_from_df(data, vars):
        state_names = {}
        for var in vars:
            state_names[var] = sorted(list(data[var].unique()))
        return state_names

    @staticmethod
    def compute_cpd(model, node, data, state_names):
        # this is a similar function to pgmpy BayesianModel.fit()
        # https://github.com/pgmpy/pgmpy
        node_cardinality = len(state_names[node])
        state_name = {node: state_names[node]}
        parents = sorted(model.get_parents(node))
        parents_cardinalities = [len(state_names[parent])
                                 for parent in parents]
        #get values
        #print('data')
        #print(data)
        if parents:
            state_name.update({parent: state_names[parent]
                              for parent in parents})
            #get values
            parents_states = [state_names[parent] for parent in parents]
            state_value_data = data.groupby(
                [node] + parents).sum().unstack(parents)
            #drop 'counts'
            state_value_data = state_value_data.droplevel(0, axis=1)
            row_index = state_names[node]
            if(len(parents) > 1):
                column_index = pd.MultiIndex.from_product(
                    parents_states, names=parents)
                state_values = state_value_data.reindex(
                    index=row_index, columns=column_index)
            state_values = state_value_data
        else:
            state_value_data = data.groupby([node]).sum()
            state_values = state_value_data.reindex(state_names[node])
        cpd = TabularCPD(
            node,
            node_cardinality,
            state_values,
            evidence=parents,
            evidence_card=parents_cardinalities,
            state_names=state_name,
        )
        cpd.normalize()
        return cpd

    @staticmethod
    def query(bayes_net, query_vars, evidence_vars):
        bayes_net_infer = VariableElimination(bayes_net)
        if evidence_vars:
            q = bayes_net_infer.query(
                variables=query_vars, evidence=evidence_vars,
                show_progress=False)
        else:
            q = bayes_net_infer.query(
                variables=query_vars, evidence=None,
                show_progress=False)
        return BayesNetHelper.convertFactorToDF(q)

    @staticmethod
    def convertFactorToDF(phi):
        a = phi.assignment(np.arange(np.prod(phi.cardinality)))
        data = []
        for line in a:
            row = []
            for (_, state_name) in line:
                if isinstance(state_name, tuple):
                    row.append(str(state_name))
                else:
                    row.append(state_name)
            data.append(row)
        data = np.hstack((np.array(data), np.array(phi.values.reshape(-1, 1))))
        header = phi.scope().copy()
        header.append("Probability({variables})".format(variables=","
                      .join(header)))
        df = pd.DataFrame(columns=header, data=data)
        df[header[-1]] = df[header[-1]].astype('float')
        return df

    @staticmethod
    def removeRelatedCpds(bayes_net, mismatchColumn):
        #remove all cpds related to the mismatch variable and then
        #re-assign them
        bayes_net_copy = bayes_net.copy()
        cpd_node = bayes_net_copy.get_cpds(node=mismatchColumn)
        bayes_net_copy.remove_cpds(cpd_node)
        children = bayes_net_copy.get_children(node=mismatchColumn)
        for c in children:
            bayes_net_copy.remove_cpds(bayes_net_copy.get_cpds(node=c))
        return bayes_net_copy

    @staticmethod
    def mapCondtionalCpd(bayes_net, bayes_net_copy, mapping, mismatchColumn):
        for c_node in bayes_net_copy.get_children(node=mismatchColumn):
            old_cpd = bayes_net.get_cpds(node=c_node)
            evidences = old_cpd.variables[1:]
            if len(evidences) > 1:
                if evidences[0] != mismatchColumn:
                    evidences.remove(mismatchColumn)
                    evidences = [mismatchColumn]+evidences
                    old_cpd.reorder_parents(evidences)
            c_node_card = old_cpd.variable_card
            new_cpd_array = np.empty(shape=[c_node_card, 0])

            old_cpd_array = np.array(
                old_cpd.values).flatten('C').reshape((c_node_card, -1))
            index = 0
            mismatch_col_card = np.sum(
                [len(mapping[i]) for i in mapping.keys()])
            for _ in range(np.prod(old_cpd.cardinality[2:])):
                for old_entry, new_entries in mapping.items():
                    num_of_sub_entries = len(new_entries)
                    # mismatch_col_card += num_of_sub_entries
                    v = old_cpd_array[:, index]
                    #if 1-d, reshape
                    if len(v.shape) == 1:
                        v = v.reshape(-1, 1)
                    for _ in range(0, num_of_sub_entries):
                        new_cpd_array = np.hstack((new_cpd_array, v))
                    index += 1
            evidence_cards = [mismatch_col_card] + list(
                old_cpd.cardinality[2:])
            new_state_names = old_cpd.state_names.copy()
            new_state_names.update(bayes_net_copy.get_cpds(
                node=mismatchColumn).state_names)
            cpd = TabularCPD(c_node, c_node_card, new_cpd_array,
                             evidence=evidences, evidence_card=evidence_cards,
                             state_names=new_state_names)
            bayes_net_copy.add_cpds(cpd)
        return bayes_net_copy
