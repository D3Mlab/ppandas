from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import pandas as pd
import copy
import itertools

class BayesNetHelper():
    @staticmethod
    def rebuild_categorical_child_cpds(old_bayes, new_bayes, mapping, node, canonical_state_names=None):
        children = old_bayes.get_children(node)
        for child in children:
            old_cpd = old_bayes.get_cpds(child)
            if old_cpd is None:
                continue
            state_names = copy.deepcopy(old_cpd.state_names)
            if canonical_state_names is not None:
                state_names[node] = list(canonical_state_names)
            elif mapping and node in mapping:
                state_names[node] = [str(s) for s in state_names[node]]
            evidence_vars = old_cpd.get_evidence()
            parent_card = [len(state_names[p]) for p in evidence_vars] if evidence_vars else []
            parent_states_new = [state_names[p] for p in evidence_vars] if evidence_vars else [[]]
            parent_combinations_new = list(itertools.product(*parent_states_new)) if evidence_vars else [()]
            parent_states_old = [old_cpd.state_names[p] for p in evidence_vars] if evidence_vars else [[]]
            parent_combinations_old = list(itertools.product(*parent_states_old)) if evidence_vars else [()]
            child_card = len(state_names[old_cpd.variable])
            new_values_array = np.zeros((child_card, len(parent_combinations_new)))
            for j, combo_new in enumerate(parent_combinations_new):
                matching_old_indices = []
                for old_idx, combo_old in enumerate(parent_combinations_old):
                    if old_idx >= old_cpd.values.shape[1]:
                        continue
                    match = True
                    for i, var in enumerate(evidence_vars):
                        if var == node and mapping and node in mapping:
                            if combo_old[i] in mapping and combo_new[i] in mapping[combo_old[i]]:
                                continue
                            else:
                                match = False
                                break
                        else:
                            if combo_new[i] != combo_old[i]:
                                match = False
                                break
                    if match:
                        matching_old_indices.append(old_idx)
                for new_idx, new_child_state in enumerate(state_names[old_cpd.variable]):
                    old_child_indices = []
                    if old_cpd.variable in mapping:
                        for old_child_idx, old_child_state in enumerate(old_cpd.state_names[old_cpd.variable]):
                            if new_child_state in mapping[old_child_state]:
                                old_child_indices.append(old_child_idx)
                    else:
                        try:
                            old_child_indices = [old_cpd.state_names[old_cpd.variable].index(new_child_state)]
                        except ValueError:
                            old_child_indices = []
                    val = 0.0
                    count = 0
                    for old_idx in matching_old_indices:
                        for old_child_idx in old_child_indices:
                            if old_child_idx >= old_cpd.values.shape[0]:
                                continue
                            v = old_cpd.values[old_child_idx, old_idx]
                            if isinstance(v, (np.ndarray, list)):
                                v = np.sum(v)
                            val += v
                            count += 1
                    if count > 0:
                        new_values_array[new_idx, j] = val / count
                    else:
                        new_values_array[new_idx, j] = 0.0
            col_sums = new_values_array.sum(axis=0, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                new_values_array = np.divide(new_values_array, col_sums, where=col_sums > 0)
            new_values_array = np.nan_to_num(new_values_array)
            # Ensure evidence and evidence_card are set correctly for conditional CPDs
            evidence = evidence_vars if evidence_vars and len(evidence_vars) > 0 else None
            evidence_card = parent_card if evidence_vars and len(evidence_vars) > 0 else None
            new_cpd = TabularCPD(
                variable=old_cpd.variable,
                variable_card=child_card,
                values=new_values_array,
                evidence=evidence,
                evidence_card=evidence_card,
                state_names=state_names
            )
            if child in [cpd.variable for cpd in new_bayes.get_cpds()]:
                new_bayes.remove_cpds(child)
            new_bayes.add_cpds(new_cpd)
        return new_bayes

    @staticmethod
    def join(reference_bayes, second_bayes, new_dependent_vars,
             new_independent_vars, ref_num_of_records, second_num_of_records):
        final_bayes = DiscreteBayesianNetwork()
        # all independent variables should stay the same
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

            # If any parent is a mismatch variable, always use reference CPD
            mismatch_vars = set()
            # Try to infer mismatch variables from state name differences
            if node in reference_bayes:
                ref_cpd = reference_bayes.get_cpds(node=node)
                if hasattr(ref_cpd, 'state_names') and ref_cpd.state_names:
                    for parent in ref_parents:
                        if parent in ref_cpd.state_names:
                            ref_states = ref_cpd.state_names[parent]
                            if node in second_bayes:
                                sec_cpd = second_bayes.get_cpds(node=node)
                                if sec_cpd and hasattr(sec_cpd, 'state_names') and sec_cpd.state_names:
                                    sec_states = sec_cpd.state_names.get(parent, ref_states)
                                    if ref_states != sec_states:
                                        mismatch_vars.add(parent)
            # If any parent is a mismatch variable, always use reference CPD
            if len(ref_parents & mismatch_vars) > 0:
                final_bayes.add_edges_from([(parent, node) for parent in ref_parents])
                # If the state space has changed, use the rebuilt CPD from the handler if available
                ref_cpd = reference_bayes.get_cpds(node=node)
                # Check if the state_names in ref_cpd match the joined state space
                joined_state_names = ref_cpd.state_names.get(node, []) if hasattr(ref_cpd, 'state_names') else []
                # If the cardinality does not match the expected, raise or skip
                if hasattr(final_bayes, 'expected_state_names') and node in final_bayes.expected_state_names:
                    expected = set(final_bayes.expected_state_names[node])
                    actual = set(joined_state_names)
                    if expected != actual:
                        # Try to get the rebuilt CPD from the reference_bayes (if handler replaced it)
                        rebuilt_cpd = reference_bayes.get_cpds(node=node)
                        final_bayes.add_cpds(rebuilt_cpd)
                    else:
                        final_bayes.add_cpds(ref_cpd)
                else:
                    final_bayes.add_cpds(ref_cpd)
            elif(len(ref_parents) == 0):
                final_bayes.add_edges_from([(parent, node) for parent in second_parents])
                final_bayes.add_cpds(second_bayes.get_cpds(node=node))
            else:
                final_bayes.add_edges_from([(parent, node) for parent in ref_parents])
                if len(second_parents - ref_parents) > 0:
                    raise ValueError('This join can not be performed since the\
                         second distribution contains new independent variable\
                         (s) for node {}. Please consider dropping these new \
                         dependencies or switching reference distribution. '
                                     .format(str(node)))
                elif ref_parents == second_parents:
                    ref_cpd = reference_bayes.get_cpds(node=node)
                    sec_cpd = second_bayes.get_cpds(node=node)
                    # Only combine if shapes match
                    if (
                        ref_cpd.values.shape == sec_cpd.values.shape and
                        ref_cpd.state_names == sec_cpd.state_names
                    ):
                        new_cpd = BayesNetHelper.calculate_weighted_cpds(
                            ref_cpd, sec_cpd, ref_num_of_records, second_num_of_records)
                        final_bayes.add_cpds(new_cpd)
                    else:
                        # Fallback: use reference CPD
                        final_bayes.add_cpds(ref_cpd)
                else:
                    final_bayes.add_cpds(reference_bayes.get_cpds(node=node))
        # Ensure all CPDs are valid: for all-zero columns, use reference CPD if possible, else uniform
        for cpd in final_bayes.get_cpds():
            values = np.array(cpd.values)
            ref_cpd = None
            if cpd.variable in reference_bayes.nodes:
                ref_cpd = reference_bayes.get_cpds(node=cpd.variable)
            if values.ndim == 1:
                s = values.sum()
                if np.allclose(s, 0):
                    if ref_cpd is not None and ref_cpd.values.shape == values.shape and not np.allclose(ref_cpd.values, 0):
                        values[:] = ref_cpd.values[:]
                    else:
                        values[:] = 1.0 / values.shape[0]
            else:
                col_sums = values.sum(axis=0)
                for j, s in enumerate(col_sums):
                    if np.allclose(s, 0):
                        use_ref = False
                        if ref_cpd is not None and ref_cpd.values.shape == values.shape:
                            ref_col = ref_cpd.values[:, j]
                            if not np.allclose(ref_col, 0):
                                values[:, j] = ref_col
                                use_ref = True
                        if not use_ref:
                            values[:, j] = 1.0 / values.shape[0]
            # Final fallback: set any remaining all-zero or NaN columns to uniform
            if values.ndim == 1:
                if np.allclose(values, 0) or np.any(np.isnan(values)):
                    values[:] = 1.0 / values.shape[0]
            else:
                for j in range(values.shape[1]):
                    if np.allclose(values[:, j], 0) or np.any(np.isnan(values[:, j])):
                        values[:, j] = 1.0 / values.shape[0]
            cpd.values = values
            if hasattr(cpd, 'normalize'):
                cpd.normalize()
        return final_bayes

    @staticmethod
    def calculate_weighted_cpds(cpd1, cpd2, n1, n2):
        new_cpd = (n1 / (n1 + n2)) * cpd1 + (n2 / (n1 + n2)) * cpd2
        new_cpd.state_names = cpd1.state_names
        new_cpd.state_names.update(cpd2.state_names)
        return new_cpd

    @staticmethod
    def single_bayes_net(df, independent_vars, dependent_vars):
        df = df.copy().astype(str)
        model = DiscreteBayesianNetwork()
        model.add_nodes_from(independent_vars | dependent_vars)
        for independent_var in independent_vars:
            for dependent_var in dependent_vars:
                model.add_edge(independent_var, dependent_var)

        # Initialize BayesianEstimator with model and data
        estimator = BayesianEstimator(model, df)
        # Get the CPDs using Bayesian estimation
        bayesian_cpds = estimator.get_parameters()
        # Add the CPDs to the model
        model.add_cpds(*bayesian_cpds)

        # Manually add CPDs for any independent vars that are missing
        for var in independent_vars:
            if model.get_cpds(var) is None:
                state_names = sorted(df[var].unique())
                counts = df[var].value_counts().reindex(state_names, fill_value=0)
                probs = (counts / counts.sum()).values
                cpd = TabularCPD(
                    variable=var,
                    variable_card=len(state_names),
                    values=np.array(probs).reshape(-1, 1),
                    evidence=None,
                    evidence_card=None,
                    state_names={var: state_names},
                )
                model.add_cpds(cpd)

        return model

    @staticmethod
    def bayes_net_from_populational_data(data, independent_vars,
                                         dependent_vars):
        model = DiscreteBayesianNetwork()
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
        # this is a similar function to pgmpy DiscreteBayesianNetwork.fit()
        # https://github.com/pgmpy/pgmpy
        node_cardinality = len(state_names[node])
        state_name = {node: state_names[node]}
        parents = sorted(model.get_parents(node))
        parents_cardinalities = [len(state_names[parent])
                                 for parent in parents]
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
        evidence = parents if parents and len(parents) > 0 else None
        evidence_card = parents_cardinalities if parents and len(parents) > 0 else None
        cpd = TabularCPD(
            node,
            node_cardinality,
            state_values,
            evidence=evidence,
            evidence_card=evidence_card,
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
    def map_query(bayes_net, query_vars, evidence_vars):
        bayes_net_infer = VariableElimination(bayes_net)
        if evidence_vars:
            q = bayes_net_infer.map_query(
                variables=query_vars, evidence=evidence_vars,
                show_progress=False)
        else:
            q = bayes_net_infer.map_query(
                variables=query_vars, evidence=None,
                show_progress=False)
        return q

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
    def mapConditionalCpd(old_bayes, new_bayes, mapping, node):
        children = old_bayes.get_children(node)
        for child in children:
            old_cpd = old_bayes.get_cpds(child)
            if old_cpd is None:
                continue

            state_names = copy.deepcopy(old_cpd.state_names)
            if mapping and node in mapping:
                state_names[node] = [str(s).strip() for s in state_names[node]]

            evidence_vars = old_cpd.get_evidence()
            parent_card = [len(state_names[p]) for p in evidence_vars] if evidence_vars else []

            old_values = np.array(old_cpd.values)
            if old_values.ndim == 1:
                old_values = old_values.reshape((-1, 1))
            elif old_values.shape[0] != len(old_cpd.state_names[old_cpd.variable]):
                old_values = old_values.T

            # New child states
            child_card = len(state_names[old_cpd.variable])
            # Build mapping from new child state to list of old child states
            if old_cpd.variable in mapping:
                child_mapping = {new: [] for new in state_names[old_cpd.variable]}
                for old_s, new_s_list in mapping[old_cpd.variable].items():
                    for new_s in new_s_list:
                        if new_s in child_mapping:
                            child_mapping[new_s].append(old_s)
            else:
                child_mapping = {s: [s] for s in state_names[old_cpd.variable]}

            # Generate new parent combinations
            parent_states_new = [state_names[p] for p in evidence_vars] if evidence_vars else [[]]
            parent_combinations_new = list(itertools.product(*parent_states_new)) if evidence_vars else [()]

            # Generate old parent combinations
            parent_states_old = [old_cpd.state_names[p] for p in evidence_vars] if evidence_vars else [[]]
            parent_combinations_old = list(itertools.product(*parent_states_old)) if evidence_vars else [()]

            # Map new parent combination -> old columns
            col_map = []
            for combo_new in parent_combinations_new:
                old_indices = []
                for old_idx, combo_old in enumerate(parent_combinations_old):
                    match = True
                    for i, var in enumerate(evidence_vars):
                        val_new = combo_new[i]
                        val_old = combo_old[i]
                        if var == node and mapping and node in mapping:
                            # If mapping is not exhaustive, treat as no match
                            if val_old in mapping and val_new in mapping[val_old]:
                                continue
                            else:
                                match = False
                                break
                        else:
                            if val_new != val_old:
                                match = False
                                break
                    if match:
                        old_indices.append(old_idx)
                col_map.append(old_indices)

            # Build new CPD values
            new_values_array = np.zeros((child_card, len(parent_combinations_new)))

            for j, old_cols in enumerate(col_map):
                col_data_new = np.zeros(child_card)
                for old_idx in old_cols:
                    old_col = old_values[:, old_idx]
                    for new_idx, new_state in enumerate(state_names[old_cpd.variable]):
                        old_states = child_mapping.get(new_state, [])
                        if not old_states:
                            continue  # No mapping, leave as zero
                        old_indices = [old_cpd.state_names[old_cpd.variable].index(s)
                                    for s in old_states if s in old_cpd.state_names[old_cpd.variable]]
                        if old_indices:
                            col_data_new[new_idx] += old_col[old_indices].sum()
                new_values_array[:, j] = col_data_new

            # Normalize columns to sum to 1, but only if sum > 0
            col_sums = new_values_array.sum(axis=0, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                new_values_array = np.divide(new_values_array, col_sums, where=col_sums > 0)
            new_values_array = np.nan_to_num(new_values_array)

            # Create new CPD
            new_cpd = TabularCPD(
                variable=old_cpd.variable,
                variable_card=child_card,
                values=new_values_array,
                evidence=evidence_vars if evidence_vars else None,
                evidence_card=parent_card if parent_card else None,
                state_names=state_names
            )

            # Replace in new_bayes
            if child in [cpd.variable for cpd in new_bayes.get_cpds()]:
                new_bayes.remove_cpds(child)
            new_bayes.add_cpds(new_cpd)

        return new_bayes
    
