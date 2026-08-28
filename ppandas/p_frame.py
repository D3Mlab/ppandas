import pandas as pd
import numpy as np
import ast
from shapely.geometry import Point
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from .helper.bayes_net_helper import BayesNetHelper
from .helper.query_helper import QueryHelper
from .mismatch_handler import\
    NumericalHandler, categoricalHandler, spatialHandler

# Well-Known Text (WKT) converters so that all the mapping works as expected

def convert_location_states_to_wkt(bayes_net, node='Location'):
    cpd = bayes_net.get_cpds(node)
    original_states = cpd.state_names[node] if cpd.state_names else []

    # Convert tuple states to WKT strings
    new_states = []
    for st in original_states:
        if isinstance(st, tuple):
            new_states.append(Point(st).wkt)
        else:
            # Try to parse string tuple like '(0.5, 0.5)'
            try:
                val = ast.literal_eval(st)
                if isinstance(val, tuple) and len(val) == 2:
                    new_states.append(Point(val).wkt)
                else:
                    new_states.append(st)
            except Exception:
                new_states.append(st)

    # Only rebuild CPD if there's a change
    if new_states != original_states:
        # Set new state names
        new_state_names = cpd.state_names.copy() if cpd.state_names else {}
        new_state_names[node] = new_states

        # Determine evidence and evidence_card
        evidence = cpd.get_evidence()
        if evidence is not None and cpd.state_names is not None:
            evidence_card = [len(cpd.state_names[ev]) for ev in evidence]
        else:
            evidence_card = None

        # Convert values to expected shape
        values = np.array(cpd.values)
        if evidence_card:
            expected_shape = (cpd.variable_card, int(np.prod(evidence_card)))
        else:
            expected_shape = (cpd.variable_card, 1)

        if values.shape != expected_shape:
            values = values.reshape(expected_shape)

        # Build new CPD with or without evidence_card depending on evidence
        if evidence is not None and cpd.state_names is not None:
            new_cpd = TabularCPD(
                variable=cpd.variable,
                variable_card=cpd.variable_card,
                values=values,
                evidence=evidence,
                evidence_card=evidence_card,
                state_names=new_state_names
            )
        else:
            new_cpd = TabularCPD(
                variable=cpd.variable,
                variable_card=cpd.variable_card,
                values=values,
                state_names=new_state_names
            )

        bayes_net.remove_cpds(cpd)
        bayes_net.add_cpds(new_cpd)


def convert_geo_dict_keys_and_values_to_wkt(mapping):
    new_mapping = {}
    for k, v_list in mapping.items():
        k_wkt = Point(k).wkt if isinstance(k, tuple) else k
        new_v_list = [
            Point(v).wkt if isinstance(v, tuple) else v
            for v in v_list
        ]
        new_mapping[k_wkt] = new_v_list
    return new_mapping


class PDataFrame():

    def __init__(self, independent_vars, data, **kwargs):
        """Constructor
        Arguments: 1 DataFrame, list of independent variable columns (columns
        not included are assumed to be dependent variables), specified edges
        of Bayes Net representation
        Find # samples (rows) in the given Dataframe
        Uses pgmpy to calculate conditional probability table (CPD) of each
        specified edge
        Creates Bayes Net
        """
        if isinstance(data, pd.DataFrame):
            self.num_of_records = data.shape[0]
            self.independent_vars = set(independent_vars)
            self.vars = set(list(data))
            # Check if any self.indep_vars not in self.vars
            vars_not_in_dataframe = self.independent_vars - self.vars
            if len(vars_not_in_dataframe) > 0:
                raise ValueError("Specified independent variables not found \
                in DataFrame: {} " .format(str(vars_not_in_dataframe)[1:-1]))
            self.dependent_vars = set([var for var in self.vars if var not in
                                      self.independent_vars])
            self.bayes_net = BayesNetHelper.\
                single_bayes_net(data, self.independent_vars,
                                 self.dependent_vars)
            # Debug print state names for 'cat' if present
            if 'cat' in self.independent_vars:
                cpd = self.bayes_net.get_cpds('cat')
                if cpd is not None and hasattr(cpd, 'state_names'):
                    print(f"[DEBUG] PDataFrame.__init__: 'cat' state_names = {cpd.state_names['cat'] if 'cat' in cpd.state_names else None}")
            self.atomic = True

        elif isinstance(data, DiscreteBayesianNetwork):
            self.num_of_records = kwargs['num_of_records']
            self.independent_vars = independent_vars
            self.dependent_vars = kwargs['dependent_vars']
            self.vars = self.dependent_vars | self.independent_vars
            self.bayes_net = data
            self.atomic = kwargs['atomic']
            if not self.atomic:
                self.reference_mapping = kwargs['reference_mapping']
                self.second_mapping = kwargs['second_mapping']

    @classmethod
    def from_populational_data(cls, independent_vars, data, num_of_records):
        """
        Create PDataFrame from populational data
        Gender |  Age  | Gun Control | Counts/percentage
        ---------------------------------------------
        Female | Young |     For     | 50
        ----------------------------------------------
        etc.
        """
        independent_vars = set(independent_vars)
        variables = set((list(data)[:-1]))
        vars_not_in_dataframe = independent_vars - variables
        if len(vars_not_in_dataframe) > 0:
            raise ValueError("Specified independent variables not found\
             in DataFrame: {} " .format(str(vars_not_in_dataframe)[1:-1]))
        dependent_vars = set([var for var in variables if var not in
                             independent_vars])
        bayes_net = BayesNetHelper.bayes_net_from_populational_data(
            data, independent_vars, dependent_vars)
        return PDataFrame(independent_vars, bayes_net,
                          num_of_records=num_of_records,
                          dependent_vars=dependent_vars,
                          atomic=True)

    def pjoin(self, df2, mismatches=None):
        overlap = self.independent_vars.intersection(df2.dependent_vars)
        if len(overlap) > 0:
            raise ValueError('This join can not be performed since independent\
                             variable(s): {} in reference distribution are \
                            dependent in the second distribution. Please \
                            consider dropping these new dependencies or \
                            switching reference distribution.' .format(str(
                overlap)[1:-1]))

        new_num_of_records = self.num_of_records + df2.num_of_records
        new_dependent_vars = self.dependent_vars | df2.dependent_vars - \
            self.independent_vars
        new_independent_vars = self.independent_vars | df2.independent_vars\
            - new_dependent_vars
        reference_bayes = self.bayes_net.copy()
        second_bayes = df2.bayes_net.copy()

        reference_mapping = {}
        second_mapping = {}
        if mismatches:
            for mNode, mType in mismatches.items():
                if mNode not in self.independent_vars:
                    raise ValueError("Only mismatches across independent variables can be handled.")
                elif mType == 'numerical':
                    handler = NumericalHandler(mNode)
                elif mType == 'categorical':
                    handler = categoricalHandler(mNode)
                elif mType == 'spatial':
                    handler = spatialHandler(mNode)
                else:
                    raise ValueError(f"mismatch type {mType} is currently not supported")
                convert_location_states_to_wkt(reference_bayes, node=mNode)
                convert_location_states_to_wkt(second_bayes, node=mNode)
                ref_mapping, sec_mapping = handler.computeMapping(reference_bayes, second_bayes)
                ref_mapping = convert_geo_dict_keys_and_values_to_wkt(ref_mapping)
                sec_mapping = convert_geo_dict_keys_and_values_to_wkt(sec_mapping)
                reference_mapping[mNode] = ref_mapping
                second_mapping[mNode] = sec_mapping
                reference_bayes = handler.replaceMismatchNode(reference_bayes, ref_mapping)
                second_bayes = handler.replaceMismatchNode(second_bayes, sec_mapping)

        final_bayes = BayesNetHelper.join(
            reference_bayes, second_bayes, new_dependent_vars,
            new_independent_vars, self.num_of_records, df2.num_of_records)

        pdataframe_joined = PDataFrame(
            new_independent_vars,
            final_bayes, num_of_records=new_num_of_records,
            dependent_vars=new_dependent_vars,
            reference_mapping=reference_mapping,
            second_mapping=second_mapping, atomic=False)
        return pdataframe_joined

    def query(self, query_vars, evidence_vars=None, entries='reference'):
        if not self.atomic:
            if entries == 'reference':
                q_helper = QueryHelper(self.reference_mapping)
            else:
                q_helper = QueryHelper(self.second_mapping)
            res = q_helper.query(self.bayes_net, query_vars, evidence_vars)
        else:
            res = BayesNetHelper.query(self.bayes_net, query_vars,
                                       evidence_vars)
        return res

    def map_query(self, query_vars, evidence_vars=None, entries='reference'):
        if not self.atomic:
            if entries == 'reference':
                q_helper = QueryHelper(self.reference_mapping)
            else:
                q_helper = QueryHelper(self.second_mapping)
            res = q_helper.map_query(self.bayes_net, query_vars, evidence_vars)
        else:
            res = BayesNetHelper.map_query(self.bayes_net, query_vars,
                                       evidence_vars)
        return res

    def predict(self, test_df, query_var, evidence_vars, query_var_state=None):
        test_df = test_df.astype(str)
        features = list(test_df.columns.values)
        valid_evidence_vars = []
        probability_column_name = "Probability("+str(query_var)+")"
        for evidence_var in evidence_vars:
            if evidence_var in features:
                valid_evidence_vars.append(evidence_var)
        for index, row in test_df.iterrows():
            test_row_evidence_dict = {}
            for evidence_var in valid_evidence_vars:
                test_row_evidence_dict[evidence_var] = row[evidence_var]
            test_row_query_df = self.query(query_var,test_row_evidence_dict)
            test_df.loc[index,query_var[0]] = self.parseQuery(query_var[0],
                                                test_row_query_df,
                                                probability_column_name,
                                                query_var_state)
            if index > 1 and index % 1000 == 0 :
                break
        return test_df
    #Returns a state value or a real-valued probability for a given state
    def parseQuery(self, query_var, query_result_df,probability_column_name,query_var_state):
        #Find query_var state with highest Probability
        probability_column_name = "Probability("+str(query_var)+")"
        if not query_var_state:
            query_var_state_row_index = query_result_df[probability_column_name]\
                                    .idxmax()
            #Select row by index, column by name
            query_var_state = query_result_df.iloc[query_var_state_row_index,:]\
                            [query_var]
            return query_var_state
        else:
            return query_result_df.loc[query_result_df[query_var]==query_var_state, probability_column_name].values[0]
        
    
    def visualise(self, output_file=None, show_tables=False):
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np

        plt.figure(figsize=(8, 6))
        plt.clf()

        try:
            pos = nx.circular_layout(self.bayes_net)
            for node, coord in pos.items():
                if any(np.isnan(coord)) or any(np.isinf(coord)):
                    print(f"Invalid position for node {node}: {coord}")

            nx.draw(self.bayes_net, pos=pos, with_labels=True,
                    node_color='skyblue', node_size=2000, arrows=True)

        except Exception as e:
            print(f"Exception during network drawing: {e}")

        if output_file:
            plt.savefig(output_file + '.png')
        plt.show()

        if show_tables:
            print("=== CPDs ===")
            for var in self.bayes_net.nodes():
                cpd = self.bayes_net.get_cpds(var)
                if cpd is not None:
                    print(cpd)
                else:
                    print(f"No CPD found for: {var}")
