import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ppandas.p_frame import PDataFrame
import pytest
import traceback
"""
Tests (re)written by me (Elise) for ppandas scenarios relevant to piccard :)
"""

def test_categorical():
    

    # Synthetic Data 1: Age as numerical bins, Gender, Gun Control (non-uniform)
    np.random.seed(42)
    ages1 = np.random.choice(['[20,40)', '[40,60)', '[60,80)'], size=300, p=[0.4, 0.4, 0.2])
    genders1 = np.random.choice(['male', 'female'], size=300, p=[0.5, 0.5])
    gun_control1 = [
        np.random.choice(['For', 'Against'], p=[0.8, 0.2]) if a == '[20,40)' and g == 'female'
        else np.random.choice(['For', 'Against'], p=[0.3, 0.7]) if a == '[40,60)' and g == 'male'
        else np.random.choice(['For', 'Against'], p=[0.6, 0.4])
        for a, g in zip(ages1, genders1)
    ]
    df1 = pd.DataFrame({'Age': ages1, 'Gender': genders1, 'Gun Control': gun_control1})

    # Synthetic Data 2: Age as different bins (mismatch), Gender, Gun Control (non-uniform)
    ages2 = np.random.choice(['[20,50)', '[50,80)'], size=300, p=[0.7, 0.3])
    genders2 = np.random.choice(['male', 'female'], size=300, p=[0.5, 0.5])
    gun_control2 = [
        np.random.choice(['For', 'Against'], p=[0.7, 0.3]) if a == '[20,50)' and g == 'female'
        else np.random.choice(['For', 'Against'], p=[0.2, 0.8]) if a == '[50,80)' and g == 'male'
        else np.random.choice(['For', 'Against'], p=[0.5, 0.5])
        for a, g in zip(ages2, genders2)
    ]
    df2 = pd.DataFrame({'Age': ages2, 'Gender': genders2, 'Gun Control': gun_control2})

    pd1 = PDataFrame(["Gender","Age"],df1)
    pd2 = PDataFrame(["Gender","Age"],df2)

    pd_join = pd1.pjoin(pd2,mismatches={"Age":'categorical'})

    # Print full CPD object, .evidence, and .values shape for Gun Control in all networks
    def print_cpd_details(cpd, label):
        print(f"{label} CPD object: {cpd}")
        if cpd is not None:
            print(f"{label} CPD evidence: {cpd.get_evidence()}")
            print(f"{label} CPD values shape: {getattr(cpd, 'values', None).shape if hasattr(cpd, 'values') else None}")
            print(f"{label} CPD values: {getattr(cpd, 'values', None)}")
        else:
            print(f"{label} CPD is None.")

    print_cpd_details(pd1.bayes_net.get_cpds('Gun Control'), "Original network 1")
    print_cpd_details(pd2.bayes_net.get_cpds('Gun Control'), "Original network 2")
    print_cpd_details(pd_join.bayes_net.get_cpds('Gun Control'), "Joined network")

    # Print parents and network structure for both original networks
    for i, pdx in enumerate([pd1, pd2], 1):
        net = pdx.bayes_net
        print(f"Original network {i} edges:", list(net.edges()))
        if 'Gun Control' in net.nodes():
            print(f"Original network {i} parents of Gun Control:", net.get_parents('Gun Control'))
        else:
            print(f"Original network {i} has no Gun Control node.")
    # Print all parent state combinations in original and joined CPDs for Gun Control
    def print_parent_combos(cpd, label):
        if cpd is None:
            print(f"{label}: No CPD found.")
            return
        values = np.array(cpd.values)
        parent_vars = cpd.get_evidence()
        if parent_vars and values.ndim == 2:
            parent_states = [cpd.state_names[p] for p in parent_vars]
            parent_combos = list(pd.MultiIndex.from_product(parent_states, names=parent_vars))
            present = []
            for j in range(values.shape[1]):
                if not np.allclose(values[:, j], 0):
                    present.append(parent_combos[j])
            print(f"{label} - Present parent state combinations:")
            for combo in present:
                print(dict(zip(parent_vars, combo)))
        else:
            print(f"{label}: No parents or not a 2D CPD.")

    print_parent_combos(pd1.bayes_net.get_cpds('Gun Control'), "Original CPD 1")
    print_parent_combos(pd2.bayes_net.get_cpds('Gun Control'), "Original CPD 2")
    print_parent_combos(pd_join.bayes_net.get_cpds('Gun Control'), "Joined CPD")


    # Diagnostic: print missing parent state combinations (all-zero columns) in Gun Control CPD
    cpd = pd_join.bayes_net.get_cpds('Gun Control')
    values = np.array(cpd.values)
    # If CPD has parents, columns correspond to parent state combinations
    if values.ndim == 2:
        parent_vars = cpd.get_evidence()
        parent_states = [cpd.state_names[p] for p in parent_vars]
        parent_combos = list(pd.MultiIndex.from_product(parent_states, names=parent_vars))
        missing = []
        for j in range(values.shape[1]):
            if np.allclose(values[:, j], 0):
                missing.append(parent_combos[j])
        if missing:
            print("Missing parent state combinations (all-zero columns) in Gun Control CPD:")
            for combo in missing:
                print(dict(zip(parent_vars, combo)))
        else:
            print("No missing parent state combinations in Gun Control CPD.")

    # Print a table with Age, Gun Control, and Probability for all Age groups
    age_states = pd_join.bayes_net.get_cpds('Age').state_names['Age']
    rows = []
    for age_state in age_states:
        try:
            queryResult = pd_join.query(['Gun Control'], {"Age": age_state})
            for _, row in queryResult.iterrows():
                rows.append({
                    'Age': age_state,
                    'Gun Control': row['Gun Control'],
                    'Probability': row['Probability(Gun Control)']
                })
        except Exception as e:
            rows.append({'Age': age_state, 'Gun Control': 'ERROR', 'Probability': str(e)})
    result_df = pd.DataFrame(rows)
    print(result_df)

def test_numerical():

    # Synthetic Data 1: Age as numerical bins, Gender, Gun Control (non-uniform)
    np.random.seed(42)
    ages1 = np.random.choice(['[20,40)', '[40,60)', '[60,80)'], size=300, p=[0.4, 0.4, 0.2])
    genders1 = np.random.choice(['male', 'female'], size=300, p=[0.5, 0.5])
    gun_control1 = [
        np.random.choice(['For', 'Against'], p=[0.8, 0.2]) if a == '[20,40)' and g == 'female'
        else np.random.choice(['For', 'Against'], p=[0.3, 0.7]) if a == '[40,60)' and g == 'male'
        else np.random.choice(['For', 'Against'], p=[0.6, 0.4])
        for a, g in zip(ages1, genders1)
    ]
    df1 = pd.DataFrame({'Age': ages1, 'Gender': genders1, 'Gun Control': gun_control1})

    # Synthetic Data 2: Age as different bins (mismatch), Gender, Gun Control (non-uniform)
    ages2 = np.random.choice(['[20,50)', '[50,80)'], size=300, p=[0.7, 0.3])
    genders2 = np.random.choice(['male', 'female'], size=300, p=[0.5, 0.5])
    gun_control2 = [
        np.random.choice(['For', 'Against'], p=[0.7, 0.3]) if a == '[20,50)' and g == 'female'
        else np.random.choice(['For', 'Against'], p=[0.2, 0.8]) if a == '[50,80)' and g == 'male'
        else np.random.choice(['For', 'Against'], p=[0.5, 0.5])
        for a, g in zip(ages2, genders2)
    ]
    df2 = pd.DataFrame({'Age': ages2, 'Gender': genders2, 'Gun Control': gun_control2})

    pd1 = PDataFrame(["Gender","Age"],df1)
    pd2 = PDataFrame(["Gender","Age"],df2)

    pd_join = pd1.pjoin(pd2,mismatches={"Age":'numerical'})

    # Diagnostics: print structure and CPD for Gun Control
    print('Nodes in joined Bayes net:', list(pd_join.bayes_net.nodes()))
    print('Edges in joined Bayes net:', pd_join.bayes_net.edges())
    print('All CPDs in joined Bayes net:')
    for cpd in pd_join.bayes_net.get_cpds():
        print(cpd)
    if 'Gun Control' in pd_join.bayes_net.nodes():
        print('Parents of Gun Control:', pd_join.bayes_net.get_parents('Gun Control'))
        try:
            print('CPD for Gun Control:')
            print(pd_join.bayes_net.get_cpds('Gun Control'))
        except Exception as e:
            print('Could not get CPD for Gun Control:', e)
    else:
        print('Gun Control is not a node in the joined Bayes net!')

    # Print conditional probabilities for all combinations of Age and Gender
    age_states = pd_join.bayes_net.get_cpds('Age').state_names['Age']
    gender_states = pd_join.bayes_net.get_cpds('Gender').state_names['Gender']
    print('Joined Age states:', age_states)
    print('Joined Gender states:', gender_states)
    cpd = pd_join.bayes_net.get_cpds('Gun Control')
    valid_parent_combos = set(zip(
        cpd.state_names['Age'] * len(cpd.state_names['Gender']),
        sum([[g]*len(cpd.state_names['Age']) for g in cpd.state_names['Gender']], [])
    ))
    for age_state in age_states:
        for gender_state in gender_states:
            if (age_state, gender_state) not in valid_parent_combos:
                print(f'Skipping invalid combination: Gender={gender_state}, Age={age_state}')
                continue
            try:
                queryResult = pd_join.query(['Gun Control'],{"Gender": gender_state, "Age": age_state})
                print(f'conditional query for Gender={gender_state}, Age={age_state}')
                print(queryResult)
            except KeyError as e:
                print(f'KeyError for Gender={gender_state}, Age={age_state}: {e}')

def test_pjoin_categorical_mismatch_cardinality():
    # Table 1: bins d1_Q1, d1_Q2, d1_Q3, d1_Q4
    df1 = pd.DataFrame({
        'bin': ['d1_Q1', 'd1_Q2', 'd1_Q3', 'd1_Q4', 'd1_Q1', 'd1_Q2'],
        'val': [1, 2, 3, 4, 5, 6]
    })
    # Table 2: bins d2_Q1, d2_Q2, d2_Q3
    df2 = pd.DataFrame({
        'bin': ['d2_Q1', 'd2_Q2', 'd2_Q3', 'd2_Q1', 'd2_Q2', 'd2_Q3'],
        'val': [10, 20, 30, 40, 50, 60]
    })

    pdf1 = PDataFrame(independent_vars=['bin'], data=df1)
    pdf2 = PDataFrame(independent_vars=['bin'], data=df2)

    joined = pdf1.pjoin(pdf2, mismatches={'bin': 'categorical'})
    all_bins = joined.bayes_net.get_cpds('bin').state_names['bin']
    expected_bins = ['d1_Q1', 'd1_Q2', 'd1_Q3', 'd1_Q4', 'd2_Q1', 'd2_Q2', 'd2_Q3']

    # The joined state space should be the union of all bins
    assert set(all_bins) == set(expected_bins)

    # You should be able to query for any bin without error
    for bin_name in expected_bins:
        try:
            result = joined.query(['val'], evidence_vars={'bin': bin_name})
            print(f"Query for bin {bin_name} succeeded: {result}")
        except Exception as e:
            raise AssertionError(f"Query failed for bin {bin_name}: {e}")


def test_pjoin_categorical_distribution_nonuniform_weighted():
    # df1: 100 rows, mostly 'A', some 'B'
    df1 = pd.DataFrame({
        'bin': ['A'] * 90 + ['B'] * 10,
        'val': list(range(1, 101))
    })
    # df2: 10 rows, mostly 'B', one 'A'
    df2 = pd.DataFrame({
        'bin': ['A'] * 1 + ['B'] * 9,
        'val': list(range(201, 211))
    })

    pdf1 = PDataFrame(independent_vars=['bin'], data=df1)
    pdf2 = PDataFrame(independent_vars=['bin'], data=df2)
    print('pdf1 bin CPD:', pdf1.bayes_net.get_cpds('bin').values.flatten())
    print('pdf2 bin CPD:', pdf2.bayes_net.get_cpds('bin').values.flatten())
    joined = pdf1.pjoin(pdf2, mismatches={'bin': 'categorical'})
    cpd = joined.bayes_net.get_cpds('bin')
    probs = cpd.values.flatten()
    print('joined bin CPD:', probs)
    print('joined bin CPD sum:', probs.sum())
    for i, p in enumerate(probs):
        print(f'joined bin CPD value {i}: {p}')
    # Check that not all probabilities are equal (not uniform)
    assert not all(abs(p - probs[0]) < 1e-8 for p in probs), \
        f"CPD for 'bin' is uniform: {probs}"