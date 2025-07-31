import sys
try:
    import datgan
    print(datgan.__version__)

    from datgan import DATGAN, advise
except Exception as e:
    print('this is the preferered structure')
    from datgan import  datgan
    from  datgan.datgan import DATGAN, advise
import numpy as np
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import tensorflow as tf


## HELPER CODE ####
def build_evidence_conditions(df):
    """
    Build evidence conditions dynamically from the original data.
    """
    evidence = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # For continuous variables, define a range based on quantiles
            lower_bound = df[col].quantile(0.1)  # 10th percentile
            upper_bound = df[col].quantile(0.9)  # 90th percentile
            evidence[col] = lambda x, lb=lower_bound, ub=upper_bound: lb <= x <= ub
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            # For categorical variables, use the most frequent value
            mode = df[col].mode()[0]
            evidence[col] = lambda x, m=mode: x == m
    return evidence

###


df = pd.read_csv('LINKED_DATA_dropped.csv', index_col=False)


GET_REQUIREMENTS = False


if GET_REQUIREMENTS:
    import subprocess
    # Save requirements to a file
    with open('requirements.txt', 'w') as f:
        subprocess.run(['pip', 'freeze'], stdout=f, text=True)


print(df.head())



remove =  [  "ABSHID_x", "ABSFID", "ABSHID_y", "ABSPID",  "ANC1P", "ANC2P", "ANCRP", "AREAENUM", "AREAENUM_x", "REGUCP", "REGU1P", "REGU5P",
    "AREAENUM_y", "ASSNP", "BPFP", "BPLP", "BPMP", "BPPP", "CHCAREP", "CITP", "CLTHP", "CPAD",
    "CPAF", "CPLTHRD", "CTPP", "DOMP", "DWTD", "EMPP", "ENGLP", "FINASF", "FNOF", "FPIP",
    "FRLF", "FTCP", "GNGP", "HARTP", "HASTP", "HCANP", "HDEMP", "HDIAP", "HHEDP",
    "HIND", "HINASD", "HKIDP", "HLTHP", "HLUNP", "HMHCP", "HOLHP", "HSTRP", "HRSP",
 "INGDWTD", "INGP", "LANP", "LFHRP", "LLDD", "MDCP",
    "MTWP", "MV1D", "MV5D", "NPRD", "OCSKP", "RELP", "RLCP", "RPIP", "SPIP",
    "SPLF", "SSCF", "STATEUR", "STATE_x", "STATE_y",  "TEND", "TENLLD", "TISP",
    "UAICP", "UAI1P", "UAI5P", "UNCAREP", "VOLWP", "YARP"
]
remove = ['ABSHID_x', 'ABSHID', 'ABSFID', 'ADBSPID']

# Drop only columns that exist in the DataFrame
df = df.drop(columns=[col for col in remove if col in df.columns])

print('headers')
print(df.columns)
# First, define the specificities of continuous variables
data_info = {
    'distance': {
        'type': 'continuous',
        'bounds': [0.0, np.infty],
        'discrete': False,
        'apply_func': (lambda x: np.log(x+1)),
    },
    'age': {
        'type': 'continuous',
        'bounds': [0, 100],
        'enforce_bounds': True,
        'discrete': True
    },
    'departure_time': {
        'type': 'continuous',
        'bounds': [0, 23.999],
        'discrete': False
    }
}

data_info = {}

# Add the other variables as categorical
for c in df.columns:
    if c not in data_info.keys():
        data_info[c] = {'type': 'categorical'}




graph = nx.DiGraph()
for column in df.columns:
    graph.add_node(column)


#advise(df, graph, plot_graphs=True)
#plt.show()
# Step 4: Add edges between categories based on relationships
# Example: Add edges manually (custom logic based on your category relationships)
# Add edges only if both nodes exist in the DataFrame

'''



Suggested Edges (Household):
ABSHID → AREAENUM: The household identifier (ABSHID) determines the area of enumeration (AREAENUM).
ABSHID → BEDRD: The household identifier determines the number of bedrooms.
HHCD → BEDRD: Household composition influences the number of bedrooms.
HHCD → VEHRD: Household composition influences the number of vehicles.
NPRD → VEHRD: The number of persons usually resident in the dwelling influences the number of vehicles.
HIND → MRERD: Total household income determines mortgage repayments.
HIND → RNTRD: Total household income determines rent.
HIND → VEHRD: Total household income influences the number of vehicles.
MRERD → STRD: Mortgage repayments depend on the structure of the dwelling.
TEND → LLDD: Tenure type determines the landlord type.
INGDWTD → AREAENUM: Indigenous household status is specific to a particular area.

(FAMILY)
CPRF → CACF: The count of persons in the family determines the count of children.
FINASF → FMCF: Total family income influences the family composition.
CACF → FMCF: The number of children influences the family composition.
FRLF → FMCF: Relationships between families influence family composition (e.g., extended families).
SPLF → CPRF: The location of a spouse affects the count of persons in the family.

Suggested Edges (Person
AGEP → STUP: Age determines whether a person is a full/part-time student.
AGEP → HSCP: Age determines the highest year of school completed.
AGEP → TISP: Age influences the number of children ever born.
AGEP → EMPP: Age determines the likelihood of employment.
INCP → HIND: Personal income contributes to total household income.
INCP → FINASF: Personal income contributes to total family income.
OCCP → INCP: Occupation influences personal income.
OCCP → OCSP: Occupation determines occupation skill level.
HIED → INCP: Household income equivalence influences personal income.
SPIP → RPIP: Spouse/partner status determines the family/household reference person.
HIND → RPIP: Household income influences the reference person in the household.



Suggested Cross-Category Edges:
ABSHID → ABSFID: A household identifier determines a family identifier.
ABSHID → ABSPID: A household identifier determines a person identifier.
ABSFID → ABSPID: A family identifier determines a person identifier.
CPRF → NPRD: The count of persons in a family contributes to the count of persons in a dwelling.
CACF → NPRD: The count of children in the family contributes to the number of persons in the dwelling.
HIND → INCP: Total household income influences personal income.
FINASF → INCP: Total family income influences personal income.


'''
edges = [
    ("HEAP", "OCCP"),
    ("INCP", "DWTD"),
    ("INCP", "RNTRD"),
    ("INCP", "MRERD"),
    ("DWTD", "BEDRD"),
    ("BEDRD", "VEHRD"),
    ("AGEP", "STUP"),
    ("HHCD", "VEHRD"),
    ("CPRF", "VEHRD"),
    ("CACF", "BEDRD"),
    ("HIND", "VEHRD"),
    ("FINASF", "CPRF"),
    ('INDP', 'INCP'),

# Household edges
("HHCD", "BEDRD"),
("HHCD", "VEHRD"),
("NPRD", "VEHRD"),
("HIND", "MRERD"),
("HIND", "RNTRD"),
("HIND", "VEHRD"),
("MRERD", "STRD"),
('AREAENUM', 'HIND'),
('AREAENUM', 'MTWP'),

#("TEND", "LLDD"),


# Family edges
("CPRF", "CACF"),
("FINASF", "FMCF"),
("CACF", "FMCF"),
("FRLF", "FMCF"),
#("SPLF", "CPRF"),

# Person edges
("AGEP", "STUP"),
("AGEP", "HSCP"),
("AGEP", "TISP"),
("AGEP", "EMPP"),
("INCP", "HIND"),
("INCP", "FINASF"),
("OCCP", "INCP"),
("OCCP", "OCSKP"),
#("HIED", "INCP"),
('SSCF', 'SEXP'),

#("SPIP", "RLHP"),
("HIND", "RLHP"),
('HRSP', 'INCP'),
('LFHRP', 'INCP'),
('SIEMP', 'INCP'),
('RLHP', 'STUP'),



# Cross-category edges
("CPRF", "NPRD"),
("CACF", "NPRD"),

]


# Add edges only if both nodes exist in the DataFrame
for edge in edges:
    if edge[0] in df.columns and edge[1] in df.columns:
        graph.add_edge(*edge)
    else:
        print(f"Skipping edge {edge} as one or both nodes are not in the DataFrame columns")

batch_size = 1116
output_folder = './core_brisbane/output/'
directory_2 = './core_brisbane/output/'
if not os.path.exists(directory_2):
    os.makedirs(directory_2)
datgan = DATGAN(output=output_folder, batch_size=batch_size, num_epochs=10)
datgan.preprocess(df, data_info, preprocessed_data_path='./core_brisbane/encoded_data')
datgan.fit(df, data_info, graph, preprocessed_data_path='./core_brisbane/encoded_data')
samples = datgan.sample(len(df))

# Directory path
directory = './core_brisbane/data/'
if not os.path.exists(directory):
    os.makedirs(directory)


# Save the DataFrame to the CSV file
df.to_csv(os.path.join(directory, 'DATGAN.csv'), index=False)

samples.to_csv('./core_brisbane/data/DATGAN.csv', index=False)