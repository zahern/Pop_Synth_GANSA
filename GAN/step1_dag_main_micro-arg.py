
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Set the working directory to the script's directory
os.chdir(script_directory)

# Check the current working directory
print("Current working directory:", os.getcwd())

import tensorflow as tf
tf.config.run_functions_eagerly(True)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
import argparse
import random
import  os
import psutil

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"



# Get the first available GPU
try:
    import GPUtil
    DEVICE_ID_LIST = GPUtil.getFirstAvailable()
    DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except:
    print('COuld not grab')


import datgan
print(datgan.__version__)

from datgan import DATGAN
from datgan.utils.dag import advise
import numpy as np
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import  datetime
import matplotlib.pyplot as plt
import tensorflow as tf
print('numpy', np.__version__)
print('tensorflow version', tf.__version__)


def build_evidence_conditions(df, dependent_categories=None):
    """
    Build evidence conditions dynamically from the original data, allowing dependencies
    between numeric columns and all categorical columns.

    :param df: The input DataFrame.
    :param dependent_categories: List of categorical columns on which numeric columns may depend.
    """
    # If no specific dependent categories are provided, use all categorical columns
    if dependent_categories is None:
        dependent_categories = [
            col for col in df.columns
            if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object'
        ]

    evidence = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # For numeric columns, create a condition dependent on categorical variables
            def numeric_condition(x, row, col=col):
                for category_col in dependent_categories:
                    if category_col in row:
                        category_value = row[category_col]
                        # Filter the DataFrame based on the category value
                        category_group = df[df[category_col] == category_value]
                        if not category_group.empty:
                            lower_bound = category_group[col].quantile(0.1)
                            upper_bound = category_group[col].quantile(0.9)
                            return lower_bound <= x <= upper_bound
                # Fallback: use global quantiles if no category matches
                lower_bound = df[col].quantile(0.1)
                upper_bound = df[col].quantile(0.9)
                return lower_bound <= x <= upper_bound

            evidence[col] = numeric_condition

        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            # For categorical columns, use the most frequent value overall or by group
            def categorical_condition(x, row, col=col):
                for category_col in dependent_categories:
                    if category_col in row:
                        category_value = row[category_col]
                        # Filter the DataFrame based on the category value
                        category_group = df[df[category_col] == category_value]
                        if not category_group.empty:
                            mode = category_group[col].mode()[0]
                            return x == mode
                # Fallback to global mode
                mode = df[col].mode()[0]
                return x == mode

            evidence[col] = categorical_condition

    return evidence





def prepare_data(edges, data_source = 'census_micro_merged.csv',remove_c = ['f'], **kwargs):

    data_source_name = kwargs.get('data_source_name', 'hhts')

    df = pd.read_csv(data_source, index_col=False)
    print('please delete this, this is just a test')
    #df = df.sample(frac=0.01, random_state=42)
    #print(df['GNGP'])
    #how to only access like 5% of the data


   # df = df.sample(frac=0.005, random_state=42)  # ra
    df = df.drop(columns=[col for col in remove_c if col in df.columns])
    max_categories = 40
    remove_c = [
        col for col in df.columns
        if df[col].nunique() > max_categories
    ]

    # Drop high-cardinality columns from the DataFrame
    df = df.drop(columns=[col for col in remove_c if col in df.columns])
    print(df['GNGP'])
    print('dd')
    '''
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
    '''
    print('headers')
    print(df.columns)
    # First, define the specificities of continuous variables
    data_info = {
        'distance': {
            'type': 'continuous',
            'bounds': [0.0, np.inf],
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



    # Step 4: Add edges between categories based on relationships
    # Example: Add edges manually (custom logic based on your category relationships)
    # Add edges only if both nodes exist in the DataFrame




    # Add edges only if both nodes exist in the DataFrame
    for edge in edges:
        if edge[0] in df.columns and edge[1] in df.columns:
            graph.add_edge(*edge)
        else:
            print(f"Skipping edge {edge} as one or both nodes are not in the DataFrame columns")
    #was 1116
    evidence_conditions = build_evidence_conditions(df)
    for i in evidence_conditions:
        print(i)
    print('need to check this')
    evidence_conditions['AGEP']
    advise(df, graph, plot_graphs=False)
   # plt.savefig('plot.png')  # Save as a PNG file
    #plt.show()


    MY_TEST_PLOT = False
    if MY_TEST_PLOT:
        # Position nodes using a layout (e.g., spring layout or shell layout)
        pos = graphviz_layout(graph, prog = 'dot')  # You can try other layouts like nx.shell_layout

        # Draw the graph
        plt.figure(figsize=(12, 8))  # Set the figure size
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=3000,
            node_color="lightblue",
            font_size=10,
        # font_color="black",
            edge_color="gray",
            #arrows=True,  # Show directed edges
        # arrowstyle="-|>",
            #arrowsize=12,
        )
        plt.title("Directed Acyclic Graph (DAG)", fontsize=16)
        plt.savefig("graph_visualization.png")  # Save the graph as a PNG file
        plt.show()

    batch_size =kwargs.get('batch_size',1600)
    num_epochs = kwargs.get('epoch_size', 70)
    #samples_bigger = kwargs.get()
    runtime_code = datetime.now().strftime('%d_%m_%H%M')

    # List of funny names to choose from
    funny_names = ["Bismuth", "ScottStrugglers", "MrBeast",
        "MythicalKitchen", "KurtisConnor", "Rhett_and_Link",
        "KarlJobst", "GameTheory", "GabiBelle", "FlyingPotato", "WittyPenguin", "PandaGibbon", "DougDoug", "WavyWebSearch",
        "QuintonReviews", "Ludwig", "Swoop", "Tamago2474", "Wendigoon", "SomeOrdinaryGamers"
    ]

    # Randomly pick a funny name
    funny_name = random.choice(funny_names)
    # Combine timestamp and funny name
    runtime_code = f"{runtime_code}_{funny_name}"

    output_folder = f'./{data_source_name}_{runtime_code}/output/'
    directory_2 = f'./{data_source_name}_{runtime_code}/output/'
    if not os.path.exists(directory_2):
        os.makedirs(directory_2)


    conditionals = get_conditionals(data_source_name)
    print('getting conditionals')
    datgan = DATGAN(output=output_folder, batch_size=batch_size, num_epochs=num_epochs, conditional_inputs=conditionals)
    #advise(df, graph, plot_graphs=True)
    datgan.preprocess(df, data_info, preprocessed_data_path=f'./{data_source_name}_{runtime_code}/encoded_data')
    datgan.fit(df, data_info, graph, preprocessed_data_path=f'./{data_source_name}_{runtime_code}/encoded_data'
               )
    '''
    cond_dict = {'age': lambda x: x < 30, 'gender': '0', 'AGEP': '{age} * 2 > 50',
                 'OCCP': lambda AGEP, OCCP: OCCP == 11 if AGEP < 10 else False  # Constraint for income based on age

                 }
    '''


    cond_dict = {'INCP': lambda AGEP, INCP: INCP == 18 if AGEP < 10 else False,
                 'OCCP': lambda AGEP, OCCP: OCCP == 11 if AGEP < 10 else False  # Constraint for income based on age
                 }
    ## how to make the cond_dependent on the inputs
    # Directory path
    directory = f'./{data_source_name}_{runtime_code}/data/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    inputs = df.reset_index(drop=True)
    try:
        samples = datgan.sample(len(df), inputs = inputs)
    except Exception as e:
        print('trying without conditional dictionary')
        samples = datgan.sample(len(df), inputs=inputs)

    samples.to_csv(os.path.join(directory, 'DATGAN_samples.csv'), index=False)
    '''
    try:
        samples_bigger =  datgan.sample(len(df)*kwargs.get('sample_multiplier', 5), inputs = df)
    except Exception as e:
        print(e)
        print('error could not sample')
        samples_bigger = pd.DataFrame()
        samples_bigger.to_csv('ss.csv')

    '''

    print('saving the data')
    # Save the DataFrame to the CSV file
    df.to_csv(os.path.join(directory, 'DATGAN.csv'), index=False)

    samples.to_csv(os.path.join(directory, 'DATGAN_sample.csv'), index=False)
    #samples_bigger.to_csv(os.path.join(directory, 'DATGAN_sample_big.csv'), index=False)


def get_conditionals(data_source):
    if data_source == 'census':
        conditionals = ['AGEP', 'HSCP', 'GNGP', 'SPIP', 'MDCP', 'QALLP', 'QALFP', 'HRSP','INCP', 'STUP', 'OCCP', 'SEXP', 'INDP', 'HIND','families_in_household', 'persons_in_family', 'persons_in_household', 'MTWP',
                        'TISP', 'SIEMP', 'LFHRP',  'OCSKP', 'HEAP', 'MSTP', 'RLHP', 'SSCF', 'RLCP', 'CTPP', 'EMPP', 'TYPP']
        non_working = ['GNGP', 'MDCP', 'QALLP']
        #conditionals = ['AGEP', 'HSCP', 'INCP', 'STUP', 'OCCP', 'SEXP', 'INDP', 'families_in_household',
       #                 'persons_in_family', 'persons_in_household']
        return  conditionals
        #sgep implies siemp, age p implies gngp, agep implies ocskp,


def get_edges(data_source):
    if data_source == 'census':
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


        ''' #i
        edges = [
            ("HEAP", "OCCP"),
            ("INCP", "DWTD"),
            ("INCP", "RNTRD"),
            ("INCP", "MRERD"),
            ("DWTD", "BEDRD"),
            ("BEDRD", "VEHRD"),
            ("AGEP", "INCP"),
            ('AGEP', 'HRSP'),
            ('AGEP', 'MTWP'),
            ("AGEP", "INDP"),
            ("HHCD", "VEHRD"),
            ("CPRF", "VEHRD"),
            ("CACF", "BEDRD"),
            ("HIND", "VEHRD"),
            ("CPRF", "FINASF"),
            ('INDP', 'INCP'),


            ('SPIP', 'RLCP'),
            ('GNGP', 'OCCP'),
            ('MDCP', 'SPIP'),
            ('QALLP', 'OCCP'),
            ('QALFP', 'OCCP'),
            ('AGEP', 'QALFP'),



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

            # ("TEND", "LLDD"),

            # Family edges
            ("CPRF", "CACF"),
            ("FINASF", "FMCF"),
            ("CACF", "FMCF"),
            ("FRLF", "FMCF"),
            ('FMCF', 'HIND'),

            # ("SPLF", "CPRF"),
            ('HEAP', "CPRF"),
            # Person edges
            ("AGEP", "STUP"),
            ("AGEP", "HSCP"),
            ("AGEP", "TISP"),
            ("AGEP", "EMPP"),
            ("INCP", "HIND"),
            ("INCP", "FINASF"),
            ("OCCP", "INCP"),
            ('AGEP', 'TYPP'),
            ('TYPP', 'STUP'),
            ('AGEP', 'CTPP'),
            ('CTPP', 'RLCP'),
            ('CTPP', 'EMPP'),
            ('AGEP', 'RLCP'),

            ("OCCP", "OCSKP"),
            ('OCSKP', 'STUP'),
            ('OCCP', 'HSCP'),
            ('STUP', 'HSCP'),
            ('SEXP', 'HSCP'),
            ('FPIP', 'HSCP'),
            ('AGEP', 'SPIP'),
            ('SPIP', 'MSTP'),
            ('AGEP', 'MSTP'),
            ('MSTP', 'persons_in_family'
            ),
            ('MSTP', 'RLHP'),

            # ("HIED", "INCP"),
            ('SSCF', 'SEXP'),
            ('SSCF', 'RLCP'),
            ('RLCP', 'persons_in_family'),


            # ("SPIP", "RLHP"),
            ("HIND", "RLHP"),
            ('HRSP', 'INCP'),
            ('HRSP', 'HIND'),
            ('LFHRP', 'INCP'),
            ('SIEMP', 'INCP'),
            ('RLHP', 'STUP'),

            # Cross-category edges
            ("CPRF", "NPRD"),
            ("CACF", "NPRD"),
            ('families_in_household', 'HHCD'),
            ('families_in_household', 'HIND'),
            ('persons_in_family', 'FINASF'),
            ('persons_in_household', 'NPRD'),
            ('persons_in_household', 'BEDRD'),
            ('persons_in_household', 'VEHRD'),
            ('persons_in_household', 'HIND'),
            ('persons_in_household', 'HIED'),
            ('persons_in_family', 'FMCF'),
            ('unique_persons_in_household', 'INCP'),
            ('unique_persons_in_household', 'STUP'),
            ('unique_persons_in_family', 'INCP'),
            ('unique_persons_in_family', 'STUP'),
            ('FPIP', 'FMCF'),
            ("SEXP", "TISP"),
            ('FPIP', 'TISP'),
            ('AGEP', 'CTPP'),
            ('AGEP', 'OCCP'),
            ('HEAP', 'FMCF'),
            ('HEAP', 'persons_in_household'),
            ('HEAP', 'STUP'),
            ('VEHRD',  'MTWP'),
            ('AGEP', 'MTWP'),
            ('HRSP', 'MTWP'),
            ('SEXP', 'HRSP'),
            ('SEXP', 'MTWP'),
            ('AREAENUM', 'INCP'),
            ('EMPP', 'HRSP'),
            ('EMPP', 'MTWP'),
            ('RNTRD', 'AREAENUM'),










        ]
    elif data_source == 'hhts':
        '''HHTS data make the edge links'''
        edges = [("HHSIZE", "BIKES"),
                 ("HHSIZE", "HHVEH"),
                 ("HHSIZE", "EBIKES"),
                 ("HHSIZE", "ESCOOTER"),
                 ("HHSIZE", "DWELLTYPE"),
                 ("AGEGROUP", "WORKSTATUS"),
                 ("AGEGROUP", "MAINACT"),
                 ("AGEGROUP", "CARLICENSE"),
                 ("AGEGROUP", "MCLICENSE"),
                 ("AGEGROUP", "OTHERLICENSE"),
                 ("STRATA_LGA", "TRAVDOW"),
                 ('HOME_SA1_2021', 'STRATA_LGA'),
                 ('STUDYING', 'ED_TYPE'),
                 ('WORKSTATUS', 'INDUSTRY'),
                 ('WORKSTATUS', 'ANZSCO_3-digit'),
                 ('SEX', 'INDUSTRY'),
                 ('WORKSTATUS', 'WFHDAY'),
                 ('AGEGROUP', 'type'),
                 ('AGEGROUP', 'RELATIONSHIP'),
                 ('INDUSTRY', 'INCOME'),
                 ('INDUSTRY', 'ANZSCO_3-digit'),
                 ('INDUSTRY', 'ANYSTOPS'),

                 ('INCOME', 'type'),
                 ('INCOME', 'fuelType'),
                 ('type', 'fuelType'),
                 ('INCOME', 'year'),


                 ]
    else:
        edges = None
    return  edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data preparation script on an HPC cluster.")
    
    print("TensorFlow version:", tf.__version__)
    # Get available memory (in GB)
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

    # Set defaults based on available memory
    default_batch_size = 64 if available_memory_gb < 8 else 116
    default_epoch_size = 100

    # Add arguments
    parser.add_argument('--data_source_name', type=str, default='census',
                        help='Name of the data source (e.g., hhts)')
    parser.add_argument('--batch-size', type = int, default=128, help = 'help network size')
    parser.add_argument('--epoch-size', type = int, default=100, help = 'number of epochs')
    parser.add_argument('--sample-multiplier', type=int, default=50, help='number of increases in the data')

    # Parse the arguments
    args = parser.parse_args()


    data_source_name = args.data_source_name
    if data_source_name =='hhts':
        data_source = 'merged_table_seq.csv'
        remove_columns = ['HHID', 'SURVEYWEEK', 'TRAVMONTH', 'TRAVDATE', 'HHWGT_22', 'PERSID', 'ANZSCO_1-digit',
                          'DEMOGWGT22', 'PERSWGT22', 'VEHID']
    else:
        data_source_name = 'census'
        data_source ='census_micro_merged.csv'
        remove_columns =['ABSHID', 'ABSPID', 'ABSFID']

    #get the edges
    edges = get_edges(data_source_name)
    #TODO get output naming folder, get datasource name, defined unique edge
    prepare_data(edges, data_source, remove_columns, **vars(args))

