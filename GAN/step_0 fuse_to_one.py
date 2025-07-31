import pandas as pd


source = '.'
dwell = pd.read_csv(f'{source}/CENSUS_2021_BASIC_dwelling.csv')
person = pd.read_csv(f'{source}/CENSUS_2021_BASIC_person.csv')
family = pd.read_csv(f'{source}/CENSUS_2021_BASIC_family.csv')

# Step 1: Merge dwelling with family on 'hhid'
person_dwell = pd.merge(person, dwell, on='ABSHID', how='left')

# Step 2: Merge the resulting dataset with person on 'familyid'
linked_data = pd.merge(person_dwell, family, on='ABSFID', how='left')

# The resulting dataset 'linked_data' now contains all attributes
print(linked_data.head())
#Now lets drop all the infor that doesn't really matter.

# ONLY CONSIDER Brisbane and Souroundinh regions.

BRISBANE_AREA =[31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42]
#
# Filter the DataFrame So we only consider Brisbane
linked_data = linked_data[linked_data['AREAENUM'].isin(BRISBANE_AREA)]

#Now lets drop categories we don't care about
diction_drop = {
    'REGUCP': 'same as AREA_NUM',
    'REGU5P': 'dont care about moving yet',
    'REGU1P': 'no moving',
    'ANC1P' : 'best case of ancestry, probably useful',
    'ANC2P' :'dk',
    'ASSNP': 'could be useful for disabled people',
    'ANCRP': 'dont know',
    'BPFP': 'birthplace mother',
    'BPLP': 'birth of person, probably useful',
    'BPMP': 'birth of father',
    'BPPP': 'both parents',
    'CITP': 'citi',
    'ENGLP': 'eng',
    'FTCP': 'form',
    'HARTP': 'ill',
    'HASTP':'ill',
    'HCANP':'ill',
    'HDEMP':'ill',
    'HDIAP': 'ill',
    'HHEDP': 'ill',
    'HKIDP': 'ill',
    'HLTHP': 'ill',
    'HLUNP':'ill',
    'HMHCP': 'ill',
    'HOLHP': 'ill',
    'HSTRP': 'ill',
    'INGP': 'do we know how indigenous people travel',
    'LANP': 'do we know how languages travel',
    'LFSP': 'bigger list from lfhrp',
    'RELP': 'do we know how religion travels',
    'STATEUR': 'capture in other',
    'UAICP': 'dk',
    'UAI1p': 'dk',
    'UAI5p': 'dk',
    'UNCAREP': 'dk',
    'VOLWP': 'dk',
    'YARP': 'dk',
    'CPAF': 'dk',
    'SPLF': 'dk',
    'CPAD': 'dk',
    'CPLTHRD': 'dk',
    'HIED': 'dk',

    'HINASD': 'dk',
    'INGDWTD': 'dk',
    'LLDD': 'dk',
    'TEND': 'captureed with other',
    'STATE': 'dk',
    'STATE_x': 'dk',
    'STATE_y': 'dk',
    'ABSPID': 'dk',
    'ABSHID_y': 'dk',
    'CHCAREP': 'consider later',
    'FNOF': 'dk',
    'ADCP': 'dk',
    'ADFP': 'dl',
    'CLTHP': 'dk',
    'DOMP': 'sk',
    'DWIP': 'family level',

    'UAI1P': 'too hard',
    'UAI5P': 'too hard',
    'AREAENUM_y': 'same',
    'AREAENUM_x': 'same',
    'RPIP' : 'dont know',
    'MV1D': 'too hard',
    'MV5D': 'too jard',


    }

#for all names in dictionary drop columns

# Drop columns that are keys in the dictionary
linked_data = linked_data.drop(columns=[col for col in diction_drop.keys() if col in linked_data.columns])

linked_data = linked_data.loc[:, linked_data.columns.str.strip() != '']


def drop_unnamed_columns(df):
    """
    Drop unnamed or blank columns from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: A new DataFrame without unnamed or blank columns.
    """
    return df.loc[:, ~df.columns.str.match(r'^\s*$|^Unnamed')]


# Usage example
linked_data = drop_unnamed_columns(linked_data)
#SAVE THE LINKED DATA
linked_data.to_csv('LINKED_DATA_dropped.csv', index = False)
