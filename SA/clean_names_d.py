import pandas as pd

# Define the dictionary of column names and their new names
column_mapping = {
    "household_id": "abshid",
    "family_id": "absfid",
    "id": "abspid",
    "AREAENUM":"regucp"
}

# Read the CSV file
input_csv = "hhid_fid.csv"  # Replace with the path to your CSV file
df = pd.read_csv(input_csv)
df['id'] = range(1, len(df) + 1)
#how to have an index column called id
# Rename columns if they exist in the dataset
df.rename(columns={old_name: new_name for old_name, new_name in column_mapping.items() if old_name in df.columns}, inplace=True)

# Keep only the columns specified in the dictionary (new names)
columns_to_keep = list(column_mapping.values())
columns_to_keep = columns_to_keep + ['AGEP', 'MSTP', 'MTWP', 'OCCP', 'RLHP', 'SEXP', 'HEAP', 'HIND', 'MRERD', 'NPRD', 'RNTRD', 'VEHRD']
columns_to_keep = columns_to_keep + ['GNGP', 'EMPP', 'HRSP', 'INCP', 'INDP', 'QALFP', 'QALLP','STRD', 'SIEMP', 'STUP', 'TYPP', 'SPIP', 'DWTD', 'TENLLD']
columns_to_keep = columns_to_keep + ['LFHRP', 'HSCP', 'FPIP', 'MDCP', 'OCSKP', 'FINASF', 'FMCF', 'FRLF', 'CACF', 'CPRF', 'SSCF', 'HINASD', 'HHCD']
df = df[columns_to_keep]

# Output the resulting DataFrame
output_csv = "hhid_fid_fixed.csv"  # Replace with the desired output file name
df.to_csv(output_csv, index=False)

print(f"Processed CSV saved to: {output_csv}")
