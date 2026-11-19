import pandas as pd
import matplotlib.pyplot as plt 

conditions = pd.read_excel("datasets/H255.xlsx", usecols=["DUPERSID", "ICD10CDX"])
patients = pd.read_excel("datasets/H256.xlsx", usecols=["DUPERSID","DIABDX_M18","AGE24X",
                                                        "POVCAT24","INSCOV24",
                                                        "TOTEXP24","TOTSLF24",
                                                        "PERWT24F","VARSTR","VARPSU"])

# Dimensions of datasets
conditions_dimensions = conditions.shape
patients_dimensions = patients.shape 

print(conditions_dimensions, patients_dimensions)

# List of Columns 
print(conditions.columns.tolist())
print(patients.columns.tolist())

# Sum of missing values 
missing_values_conditions = conditions.isna().sum()
missing_values_patients = patients.isna().sum()

print("Missing Condition Values:", missing_values_conditions)
print("Missing Patient Values:", missing_values_patients)

# Value Counts 
print(patients["DIABDX_M18"].value_counts())

filtered_data =  patients[(patients["DIABDX_M18"] == 1) & 
(patients["AGE24X"] >= 18) & 
(patients["PERWT24F"] > 0)
]

print(filtered_data.shape)

# Conditions Dataset 
diabetes_codes = ["E08","E09","E10","E11","E13"]

filtered_data_conditions = conditions[conditions["ICD10CDX"].isin(diabetes_codes)]
print(filtered_data_conditions.shape)

print(filtered_data_conditions["DUPERSID"].nunique())
print(filtered_data_conditions["ICD10CDX"].value_counts())

unique_conditions = filtered_data_conditions.drop_duplicates(subset=["DUPERSID"])

print(unique_conditions)