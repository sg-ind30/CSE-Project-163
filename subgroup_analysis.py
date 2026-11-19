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

filtered_data_patients =  patients[(patients["DIABDX_M18"] == 1) & 
(patients["AGE24X"] >= 18) & 
(patients["PERWT24F"] > 0)
]

print(filtered_data_patients.shape)

# Conditions Dataset 
diabetes_codes = ["E08","E09","E10","E11","E13"]

filtered_data_conditions = conditions[conditions["ICD10CDX"].isin(diabetes_codes)]
print(filtered_data_conditions.shape)

print(filtered_data_conditions["DUPERSID"].nunique())
print(filtered_data_conditions["ICD10CDX"].value_counts())

unique_conditions = filtered_data_conditions.drop_duplicates(subset=["DUPERSID"])

print(unique_conditions)

# Merging Datasets 

merged_data = filtered_data_patients.merge(unique_conditions, on="DUPERSID", how="inner")
print(merged_data)

print(merged_data["DUPERSID"].nunique())
print(merged_data.isna().sum())
print(merged_data["POVCAT24"].value_counts())
print(merged_data["INSCOV24"].value_counts())
print(merged_data["TOTEXP24"].min())
print(merged_data["TOTSLF24"].min())

print(merged_data[["AGE24X","TOTEXP24","TOTSLF24"]].describe())

merged_data["Age_Group"] = pd.cut(
    merged_data["AGE24X"],
    bins = [18,44,64,85],
    labels = ["Young Adults", "Middle Aged Adults", "Older Adults"],
    include_lowest=True
)
print(merged_data["Age_Group"].value_counts())

income_labels = {
    1: "Poor",
    2: "Near Poor",
    3: "Low Income",
    4: "Middle Income",
    5: "High Income"
    
}

insurance_labels = {
    1: "Private Insurance",
    2: "Public Insurance",
    3: "Uninsured"
}

merged_data["Income_Group"] = merged_data["POVCAT24"].map(income_labels)
merged_data["Insurance_Group"] = merged_data["INSCOV24"].map(insurance_labels)

print(merged_data["Income_Group"].value_counts())
print(merged_data["Insurance_Group"].value_counts())

income_age_spending = merged_data.groupby(["Age_Group","Income_Group"])[["TOTEXP24","TOTSLF24"]].mean()

print(income_age_spending)

income_insurance_spending = merged_data.groupby(["Insurance_Group","Income_Group"])[["TOTEXP24","TOTSLF24"]].mean()

print(income_insurance_spending)

income_age_count = merged_data.groupby(["Age_Group","Income_Group"]).size()

income_insurance_count = merged_data.groupby(["Insurance_Group","Income_Group"]).size()

print(income_age_count)
print(income_insurance_count)

merged_data["Weighted_Total_Expenditure"] = merged_data["TOTEXP24"] * merged_data["PERWT24F"]

merged_data["Weighted_Out_Of_Pocket_Expenditure"] = merged_data["TOTSLF24"] * merged_data["PERWT24F"]

print(merged_data["Weighted_Total_Expenditure"].head())
print(merged_data["Weighted_Out_Of_Pocket_Expenditure"].head())