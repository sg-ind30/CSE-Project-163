"""
Tanisha Deka, Yuktha Pereddy, Shripad Guntur
CSE 163
Autumn 2026
EDA Project 2
Explores healthcare affordability and cost-related medical
barriers among adults with diabetes across different income,
age, and insurance groups done via the NHIS 2023 dataset.
Performs exploratory data analysis on the 2022 NHIS Sample Adult dataset,
computing missingness, variable summaries, and cost-barrier visualizations
for adults with diagnosed diabetes.
Also includes the MEPS HC-255 + HC-256 merged adult diabetes EDA.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

# Constants

# NHIS 2023
DATA_FILE = "datasets/adult23.csv"

VARIABLES = [
    "DIBEV_A",
    "AGEP_A",
    "POVRATTC_A",
    "HICOV_A",
    "MEDDL12M_A",
    "MEDNG12M_A",
    "RXDG12M_A",
    "RXSK12M_A",
    "WTFA_A"
]

QUANTITATIVE_VARIABLES = [
    "AGEP_A",
    "POVRATTC_A",
    "WTFA_A"
]

CATEGORICAL_VARIABLES = [
    "DIBEV_A",
    "HICOV_A",
    "MEDDL12M_A",
    "MEDNG12M_A",
    "RXDG12M_A",
    "RXSK12M_A"
]

COLORS = ["#1f77b4", "#2ca02c"]

# NHIS 2022
DATA_PATH = 'datasets/adult22.csv'
CAT_COLS = ['HICOV_A', 'MEDDL12M_A', 'MEDNG12M_A', 'RXDG12M_A', 'RXSK12M_A']
QUANT_COLS = ['AGEP_A', 'POVRATTC_A', 'WTFA_A']

# MEPS
H255_FILE = 'datasets/H255.csv'
H256_FILE = 'datasets/h256.csv'

# NHIS 2023 functions

def explore_data(data):
    """
    Displays the Dimensions, missing values, & percentange of missing values.
    """
    print("Dataset dimensions:", data.shape)

    missing_values = data.isna().sum().sum()
    missing_percentage = missing_values / data.size * 100

    print("Missing values:", missing_values)
    print("Missing percentage:", missing_percentage)

    print("\nVariables of Interest:")

    for variable in VARIABLES:
        print(variable, variable in data.columns)


def filter_diabetes(data):
    """
    Returns a DataFrame containing adults diagnosed with diabetes
    who are at least 18 years old and have positive survey weights.
    """
    diabetes_data = data[
        (data["DIBEV_A"] == 1) &
        (data["AGEP_A"] >= 18) &
        (data["WTFA_A"] > 0)
    ][VARIABLES].copy()

    return diabetes_data


def summarize_data(data):
    """
    Reports missing values, quantitative summary statistics,
    and categorical frequency counts for adults with diabetes.
    """
    print("\nFiltered dataset:", data.shape)

    print("\nMissing Values:")
    print(data.isna().sum())

    print("\nQuantitative Summary:")
    print(data[QUANTITATIVE_VARIABLES].describe())

    print("\nCategorical Summary:")

    for variable in CATEGORICAL_VARIABLES:
        print("\n", variable)
        print(data[variable].value_counts(dropna=False))


def clean_data(data):
    """
    Converts invalid categorical responses and age codes
    into missing values while retaining valid responses.
    """
    result = data.copy()

    for variable in CATEGORICAL_VARIABLES:
        result[variable] = result[variable].where(
            result[variable].isin([1, 2])
        )

    result["AGEP_A"] = result["AGEP_A"].where(
        result["AGEP_A"].between(18, 85)
    )

    print("\nMissing Values After Cleaning:")
    print(result.isna().sum())

    return result


def categorize_data(data):
    """
    Categorizes participants according to family income relative
    to the federal poverty level, age, and insurance coverage.
    Converts healthcare barrier responses into binary indicators.
    """
    result = data.copy()

    # Income groups
    result["Income_Group"] = pd.cut(
        result["POVRATTC_A"],
        bins=[0, 1, 2, 4, float("inf")],
        labels=["Poor", "Low", "Middle", "High"],
        include_lowest=True,
        right=False
    )

    # Age groups
    result["Age_Group"] = pd.cut(
        result["AGEP_A"],
        bins=[18, 45, 65, float("inf")],
        labels=["18-44", "45-64", "65+"],
        include_lowest=True,
        right=False
    )

    # Insurance groups
    result["Insurance_Group"] = result["HICOV_A"].map({
        1: "Insured",
        2: "Uninsured"
    })

    # Healthcare barrier indicators
    result["Delayed_Care"] = (
        result["MEDDL12M_A"].map({1: 1, 2: 0})
    )

    result["Forgone_Care"] = (
        result["MEDNG12M_A"].map({1: 1, 2: 0})
    )

    result["Missing_Prescription"] = (
        result["RXDG12M_A"].map({1: 1, 2: 0})
    )

    result["Skipped_Medication"] = (
        result["RXSK12M_A"].map({1: 1, 2: 0})
    )

    print("\nIncome Group Counts:")
    print(result["Income_Group"].value_counts())

    print("\nAge Group Counts:")
    print(result["Age_Group"].value_counts())

    print("\nInsurance Group Counts:")
    print(result["Insurance_Group"].value_counts())

    return result


def plot_income_insurance_barriers(data):
    """
    Generates a grouped bar chart comparing the percentage of
    respondents reporting delayed medical care across income
    groups and insurance coverage categories.
    """
    income_insurance_barriers = data.groupby(
        ["Income_Group", "Insurance_Group"],
        observed=True
    )["Delayed_Care"].mean() * 100

    income_insurance_barriers = (
        income_insurance_barriers.unstack()
    )

    income_insurance_barriers = (
        income_insurance_barriers.reindex(
            index=["Poor", "Low", "Middle", "High"],
            columns=["Insured", "Uninsured"]
        )
    )

    print("\nDelayed Medical Care by Income and Insurance:")
    print(income_insurance_barriers)

    income_insurance_barriers.plot(
        kind="bar",
        figsize=(10, 6),
        color=COLORS
    )

    plt.title(
        "Delayed Medical Care by Income and Insurance "
        "(2023 NHIS)"
    )

    plt.xlabel("Income Group")
    plt.ylabel("Respondents Reporting Delayed Care (%)")
    plt.legend(title="Insurance Coverage")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        "nhis23_income_insurance_barriers.png",
        dpi=300
    )

    plt.close()


def plot_age_prescription_barriers(data):
    """
    Generates a grouped bar chart comparing the percentage of
    adults with diabetes who could not obtain medication or
    skipped prescription doses due to cost across age groups.
    """
    age_barriers = data.groupby(
        "Age_Group",
        observed=True
    )[[
        "Missing_Prescription",
        "Skipped_Medication"
    ]].mean() * 100

    age_barriers = age_barriers.reindex([
        "18-44",
        "45-64",
        "65+"
    ])

    print("\nPrescription Medication Barriers by Age:")
    print(age_barriers)

    age_barriers.plot(
        kind="bar",
        figsize=(10, 6),
        color=COLORS
    )

    plt.title(
        "Cost-Related Prescription Barriers by Age "
        "(2023 NHIS)"
    )

    plt.xlabel("Age Group")
    plt.ylabel("Percentage of Valid Respondents (%)")
    plt.legend([
        "Did Not Obtain Medication",
        "Skipped Doses"
    ])

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        "nhis23_age_prescription_barriers.png",
        dpi=300
    )

    plt.close()


def summarize_subgroups(data):
    """
    Reports the number of valid respondents contributing to
    each healthcare barrier percentage in the visualizations.
    """
    print("\nIncome and Insurance Sample Sizes:")

    print(
        data.groupby(
            ["Income_Group", "Insurance_Group"],
            observed=True
        )["Delayed_Care"].count()
    )

    print("\nAge Group Sample Sizes:")

    print(
        data.groupby(
            "Age_Group",
            observed=True
        )[[
            "Missing_Prescription",
            "Skipped_Medication"
        ]].count()
    )

def validate_rq3(data):
    """
    Checks the expected frequency assumption for chi-square
    tests of income and cost-related medical care barriers.
    """
    outcomes = ["Delayed_Care", "Forgone_Care"]

    for outcome in outcomes:
        table = pd.crosstab(
            data["Income_Group"],
            data[outcome]
        )

        chi2, p_value, dof, expected = chi2_contingency(table)

        print("\nRQ3 Result Validity:", outcome)

        print("Observed Frequencies:")
        print(table)

        print("Expected Frequencies:")
        print(expected)

        print("Minimum Expected Frequency:", expected.min())
        print("All Expected Frequencies >= 5:",
              (expected >= 5).all())

        print("Chi-square Statistic:", chi2)
        print("Degrees of Freedom:", dof)
        print("P-value:", p_value)


# NHIS 2022 functions

def clean_nhis_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters an NHIS DataFrame to adults (18+) with diabetes (DIBEV_A == 1),
    replaces survey non-response codes (7, 8, 9) with NaN, and categorizes
    POVRATTC_A into four federal poverty income brackets.
    """
    filtered = df[(df['DIBEV_A'] == 1) & (df['AGEP_A'] >= 18)].copy()

    for col in CAT_COLS:
        if col in filtered.columns:
            filtered[col] = filtered[col].replace([7, 8, 9], np.nan)

    if 'POVRATTC_A' in filtered.columns:
        filtered['INCOME_GROUP'] = pd.cut(
            filtered['POVRATTC_A'],
            bins=[-1, 0.99, 1.99, 3.99, 100],
            labels=[
                'Poor (<100% FPL)',
                'Low Income (100–199%)',
                'Middle Income (200–399%)',
                'High Income (400%+)'
            ]
        )

    return filtered


def print_summaries(raw_df: pd.DataFrame, diab_df: pd.DataFrame) -> None:
    """
    Prints overall missingness, sample size, seven-number summaries for
    quantitative variables, and value counts for categorical variables.
    """
    total_missing_pct = (raw_df.isna().sum().sum() / raw_df.size) * 100
    print(f"Overall dataset missingness: {total_missing_pct:.2f}%")
    print(f"Total diabetic adult sample size: {len(diab_df)}\n")

    summary_7 = diab_df[QUANT_COLS].describe().T[
        ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    ]
    print("--- QUANTITATIVE VARIABLES (7-NUMBER SUMMARY) ---")
    print(summary_7.round(2))

    print("\n--- CATEGORICAL VARIABLES (UNIQUE VALUES & COUNTS) ---")
    for col in CAT_COLS:
        print(f"\n{col}:")
        print(diab_df[col].value_counts(dropna=False))


def plot_income_barriers(diab_df: pd.DataFrame) -> None:
    """
    Generates and saves a grouped bar chart comparing cost-related delayed
    and forgone medical care across federal poverty income brackets.
    """
    plot_df = diab_df.dropna(
        subset=['INCOME_GROUP', 'MEDDL12M_A', 'MEDNG12M_A']
    ).copy()

    delayed = (plot_df['MEDDL12M_A'] == 1).astype(int) * 100
    forgone = (plot_df['MEDNG12M_A'] == 1).astype(int) * 100

    plot_df['Delayed Care (%)'] = delayed
    plot_df['Forgone Care (%)'] = forgone

    melted = plot_df.melt(
        id_vars='INCOME_GROUP',
        value_vars=['Delayed Care (%)', 'Forgone Care (%)'],
        var_name='Barrier Type',
        value_name='Percentage'
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=melted,
        x='INCOME_GROUP',
        y='Percentage',
        hue='Barrier Type',
        errorbar=None,
        palette=['#4C72B0', '#DD8452']
    )
    plt.title(
        'Cost-Related Delayed and Forgone Medical Care '
        'by Income Level (2022 NHIS)'
    )
    plt.xlabel('Family Income Relative to Federal Poverty Level (POVRATTC_A)')
    plt.ylabel('Diabetic Adults Reporting Barrier (%)')
    plt.legend(title='Care Barrier')
    plt.tight_layout()
    plt.savefig('nhis22_plot1_income_barriers.png', dpi=300)
    plt.close()


def plot_insurance_rx(diab_df: pd.DataFrame) -> None:
    """
    Generates and saves a grouped bar chart comparing cost-related
    prescription barriers between insured and uninsured diabetic adults.
    """
    plot_df = diab_df.dropna(
        subset=['HICOV_A', 'RXDG12M_A', 'RXSK12M_A']
    ).copy()
    plot_df['Insurance Status'] = plot_df['HICOV_A'].map(
        {1.0: 'Insured', 2.0: 'Uninsured'}
    )
    plot_df['Did Not Get Rx (%)'] = (
        (plot_df['RXDG12M_A'] == 1).astype(int) * 100
    )
    plot_df['Skipped Doses (%)'] = (
        (plot_df['RXSK12M_A'] == 1).astype(int) * 100
    )

    melted = plot_df.melt(
        id_vars='Insurance Status',
        value_vars=['Did Not Get Rx (%)', 'Skipped Doses (%)'],
        var_name='Medication Barrier',
        value_name='Percentage'
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=melted,
        x='Insurance Status',
        y='Percentage',
        hue='Medication Barrier',
        errorbar=None,
        palette=['#55A868', '#C44E52']
    )
    plt.title(
        'Cost-Related Prescription Medication Barriers '
        'by Insurance Coverage (2022 NHIS)'
    )
    plt.xlabel('Health Insurance Coverage Status (HICOV_A)')
    plt.ylabel('Diabetic Adults Reporting Barrier (%)')
    plt.legend(title='Prescription Barrier')
    plt.tight_layout()
    plt.savefig('nhis22_plot2_insurance_rx.png', dpi=300)
    plt.close()


def check_chi_square_assumptions(diab_df: pd.DataFrame) -> None:
    """
    Verifies that expected cell frequencies exceed 5 for Chi-Square
    testing between income groups and cost-related care barriers.
    """
    for col in ['MEDDL12M_A', 'MEDNG12M_A']:
        ct = pd.crosstab(diab_df['INCOME_GROUP'], diab_df[col])
        _, _, _, expected = chi2_contingency(ct)
        print(f"Min expected cell count for {col}: {expected.min():.2f}")

# MEPS functions

def get_valid_age(series):
    """
    Returns the first valid non-negative age from a Series.

    Parameters:
        series: A pandas Series containing age values.

    Returns:
        The first valid age, or None if no valid age is found.
    """
    valid = series[series >= 0]

    if len(valid) == 0:
        return None

    return valid.iloc[0]


def prepare_h255(h255):
    """
    Filters HC-255 to diabetes records and reduces the
    condition-level file to one row per person.

    Parameters:
        h255: A pandas DataFrame containing HC-255 data.

    Returns:
        A DataFrame with one row per person with diabetes.
    """
    diabetes_codes = ['E08', 'E09', 'E10', 'E11', 'E13']

    diabetes = h255[
        h255['ICD10CDX'].isin(diabetes_codes)
    ].copy()

    person_diabetes = diabetes.groupby(
        'DUPERSID',
        as_index=False
    ).agg(
        ICD10CDX=('ICD10CDX', 'first'),
        AGEDIAG=('AGEDIAG', get_valid_age),
        E11_RECORDS=('ICD10CDX', 'size')
    )

    return person_diabetes


def create_merged_dataset(h255, h256):
    """
    Creates the final merged adult diabetes dataset.

    Parameters:
        h255: A pandas DataFrame containing HC-255 data.
        h256: A pandas DataFrame containing HC-256 data.

    Returns:
        A merged DataFrame containing eligible adults with diabetes.
    """
    person_diabetes = prepare_h255(h255)

    merged = h256.merge(
        person_diabetes,
        on='DUPERSID',
        how='inner',
        validate='one_to_one'
    )

    merged = merged[
        (merged['AGE24X'] >= 18) &
        (merged['PERWT24F'] > 0) &
        (merged['DIABDX_M18'] == 1)
    ].copy()

    return merged


def add_age_groups(data):
    """
    Adds age groups used for Research Question 4.

    Parameters:
        data: A pandas DataFrame containing AGE24X.

    Returns:
        A copy of the DataFrame with an Age Group column.
    """
    result = data.copy()

    result['Age Group'] = pd.cut(
        result['AGE24X'],
        bins=[17, 44, 64, float('inf')],
        labels=['18-44', '45-64', '65+']
    )

    return result


def analyze_merged_data(data):
    """
    Prints descriptive statistics and summaries for the merged dataset.

    Parameters:
        data: A pandas DataFrame containing the merged MEPS data.

    Returns:
        None.
    """
    print('MERGED HC-255 + HC-256 DATASET')
    print('Rows:', data.shape[0])
    print('Columns:', data.shape[1])

    variables = [
        'INSCOV24',
        'POVCAT24',
        'AGE24X',
        'RXEXP24',
        'RXSLF24',
        'RXTOT24',
        'TOTEXP24',
        'TOTSLF24'
    ]

    print('\nMissing values:')
    print(data[variables].isna().sum())

    print('\nMinimum values:')
    print(data[variables].min())

    print('\nInsurance counts:')
    print(
        data['INSCOV24']
        .value_counts()
        .sort_index()
    )

    print('\nIncome category counts:')
    print(
        data['POVCAT24']
        .value_counts()
        .sort_index()
    )

    print('\nAge group counts:')
    print(
        data['Age Group']
        .value_counts()
        .sort_index()
    )

    quantitative = [
        'AGE24X',
        'RXEXP24',
        'RXSLF24',
        'RXTOT24',
        'TOTEXP24',
        'TOTSLF24'
    ]

    print('\nSeven-number summaries:')
    print(
        data[quantitative]
        .describe()
        .loc[
            [
                'mean',
                'std',
                'min',
                '25%',
                '50%',
                '75%',
                'max'
            ]
        ]
    )

    print('\nMean values by insurance:')
    print(
        data.groupby('INSCOV24')[
            quantitative
        ].mean()
    )

    print('\nMedian values by insurance:')
    print(
        data.groupby('INSCOV24')[
            quantitative
        ].median()
    )


def add_insurance_labels(data):
    """
    Adds readable insurance category labels to the dataset.

    Parameters:
        data: A pandas DataFrame containing the INSCOV24 column.

    Returns:
        A copy of the DataFrame with an Insurance column added.
    """
    result = data.copy()

    insurance_names = {
        1: 'Private',
        2: 'Public Only',
        3: 'Uninsured'
    }

    result['Insurance'] = (
        result['INSCOV24']
        .map(insurance_names)
    )

    return result


def add_income_labels(data):
    """
    Adds readable income category labels.

    Parameters:
        data: A pandas DataFrame containing POVCAT24.

    Returns:
        A copy of the DataFrame with an Income Category column.
    """
    result = data.copy()

    income_names = {
        1: 'Poor/Negative',
        2: 'Near Poor',
        3: 'Low Income',
        4: 'Middle Income',
        5: 'High Income'
    }

    result['Income Category'] = (
        result['POVCAT24']
        .map(income_names)
    )

    return result


def plot_out_of_pocket(data):
    """
    Creates and saves a bar chart of mean out-of-pocket
    prescription spending by insurance group.

    Parameters:
        data: A pandas DataFrame containing insurance and
        prescription spending data.

    Returns:
        None.
    """
    labeled = add_insurance_labels(data)

    means = (
        labeled.groupby('Insurance')['RXSLF24']
        .mean()
        .reindex(
            ['Private', 'Public Only', 'Uninsured']
        )
    )

    means.plot(kind='bar')

    plt.title(
        'Mean Out-of-Pocket Prescription Spending '
        'by Insurance'
    )
    plt.xlabel('Insurance Coverage')
    plt.ylabel(
        'Mean Annual Out-of-Pocket Spending ($)'
    )
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        'merged_out_of_pocket_by_insurance.png'
    )

    plt.clf()


def plot_spending_by_income_insurance(data):
    """
    Creates and saves a grouped bar chart of mean annual
    out-of-pocket healthcare spending by income and insurance.

    Parameters:
        data: A pandas DataFrame containing income,
        insurance, and healthcare spending data.

    Returns:
        None.
    """
    labeled = add_insurance_labels(data)
    labeled = add_income_labels(labeled)

    income_order = [
        'Poor/Negative',
        'Near Poor',
        'Low Income',
        'Middle Income',
        'High Income'
    ]

    insurance_order = [
        'Private',
        'Public Only',
        'Uninsured'
    ]

    means = (
        labeled.groupby(
            ['Income Category', 'Insurance'],
            observed=True
        )['TOTSLF24']
        .mean()
        .unstack()
        .reindex(
            index=income_order,
            columns=insurance_order
        )
    )

    means.plot(kind='bar')

    plt.title(
        'Mean Out-of-Pocket Healthcare Spending '
        'by Income and Insurance'
    )
    plt.xlabel('Income Category')
    plt.ylabel(
        'Mean Annual Out-of-Pocket '
        'Healthcare Spending ($)'
    )
    plt.xticks(rotation=20)
    plt.legend(title='Insurance Coverage')
    plt.tight_layout()

    plt.savefig(
        'merged_spending_by_income_insurance.png'
    )

    plt.clf()


def main():
    """
    Executes the NHIS 2023 exploratory data analysis, the MEPS
    HC-255 + HC-256 merged analysis, and the NHIS 2022 exploratory
    data analysis.
    """
    # ----- NHIS 2023 -----
    data = pd.read_csv(DATA_FILE)

    explore_data(data)

    diabetes_data = filter_diabetes(data)

    summarize_data(diabetes_data)

    diabetes_data = clean_data(diabetes_data)

    print("\nCleaned Quantitative Summary:")
    print(diabetes_data[QUANTITATIVE_VARIABLES].describe())

    diabetes_data = categorize_data(diabetes_data)

    plot_income_insurance_barriers(diabetes_data)

    plot_age_prescription_barriers(diabetes_data)

    summarize_subgroups(diabetes_data)

    validate_rq3(diabetes_data)

    # MEPS (HC-255 + HC-256)
    h255 = pd.read_csv(H255_FILE)
    h256 = pd.read_csv(H256_FILE)

    merged = create_merged_dataset(
        h255,
        h256
    )

    merged = add_age_groups(merged)

    analyze_merged_data(merged)

    plot_out_of_pocket(merged)
    plot_spending_by_income_insurance(merged)

    # NHIS 2022 

    sns.set_theme(style='whitegrid')
    raw_data = pd.read_csv(DATA_PATH)
    diab_df = clean_nhis_data(raw_data)

    print_summaries(raw_data, diab_df)
    plot_income_barriers(diab_df)
    plot_insurance_rx(diab_df)
    check_chi_square_assumptions(diab_df)


if __name__ == "__main__":
    main()