# Name: Shripad Guntur
# CSE 163
# Autumn 2026
# EDA Project 2
# Explores healthcare affordability and cost-related medical
# barriers among adults with diabetes across different income,
# age, and insurance groups done via the NHIS 2023 dataset.

import pandas as pd
import matplotlib.pyplot as plt

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
    Converts invalid categorical survey responses into missing
    values while retaining valid responses for each participant.
    """
    result = data.copy()

    for variable in CATEGORICAL_VARIABLES:
        result[variable] = result[variable].where(
            result[variable].isin([1, 2])
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


def main():
    """
    Executes the exploratory data analysis for the 2023
    National Health Interview Survey dataset.
    """
    data = pd.read_csv(DATA_FILE)

    explore_data(data)

    diabetes_data = filter_diabetes(data)

    summarize_data(diabetes_data)

    diabetes_data = clean_data(diabetes_data)

    diabetes_data = categorize_data(diabetes_data)

    plot_income_insurance_barriers(diabetes_data)

    plot_age_prescription_barriers(diabetes_data)

    summarize_subgroups(diabetes_data)


if __name__ == "__main__":
    main()