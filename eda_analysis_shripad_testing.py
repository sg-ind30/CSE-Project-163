# Name: Shripad Guntur
# CSE 163
# Autumn 2026
# EDA Project 2 - NHIS 2023 Testing
# Tests data filtering, cleaning, categorization,
# and visualization functions.

import os

import pandas as pd

from eda_analysis_shripad import (
    filter_diabetes,
    clean_data,
    categorize_data,
    plot_income_insurance_barriers,
    plot_age_prescription_barriers
)


def test_filter_diabetes():
    """
    Tests whether the dataset is correctly filtered
    to adults with diabetes and positive survey weights.
    """
    data = pd.DataFrame({
        "DIBEV_A": [1, 2, 1, 1],
        "AGEP_A": [25, 50, 17, 65],
        "POVRATTC_A": [0.5, 2, 3, 4],
        "HICOV_A": [1, 2, 1, 2],
        "MEDDL12M_A": [1, 2, 1, 2],
        "MEDNG12M_A": [1, 2, 1, 2],
        "RXDG12M_A": [1, 2, 1, 2],
        "RXSK12M_A": [2, 1, 2, 1],
        "WTFA_A": [100, 200, 300, 0]
    })

    result = filter_diabetes(data)

    assert len(result) == 1
    assert (result["DIBEV_A"] == 1).all()
    assert (result["AGEP_A"] >= 18).all()
    assert (result["WTFA_A"] > 0).all()


def test_clean_data():
    """
    Tests whether invalid categorical responses
    are converted into missing values.
    """
    data = pd.DataFrame({
        "DIBEV_A": [1, 1, 1],
        "HICOV_A": [1, 2, 9],
        "MEDDL12M_A": [1, 2, 8],
        "MEDNG12M_A": [1, 2, 9],
        "RXDG12M_A": [1, 2, 8],
        "RXSK12M_A": [1, 2, None]
    })

    result = clean_data(data)

    assert len(result) == 3
    assert result["HICOV_A"].isna().sum() == 1
    assert result["MEDDL12M_A"].isna().sum() == 1
    assert result["MEDNG12M_A"].isna().sum() == 1
    assert result["RXDG12M_A"].isna().sum() == 1
    assert result["RXSK12M_A"].isna().sum() == 1


def test_categorize_data():
    """
    Tests income, age, insurance, and healthcare
    barrier categorization.
    """
    data = pd.DataFrame({
        "POVRATTC_A": [0.5, 1, 2, 4],
        "AGEP_A": [25, 44, 45, 65],
        "HICOV_A": [1, 2, 1, 2],
        "MEDDL12M_A": [1, 2, 1, 2],
        "MEDNG12M_A": [2, 1, 2, 1],
        "RXDG12M_A": [1, 2, 1, 2],
        "RXSK12M_A": [2, 1, 2, 1]
    })

    result = categorize_data(data)

    assert result["Income_Group"].tolist() == [
        "Poor", "Low", "Middle", "High"
    ]

    assert result["Age_Group"].tolist() == [
        "18-44", "18-44", "45-64", "65+"
    ]

    assert result["Insurance_Group"].tolist() == [
        "Insured", "Uninsured", "Insured", "Uninsured"
    ]

    assert result["Delayed_Care"].tolist() == [1, 0, 1, 0]
    assert result["Forgone_Care"].tolist() == [0, 1, 0, 1]
    assert result["Missing_Prescription"].tolist() == [1, 0, 1, 0]
    assert result["Skipped_Medication"].tolist() == [0, 1, 0, 1]


def test_plot_income_insurance_barriers():
    """
    Tests whether the income and insurance visualization
    is successfully generated and saved.
    """
    filename = "nhis23_income_insurance_barriers.png"

    data = pd.DataFrame({
        "Income_Group": ["Poor", "Poor", "Low", "Low",
                         "Middle", "Middle", "High", "High"],
        "Insurance_Group": ["Insured", "Uninsured"] * 4,
        "Delayed_Care": [0, 1, 0, 1, 0, 1, 0, 0]
    })

    plot_income_insurance_barriers(data)

    assert os.path.exists(filename)
    assert os.path.getsize(filename) > 0


def test_plot_age_prescription_barriers():
    """
    Tests whether the prescription medication visualization
    is successfully generated and saved.
    """
    filename = "nhis23_age_prescription_barriers.png"

    data = pd.DataFrame({
        "Age_Group": ["18-44", "18-44", "45-64",
                      "45-64", "65+", "65+"],
        "Missing_Prescription": [1, 1, 1, 0, 0, 0],
        "Skipped_Medication": [1, 0, 0, 1, 0, 0]
    })

    plot_age_prescription_barriers(data)

    assert os.path.exists(filename)
    assert os.path.getsize(filename) > 0


def main():
    """
    Executes all NHIS 2023 EDA tests.
    """
    test_filter_diabetes()
    test_clean_data()
    test_categorize_data()
    test_plot_income_insurance_barriers()
    test_plot_age_prescription_barriers()

    print("All NHIS 2023 tests passed!")


if __name__ == "__main__":
    main()