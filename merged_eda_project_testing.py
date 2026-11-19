"""
Tanisha, Shripad, Yuktha
CSE 163
Autumn 2026
EDA Project 2 - NHIS 2023 Testing
Tests data filtering, cleaning, categorization,
and visualization functions.
Tests data filtering, missing-value recoding, and income categorization
in eda.py using a controlled synthetic DataFrame.
Also includes the MEPS HC-255 + HC-256 merged dataset tests.
"""

import os

import pandas as pd

from merged_eda_project_2 import (
    # NHIS 2023
    filter_diabetes,
    clean_data,
    categorize_data,
    plot_income_insurance_barriers,
    plot_age_prescription_barriers,
    # MEPS
    get_valid_age,
    prepare_h255,
    create_merged_dataset,
    add_age_groups,
    add_insurance_labels,
    add_income_labels,
    plot_out_of_pocket,
    plot_spending_by_income_insurance,
    # NHIS 2022
    clean_nhis_data,
)

# NHIS 2023 tests


def test_filter_diabetes():
    """
    Tests whether the dataset is correctly filtered
    to adults with diabetes and positive survey weights.
    """
    data = pd.DataFrame(
        {
            "DIBEV_A": [1, 2, 1, 1],
            "AGEP_A": [25, 50, 17, 65],
            "POVRATTC_A": [0.5, 2, 3, 4],
            "HICOV_A": [1, 2, 1, 2],
            "MEDDL12M_A": [1, 2, 1, 2],
            "MEDNG12M_A": [1, 2, 1, 2],
            "RXDG12M_A": [1, 2, 1, 2],
            "RXSK12M_A": [2, 1, 2, 1],
            "WTFA_A": [100, 200, 300, 0],
        }
    )

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
    data = pd.DataFrame(
        {
            "AGEP_A": [25, 65, 97],
            "DIBEV_A": [1, 1, 1],
            "HICOV_A": [1, 2, 9],
            "MEDDL12M_A": [1, 2, 8],
            "MEDNG12M_A": [1, 2, 9],
            "RXDG12M_A": [1, 2, 8],
            "RXSK12M_A": [1, 2, None],
        }
    )

    result = clean_data(data)

    assert len(result) == 3
    assert result["HICOV_A"].isna().sum() == 1
    assert result["MEDDL12M_A"].isna().sum() == 1
    assert result["MEDNG12M_A"].isna().sum() == 1
    assert result["RXDG12M_A"].isna().sum() == 1
    assert result["RXSK12M_A"].isna().sum() == 1
    assert result["AGEP_A"].isna().sum() == 1
    assert result["AGEP_A"].iloc[0] == 25


def test_categorize_data():
    """
    Tests income, age, insurance, and healthcare
    barrier categorization.
    """
    data = pd.DataFrame(
        {
            "POVRATTC_A": [0.5, 1, 2, 4],
            "AGEP_A": [25, 44, 45, 65],
            "HICOV_A": [1, 2, 1, 2],
            "MEDDL12M_A": [1, 2, 1, 2],
            "MEDNG12M_A": [2, 1, 2, 1],
            "RXDG12M_A": [1, 2, 1, 2],
            "RXSK12M_A": [2, 1, 2, 1],
        }
    )

    result = categorize_data(data)

    assert result["Income_Group"].tolist() == ["Poor", "Low", "Middle", "High"]

    assert result["Age_Group"].tolist() == ["18-44", "18-44", "45-64", "65+"]

    assert result["Insurance_Group"].tolist() == [
        "Insured",
        "Uninsured",
        "Insured",
        "Uninsured",
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

    data = pd.DataFrame(
        {
            "Income_Group": [
                "Poor",
                "Poor",
                "Low",
                "Low",
                "Middle",
                "Middle",
                "High",
                "High",
            ],
            "Insurance_Group": ["Insured", "Uninsured"] * 4,
            "Delayed_Care": [0, 1, 0, 1, 0, 1, 0, 0],
        }
    )

    plot_income_insurance_barriers(data)

    assert os.path.exists(filename)
    assert os.path.getsize(filename) > 0


def test_plot_age_prescription_barriers():
    """
    Tests whether the prescription medication visualization
    is successfully generated and saved.
    """
    filename = "nhis23_age_prescription_barriers.png"

    data = pd.DataFrame(
        {
            "Age_Group": ["18-44", "18-44", "45-64", "45-64", "65+", "65+"],
            "Missing_Prescription": [1, 1, 1, 0, 0, 0],
            "Skipped_Medication": [1, 0, 0, 1, 0, 0],
        }
    )

    plot_age_prescription_barriers(data)

    assert os.path.exists(filename)
    assert os.path.getsize(filename) > 0


# MEPS tests


def test_get_valid_age():
    """
    Tests get_valid_age using valid and invalid age values.

    Parameters:
        None.

    Returns:
        None.
    """
    values = pd.Series([-1, -8, 45, 50])
    assert get_valid_age(values) == 45

    invalid = pd.Series([-1, -8])
    assert get_valid_age(invalid) is None


def test_prepare_h255():
    """
    Tests that HC-255 diabetes records are filtered and
    reduced to one row per person correctly.

    Parameters:
        None.

    Returns:
        None.
    """
    h255 = pd.DataFrame(
        {
            "DUPERSID": ["1", "1", "2", "3"],
            "ICD10CDX": ["E11", "E11", "E11", "A10"],
            "AGEDIAG": [40, 42, 55, 20],
        }
    )

    result = prepare_h255(h255)

    assert len(result) == 2
    assert result["DUPERSID"].nunique() == 2

    person1 = result[result["DUPERSID"] == "1"].iloc[0]

    assert person1["AGEDIAG"] == 40
    assert person1["E11_RECORDS"] == 2


def test_create_merged_dataset():
    """
    Tests that the merged dataset correctly applies
    age, survey-weight, and diabetes filters.

    Parameters:
        None.

    Returns:
        None.
    """
    h255 = pd.DataFrame(
        {
            "DUPERSID": ["1", "2", "3", "4"],
            "ICD10CDX": ["E11", "E11", "E11", "E11"],
            "AGEDIAG": [40, 50, 60, 30],
        }
    )

    h256 = pd.DataFrame(
        {
            "DUPERSID": ["1", "2", "3", "4"],
            "AGE24X": [30, 17, 60, 45],
            "PERWT24F": [1, 1, 0, 1],
            "DIABDX_M18": [1, 1, 1, 2],
            "INSCOV24": [1, 2, 2, 3],
            "POVCAT24": [5, 2, 3, 1],
            "RXEXP24": [100, 200, 300, 400],
            "RXSLF24": [20, 30, 40, 50],
            "RXTOT24": [5, 6, 7, 8],
            "TOTEXP24": [1000, 2000, 3000, 4000],
            "TOTSLF24": [100, 200, 300, 400],
        }
    )

    result = create_merged_dataset(h255, h256)

    assert len(result) == 1
    assert result.iloc[0]["DUPERSID"] == "1"

    assert (result["AGE24X"] >= 18).all()
    assert (result["PERWT24F"] > 0).all()
    assert (result["DIABDX_M18"] == 1).all()


def test_add_age_groups():
    """
    Tests that respondents are assigned to the
    correct age groups.

    Parameters:
        None.

    Returns:
        None.
    """
    data = pd.DataFrame({"AGE24X": [18, 44, 45, 64, 65, 80]})

    result = add_age_groups(data)

    assert result["Age Group"].astype(str).tolist() == [
        "18-44",
        "18-44",
        "45-64",
        "45-64",
        "65+",
        "65+",
    ]


def test_add_insurance_labels():
    """
    Tests that insurance codes are mapped to the
    correct readable insurance labels.

    Parameters:
        None.

    Returns:
        None.
    """
    data = pd.DataFrame({"INSCOV24": [1, 2, 3]})

    result = add_insurance_labels(data)

    assert result["Insurance"].tolist() == [
        "Private",
        "Public Only",
        "Uninsured",
    ]


def test_add_income_labels():
    """
    Tests that income codes are mapped to the
    correct readable income labels.

    Parameters:
        None.

    Returns:
        None.
    """
    data = pd.DataFrame({"POVCAT24": [1, 2, 3, 4, 5]})

    result = add_income_labels(data)

    assert result["Income Category"].tolist() == [
        "Poor/Negative",
        "Near Poor",
        "Low Income",
        "Middle Income",
        "High Income",
    ]


def test_plot_out_of_pocket():
    """
    Tests that the out-of-pocket prescription spending
    plot is successfully created and saved.

    Parameters:
        None.

    Returns:
        None.
    """
    filename = "merged_out_of_pocket_by_insurance.png"

    if os.path.exists(filename):
        os.remove(filename)

    data = pd.DataFrame(
        {"INSCOV24": [1, 1, 2, 2, 3], "RXSLF24": [100, 200, 50, 70, 300]}
    )

    plot_out_of_pocket(data)

    assert os.path.exists(filename)


def test_plot_spending_by_income_insurance():
    """
    Tests that the income and insurance healthcare
    spending plot is successfully created and saved.

    Parameters:
        None.

    Returns:
        None.
    """
    filename = "merged_spending_by_income_insurance.png"

    if os.path.exists(filename):
        os.remove(filename)

    data = pd.DataFrame(
        {
            "INSCOV24": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "POVCAT24": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            "TOTSLF24": [
                100,
                80,
                150,
                120,
                90,
                170,
                140,
                100,
                190,
                160,
                110,
                210,
                180,
                120,
                230,
            ],
        }
    )

    plot_spending_by_income_insurance(data)

    assert os.path.exists(filename)


# NHIS 2022 tests


def test_clean_nhis_data() -> None:
    """
    Tests adult diabetes filtering, NaN recoding, and income bracket
    categorization on a small synthetic DataFrame.
    """
    test_df = pd.DataFrame(
        {
            "DIBEV_A": [1, 2, 1, 1],  # Row 1 has no diabetes -> drop
            "AGEP_A": [45, 50, 16, 70],  # Row 2 is under 18 -> drop
            "POVRATTC_A": [0.85, 2.50, 1.20, 4.50],
            "HICOV_A": [1, 1, 2, 9],  # 9 -> NaN
            "MEDDL12M_A": [1, 2, 1, 7],  # 7 -> NaN
            "MEDNG12M_A": [2, 2, 1, 8],  # 8 -> NaN
            "RXDG12M_A": [1, 2, 2, 2],
            "RXSK12M_A": [2, 2, 1, 1],
        }
    )

    result = clean_nhis_data(test_df)

    assert len(result) == 2
    assert pd.isna(result.iloc[1]["HICOV_A"])
    assert pd.isna(result.iloc[1]["MEDDL12M_A"])
    assert pd.isna(result.iloc[1]["MEDNG12M_A"])
    assert result.iloc[0]["INCOME_GROUP"] == "Poor (<100% FPL)"
    assert result.iloc[1]["INCOME_GROUP"] == "High Income (400%+)"


def main():
    """
    Executes all NHIS 2023, MEPS, and NHIS 2022 EDA tests.
    """
    # NHIS 2023
    test_filter_diabetes()
    test_clean_data()
    test_categorize_data()
    test_plot_income_insurance_barriers()
    test_plot_age_prescription_barriers()

    # MEPS (HC-255 + HC-256)
    test_get_valid_age()
    test_prepare_h255()
    test_create_merged_dataset()
    test_add_age_groups()
    test_add_insurance_labels()
    test_add_income_labels()
    test_plot_out_of_pocket()
    test_plot_spending_by_income_insurance()

    # NHIS 2022
    test_clean_nhis_data()

    print("All tests passed!")


if __name__ == "__main__":
    main()
