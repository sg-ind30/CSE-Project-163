import os

import pandas as pd

from eda_analysis_merged import (
    get_valid_age,
    prepare_h255,
    create_merged_dataset,
    add_age_groups,
    add_insurance_labels,
    add_income_labels,
    plot_out_of_pocket,
    plot_spending_by_income_insurance
)


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
    h255 = pd.DataFrame({
        'DUPERSID': ['1', '1', '2', '3'],
        'ICD10CDX': ['E11', 'E11', 'E11', 'A10'],
        'AGEDIAG': [40, 42, 55, 20]
    })

    result = prepare_h255(h255)

    assert len(result) == 2
    assert result['DUPERSID'].nunique() == 2

    person1 = result[
        result['DUPERSID'] == '1'
    ].iloc[0]

    assert person1['AGEDIAG'] == 40
    assert person1['E11_RECORDS'] == 2


def test_create_merged_dataset():
    """
    Tests that the merged dataset correctly applies
    age, survey-weight, and diabetes filters.

    Parameters:
        None.

    Returns:
        None.
    """
    h255 = pd.DataFrame({
        'DUPERSID': ['1', '2', '3', '4'],
        'ICD10CDX': ['E11', 'E11', 'E11', 'E11'],
        'AGEDIAG': [40, 50, 60, 30]
    })

    h256 = pd.DataFrame({
        'DUPERSID': ['1', '2', '3', '4'],
        'AGE24X': [30, 17, 60, 45],
        'PERWT24F': [1, 1, 0, 1],
        'DIABDX_M18': [1, 1, 1, 2],
        'INSCOV24': [1, 2, 2, 3],
        'POVCAT24': [5, 2, 3, 1],
        'RXEXP24': [100, 200, 300, 400],
        'RXSLF24': [20, 30, 40, 50],
        'RXTOT24': [5, 6, 7, 8],
        'TOTEXP24': [1000, 2000, 3000, 4000],
        'TOTSLF24': [100, 200, 300, 400]
    })

    result = create_merged_dataset(h255, h256)

    assert len(result) == 1
    assert result.iloc[0]['DUPERSID'] == '1'

    assert (result['AGE24X'] >= 18).all()
    assert (result['PERWT24F'] > 0).all()
    assert (result['DIABDX_M18'] == 1).all()


def test_add_age_groups():
    """
    Tests that respondents are assigned to the
    correct age groups.

    Parameters:
        None.

    Returns:
        None.
    """
    data = pd.DataFrame({
        'AGE24X': [18, 44, 45, 64, 65, 80]
    })

    result = add_age_groups(data)

    assert result['Age Group'].astype(str).tolist() == [
        '18-44',
        '18-44',
        '45-64',
        '45-64',
        '65+',
        '65+'
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
    data = pd.DataFrame({
        'INSCOV24': [1, 2, 3]
    })

    result = add_insurance_labels(data)

    assert result['Insurance'].tolist() == [
        'Private',
        'Public Only',
        'Uninsured'
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
    data = pd.DataFrame({
        'POVCAT24': [1, 2, 3, 4, 5]
    })

    result = add_income_labels(data)

    assert result['Income Category'].tolist() == [
        'Poor/Negative',
        'Near Poor',
        'Low Income',
        'Middle Income',
        'High Income'
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
    filename = (
        'merged_out_of_pocket_by_insurance.png'
    )

    if os.path.exists(filename):
        os.remove(filename)

    data = pd.DataFrame({
        'INSCOV24': [1, 1, 2, 2, 3],
        'RXSLF24': [100, 200, 50, 70, 300]
    })

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
    filename = (
        'merged_spending_by_income_insurance.png'
    )

    if os.path.exists(filename):
        os.remove(filename)

    data = pd.DataFrame({
        'INSCOV24': [
            1, 2, 3,
            1, 2, 3,
            1, 2, 3,
            1, 2, 3,
            1, 2, 3
        ],
        'POVCAT24': [
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
            4, 4, 4,
            5, 5, 5
        ],
        'TOTSLF24': [
            100, 80, 150,
            120, 90, 170,
            140, 100, 190,
            160, 110, 210,
            180, 120, 230
        ]
    })

    plot_spending_by_income_insurance(data)

    assert os.path.exists(filename)


def main():
    """
    Runs all EDA tests and prints a confirmation
    message if every test passes.

    Parameters:
        None.

    Returns:
        None.
    """
    test_get_valid_age()
    test_prepare_h255()
    test_create_merged_dataset()
    test_add_age_groups()
    test_add_insurance_labels()
    test_add_income_labels()
    test_plot_out_of_pocket()
    test_plot_spending_by_income_insurance()

    print('All tests passed!')


if __name__ == '__main__':
    main()