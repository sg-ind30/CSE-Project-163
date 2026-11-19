import os

import pandas as pd

from eda_analysis_rq2 import (
    get_valid_age,
    prepare_h255,
    create_merged_dataset,
    add_insurance_labels,
    plot_out_of_pocket,
    plot_prescription_use
)


def test_get_valid_age():
    values = pd.Series([-1, -8, 45, 50])
    assert get_valid_age(values) == 45

    invalid = pd.Series([-1, -8])
    assert get_valid_age(invalid) is None


def test_prepare_h255():
    h255 = pd.DataFrame({
        'DUPERSID': ['1', '1', '2', '3'],
        'ICD10CDX': ['E11', 'E11', 'E11', 'A10'],
        'AGEDIAG': [40, 42, 55, 20]
    })

    result = prepare_h255(h255)

    assert len(result) == 2
    assert result['DUPERSID'].nunique() == 2

    person1 = result[result['DUPERSID'] == '1'].iloc[0]

    assert person1['AGEDIAG'] == 40
    assert person1['E11_RECORDS'] == 2


def test_create_merged_dataset():
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
        'RXTOT24': [5, 6, 7, 8]
    })

    result = create_merged_dataset(h255, h256)

    assert len(result) == 1
    assert result.iloc[0]['DUPERSID'] == '1'

    assert (result['AGE24X'] >= 18).all()
    assert (result['PERWT24F'] > 0).all()
    assert (result['DIABDX_M18'] == 1).all()


def test_add_insurance_labels():
    data = pd.DataFrame({
        'INSCOV24': [1, 2, 3]
    })

    result = add_insurance_labels(data)

    assert result['Insurance'].tolist() == [
        'Private',
        'Public Only',
        'Uninsured'
    ]


def test_plot_out_of_pocket():
    filename = 'merged_out_of_pocket_by_insurance.png'

    if os.path.exists(filename):
        os.remove(filename)

    data = pd.DataFrame({
        'INSCOV24': [1, 1, 2, 2, 3],
        'RXSLF24': [100, 200, 50, 70, 300]
    })

    plot_out_of_pocket(data)

    assert os.path.exists(filename)


def test_plot_prescription_use():
    filename = 'merged_prescription_use_by_insurance.png'

    if os.path.exists(filename):
        os.remove(filename)

    data = pd.DataFrame({
        'INSCOV24': [1, 1, 2, 2, 3],
        'RXTOT24': [10, 20, 30, 40, 5]
    })

    plot_prescription_use(data)

    assert os.path.exists(filename)


def main():
    test_get_valid_age()
    test_prepare_h255()
    test_create_merged_dataset()
    test_add_insurance_labels()
    test_plot_out_of_pocket()
    test_plot_prescription_use()

    print('All tests passed!')


if __name__ == '__main__':
    main()