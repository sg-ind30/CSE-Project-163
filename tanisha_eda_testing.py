"""
Tanisha Deka
CSE 163 Final Project - Part 2 (Testing)

Tests data filtering, missing-value recoding, and income categorization
in eda.py using a controlled synthetic DataFrame.
"""

import pandas as pd

from eda import clean_nhis_data


def test_clean_nhis_data() -> None:
    """
    Tests adult diabetes filtering, NaN recoding, and income bracket
    categorization on a small synthetic DataFrame.
    """
    test_df = pd.DataFrame({
        'DIBEV_A': [1, 2, 1, 1],        # Row 1 has no diabetes -> drop
        'AGEP_A': [45, 50, 16, 70],     # Row 2 is under 18 -> drop
        'POVRATTC_A': [0.85, 2.50, 1.20, 4.50],
        'HICOV_A': [1, 1, 2, 9],        # 9 -> NaN
        'MEDDL12M_A': [1, 2, 1, 7],     # 7 -> NaN
        'MEDNG12M_A': [2, 2, 1, 8],     # 8 -> NaN
        'RXDG12M_A': [1, 2, 2, 2],
        'RXSK12M_A': [2, 2, 1, 1]
    })

    result = clean_nhis_data(test_df)

    assert len(result) == 2
    assert pd.isna(result.iloc[1]['HICOV_A'])
    assert pd.isna(result.iloc[1]['MEDDL12M_A'])
    assert pd.isna(result.iloc[1]['MEDNG12M_A'])
    assert result.iloc[0]['INCOME_GROUP'] == 'Poor (<100% FPL)'
    assert result.iloc[1]['INCOME_GROUP'] == 'High Income (400%+)'


def main() -> None:
    test_clean_nhis_data()
    print("All NHIS tests passed!")


if __name__ == '__main__':
    main()
