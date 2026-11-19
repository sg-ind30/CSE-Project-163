"""
Tanisha Deka
CSE 163 Final Project - Part 2 (EDA)

Performs exploratory data analysis on the 2022 NHIS Sample Adult dataset,
computing missingness, variable summaries, and cost-barrier visualizations
for adults with diagnosed diabetes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

DATA_PATH = 'adult22.csv'
CAT_COLS = ['HICOV_A', 'MEDDL12M_A', 'MEDNG12M_A', 'RXDG12M_A', 'RXSK12M_A']
QUANT_COLS = ['AGEP_A', 'POVRATTC_A', 'WTFA_A']


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


def main() -> None:
    sns.set_theme(style='whitegrid')
    raw_data = pd.read_csv(DATA_PATH)
    diab_df = clean_nhis_data(raw_data)

    print_summaries(raw_data, diab_df)
    plot_income_barriers(diab_df)
    plot_insurance_rx(diab_df)
    check_chi_square_assumptions(diab_df)


if __name__ == '__main__':
    main()
