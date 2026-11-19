import pandas as pd
import matplotlib.pyplot as plt


H255_FILE = 'H255.csv'
H256_FILE = 'h256.csv'


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
        'RXEXP24',
        'RXSLF24',
        'RXTOT24'
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

    quantitative = [
        'RXEXP24',
        'RXSLF24',
        'RXTOT24'
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


def plot_prescription_use(data):
    """
    Creates and saves a bar chart of mean annual prescription
    fills and refills by insurance group.

    Parameters:
        data: A pandas DataFrame containing insurance and
        prescription use data.

    Returns:
        None.
    """
    labeled = add_insurance_labels(data)

    means = (
        labeled.groupby('Insurance')['RXTOT24']
        .mean()
        .reindex(
            ['Private', 'Public Only', 'Uninsured']
        )
    )

    means.plot(kind='bar')

    plt.title(
        'Mean Annual Prescription Fills '
        'by Insurance'
    )
    plt.xlabel('Insurance Coverage')
    plt.ylabel(
        'Mean Annual Prescription Fills/Refills'
    )
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        'merged_prescription_use_by_insurance.png'
    )

    plt.clf()


def main():
    """
    Loads the MEPS datasets, creates the merged dataset,
    performs the EDA, and generates the visualizations.

    Parameters:
        None.

    Returns:
        None.
    """
    h255 = pd.read_csv(H255_FILE)
    h256 = pd.read_csv(H256_FILE)

    merged = create_merged_dataset(
        h255,
        h256
    )

    analyze_merged_data(merged)

    plot_out_of_pocket(merged)
    plot_prescription_use(merged)


if __name__ == '__main__':
    main()