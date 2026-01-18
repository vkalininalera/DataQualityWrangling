import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from scipy.stats import boxcox

data = pd.read_csv('AmesHousing.csv', encoding='windows-1252')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Feature selection. Appendix A
main_features = data.drop(columns=['Condition 1', 'Condition 2', 'Sale Type', 'Sale Condition',
                                   'Lot Shape', 'Alley', 'Land Contour', 'Land Slope', 'Bsmt Qual', 'Bsmt Cond',
                                   'Utilities', 'MS SubClass', 'Exter Cond', 'Mas Vnr Type',
                                   'Mas Vnr Area', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Bsmt Unf SF',
                                   'Bsmt Exposure', 'BsmtFin SF 1', 'BsmtFin SF 2',
                                   'Bsmt Full Bath', 'Bsmt Half Bath', 'Heating QC', 'Central Air',
                                   'Electrical', 'Low Qual Fin SF', 'Kitchen Qual', 'Functional',
                                   'Fireplace Qu', 'Garage Finish', 'Garage Qual', 'Garage Cond', 'Pool QC',
                                   'Paved Drive', 'Wood Deck SF'])
main_features.to_csv('main_features.csv', sep=';', index=False)
print(main_features.info())

duplicate_count = main_features.duplicated().sum()

# Handling missing values
mean_lot_frontage = round(main_features['Lot Frontage'].mean(), 2)

missing_inside = (
        main_features['Lot Frontage'].isna() &
        (main_features['Lot Config'] == 'Inside')
)

missing_other = (
        main_features['Lot Frontage'].isna() &
        main_features['Lot Config'].isin(['Corner', 'CulDSac', 'FR2', 'FR3'])
)

main_features.loc[missing_inside, 'Lot Frontage'] = mean_lot_frontage
main_features.loc[missing_other, 'Lot Frontage'] = 0

remaining_missing = main_features.index[
    main_features['Lot Frontage'].isna()
]
main_features['Total Bsmt SF'] = main_features['Total Bsmt SF'].fillna(0)
main_features['Garage Type'] = main_features['Garage Type'].fillna('NoGarage')
main_features['Garage Yr Blt'] = main_features['Garage Yr Blt'].fillna(0)
main_features['Garage Cars'] = main_features['Garage Cars'].fillna(0)
main_features['Garage Area'] = main_features['Garage Area'].fillna(0)
main_features['Fence'] = main_features['Fence'].fillna('NoFence')
main_features['Misc Feature'] = main_features['Misc Feature'].fillna('NoFeature')

# Feature transformation
main_features['SalePrice_log'] = np.log(main_features['SalePrice'])
main_features['Lot Area_log'] = np.log(main_features['Lot Area'])
main_features['Total Bsmt SF_log'] = np.log1p(main_features['Total Bsmt SF'])

# Correlation check after cleaning missing values

numeric_features = main_features.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_features.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)

attributes = ['SalePrice_log',
              'Total Bsmt SF_log', 'Year Built',
              'Overall Qual', 'Lot Area_log']
scatter_matrix(numeric_features[attributes], figsize=(12, 8))
plt.tight_layout()
plt.show()

# Figure 2
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(main_features['Lot Area'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Lot Area (original)')

sns.histplot(main_features['Lot Area_log'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Lot Area (log)')

sns.histplot(main_features['SalePrice'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('SalePrice (original)')

sns.histplot(main_features['SalePrice_log'], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('SalePrice (log)')

plt.tight_layout()
plt.show()

# Feature aggregation
main_features['TotalPorchSF'] = (
        main_features['Open Porch SF'] +
        main_features['Enclosed Porch'] +
        main_features['3Ssn Porch'] +
        main_features['Screen Porch']
)

# Feature construction
main_features['HouseAge'] = (
        main_features['Yr Sold'] - main_features['Year Built']
)
main_features['YearsSinceRemodel'] = (
        main_features['Yr Sold'] - main_features['Year Remod/Add']
)
main_features['IsRenovated'] = (
        main_features['Year Remod/Add'] > main_features['Year Built']).astype(int)

# Complexity Reduction
main_features.drop(columns=[
    'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
    '1st Flr SF', '2nd Flr SF'
], inplace=True)

# Feature transformation. Box-Cox on LotArea and SalePrice.

main_features['GrLivArea_m2'] = (main_features['Gr Liv Area'] * 0.092903).round(2)

skewness = numeric_features.skew().sort_values(ascending=False)
# print(skewness)

main_features['LotArea_boxcox'], lambda_lot = boxcox(
    main_features['Lot Area']
)

main_features['SalePrice_boxcox'], lambda_price = boxcox(
    main_features['SalePrice']
)

attributes2 = ['SalePrice_boxcox',
               'Total Bsmt SF_log', 'Year Built',
               'Overall Qual', 'LotArea_boxcox']

numeric_features = main_features.select_dtypes(
    include=['int64', 'float64']
)

scatter_matrix(
    numeric_features[attributes2],
    figsize=(12, 8)
)
plt.tight_layout()
plt.show()

# Figure 2
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(main_features['Lot Area'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Lot Area (original)')

sns.histplot(main_features['LotArea_boxcox'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Lot Area (Box–Cox)')

sns.histplot(main_features['SalePrice'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('SalePrice (original)')

sns.histplot(main_features['SalePrice_boxcox'], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('SalePrice (Box–Cox)')

plt.tight_layout()
plt.show()

main_features.to_csv('data_cleaned.csv', sep=';', index=False)
