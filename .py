# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import t

data = pd.read_csv('QVI_data.csv')

data.head()

data.describe()

sns.set_style("ticks")
plt.rc("figure", figsize=(8, 4))
plt.rc("font", size=14)
plt.rc("axes", labelsize=14)
sns.set_palette("dark")

data.dtypes

# Convert DATE column to datetime format
data['DATE'] = pd.to_datetime(data['DATE'], format='%Y-%m-%d')
print(data['DATE'].head())

# Convert DATE column to datetime format
data['MONTH_ID'] = data['DATE'].dt.strftime('%Y-%m')
print(data['MONTH_ID'].head())

data.dtypes

data = data.sort_values(by=['DATE'])
data.head(50)

measureOverTime = data.groupby(['STORE_NBR', 'MONTH_ID']).agg(
    totSales=('TOT_SALES', 'sum'),
    nCustomers=('LYLTY_CARD_NBR', 'nunique'),
    nTxnPerCust=('TXN_ID', lambda x: x.count() / x.nunique()),
    nChipsPerTxn=('PROD_QTY', lambda x: x.sum() / x.count()),
    avgPricePerUnit=('TOT_SALES', lambda x: x.sum() / x.count())
).reset_index()

measureOverTime.head(50)

storesWithFullObs = measureOverTime.groupby('STORE_NBR').filter(lambda x: x['MONTH_ID'].nunique() == 12)['STORE_NBR'].unique()
preTrialMeasures = measureOverTime[(measureOverTime['MONTH_ID'] < '2019-02') & (measureOverTime['STORE_NBR'].isin(storesWithFullObs))]

print(storesWithFullObs)

preTrialMeasures.sort_values(by=['MONTH_ID'], inplace=True)
preTrialMeasures.head(2000)

def calculateCorrelation(inputTable, metricCol, storeComparison):
    calcCorrTable = pd.DataFrame(columns=['Store1', 'Store2', 'corr_measure'])
    storeNumbers = inputTable['STORE_NBR'].unique()
    for i in storeNumbers:
        calculatedMeasure = pd.DataFrame({
            'Store1': storeComparison,
            'Store2': i,
            'corr_measure': inputTable[inputTable['STORE_NBR'] == storeComparison][metricCol].corr(inputTable[inputTable['STORE_NBR'] == i][metricCol])
        }, index=[0])
        calcCorrTable = pd.concat([calcCorrTable, calculatedMeasure], ignore_index=True)
    return calcCorrTable

def calculateMagnitudeDistance(inputTable, metricCol, storeComparison):
    calcDistTable = pd.DataFrame(columns=['Store1', 'Store2', 'MONTH_ID', 'measure'])
    storeNumbers = inputTable['STORE_NBR'].unique()
    for i in storeNumbers:
        calculatedMeasure = pd.DataFrame({
            'Store1': storeComparison,
            'Store2': i,
            'MONTH_ID': inputTable[inputTable['STORE_NBR'] == storeComparison]['MONTH_ID'],
            'measure': abs(inputTable[inputTable['STORE_NBR'] == storeComparison][metricCol].values - inputTable[inputTable['STORE_NBR'] == i][metricCol].values)
        })
        calcDistTable = pd.concat([calcDistTable, calculatedMeasure], ignore_index=True)

    # Standardize the magnitude distance so that the measure ranges from 0 to 1
    minMaxDist = calcDistTable.groupby(['Store1', 'MONTH_ID'])['measure'].agg(['min', 'max']).reset_index().rename(columns={'min': 'minDist', 'max': 'maxDist'})
    distTable = pd.merge(calcDistTable, minMaxDist, on=['Store1', 'MONTH_ID'])
    distTable['magnitudeMeasure'] = 1 - (distTable['measure'] - distTable['minDist']) / (distTable['maxDist'] - distTable['minDist'])

    finalDistTable = distTable.groupby(['Store1', 'Store2'])['magnitudeMeasure'].mean().reset_index().rename(columns={'magnitudeMeasure': 'mag_measure'})
    return finalDistTable

# Use the function you created to calculate correlations against store 77 using total sales and number of customers.
trial_store = 77
corr_nSales = calculateCorrelation(preTrialMeasures, 'totSales', trial_store)
corr_nCustomers = calculateCorrelation(preTrialMeasures, 'nCustomers', trial_store)

# Then, use the functions for calculating magnitude.
magnitude_nSales = calculateMagnitudeDistance(preTrialMeasures, 'totSales', trial_store)
magnitude_nCustomers = calculateMagnitudeDistance(preTrialMeasures, 'nCustomers', trial_store)

# Create a combined score composed of correlation and magnitude, by first merging the correlations table with the magnitude table.
corr_weight = 0.5
score_nSales = pd.merge(corr_nSales, magnitude_nSales, on=['Store1', 'Store2'])
score_nSales['scoreNSales'] = corr_weight * score_nSales['corr_measure'] + (1 - corr_weight) * score_nSales['mag_measure']

score_nCustomers = pd.merge(corr_nCustomers, magnitude_nCustomers, on=['Store1', 'Store2'])
score_nCustomers['scoreNCust'] = corr_weight * score_nCustomers['corr_measure'] + (1 - corr_weight) * score_nCustomers['mag_measure']

# Combine scores across the drivers by first merging our sales scores and customer scores into a single table
score_Control = pd.merge(score_nSales, score_nCustomers, on=['Store1', 'Store2'])
score_Control['finalControlScore'] = score_Control['scoreNSales'] * 0.5 + score_Control['scoreNCust'] * 0.5

control_store = score_Control[score_Control['Store1'] == 77].sort_values(by='finalControlScore', ascending=False).iloc[1]['Store2']
print(control_store)

# Visual checks on trends based on the drivers
measureOverTimeSales = measureOverTime.copy()
measureOverTimeSales['Store_type'] = measureOverTimeSales['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))
measureOverTimeSales = measureOverTimeSales.groupby(['MONTH_ID', 'Store_type'])['totSales'].mean().reset_index()
measureOverTimeSales['TransactionMonth'] = pd.to_datetime(measureOverTimeSales['MONTH_ID'], format='%Y-%m')
pastSales = measureOverTimeSales[measureOverTimeSales['MONTH_ID'] < '2019-03']

# Plot total sales by month
ax1 = sns.lineplot(data=pastSales, x='TransactionMonth', y='totSales', hue='Store_type')
plt.xlabel('Month of operation')
plt.ylabel('Total sales')
plt.title('Total sales by month')

# Move hue legend to the top right corner outside plot
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

plt.show()

# Conduct visual checks on customer count trends by comparing the trial store to the control store and other stores.
measureOverTimeCusts = measureOverTime.copy()
measureOverTimeCusts['Store_type'] = measureOverTimeCusts['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))
measureOverTimeCusts = measureOverTimeCusts.groupby(['MONTH_ID', 'Store_type'])['nCustomers'].mean().reset_index()
measureOverTimeCusts['TransactionMonth'] = pd.to_datetime(measureOverTimeCusts['MONTH_ID'], format='%Y-%m')
pastCustomers = measureOverTimeCusts[measureOverTimeCusts['MONTH_ID'] < '2019-03']

# Plot number of customers by month
ax2 = sns.lineplot(data=pastCustomers, x='TransactionMonth', y='nCustomers', hue='Store_type')
plt.xlabel('Month of operation')
plt.ylabel('Number of customers')
plt.title('Number of customers by month')

# Move hue legend to the top right corner outside plot
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

plt.show()

# Scale pre-trial control sales to match pre-trial trial store sales
scalingFactorForControlSales = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == trial_store) & (preTrialMeasures['MONTH_ID'] < '2019-02')]['totSales'].sum() / preTrialMeasures[(preTrialMeasures['STORE_NBR'] == control_store) & (preTrialMeasures['MONTH_ID'] < '2019-02')]['totSales'].sum()

# Apply the scaling factor
measureOverTimeSales = measureOverTime.copy()
scaledControlSales = measureOverTimeSales[measureOverTimeSales['STORE_NBR'] == control_store].copy()
scaledControlSales['controlSales'] = scaledControlSales['totSales'] * scalingFactorForControlSales

# Calculate the percentage difference between scaled control sales and trial sales
percentageDiff = pd.merge(scaledControlSales, measureOverTimeSales[measureOverTimeSales['STORE_NBR'] == trial_store], on='MONTH_ID')
percentageDiff['percentageDiff'] = (abs(percentageDiff['controlSales'] - percentageDiff['totSales_y']) / percentageDiff[['controlSales', 'totSales_y']].mean(axis=1)) * 100

# As our null hypothesis is that the trial period is the same as the pre-trial period, let's take the standard deviation based on the scaled percentage difference in the pre-trial period
stdDev = percentageDiff[percentageDiff['MONTH_ID'] < '2019-02']['percentageDiff'].std()

# Note that there are 8 months in the pre-trial period hence 8 - 1 = 7 degrees of freedom
degreesOfFreedom = 7

# We will test with a null hypothesis of there being 0 difference between trial and control stores.
# Calculate the t-values for the trial months
percentageDiff['tValue'] = (percentageDiff['percentageDiff'] - 0) / stdDev

critical_t_value = t.isf(0.05, df=degreesOfFreedom)

# Trial and control store total sales
pastSales = measureOverTimeSales.copy()
pastSales['Store_type'] = pastSales['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else 'Control' if x == control_store else np.nan)
pastSales['TransactionMonth'] = pastSales['MONTH_ID']
pastSales = pastSales[pastSales['Store_type'].isin(['Trial', 'Control'])]

# Control store 95th percentile
pastSales_Controls95 = pastSales[pastSales['Store_type'] == 'Control'].copy()
pastSales_Controls95['totSales'] = pastSales_Controls95['totSales'] * (1 + stdDev * 2)
pastSales_Controls95['Store_type'] = 'Control 95th % confidence interval'

# Control store 5th percentile
pastSales_Controls5 = pastSales[pastSales['Store_type'] == 'Control'].copy()
pastSales_Controls5['totSales'] = pastSales_Controls5['totSales'] * (1 - stdDev * 2)
pastSales_Controls5['Store_type'] = 'Control 5th % confidence interval'

trialAssessment = pd.concat([pastSales, pastSales_Controls95, pastSales_Controls5])

# Plotting these in one nice graph
trialAssessment['TransactionMonth'] = pd.to_datetime(trialAssessment['TransactionMonth'], format='%Y-%m')
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=trialAssessment, x='TransactionMonth', y='totSales', hue='Store_type', ax=ax)
ax.axvspan(pd.to_datetime('2019-02', format='%Y-%m'), pd.to_datetime('2019-04', format='%Y-%m'), alpha=0.2, color='grey')
ax.set(xlabel='Month of operation', ylabel='Total sales', title='Total sales by month')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
plt.show()

# As our null hypothesis is that the trial period is the same as the pre-trial period, let's take the standard deviation based on the scaled percentage difference in the pre-trial period
stdDev = percentageDiff[percentageDiff['MONTH_ID'] < '2019-02']['percentageDiff'].std()
degreesOfFreedom = 7

# Trial and control store number of customers
pastCustomers = measureOverTimeCusts.copy()
pastCustomers['nCusts'] = pastCustomers.groupby(['MONTH_ID', 'Store_type'])['nCustomers'].transform('mean')
pastCustomers = pastCustomers[pastCustomers['Store_type'].isin(['Trial', 'Control'])]

# Control store 95th percentile
pastCustomers_Controls95 = pastCustomers[pastCustomers['Store_type'] == 'Control'].copy()
pastCustomers_Controls95['nCusts'] = pastCustomers_Controls95['nCusts'] * (1 + stdDev * 2)
pastCustomers_Controls95['Store_type'] = 'Control 95th % confidence interval'

# Control store 5th percentile
pastCustomers_Controls5 = pastCustomers[pastCustomers['Store_type'] == 'Control'].copy()
pastCustomers_Controls5['nCusts'] = pastCustomers_Controls5['nCusts'] * (1 - stdDev * 2)
pastCustomers_Controls5['Store_type'] = 'Control 5th % confidence interval'

trialAssessment = pd.concat([pastCustomers, pastCustomers_Controls95, pastCustomers_Controls5])

# Plot everything into one nice graph
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=trialAssessment, x='TransactionMonth', y='nCusts', hue='Store_type', ax=ax)
ax.axvspan(pd.to_datetime('2019-02', format='%Y-%m'), pd.to_datetime('2019-04', format='%Y-%m'), alpha=0.2, color='grey')
ax.set(xlabel='Month of operation', ylabel='Number of customers', title='Number of customers by month')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
plt.show()

# Calculate the metrics below as we did for the first trial store.
measureOverTime = data.groupby(['STORE_NBR', 'MONTH_ID']).agg(
    nCustomers=('LYLTY_CARD_NBR', 'nunique'),
    nTxn=('TXN_ID', 'nunique'),
    nChips=('PROD_QTY', 'sum'),
    totSales=('TOT_SALES', 'sum')
).reset_index()
measureOverTime['nTxnPerCust'] = measureOverTime['nTxn'] / measureOverTime['nCustomers']
measureOverTime['nChipsPerTxn'] = measureOverTime['nChips'] / measureOverTime['nTxn']
measureOverTime['avgPricePerUnit'] = measureOverTime['totSales'] / measureOverTime['nChips']
measureOverTime = measureOverTime.sort_values(by=['STORE_NBR', 'MONTH_ID'])

# Use the functions we created earlier to calculate correlations and magnitude for each potential control store
trial_store = 86
corr_nSales = calculateCorrelation(preTrialMeasures, 'totSales', trial_store)
corr_nCustomers = calculateCorrelation(preTrialMeasures, 'nCustomers', trial_store)
magnitude_nSales = calculateMagnitudeDistance(preTrialMeasures, 'totSales', trial_store)
magnitude_nCustomers = calculateMagnitudeDistance(preTrialMeasures, 'nCustomers', trial_store)

# Now, create a combined score composed of correlation and magnitude
corr_weight = 0.5
score_nSales = corr_nSales.copy()
score_nSales['score'] = corr_weight * score_nSales['corr_measure'] + (1 - corr_weight) * score_nSales['magnitude_measure']
score_nCustomers = corr_nCustomers.copy()
score_nCustomers['score'] = corr_weight * score_nCustomers['corr_measure'] + (1 - corr_weight) * score_nCustomers['magnitude_measure']

# Finally, combine scores across the drivers using a simple average.
score_Control = pd.merge(score_nSales, score_nCustomers, on=['Store1', 'Store2'])
score_Control['finalControlScore'] = (score_Control['score_x'] + score_Control['score_y']) / 2

# Select control stores based on the highest matching store (closest to 1 but not the store itself, i.e. the second-ranked highest store)
# Select control store for trial store 86
control_store = score_Control[score_Control['Store1'] == trial_store].sort_values(by='finalControlScore', ascending=False).iloc[1]['Store2']
print(control_store)
