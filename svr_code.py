import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from scipy.stats import skew
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlxtend.evaluate import bias_variance_decomp


df = pd.read_csv(r"Life Expectancy Data.csv")

print(df.head())
print(df.isnull().sum())

print(df["Life expectancy "])

print(df.describe(include='all'))

# plot the frequency distribution histogram
df["Life expectancy "].plot.hist(grid=True, bins=30, rwidth=0.8)
plt.title('test')
plt.ylabel('Count')
plt.xlabel('Age')
plt.legend(facecolor='white', edgecolor='grey',
           loc='best', title='Legend', frameon=True, fontsize='medium')

plt.show()
#

# Measure the skewness our target variable
skewness = skew(df["Life expectancy "])

print('The skewness of our target variable is:', skewness)

print(df['Status'].unique())

# LabelEncoder() is used to transform categorical values to numerical

label_encode = LabelEncoder()

labels_status = label_encode.fit_transform(df['Status'])
labels_country = label_encode.fit_transform(df['Country'])

# We assign the new label encoded values to their original feature names

df['Status'] = labels_status
df['Country'] = labels_country

print("final Head")
print(df.head())


# Assuming df is your DataFrame containing the data
# Generate the correlation matrix
correlation_matrix = df.corr()

# Generate the heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix[['Life expectancy ']].sort_values(by='Life expectancy ', ascending=False),
                      vmin=-1, vmax=1, annot=True, cmap='coolwarm')
heatmap.set_title('Features Correlating with Life Expectancy', fontdict={'fontsize': 18}, pad=16)
plt.show()

# Identify features with high absolute correlation values
absolute_correlation = correlation_matrix['Life expectancy '].abs().sort_values(ascending=False)

# Print the top n features with highest absolute correlation
n = 10  # Adjust this value as per your requirement
top_features = absolute_correlation[1:n+1]  # Exclude the target variable itself
print("Top", n, "features with highest absolute correlation with Life expectancy:")
print(top_features)



#
#
df2 = pd.read_csv(r"Life Expectancy Data.csv")

# Define lists of countries for each region in Asia
asian_countries = [
    'Azerbaijan', "Afghanistan","Armenia",'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei Darussalam', 'Cambodia', 'China', 'Cyprus',
    "Democratic People's Republic of Korea", 'Georgia', 'India', 'Indonesia', 'Iran (Islamic Republic of)', 'Iraq',
    'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kyrgyzstan', 'Lao Peoples Democratic Republic', 'Lebanon',
    'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'Oman', 'Pakistan', 'Philippines', 'Qatar',
    'Republic of Korea', 'Saudi Arabia', 'Singapore', 'Sri Lanka', 'Syrian Arab Republic', 'Tajikistan', 'Thailand',
    'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Viet Nam', 'Yemen'
]
# Filter the DataFrame to include only Asian countries
df_asia = df2[df2['Country'].isin(asian_countries)]

# Save the downsized dataset to a new CSV file or use it as needed
df_asia.to_csv('asia_dataset.csv', index=False)

new_df = pd.read_csv(r'asia_dataset.csv')

# Print the head of the new edited dataset
print(new_df.head())


# Feature selection
selected_features = ['Life expectancy ','Polio','Status', ' HIV/AIDS', 'Income composition of resources', 'Adult Mortality',
                     ' thinness 5-9 years', ' thinness  1-19 years',' BMI ', 'Diphtheria ', 'Schooling']
# fit_transform fits and transform the categorical values of 'Status' & 'Country' to numerical in 1 step

labels_status = label_encode.fit_transform(new_df['Status'])
labels_country = label_encode.fit_transform(new_df['Country'])

new_df = new_df[selected_features]

# We assign the new label encoded values to their original feature names

new_df['Status'] = labels_status
new_df['Country'] = labels_country

print("final Head")
print(new_df.head())
#
# .drop() function to drop unwanted features

X = new_df.drop(columns = ['Life expectancy '], axis = 1)
Y = new_df['Life expectancy '].values


print(f"There are currently {X.shape} rows and columns in X \n {Y.shape} rows and columns in Y respectively")

# mean and standard deviation of our features

print('The mean of each feature is the following: ', '\n', X.mean(), '\n'*3,
      'The standard deviation of each feature is the following: ', '\n', X.std())

# # We filled the NaN values using the mean() function

mean = X.mean()
X.fillna(mean, inplace=True)
print(new_df.isna().sum())

# standardize and transform X using the StandardScaler().fit_transform function

standardized_data = StandardScaler().fit_transform(X)
X = standardized_data

# Split data into training and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Initialize Support Vector Regression (SVR) model
svr = SVR()

#
# Setting the values of our hyperparameters
# C (Cost Parameter)

param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['linear', 'rbf']}

# We will use a cross validation with 5 folds (cv=5)

grid_search = GridSearchCV(svr, param_grid, cv=5, n_jobs=-1)

grid_search.fit(X_train, Y_train)


# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Predict on the test data
Y_pred = grid_search.predict(X_train)

print("Train Accuracy: ", grid_search.score(X_train, Y_train))

# Predict on the test data
Y_pred = grid_search.predict(X_test)

print("Test Accuracy: ", grid_search.score(X_test, Y_test))
# Evaluate the model's performance metrics using MSE, MAE

MSE = mean_squared_error(Y_test, Y_pred)
MAE = mean_absolute_error(Y_test, Y_pred)

print("Mean Squared Error:", MSE)
print("Mean Absolute Error:", MAE)

# Estimating the Bias and Variance using bias_variance_decomp() from mlxtend.evaluate library

mse, bias, var = bias_variance_decomp(svr, X_train, Y_train, X_test, Y_test, loss='mse')

# summarize results
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)

# Assuming you have already trained your model 'svr' and obtained predictions
Y_pred_train = grid_search.predict(X_train)
Y_pred_test = grid_search.predict(X_test)

# Plot Bias-Variance Tradeoff Curve
complexity_values = np.arange(1, 2, 1)  # Adjust this range as per your requirement
train_errors = []
test_errors = []

for complexity in complexity_values:
    svr = SVR(C=complexity)
    mse, bias, var = bias_variance_decomp(svr, X_train, Y_train, X_test, Y_test, loss='mse')
    train_errors.append(bias + var)
    test_errors.append(mse)

plt.plot(complexity_values, train_errors, label='Bias + Variance (Train)')
plt.plot(complexity_values, test_errors, label='Test Error')
plt.xlabel('Complexity')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.show()

# Visualizing Bias and Variance
plt.scatter(Y_test, Y_pred_test, color='blue', label='Test Data')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Bias-Variance Visualization')
plt.legend()
plt.show()