import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

csv_file = Path('C:/Users/lexpaton/Desktop/archive/vgchartz-2024.csv')

# Read the CSV file
df = pd.read_csv(csv_file)

# Display the dataframe
print(df.head())

# Analyze the data structure 
df.shape
df.info()
df.describe().T
print(df.columns)

# Fixed hot coded column issue (ignore this)
def identify_one_hot_columns(df):
    prefixes = ['console', 'genre', 'publisher', 'developer']
    one_hot_columns = []
    
    for prefix in prefixes:
        # Check columns that start with the prefix and contain '_'
        columns_with_prefix = [col for col in df.columns if col.startswith(f'{prefix}_')]
        if columns_with_prefix:
            one_hot_columns.extend(columns_with_prefix)
    
    return one_hot_columns

def revert_one_hot_encoding(df, one_hot_columns):
    prefixes = ['console', 'genre', 'publisher', 'developer']
    
    for prefix in prefixes:
        columns_to_merge = [col for col in one_hot_columns if col.startswith(f'{prefix}_')]
        
        if columns_to_merge:
            original_column = f'{prefix}'
            
            # Reverse the one-hot encoding
            df[original_column] = df[columns_to_merge].idxmax(axis=1).str.replace(f'{prefix}_', '')
            
            # Drop the one-hot encoded columns
            df.drop(columns=columns_to_merge, inplace=True)
    
    return df

# Identify one-hot encoded columns
one_hot_columns = identify_one_hot_columns(df)

# Revert one-hot encoding
df_reverted = revert_one_hot_encoding(df, one_hot_columns)

# Drop columns
columns_to_drop = ['release_date']  # Add more columns to drop 
df_reverted.drop(columns=columns_to_drop, errors='ignore', inplace=True)

# Display the updated DataFrame
print(df_reverted.head())
print(df_reverted.columns)

# -----------------------
# Data preprocessing stage
# Handle missing values 

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
categorical_columns = ['console', 'genre', 'publisher', 'developer']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Converting release date column 
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df.drop('release_date', axis=1, inplace=True)

# Fill missing values 
df.fillna(df.mean(), inplace=True)

# Checking data structure
print(df.head())

# Initialize the LabelEncoder for each categorical column
le_console = LabelEncoder()
le_genre = LabelEncoder()
le_publisher = LabelEncoder()
le_developer = LabelEncoder()

# Apply label encoding to each column
df['console'] = le_console.fit_transform(df['console'])
df['genre'] = le_genre.fit_transform(df['genre'])
df['publisher'] = le_publisher.fit_transform(df['publisher'])
df['developer'] = le_developer.fit_transform(df['developer'])

# --------------------------
# Building linear regression predictor model
# Features and target variable
X = df.drop(['total_sales', 'na_sales', 'jp_sales'], axis=1)  # Drop columns that won't be used as features
y = df['total_sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse: .2f}')
print(f'R-squared: {r2: .2f}')

# ----------------------------

new_game = {
    'console': le_console.transform(['PS4'])[0],
    'genre': le_genre.transform(['Action'])[0],
    'publisher': le_publisher.transform(['Sony'])[0],
    'developer': le_developer.transform(['Naughty Dog'])[0],
    'critic_score': 9.0,
    'total_sales': 0, # Add dummy value for total_sales since it's not used as a feature
    'na_sales': 0,    # Add dummy value for na_sales
    'jp_sales': 0,    # Add dummy value for jp_sales
    'pal_sales': 0,   # Add dummy value for pal_sales
    'other_sales': 0, # Add dummy value for other_sales
    'release_year': 2018,
    'release_month': 5
}

# Create DataFrame with the same columns as the training data
new_game_df = pd.DataFrame([new_game])

# Make sure columns match those used during training
new_game_df = new_game_df[X.columns]

# Make prediction
predicted_sales = model.predict(new_game_df)
print(f'Predicted Total Sales: {predicted_sales[0]}')