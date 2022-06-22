"""This Module Contains common useful function related to pandas"""

from datetime import datetime, date
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd


def get_train_test_arrays_from_dfs(df, df_test, target):
  """
    Parameters
    ----------
      df      -> (pandas dataframe) train_dataframe
      df_test -> (pandas dataframe) test dataframe
      target  -> (str) target feature column name
    Returns
    -------
      4 numpy arrays
  """
    X_train = df.iloc[:, df.columns != target].to_numpy()
    X_test = df_test.iloc[:, df_test.columns != target].to_numpy()
    y_train = df.iloc[:, df.columns == target].to_numpy()[:, 0]
    y_test = df_test.iloc[:, df_test.columns == target].to_numpy()[:, 0]

    return X_train, X_test, y_train, y_test


def get_categorical_columns(df):
  """
    Parameters
    ----------
      df : (pandas dataframe) 
    Returns
    -------
      list of column Names
  """
    return list(df.select_dtypes(include=['object']).columns)


def get_numerical_columns(df):
  """
    Parameters
    ----------
      df : pandas dataframe
    Returns
    -------
      list of column Names
  """
    return list(df.select_dtypes(include=['float64', 'float32', 'int32', 'int64']).columns)


def encode_categorical_labels(df_train, df_test, categorical_columns):
  """
    Parameters
    ----------
      df_train             -> (pandas dataframe) train dataframe
      df_test              -> (pandas dataframe) test dataframe
      categorical_columns  -> (list) List of column Names
    Returns
    -------
      returns encoded test and train dataframes
  """
    new_label_dict = {i: [] for i in categorical_columns}

    def check_for_unknown_category(value, categories, colname):
        if value not in categories:
            if value not in new_label_dict[colname]:  # add label name that is not present in train
                new_label_dict[colname].append(value)
            return '<unknown>'  # return unknown as label is unseen label is present
        else:
            return value

    for col in categorical_columns:
        labelencoder = LabelEncoder()
        full_col = pd.concat([df_train[col], df_test[col]], ignore_index=True, axis=0)
        labelencoder.fit(full_col)
        df_test[col] = df_test[col].map(lambda l: check_for_unknown_category(l, labelencoder.classes_, col))
        labelencoder.classes_ = np.append(labelencoder.classes_, '<unknown>')
        df_train[col] = labelencoder.transform(df_train[col])
        df_test[col] = labelencoder.transform(df_test[col])

    decorate("line")
    cold_start_cols = []
    for col, values in new_label_dict.items():
        if values:
            cold_start_cols.append(col)
            print("For Feature -", col, "Following variables are new in test -", values)
            decorate("line")

    print("Columns with problem -\n", cold_start_cols)

    return df_train, df_test


def scale_numerical_columns(df_train, df_test, numerical_columns):
  """
    Parameters
    ----------
      df_train             -> (pandas dataframe) train dataframe
      df_test              -> (pandas dataframe) test dataframe
      numerical_columns    -> (list) List of column Names
    Returns
    -------
      returns encoded test and train dataframes
  """
    scaler = StandardScaler()
    full_df = pd.concat([df_train.loc[:, numerical_columns], df_test.loc[:, numerical_columns]], ignore_index=True,
                        axis=0)
    scaler.fit(full_df)
    df_train.loc[:, numerical_columns] = scaler.transform(df_train.loc[:, numerical_columns])
    df_test.loc[:, numerical_columns] = scaler.transform(df_test.loc[:, numerical_columns])

    return df_train, df_test
  
  
def split_df_by_date_range(train_date_range, test_date_range, df, date_column_to_split):
    """
    Parameters
    ----------
    train_date_range : (list) list of two dates (for training) which are of datetime datatype
    test_date_range  : (list) list of two dates (for testingg) which are of datetime datatype
    df : (pandas dataframe) Pandas DataFrame to Split.
    date_column_to_split : (str) Name of the Date Column on which split needs to be performed.
    
    Returns
    -------
      tuple of pandas dataframes (train and test dataframes)
      
    Use the following code to format date to yyyy-mm-dd Format.
    train_start_date = datetime.strptime(train_date_range[0].strftime('%Y-%m-%d'), '%Y-%m-%d')
    train_end_date = datetime.strptime(train_date_range[1].strftime('%Y-%m-%d'), '%Y-%m-%d')
    test_start_date = datetime.strptime(test_date_range[0].strftime('%Y-%m-%d'), '%Y-%m-%d')
    test_end_date = datetime.strptime(test_date_range[1].strftime('%Y-%m-%d'), '%Y-%m-%d')
    """
    
    train_start_date = train_date_range[0]
    train_end_date = train_date_range[1]

    test_start_date = test_date_range[0]
    test_end_date = test_date_range[1]

    train_cond = (df[date_column_to_split] >= train_start_date) & \
                 (df[date_column_to_split] <= train_end_date)
    test_cond = (df[date_column_to_split] >= test_start_date) & \
                (df[date_column_to_split] <= test_end_date)

    # dropping date_column_to_split
    df.drop([date_column_to_split], inplace=True, axis=1)
    print("Dropped column -", date_column_to_split)

    return df.loc[train_cond], df.loc[test_cond]
