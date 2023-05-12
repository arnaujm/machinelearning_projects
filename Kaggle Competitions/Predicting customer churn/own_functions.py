## This script defines several functions that are useful for usual tasks undertaken in projects.
## Author: Arnau Juanmarti
## Last edited: 2nd May 2023


## Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

## Function that describes the main characteristics of a pandas DataFrame.
## It is similar to pandas.info() but provides more information.
def df_summary(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Describes the main characteristics of a pandas DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to be described.

        Returns:
            df_out (pd.DataFrame): DataFrame with the main characteristics of df.
    '''
    s1 = df.dtypes.rename('Type', inplace=True)
    s2 = (df.nunique()).rename('N Unique Values', inplace=True)
    s3 = (df.isnull().sum()).rename('N Missings', inplace=True)
    s4 = (df.isnull().sum() / len(df) * 100).rename('% Missings', inplace=True)
    values_cols = {}
    for col in df.columns:
        values_cols[col] = df[col][0:5].tolist()
    s5 = pd.DataFrame(list(values_cols.items()), columns=['Variable', 'Values']).set_index('Variable')
    df_out = pd.concat([s1, s2, s3, s4, s5], axis=1)
    print('Number of rows: ' + str(len(df)))
    # print(df_out)
    return df_out

## Function that describes the information on missing values of two pandas DataFrames.
def table_missings(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    '''
    Describes the information on missing values of two pandas DataFrames.

        Parameters:
            train (pd.DataFrame): A DataFrame to be described.
            test (pd.DataFrame: Another DataFrame to be described.

        Returns:
            df_out (pd.DataFrame): DataFrame with the information on missing values of the input DataFrames.
    '''
    s1 = (train.dtypes).rename('Type', inplace=True)
    s2 = (train.nunique()).rename('N Unique Values', inplace=True)
    s3 = (round(train.isna().sum() / len(train) * 100, 2)).rename('% missings train', inplace=True)
    s4 = (test.dtypes).rename('Type', inplace=True)
    s5 = (test.nunique()).rename('N Unique Values', inplace=True)
    s6 = (round(test.isna().sum() / len(test) * 100, 2)).rename('% missings test', inplace=True)
    values_cols = {}
    for col in train.columns:
        values_cols[col] = train[col][0:5].tolist()
    s7 = pd.DataFrame(list(values_cols.items()), columns=['Variable', 'First 5 values (in train)']).set_index(
        'Variable')

    df_out = pd.concat([s1, s2, s3, s4, s5, s6, s7], axis=1)

    # print(df_out)
    return df_out


## Function for label encoding. It handles unseen categories by creating the category 'Unseen'
def label_encode_columns(df: pd.DataFrame, columns: list, encoders=None):
    '''
    Encodes the specified columns of an input DataFrame using the sklearn LabelEncoder().
    If the parameter 'encoders' is not provided, it generates a dictionary of encoders to be applied in another DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame to be encoded.
            columns (list): List of column names to be encoded.

        Returns:
            df (pd.DataFrame): Input df with the columns encoded.
            encoders (dict): Dictionary of encoders.
    '''
    if encoders is None:
        encoders = {}

        for col in columns:
            unique_values = list(df[col].unique())
            unique_values.append('Unseen')
            le = LabelEncoder().fit(unique_values)
            # Attention, this df is modified by reference
            df[col] = le.transform(df[[col]])
            encoders[col] = le

        return df, encoders

    for col in columns:
        le = encoders.get(col)
        df[col] = [x if x in le.classes_ else 'Unseen' for x in df[col]]
        df[col] = le.transform(df[[col]])

    return df, encoders


## Funtion to perform One-Hot-Encoding of specific columns of a train and a test DataFrames.
## PENDING: Include option for dropping categories
def onehotencode(train: pd.DataFrame, test: pd.DataFrame, ohe_cols: list):
    '''
    Performs One-Hot-Encoding of specific columns of a train and a test DataFrames.

        Parameters:
            train (pd.DataFrame): Training set.
            test (pd.DataFrame): Testing set.
            ohe_cols (list): List of column names to be encoded.

        Returns:
            train (pd.DataFrame): Input train with the ohe_cols encoded.
            test (pd.DataFrame): Input test with the ohe_cols encoded.
    '''
    # Separate columns to encode
    train_other = train.drop(ohe_cols, axis=1)
    train_ohe = train[ohe_cols]

    test_other = test.drop(ohe_cols, axis=1)
    test_ohe = test[ohe_cols]

    # Initialize encoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # Encode columns in train set
    train_cat_encoded = encoder.fit_transform(train_ohe)
    cat_names = [f'{col}_{cat}' for i, col in enumerate(train_ohe.columns) for cat in encoder.categories_[i]]
    train_cat_encoded_df = pd.DataFrame(train_cat_encoded, columns=cat_names)

    train = train_other.join(train_cat_encoded_df)

    # Encode columns in test set
    test_cat_encoded = encoder.transform(test_ohe)
    test_cat_encoded_df = pd.DataFrame(test_cat_encoded, columns=cat_names)

    test = test_other.join(test_cat_encoded_df)

    return train, test



## Function to plot the distribution of continuous features and their relationship with a binary target
## PENDING: Add option for only one axes subplot
def plot_cont_target_all(train: pd.DataFrame, test: pd.DataFrame, cont_cols: list, target: str, rows: int, cols: int,
                         bins1=100, bins2=50, figsize=(12, 12),
                         title='Title'):
    '''
    For each specified feature, it generates a matplotlib axes subplot with a histogram of the feature (for both the train and the
    test sets) and a scatterplot of the (binned) feature with the binary target rate.

        Parameters:
            train (pd.DataFrame): Training set.
            test (pd.DataFrame): Test set.
            cont_cols (list): List of column names of the features to plot (should be columns in train and test sets).
            target (str): Name of the target (should be a column in the train set)
            rows (int): Number of rows of the matplotlib figure.
            cols (int): Number of columns of the matplotlib figure.
            bins1 (int): Number of bins in the histogram.
            bins2 (int): Number of bins in the scatterplot.
            figsize (tuple): Size of the matplotlib figure.
            title (str): Title of the figure.
    '''

    # Generate the matplotlib subplots.
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    # Loop through the axes subplots and the features.
    for col, ax in zip(cont_cols, axs.ravel()):

        # Generate the histogram.
        mi = min(train[col].min(), test[col].min())
        ma = max(train[col].max(), test[col].max())
        if train[col].dtype == 'datetime64[ns]':
            bins1 = pd.date_range(start=mi,
                                  end=ma,
                                  periods=100)
        else:
            bins1 = np.linspace(mi, ma, 100)
        ax.hist(train[col], bins=bins1, alpha=0.5, density=True, label='train')
        ax.hist(test[col], bins=bins1, alpha=0.5, density=True, label='test')
        ax.set(xlabel=col)
        if any(i == 1 for i in [rows, cols]):
            if ax == axs[0]: ax.legend(loc='upper left')
        else:
            if ax == axs[0, 0]: ax.legend(loc='upper left')

        # Generate the scatterplot in a secondary axis.
        ax2 = ax.twinx()
        if train[col].dtype == 'datetime64[ns]':
            bins2 = pd.date_range(start=mi,
                                  end=ma,
                                  periods=50)
        else:
            bins2 = np.linspace(mi, ma, 50)
        total, _ = np.histogram(train[col], bins=bins2)
        positives, _ = np.histogram(train[col][train[target] == 1], bins=bins2)
        with warnings.catch_warnings():  # ignore divide by zero for empty bins
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if train[col].dtype == 'datetime64[ns]':
                ax2.scatter(bins2[:-1], positives / total,
                            color='green', s=12, label='target rate')
            else:
                x = (bins2[1:] + bins2[:-1]) / 2
                y = positives / total
                ax2.scatter(x, y,
                            color='green', s=12, label='target rate')
                #m, b = np.polyfit(x, y, 1)
                #ax2.plot(x, m * x + b, color='green')
        ax2.set_ylim(0, 0.5)
        ax2.tick_params(axis='y', colors='green')
        if any(i == 1 for i in [rows, cols]):
            if ax == axs[0]: ax2.legend(loc='upper right')
        else:
            if ax == axs[0, 0]: ax2.legend(loc='upper right')
        if train[col].dtype != 'datetime64[ns]':
            corr = train[col].corr(train[target])
            ax2.text(0.5, 0.7, f'corr = {round(corr, 2)}', color='black', fontsize=12, fontweight='bold', transform=ax.transAxes)

    # Tight layout and title of the figure.
    plt.tight_layout(w_pad=1)
    plt.suptitle(title, fontsize=20, y=1.02)


## Function to plot the distribution of continuous features and their relationship with a continuous target
def plot_cont_target_cont_all(train: pd.DataFrame, test: pd.DataFrame, cont_cols: list, target: str,
                              rows: int, cols: int, num_bins1=100, num_bins2=20,
                              figsize=(12, 12)):
    '''
    For each specified feature, it generates a matplotlib axes subplot with a histogram of the feature (for both the train and the
    test sets) and a scatterplot of the (binned) feature with the mean of the continuous target for each bin.

        Parameters:
            train (pd.DataFrame): Training set.
            test (pd.DataFrame): Test set.
            cont_cols (list): List of column names of the features to plot (should be columns in train and test sets).
            target (str): Name of the target (should be a column in the train set)
            rows (int): Number of rows of the matplotlib figure.
            cols (int): Number of columns of the matplotlib figure.
            num_bins1 (int): Number of bins in the histogram.
            num_bins2 (int): Number of bins in the scatterplot.
            figsize (tuple): Size of the matplotlib figure.
    '''

    # Generate the axes subplots.
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    # Loop through the axes subplots and the features.
    for col, ax in zip(cont_cols, axs.ravel()):

        # Generate the histogram.
        mi = min(train[col].min(), test[col].min())
        ma = max(train[col].max(), test[col].max())
        bins1 = np.linspace(mi, ma, num_bins1)
        ax.hist(train[col], bins=bins1, alpha=0.5, density=True, label='train')
        ax.hist(test[col], bins=bins1, alpha=0.5, density=True, label='test')
        ax.set(xlabel=col)
        if any(i == 1 for i in [rows, cols]):
            if ax == axs[0]: ax.legend(loc='upper left')
        else:
            if ax == axs[0, 0]: ax.legend(loc='upper left')

        # Generate the scatterplot in a secondary axis.
        ax2 = ax.twinx()
        bins2_means = train.groupby(pd.qcut(train[col], num_bins2)).agg(col_mean=(col, np.mean),
                                                                        target_mean=(target, np.mean))
        ax2.scatter(bins2_means['col_mean'], bins2_means['target_mean'], color='green', s=10, label='target mean')
        ax2.tick_params(axis='y', colors='green')
        m, b = np.polyfit(bins2_means['col_mean'], bins2_means['target_mean'], 1)
        ax2.plot(bins2_means['col_mean'], m * bins2_means['col_mean'] + b, color='green')
        corr = train[col].corr(train[target])
        ax2.text(0.5, 0.7, f'corr = {round(corr, 2)}', transform=ax.transAxes)
        if any(i == 1 for i in [rows, cols]):
            if ax == axs[0]: ax2.legend(loc='lower left')
        else:
            if ax == axs[0, 0]: ax2.legend(loc='lower left')

    # Tight layout and title of the figure.
    plt.tight_layout(w_pad=1)
    plt.suptitle('Distribution of continuous features and relationship with target', fontsize=20, y=1.02)


## Function to plot scatterplots of continuous features with a continuous target
def scatterplot_features_target(train: pd.DataFrame, features: list, target: str, rows: int, cols: int, figsize=(12, 12)):
    '''
    For each feature, it plots a scatterplot of the feature with the target, adding a regression line.

        Parameters:
            train (pd.DataFrame): Training set.
            features (list): List of feature names to plot (should be columns in the train set.
            target (str): Name of the target (should be a column in the train set).
            rows (int): Number of rows of the matplotlib figure.
            cols (int): Number of columns of the matplotlib figure.
            figsize (tuple): Size of the matplotlib figure.
    '''

    # Generate the matplotlib subplots.
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    # Loop through the axes subplots and the features.
    for col, ax in zip(features, axs.ravel()):

        # Generate the scatterplot and add the regression line.
        ax.scatter(train[col], train[target], color='m', s=10, alpha=0.05)
        m, b = np.polyfit(train[col], train[target], 1)
        ax.plot(train[col], m * train[col] + b, color='blue')
        ax.set(xlabel=col)
        if any(i == 1 for i in [rows, cols]):
            if ax == axs[0]: ax.set(ylabel=target)
        else:
            if ax == axs[0, 0]: ax.set(ylabel=target)
        corr = train[col].corr(train[target])
        ax.text(0.5, 0.7, f'corr = {round(corr, 2)}', transform=ax.transAxes)

    # Tight layout and title of the figure.
    plt.tight_layout(w_pad=1)
    plt.suptitle('Scatterplots of features with target', fontsize=20, y=1.02)


## Function to plot the distribution of categorical features and their relationship with a binary target
## PENDING: Add option for only one axes subplot
def plot_cat_target_all(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list, target: str,
                        rows: int, cols: int, figsize=(12, 12), title='Title'):
    '''
    For each categorical feature, it generates an axes subplot with a bar plot of the feature (for both the train and the
    test sets) and plots the target rate for each category of the feature.

        Parameters:
            train (pd.DataFrame): Training set.
            test (pd.DataFrame): Test set.
            cat_cols (list): List of feature names to plot (should be colums in the train and test sets).
            target (str): Name of the target (should be a column in the train set).
            rows (int): Number of rows of the matplotlib figure.
            cols (int): Number of columns of the matplotlib figure.
            figsize (tuple): Size of the matplotlib figure.
            title (str): Title of the figure.
    '''

    # Generate the axes subplots.
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    # Loop through the axes subplots and the features.
    for col, ax in zip(cat_cols, axs.ravel()):

        # Generate the bar plots.
        vc_train = train[col].value_counts()
        vc_test = test[col].value_counts()
        values = sorted(set(vc_train.index).union(vc_test.index))
        vc_train = vc_train.reindex(values).fillna(0)
        vc_test = vc_test.reindex(values).fillna(0)
        ax.bar(range(len(values)),
               vc_train.values / len(train),
               alpha=0.5, label='train')
        ax.bar(range(len(values)),
               vc_test.values / len(test),
               alpha=0.5, label='test')
        ax.set_xticks(range(len(values)), values)
        ax.set(xlabel=col, ylabel='proportion')
        if any(i == 1 for i in [rows, cols]):
            if ax == axs[0]: ax.legend(loc='upper left')
        else:
            if ax == axs[0, 0]: ax.legend(loc='upper left')

        # Plot the target rate for each category of the feature.
        ax2 = ax.twinx()
        mean_target = train[target].groupby(train[col]).mean().reindex(values)
        ax2.scatter(range(len(values)),
                    mean_target.values,
                    color='green', label='target rate')
        ax2.set_ylim(np.min(train[target]), np.max(train[target]))
        # ax2.set_yticks(np.arange(0, 0.6, 0.1))
        ax2.tick_params(axis='y', colors='green')
        if any(i == 1 for i in [rows, cols]):
            if ax == axs[0]: ax2.legend(loc='upper right')
        else:
            if ax == axs[0, 0]: ax2.legend(loc='upper right')

    # Tight layout and title of the figure.
    plt.tight_layout(w_pad=1)
    plt.suptitle(title, fontsize=20, y=1.02)
