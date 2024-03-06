import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2
from datetime import datetime, timedelta
import scipy as sp
from scipy import stats as st
from scipy.stats import kruskal
from sklearn.model_selection import PredefinedSplit


def datetime_transformer(df, datetime_vars):
    """
    The datetime transformer

    Parameters
    ----------
    df : the dataframe
    datetime_vars : the datetime variables
    
    Returns
    ----------
    The dataframe where datetime_vars are transformed into the following 6 datetime types:
    year, month, day, hour, minute and second
    """
    
    # The dictionary with key as datetime type and value as datetime type operator
    dict_ = {'year'   : lambda x : x.dt.year,
             'month'  : lambda x : x.dt.month,
             'day'    : lambda x : x.dt.day,
             'hour'   : lambda x : x.dt.hour,
             'minute' : lambda x : x.dt.minute,
             'second' : lambda x : x.dt.second}
    
    # Make a copy of df
    df_datetime = df.copy(deep=True)
    
    # For each variable in datetime_vars
    for var in datetime_vars:
        # Cast the variable to datetime
        df_datetime[var] = pd.to_datetime(df_datetime[var])
        
        # For each item (datetime_type and datetime_type_operator) in dict_
        for datetime_type, datetime_type_operator in dict_.items():
            # Add a new variable to df_datetime where:
            # the variable's name is var + '_' + datetime_type
            # the variable's values are the ones obtained by datetime_type_operator
            df_datetime[var + '_' + datetime_type] = datetime_type_operator(df_datetime[var])
            
    # Remove datetime_vars from df_datetime
    df_datetime = df_datetime.drop(columns=datetime_vars)
                
    return df_datetime


def create_severity(df):
    """
    Create a new feature named 'Severity'

    Parameters
    ----------
    df : the dataframe
    
    Returns
    ----------
    The dataframe where 'Severity' has 3 different values:
    3: The number of fatal injuries >= 1 or the number of total injuries >= 3.
    2: The number of total injuries is more than one and less than three.
    1: The number of total injuries is zero.
    """

    # make a copy of df
    df_severity = df.copy(deep=True)
    
    # make an initial severity series
    series_severity = pd.Series([1]*df.shape[0], index=df_severity.index)
    
    # update series_severity
    for i in range(df.shape[0]):
        if df["INJURIES_FATAL"].iloc[i] > 0 or df["INJURIES_TOTAL"].iloc[i] > 2:
            series_severity.iloc[i] = 3
        elif df["INJURIES_TOTAL"].iloc[i] > 0:
            series_severity.iloc[i] = 2
            
    df_severity["Severity"] = series_severity
    
    return df_severity


def independence_test(df, feature1, feature2):
    """
    Performe the test of independence

    Parameters
    ----------
    df : the dataframe
    feature1 : the name of feature for which the test is performed
    feature2 : the name of feature for which the test is performed
    
    Returns
    ----------
    Tuple which contains chi-statistic, pvalue, degree of freedom, and expected frequencies
    """
    # initialize the cross table
    cross_table = []

    # update the cross table
    for i in df[feature1].unique():
        temp = []
        for j in df[feature2].unique():
            temp.append(df[(df[feature1]==i) & (df[feature2]==j)].shape[0])
        cross_table.append(temp)
        
    return chi2_contingency(np.array(cross_table))


def independence_test_summary(df, significance):
    """
    Create a dataframe containing the results of the test of independence

    Parameters
    ----------
    df : the dataframe
    significance : significance level
    
    Returns
    ----------
    Dataframe which contains feature name, chi-statistic, critical statistic, and p-value
    """
    cat_features = df.dtypes[df.dtypes==object].index
    stat = []
    crit = []
    pval = []
    for feature in cat_features:
        temp = independence_test(df, feature, "Severity")
        stat.append(temp[0])
        crit.append(chi2.ppf(1-significance, df=temp[2]))
        pval.append(temp[1])
    
    chi_df = pd.DataFrame(data=[cat_features, stat, crit, pval], 
                          index=["Feature", "Chi-stat", "Critical stat", "p-value"]).T

    return chi_df


def create_initial_df(original_df, start_year, start_month, start_day, end_year, end_month, end_day):
    time_frame = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-14", "14-16", "16-18", "18-20", "20-22", "22-24"]
    week_name = [2, 3, 4, 5, 6, 7, 1]

    start_date = datetime(start_year, start_month, start_day)
    end_date = datetime(end_year, end_month, end_day)
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    Year, Month, Day, Time_frame, Day_of_week, Severity, Count = [], [], [], [], [], [], []

    for severity in [1, 2, 3]:
        for day in dates:
            Year += [day.year] * 12
            Month += [day.month] * 12
            Day += [day.day] * 12
            Time_frame += time_frame
            Day_of_week += [week_name[day.weekday()]] * 12
            Severity += [severity] * 12
            Count += [0] * 12
        
    initial_df = pd.DataFrame(data=[Year, Month, Day, Time_frame, Day_of_week, Severity], 
                              index=["Year", "Month", "Day", "Time_frame", "Day_of_week", "Severity"]).T

    DF = initial_df.copy(deep=True)
    DF["Community_area"] = [original_df.Community_area.unique()[0]] * DF.shape[0]

    for comm in original_df.Community_area.unique()[1:]:
        df_temp = initial_df.copy(deep=True)
        df_temp["Community_area"] = [comm] * df_temp.shape[0]
        DF = pd.concat([DF, df_temp], ignore_index=True)

    return DF


def calculate_time_frame(hour):
    return f"{hour // 2 * 2}-{hour // 2 * 2 + 2}"


def create_count_df(initial_df, original_df):

    # create "Time_frame" 
    df = original_df.copy(deep=True)
    df['Time_frame'] = df['CRASH_DATE_hour'].apply(calculate_time_frame)

    # groupby
    grouped = df.groupby(['CRASH_DATE_year', 'CRASH_DATE_month', 'CRASH_DATE_day', 'Time_frame', 'Severity', "Community_area"]).size().reset_index(name='Count')
    grouped.rename(columns={'CRASH_DATE_year': 'Year', 'CRASH_DATE_month': 'Month', 'CRASH_DATE_day': 'Day'}, inplace=True)

    # merge
    result = pd.merge(initial_df, grouped, left_on=['Year', 'Month', 'Day', 'Time_frame', 'Severity', 'Community_area'], 
                      right_on=['Year', 'Month', 'Day', 'Time_frame', 'Severity', 'Community_area'], how='left')

    # replace NAN with 0
    result['Count'] = result['Count'].fillna(0).astype(int)

    return result


def normal_test(df, numerical_feature):
    nomal = True
    for s in [1, 2, 3]:
        temp = df[df["Severity"]==s][numerical_feature]
        if st.shapiro(temp)[1] < 0.05:
            nomal = False
            
    if nomal:
        print(f"{numerical_feature} is normaly distributed")
    else:
        print(f"{numerical_feature} is NOT normaly distributed")
        
        
def kruskal_test(df, numerical_features):
    kruskal_stat, pval = [], []
    for feature in numerical_features:
        statistic, p_value = kruskal(df[df["Severity"]==1][feature], df[df["Severity"]==2][feature], df[df["Severity"]==3][feature])
        kruskal_stat.append(statistic)
        pval.append(p_value)
        
    result = pd.DataFrame(data=[numerical_features, kruskal_stat, pval], index=["Feature", "Kruskal-stat", "p-value"]).T
    
    return result


def cyclic_encoding(df, feature_list):
    """
    Add new features, which are encoded using trigonometric function

    Parameters
    ----------
    df : the dataframe
    feature_list : the list of categorical features which shows cyclic behavior
    
    Returns
    ----------
    Dataframe which contains new encoded features instead of the original features
    """
    df_encoded = df.copy(deep=True)

    for feature in feature_list:
        df_encoded = df_encoded.astype({feature: int})
        df_encoded[f"{feature}_sin"] = np.sin(df_encoded[feature] * (2.0 * np.pi / df_encoded[feature].max()))
        df_encoded[f"{feature}_cos"] = np.cos(df_encoded[feature] * (2.0 * np.pi / df_encoded[feature].max()))
        df_encoded.drop(columns=[feature], inplace=True)
        
    return df_encoded


def get_train_val_ps(X_train, y_train, X_val, y_val):
    """
    Get the:
    feature matrix and target velctor in the combined training and validation data
    target vector in the combined training and validation data
    PredefinedSplit
    
    Parameters
    ----------
    X_train : the feature matrix in the training data
    y_train : the target vector in the training data
    X_val : the feature matrix in the validation data
    y_val : the target vector in the validation data  

    Return
    ----------
    The feature matrix in the combined training and validation data
    The target vector in the combined training and validation data
    PredefinedSplit
    """  

    # Combine the feature matrix in the training and validation data
    X_train_val = np.vstack((X_train, X_val))

    # Combine the target vector in the training and validation data
    y_train_val = np.vstack((y_train.reshape(-1, 1), y_val.reshape(-1, 1))).reshape(-1)

    # Get the indices of training and validation data
    train_val_idxs = np.append(np.full(X_train.shape[0], -1), np.full(X_val.shape[0], 0))

    # The PredefinedSplit
    ps = PredefinedSplit(train_val_idxs)

    return X_train_val, y_train_val, ps