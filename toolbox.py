import pandas as pd


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