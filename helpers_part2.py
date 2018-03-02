


def get_predictions(model, data_points):
    return list(map(np.round, np.exp(model.predict(data_points)) - 1))

def preprocessor(X):
    time_feature_maker(X, 'datetime')
    X['weekend'] = (X['datetime_day_of_week'].isin({6, 7})) & (X.holiday == 0)
    return None

def feature_importance_plot(feature_names, model, top_x=None):
    '''
    Ploting top x most important features
    
    I: feature name (list), model (fitted tree-based model), top x (int)
    Plots: bar plot of features importance with standard error among trees.
    
    *importances weights (normalized by sum) paired with names
    '''
    import numpy as np

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
    if top_x:
        indices = np.argsort(importances)[::-1][:top_x]
    else:
        indices = np.argsort(importances)[::-1]
    
    _plot_barchart_with_error_bars(importances[indices], std[indices], feature_names[indices])
        
    return None


def _plot_barchart_with_error_bars(importance_weights, errors, feature_names):
    '''
    I: importance_weights (np array), errors (np array), feature_names
    O: None
    does the actual plotting of feature weights
    goes with feature_importance_plot
    '''
    
    #if we don't reverse everything it'll plot it upside-down

    import matplotlib.pyplot as plt
    importance_weights_reversed = importance_weights[::-1] 
    errors_reversed = errors[::-1]
    feature_names_reversed = feature_names[::-1] 

    number_of_features = range(len(importance_weights_reversed))

    plt.figure(figsize=(12,10))
    plt.title("Feature importances")
    plt.barh(
        number_of_features, 
        importance_weights_reversed,
        color="r", 
        xerr=errors_reversed, 
        align="center"
    )
    plt.yticks(number_of_features, feature_names_reversed)
    plt.show()

def log_transform_distribution(data_points_list):
    from sklearn import preprocessing
    constant = (1 + - 1 * data_points_list.min())
    log_transformed_data = preprocessing.scale(np.log(data_points_list + constant))
    return log_transformed_data

def plot_distribution(data_points_list):
    '''
    kurtosis:
    Is about the fatness of the tails which is also indicative out of outliers.
    skew: 
    when a distribution "leans" to the right or left, it's called skewed the right/left. 
    Think of a skewer. This it's a indication of outliers that live on that side of the distribution.
    *these are both aggregate stats and very subjectable to the size of the target sample
    '''
   
    print(pd.Series(data_points_list).describe())
    
    skew, kurtosis = _skew_and_kurtosis(data_points_list)
    print ('skew -- ', skew)
    print ('kurtosis --', kurtosis)
    
    plot_transformation(data_points_list, 'no_transformation');
    plt.violinplot(
       data_points_list,
       showmeans=False,
       showmedians=True
    );

def plot_transformation(data, name_of_transformation):

    #setting up canvas
    figure = plt.figure(figsize=(10,5))
    
    plt.suptitle(name_of_transformation)
    
    figure.add_subplot(121)
    
    plt.hist(data, alpha=0.75, bins=100) 
    
    figure.add_subplot(122)
    plt.boxplot(data)
    
    plt.show()

def _skew_and_kurtosis(data_points_list): 
    from scipy.stats import skew, kurtosis
    return (skew(data_points_list), kurtosis(data_points_list))



def plot_no_transformation(data_points_scaled):
   
    print(pd.Series(data_points_scaled).describe())
    print (_skew_and_kurtosis(data_points_scaled))
    plot_transformation(data_points_scaled, 'no_transformation');
    plt.violinplot(
       data_points_scaled,
       showmeans=False,
       showmedians=True
    );


def time_feature_maker(df_with_time_series, name_of_date_time_col):
    '''
    I:dataframe, name of the time-series column (str)
    O:None, does transformation inplace
    parses the time-series column of it's different time attributes e.g. hour, day of week, year..
    forms them into individual series and puts them back into the same DataFrame.
    '''
    import pandas as pd    
    
    df_with_time_series[name_of_date_time_col] =  pd.to_datetime(df_with_time_series[name_of_date_time_col])
    
    df_with_time_series[name_of_date_time_col + '_hour'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).hour
    df_with_time_series[name_of_date_time_col + '_week'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).week
    df_with_time_series[name_of_date_time_col + '_day_of_week'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).dayofweek
    df_with_time_series[name_of_date_time_col + '_day'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).day
    df_with_time_series[name_of_date_time_col + '_month'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).month
    df_with_time_series[name_of_date_time_col + '_year'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).year
    df_with_time_series[name_of_date_time_col + '_quarter'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).quarter
    df_with_time_series[name_of_date_time_col + '_daysinmonth'] = pd.DatetimeIndex(df_with_time_series[name_of_date_time_col]).daysinmonth
    return None

def get_model_cv_scores(model, feat_matrix, labels, folds=10, scoring='r2'):
    
    '''
    I: persisted model object, feature matrix (numpy or pandas datafram), labels, k-folds, scoring metric
    O: mean of scores over each k-fold (float)
    '''
    from sklearn import model_selection
    import numpy as np
    scores = model_selection.cross_val_score(model, feat_matrix, labels, cv=folds,scoring=scoring, n_jobs=-1)
    return np.median(scores), np.mean(scores), scores

def get_errors(trained_model, feature_M, truths, stds):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    errors = [pred - truths[i] for i, pred in enumerate(trained_model.predict(feature_M))]
    
    
    scaler = StandardScaler()
    errors_scaled = scaler.fit_transform(pd.Series(errors))
    error_idx = [i for i, error in enumerate(errors_scaled) if (error < -1 * stds ) or (error > stds)]

    anomalies_idx_removed = [idx for idx in feature_M.index if idx not in error_idx]
    
    return anomalies_idx_removed, errors, error_idx

def missing_values_finder(df):
    '''
    finds missing values in a data frame returns to you the value counts
    '''
    import pandas as pd
    missing_vals_dict= {col : df[col].dropna().shape[0] / float(df[col].shape[0]) for col in df.columns}
    output_df = pd.DataFrame().from_dict(missing_vals_dict, orient='index').sort_index()
    return output_df

def get_all_cv_scores(df_train, df_test, params):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    import numpy as np
    kf = KFold(n_splits=10)

    scores_count=[]
    scores_casual=[]
    scores_registered=[]
    scores_combined=[]

    for train_idx, test_idx in kf.split(df_train[df_test.columns]):
        
        X_train = df_train[df_test.columns].ix[train_idx][:]
        preprocessor(X_train)
        X_train = X_train.drop(['datetime'], axis=1)
        y_train = df_train['count_log'].ix[train_idx]
        
        X_test = df_train[df_test.columns].ix[test_idx][:]
        preprocessor(X_test)
        X_test = X_test.drop(['datetime'], axis=1)
        y_test = df_train['count_log'].ix[test_idx]

        rf_count = RandomForestRegressor(**params)
        rf_count.fit(X_train, y_train)

        scores_count.append(r2_score(df_train['count'].ix[test_idx], np.exp(rf_count.predict(X_test)) - 1))

        y_train = df_train['registered_log'].ix[train_idx]
        
        X_test = df_train[df_test.columns].ix[test_idx][:]
        preprocessor(X_test)
        X_test_registered = X_test.drop(['datetime'], axis=1)
        y_test_registered = df_train['registered_log'].ix[test_idx]
        
        rf_registered = RandomForestRegressor(**params)
        rf_registered.fit(X_train, y_train)

        y_train = df_train['casual_log'].ix[train_idx]
        
        X_test = df_train[df_test.columns].ix[test_idx][:]
        preprocessor(X_test)
        X_test_casual = X_test.drop(['datetime'], axis=1)
        y_test_casual = df_train['casual_log'].ix[test_idx]

        rf_casual = RandomForestRegressor(**params)
        rf_casual.fit(X_train, y_train)

        y_true = df_train['count'].ix[test_idx]
        y_pred = (np.exp(rf_casual.predict(X_test_casual)) - 1) + \
            (np.exp(rf_registered.predict(X_test_registered)) - 1)
        
        scores_registered.append(r2_score(y_test_registered, rf_registered.predict(X_test_registered)))
        scores_casual.append(r2_score(y_test_casual, rf_casual.predict(X_test_casual)))
        scores_combined.append(r2_score(y_true, y_pred))
        

    return scores_count,scores_casual,scores_registered,scores_combined
