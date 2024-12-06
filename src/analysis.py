import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from collections import defaultdict
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

def make_dataframe(x):
    stacked = x.stack(spatial=('x','y'))
    transposed = stacked.transpose('spatial', 'band', 'time')
    # convert to dataframe
    df = transposed.to_dataframe(name='value').unstack(['band', 'time'])

    df = df.loc[:,('value')]
    df = df.reset_index()

    # flatten columns
    df.columns = [f'{x[0]}_{x[1]}' for x in df.columns]

    df.rename(columns={df.columns[0]:'x',df.columns[1]:'y'},inplace=True)

    cols_to_drop = ['x','y'] + list(df.loc[:,df.isnull().values.all(axis=0)].columns)
    df_drop = df.drop(cols_to_drop,axis=1)

    df = df_drop.rename(columns={'1_annual':'canopy'})
    # drop all na rows
    df = df[df.isna().sum(axis=1) == 0]

    print(df.shape)

    return df

def make_nyc_dataframes(root, borough, all_vars=False):
    first_array = xr.open_dataarray(root / 'data' / f'nyc_{borough}_hls_bands.nc')
    if all_vars == False:
        df = make_dataframe(first_array)
        df = df.loc[df.canopy != -9999,] # filter out weird no data values
        return df
    else:
        second_array = xr.open_dataarray(root / 'data' / f'nyc_{borough}_hls_indices.nc')
        a = xr.concat([first_array,second_array],dim='band')
        df = make_dataframe(a)
        df = df.loc[df.canopy != -9999,]
        return df

def get_cluster_vars(threshold,dist_linkage,data):

    cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features_names = data.columns[selected_features]

    return selected_features_names

def get_threshold_variables(df,root,city,t=None):

    df_sample = df.sample(n=12000,replace=False)
    X = df_sample.drop('canopy',axis=1)
    y = df_sample['canopy']
    X_train, _, _, _ = train_test_split(X, y, test_size=0.3) 

    # make correlation matrix using spearmans r
    corr = spearmanr(X_train).correlation
    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # convert the correlation matrix to a distance matrix 
    distance_matrix = 1 - np.abs(corr)
    # hierarchical clustering using Ward's linkage algorithm
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    if t != None:
        names = get_cluster_vars(dist_linkage=dist_linkage,threshold=t,data=X_train)
        return names

    thresh_07_names = get_cluster_vars(dist_linkage=dist_linkage,threshold=0.7, data=X_train)
    thresh_05_names = get_cluster_vars(dist_linkage=dist_linkage,threshold=0.5, data=X_train)
    thresh_03_names = get_cluster_vars(dist_linkage=dist_linkage,threshold=0.3, data=X_train)
    thresh_04_names = get_cluster_vars(dist_linkage=dist_linkage,threshold=0.4, data=X_train)
    thresh_02_names = get_cluster_vars(dist_linkage=dist_linkage,threshold=0.2, data=X_train)
    thresh_01_names = get_cluster_vars(dist_linkage=dist_linkage,threshold=0.1, data=X_train)
    thresh_005_names = get_cluster_vars(dist_linkage=dist_linkage,threshold=0.05, data=X_train)

    threshold_vars_df = pd.DataFrame({'thresh_07':[thresh_07_names],'thresh_05':[thresh_05_names],
                               'thresh_04':[thresh_04_names],
                               'thresh_03':[thresh_03_names], 
                               'thresh_02':[thresh_02_names],'thresh_01':[thresh_01_names],
                               'thresh_005':[thresh_005_names]})

    threshold_vars_df.to_csv(root / 'output' / city / f'{city}_threshold_all_vars.csv')

    return  [thresh_07_names,thresh_05_names,thresh_04_names,thresh_03_names,thresh_02_names,thresh_01_names,thresh_005_names]

def get_columns_to_train(df,word_list):
    mask = df.columns.str.contains('|'.join(word_list))
    return mask

def run_cross_val(data,threshold_vars_list,threshold=False):
    if threshold == True:
        selected_vars = ['thresh_07','thresh_05','thresh_04','thresh_03','thresh_02','thresh_01','thresh_005']
        vars_list = threshold_vars_list

    if threshold == False:
        annual = ['annual']
        april = ['annual','april']
        may = ['annual','april','may']
        june = ['annual','april','may','june']
        july = ['annual','april','may','june','july']
        august = ['annual','april','may','june','july','august']
        september = ['annual','april','may','june','july','august','september']
        october = ['annual','april','may','june','july','august','september','october']
        november = ['annual','april','may','june','july','august','september','october','november']

        selected_vars = ['annual','april','may','june','july','august','sepetember','october','november']
        vars_list = [annual,april,may,june,july,august,september,october,november]
    
    df_sample = data.sample(n=12000,replace=False)

    X = df_sample.drop('canopy',axis=1)
    y = df_sample['canopy']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3)
    
    mean_cv_rmse_list = []
    cv_sd_list = []

    for i in range(0,len(vars_list)):
        if threshold == True:
            print(len(vars_list[i]))
        selected_cols = get_columns_to_train(X,vars_list[i])
        X_tr = X_train.loc[:,selected_cols]
        #X_te = X_test.loc[:,selected_cols]

        model = RandomForestRegressor(n_estimators=500,max_features='sqrt',max_samples=0.5)

        kf = KFold(n_splits=10, shuffle=True)

        cv_scores = cross_val_score(model,X_tr,y_train,cv=kf,scoring='neg_root_mean_squared_error')
        mean_cv_rmse = np.mean(abs(cv_scores))
        cv_sd = np.std(abs(cv_scores))

        mean_cv_rmse_list.append(mean_cv_rmse)
        cv_sd_list.append(cv_sd)
        print(f'cross val loop {i} complete')
        print(mean_cv_rmse)


    scores = pd.DataFrame({'vars_added':selected_vars,'cv_rmse':mean_cv_rmse_list,'cv_sd':cv_sd_list})
    if threshold == True:
        scores['n_vars'] = [len(x) for x in vars_list]

    return scores

def make_variable_plots(bands_df,all_vars_df,threshold_df,title):

    fig, axs = plt.subplots(1,2,figsize=(12,6))
    fig.suptitle(f'{title}',fontsize=16)
    ax1 = axs[0]
    ax2 = axs[1]
    mn = np.round(np.min(all_vars_df.cv_rmse) - .002,3)
    mx = np.round(np.max(bands_df.cv_rmse) + .002,3)
        
    ax1.plot(bands_df.vars_added,bands_df.cv_rmse,color='red',label='Spectral bands')
    ax1.plot(all_vars_df.cv_rmse,color='blue',label='Spectral bands + Indices')
    ax1.set_ylabel('RMSE')
    ax1.set_ylim(mn,mx)
    ax1.set_yticks(np.arange(mn,mx,0.002))
    ax1.set_xlabel('Months')
    ax1.set_title('Time points added cumulatively to model',fontsize=10,loc='left')
    ax1.legend(title='Model Inputs')


    ax2.plot(threshold_df.n_vars,threshold_df.cv_rmse,color='black',marker='o',markersize=4)
    ax2.set_ylim(mn,mx)
    ax2.set_yticks(np.arange(mn,mx,0.002))
    ax2.set_xlabel('Number of Variables')
    ax2.set_xticks(np.arange(0,threshold_df.n_vars.max(),8))
    ax2.set_title('Variables selected through hierarchical clustering',fontsize=10,loc='left')


    plt.show()

def make_variable_plots_witherrorbars(bands_df,all_vars_df,threshold_df,title):
    fig, axs = plt.subplots(1,2,figsize=(12,6))
    fig.suptitle(f'{title}',fontsize=16)
    ax1 = axs[0]
    ax2 = axs[1]
    mn = np.round(np.min(all_vars_df.cv_rmse) - .01,3)
    mx = np.round(np.max(bands_df.cv_rmse) + .01,3)
        
    ax1.plot(bands_df.vars_added,bands_df.cv_rmse,color='red',label='Spectral bands')
    ax1.errorbar(bands_df.vars_added,bands_df.cv_rmse, yerr=bands_df.cv_sd, capsize=2, fmt="r--o",elinewidth=.5)

    ax1.errorbar(bands_df.vars_added,all_vars_df.cv_rmse, yerr=all_vars_df.cv_sd, capsize=2,fmt="b--o",elinewidth=.5)
    ax1.plot(all_vars_df.cv_rmse,color='blue',label='Spectral bands + Indices')

    ax1.set_ylabel('RMSE')
    ax1.set_ylim(mn,mx)
    ax1.set_yticks(np.arange(mn,mx,0.002))
    ax1.set_xlabel('Months')
    ax1.set_title('Time points added cumulatively to model',fontsize=10,loc='left')
    ax1.legend(title='Model Inputs')


    ax2.plot(threshold_df.n_vars,threshold_df.cv_rmse,color='black',marker='o',markersize=4)
    ax2.errorbar(threshold_df.n_vars,threshold_df.cv_rmse, yerr=threshold_df.cv_sd, capsize=2,fmt="k--o",elinewidth=.5)
    ax2.set_ylim(mn,mx)
    ax2.set_yticks(np.arange(mn,mx,0.002))
    ax2.set_xlabel('Number of Variables')
    ax2.set_xticks(np.arange(0,100,8))
    ax2.set_title('Variables selected through hierarchical clustering',fontsize=10,loc='left')


    plt.show()


