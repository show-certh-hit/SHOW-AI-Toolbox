import  numpy as  np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn import model_selection
# from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE, mean_absolute_percentage_error as MAPE, r2_score
import xgboost  as xgb
import catboost as cb
import lightgbm as lgb
from haversine import  haversine
# color = sns.color_palette
pd.options.mode.chained_assignment = None
# pd.set_option('display.max_columns', 15)
# pd.set_option('display.max_rows', 500)
# from time import time
from datetime import datetime
# from keras.preprocessing.sequence import TimeseriesGenerator

########### LOAD THE DATA ##################

# df_1 = pd.read_csv('Linkoping_vehicle_1_2021_12.csv')
# df_1 = pd.read_csv('./csv/Linkoping_vehicle_1_2021_12.csv')
df_1 = pd.read_csv('/home/eantypas/code/SHOW-ETA-calculation/eta_regression_gb/csv/Linkoping_vehicle_1_2021_12.csv')
df_1 = df_1.iloc[1:, :]
df_2 = df_1[['record_date', 'geo_lat','geo_lon', 'speed_value', 'acceleration_value', 'outdoors_temp_value']] ##### day of week, hours, times



########### CONVERT TIMESTAMP TO TIME #########

df_2['record_date'] = pd.to_datetime(df_2['record_date'])
df_2['value_seconds'] = df_2['record_date'].dt.second + df_2['record_date'].dt.minute*60 + df_2['record_date'].dt.hour*3600
df_2['value_date'] = df_2['record_date'].dt.strftime("%Y-%m-%d, %H:%M:%S")
df_2['value_time'] = [d.time() for d in df_2['record_date']]
df_2 = df_2[df_2['record_date'].astype(str).str.contains('2021-12-29') == False]



for i in range(df_2.shape[0]):
    if '2021-12-01' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 1.0
        continue
    elif '2021-12-02' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 2.0
        continue
    elif '2021-12-03' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 3.0
        continue
    elif '2021-12-04' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 4.0
        continue
    elif '2021-12-05' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 5.0
        continue
    elif '2021-12-06' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 6.0
        continue
    elif '2021-12-07' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 7.0
        continue
    elif '2021-12-08' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 1.0
        continue
    elif '2021-12-09' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 2.0
        continue
    elif '2021-12-10' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 3.0
        continue
    elif '2021-12-11' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 4.0
        continue
    elif '2021-12-12' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 5.0
        continue
    elif '2021-12-13' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 6.0
        continue
    elif '2021-12-14' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 7.0
        continue
    elif '2021-12-15' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 1.0
        continue
    elif '2021-12-16' in df_2.iloc[i]['value_date']:
        df_2.loc[i, 'day_zone'] = 2.0
        continue
    else:
        df_2.loc[i, 'day_zone'] = 0.0
        continue




# print(df_2.head)
# print(df_2)
# exit()
zone_value1 = 7*3600 + 53*60 + 39
# zone_value2 = 10*3600 + 53*60 + 39
zone_value3 = 12*3600 + 43*60 + 7
# zone_value4 = 15*3600 + 53*60 + 39
zone_value5 = 17*3600 + 45*60 + 37


df_2['time_zone_1'] = 0
df_2['time_zone_2'] = 0
df_2['time_zone_3'] = 0

for z in range(df_2.shape[0]):
    # print(z)
    if df_2.iloc[z]['value_seconds'] > zone_value1 and df_2.iloc[z]["value_seconds"] < zone_value3:
        df_2.loc[z,'time_zone_1'] = 1
        continue
    elif df_2.iloc[z]["value_seconds"] > zone_value3 and df_2.iloc[z]["value_seconds"] < zone_value5:
        df_2.loc[z, 'time_zone_2'] = 1
        continue
    else:
        df_2.loc[z, 'time_zone_3'] = 1
        continue

# print (df_2)
# print(df_2.describe())

# exit()
#%% preprocess
#################### CALCULATE DISTANCE WITH HAVERSINE #################################

def haver_calc(i, df, stp):
    temp = []
    while i<df.shape[0]:
        temp.append(i)
        # df.loc[i+1, 'day_zone'] = k
        lat1 = df.iloc[i,1]
        lon1 = df.iloc[i,2]
        if df.iloc[i-1, 6] > df.iloc[i, 6]: ## TAKE DAYS INTO ACCOUNT
            i = i+1
            print("new day")
            print(df.iloc[i,7])
            continue
        elif (i+1 > df.shape[0]-1):
            break
        else:
            for j in range(i+1, df.shape[0]-1):
                lat2 = df.iloc[j+1,1]
                lon2 = df.iloc[j+1,2] ## problem area
                # lat2 = df.iloc[j,1]
                # lon2 = df.iloc[j,2]
                a = haversine([lon1, lat1],[lon2, lat2], unit = 'km')
                if a > float(stp): #200m
                    df.loc[j+1, 'geo_haversine'] = a
                    break
            # print(temp)
        i = 1+j
    return temp

index = 1

stop = 0.2 # stops every 200 m
A = haver_calc(index, df_2, stop)
A = [a - 1 for a in A]

df_3 = df_2.iloc[A]

seconds_diff = df_3['value_seconds'].values[1:]- df_3['value_seconds'].values[:len(df_3)-1] ## subtract current time from previous time to get time difference
seconds_diff[seconds_diff<0] = 0 ## negative values is a new day, set to 0
seconds_diff = np.insert(seconds_diff, 0, 0, axis=0) ## first value is 0 to add to dataframe

df_3['value_seconds_2'] = seconds_diff ## add seconds difference to dataframe

## drop duplicate values from dataframe
df_3.drop_duplicates(subset=['record_date'])
df_3.drop_duplicates(subset=['value_seconds'])
# print(df_3['value_seconds'])
df_3 = df_3.sort_values(by = ['record_date'])
# df_3 = df_3[df_3['geo_haversine'].notna()]
# print(df_3)
# print(df_3.shape[0]) 
df_3 = df_3[df_3['record_date'].astype(str).str.contains('2021-12-16') == False]  ##this date contains only one entry

df_3 = df_3[df_3.value_seconds_2 != 0] ## delete all entries with zero time (first entry of every new day)
df_3 = df_3[df_3.value_seconds_2 < 600] ## if time over 10 minutes, considered noise and deleted
print(df_3.describe()) ## read the statistics of the dataframe
print(df_3)

# df_3.to_csv('final_set.csv')
# temp = df_3['record_date'].dt.date
# geo_lat_prev1 = []
# geo_lon_prev1 = []
# value_seconds_2_prev1 = []
# for i in range(2, df_3.shape[0]):
#     if temp.iloc[i] > temp.iloc[i-1]:
#         i = i+1
#         print("new day")
#         print(temp.iloc[i])
#         continue
#     else:
#         # df_3.iloc['geo_lat_prev1', i] = df_3.iloc['geo_lat', i-1]
#         # df_3.iloc['geo_lon_prev1', i] = df_3.iloc['geo_lon', i-1]
#         # df_3.iloc['value_seconds_2_prev1', i] = df_3.iloc['value_seconds_2', i-1]
#         geo_lat_prev1 = df_3['geo_lat'].iloc[i-1]
#         geo_lon_prev1 = df_3['geo_lon'].iloc[i-1]
#         value_seconds_2_prev1 = df_3['value_seconds_2'].iloc[i-1]
#         # df_3['geo_lat_prev1'] = df_3['geo_lat'].shift(1)
#         # df_3['geo_lon_prev1'] = df_3['geo_lon'].shift(1)
#         # df_3['value_seconds_2_prev1'] = df_3['value_seconds_2'].shift(1)
#     df_3['geo_lat_prev1']
#     df_3['geo_lon_prev1']
#     df_3['value_seconds_2_prev1']
# print(df_3)

# df_3['geo_lat_prev1'] = df_3[df_3['geo_lat'].astype(str).str.contains('2021-12-02') == False]

df_3['geo_lat_prev1'] = df_3.groupby(df_3['record_date'].dt.date)['geo_lat'].shift(1)
df_3['geo_lon_prev1'] = df_3.groupby(df_3['record_date'].dt.date)['geo_lon'].shift(1)
df_3['value_seconds_2_prev1'] = df_3.groupby(df_3['record_date'].dt.date)['value_seconds_2'].shift(1)

df_3 = df_3[df_3['geo_lat_prev1'].notna()]
df_3 = df_3[df_3['geo_lon_prev1'].notna()]
df_3 = df_3[df_3['value_seconds_2_prev1'].notna()]
print(df_3)

df_3.to_csv('200m_10min_1shift_final_set.csv')

exit()
## set train and valitation split to 80% ##
# train_th = round(df_3.shape[0] * 0.8) # set treshold for training
# print(train_th)

# exit()
# df_3['value_seconds_2'] = df_3['value_seconds']%(df_3.iloc[1, 6]-1) ## seconds normalization from cumulative to day
# print(df_3)

# exit()

## value_seconds for cumulative time
# X1_train_1 = df_3.iloc[0:train_th,0:df_3.shape[1]]
# # print(X1_train_1)
# # X1_train = X1_train_1[['geo_lat', 'geo_lon', 'geo_haversine']]
# X1_train = X1_train_1[['geo_lat', 'geo_lon', 'speed_value', 'acceleration_value', 'outdoors_temp_value']]
# # print(X1_train)
# y1_train = X1_train_1['value_seconds']
# # print(y1_train)
# X1_test_1 = df_3.iloc[train_th:df_3.shape[0]+1,0:df_3.shape[1]]
# # print(X1_test_1)
# # X1_test = X1_test_1[['geo_lat', 'geo_lon', 'geo_haversine']]
# X1_test = X1_test_1[['geo_lat', 'geo_lon', 'speed_value', 'acceleration_value', 'outdoors_temp_value']]
# # print(X1_test)
# y1_test = X1_test_1[['value_seconds']]
# # print(y1_test)

# X1_train_1 = df_3[df_3['record_date'].astype(str).str.contains('2021-12-11') == False]
# X1_train_1 = X1_train_1[X1_train_1['record_date'].astype(str).str.contains('2021-12-13') == False]
# X1_train_1 = X1_train_1[X1_train_1['record_date'].astype(str).str.contains('2021-12-16') == False] #remove this entry from everywhere because it contains only one
# print("x1 train")
# print(X1_train_1)


X1_test_1 = df_3[df_3['record_date'].astype(str).str.contains('2021-12-01') == False]
X1_test_1 = X1_test_1[X1_test_1['record_date'].astype(str).str.contains('2021-12-02') == False]
X1_test_1 = X1_test_1[X1_test_1['record_date'].astype(str).str.contains('2021-12-03') == False]
X1_test_1 = X1_test_1[X1_test_1['record_date'].astype(str).str.contains('2021-12-05') == False]
X1_test_1 = X1_test_1[X1_test_1['record_date'].astype(str).str.contains('2021-12-06') == False]
X1_test_1 = X1_test_1[X1_test_1['record_date'].astype(str).str.contains('2021-12-07') == False]
X1_test_1 = X1_test_1[X1_test_1['record_date'].astype(str).str.contains('2021-12-08') == False]
X1_test_1 = X1_test_1[X1_test_1['record_date'].astype(str).str.contains('2021-12-09') == False]
X1_test_1 = X1_test_1[X1_test_1['record_date'].astype(str).str.contains('2021-12-10') == False]
X1_test_1 = X1_test_1[X1_test_1['record_date'].astype(str).str.contains('2021-12-14') == False]
X1_test_1 = X1_test_1[X1_test_1['record_date'].astype(str).str.contains('2021-12-16') == False] #remove this entry from everywhere because it contains only one

print("x1 test")
print(X1_test_1)


# training for validation
X1_train_1 = df_3[df_3['record_date'].astype(str).str.contains('2021-12-11') == False]
X1_train_1 = X1_train_1[X1_train_1['record_date'].astype(str).str.contains('2021-12-09') == False] #val set
X1_train_1 = X1_train_1[X1_train_1['record_date'].astype(str).str.contains('2021-12-10') == False] #val set
X1_train_1 = X1_train_1[X1_train_1['record_date'].astype(str).str.contains('2021-12-13') == False]
X1_train_1 = X1_train_1[X1_train_1['record_date'].astype(str).str.contains('2021-12-16') == False] #remove this entry from everywhere because it contains only one
print("x1 train")
print(X1_train_1)

X1_val_1 = df_3[df_3['record_date'].astype(str).str.contains('2021-12-01') == False]
X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-02') == False]
X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-03') == False]
X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-05') == False]
X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-06') == False]
X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-07') == False]
X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-08') == False]
# X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-09') == False] #val set
# X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-10') == False] #val set
X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-11') == False] # test set
X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-13') == False] # test set
X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-14') == False]
X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-16') == False] #remove this entry from everywhere because it contains only one

print("x1 val")
print(X1_val_1)
# exit()


X1_train = X1_train_1[['geo_lat', 'geo_lon', 'geo_haversine',\
     'speed_value', 'acceleration_value', 'outdoors_temp_value',\
        'day_zone', 'time_zone_1', 'time_zone_2', 'time_zone_3',\
            'geo_lat_prev1', 'geo_lon_prev1', 'value_seconds_2_prev1']] ## keep for training 
print(X1_train)
y1_train = X1_train_1['value_seconds_2'] ## training is done with seconds_diff
print(y1_train)
X1_val = X1_val_1[['geo_lat', 'geo_lon', 'geo_haversine',\
     'speed_value', 'acceleration_value', 'outdoors_temp_value',\
         'day_zone', 'time_zone_1', 'time_zone_2', 'time_zone_3',\
            'geo_lat_prev1', 'geo_lon_prev1', 'value_seconds_2_prev1']] ## validation set
print(X1_val)
y1_val = X1_val_1[['value_seconds_2']] ## validation
print(y1_val)

# exit()

#%%
def mape2(test, pred):
    mape2 = np.mean(np.abs((np.array(test) - np.array(pred)) / np.array(test))) * 100
    return mape2


########## XGBOOST VALIDATION ###############

# xgbr = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bynode=1, colsample_bytree=1, gamma=0,
#        importance_type='gain', learning_rate=0.1, max_delta_step=0,
#        max_depth=3, min_child_weight=1, missing=1, n_estimators=5,
#        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42,
#        silent=None, subsample=1, verbosity=1) 
# print(xgbr)
xgbr = xgb.XGBRegressor(base_score=0.2, verbosity=1, objective='reg:squarederror',\
      learning_rate=0.08, eval_metric='mape', max_depth=30) #,base_score=0.2,  learning_rate=0.08, eval_metric='mape', max_depth=5)
xgbr.fit(X1_train, y1_train)

print("Testing XGBoost validation performance: ")

score = xgbr.score(X1_train, y1_train)
print("Training score: ", score)

# print("Actual values: ", y1_test['value_seconds'])

scores = model_selection.cross_val_score(xgbr, X1_train, y1_train,cv=3)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = model_selection.KFold(n_splits=3, shuffle=True)
kf_cv_scores = model_selection.cross_val_score(xgbr, X1_train, y1_train, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


y1_pred = xgbr.predict(X1_val)
mse = MSE(y1_val, y1_pred)
rmse = np.sqrt(mse)
mae = MAE(y1_val, y1_pred)
mape = MAPE(y1_val, y1_pred)
y1_pred_shape = y1_pred.reshape(len(y1_pred), 1)
mape_2 = mape2(y1_val, y1_pred_shape)
r2 = r2_score(y1_val, y1_pred)

adjusted_r_squared = 1 - (1-r2)*(len(y1_val)-1)/(len(y1_val)-X1_val.shape[1]-1)

# print("Validation prediction: ", y1_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (rmse))
print("MAE: %.2f" % mae)
print("MAPE: %.2f" % mape)
print("MAPE2: %.2f" % mape_2)
print("R2: %.2f" % r2)
# print("adjusted R2: %.2f" % adjusted_r_squared)
# exit()


############ LIGHTGBM VALIDATION##############


# lgbr = lgb.LGBMRegressor(learning_rate=0.1,max_depth=-2, objective='regression')
lgbr = lgb.LGBMRegressor(learning_rate=0.15, objective='regression', boosting_type='goss', num_leaves=22, max_depth = -13, n_estimators = 600)

lgbr.fit(X1_train,y1_train,eval_set=[(X1_val,y1_val),(X1_train,y1_train)], eval_metric='rmse')
# lgbr.fit(X1_train,y1_train, eval_metric='rmse')
print("Testing LightGBM performance: ")

score = lgbr.score(X1_train, y1_train)
print("Training score: ", score)

pred = lgbr.predict(X1_val)
# print("Training prediction: ", pred)

scores2 = model_selection.cross_val_score(lgbr, X1_train, y1_train,cv=3)
print("Mean cross-validation score: %.2f" % scores2.mean())

kfold2 = model_selection.KFold(n_splits=3, shuffle=True)
kf_cv_scores2 = model_selection.cross_val_score(lgbr, X1_train, y1_train, cv=kfold2)
print("K-fold CV average score: %.2f" % kf_cv_scores2.mean())

mse = MSE(y1_val, pred)
rmse = np.sqrt(mse)
mae = MAE(y1_val, pred)
mape = MAPE(y1_val, pred)
y1_pred_shape = pred.reshape(len(pred), 1)
mape_2 = mape2(y1_val, y1_pred_shape)
r2 = r2_score(y1_val, pred)

print('MSE: {:.2f}'.format(mse))
print('RMSE: {:.2f}'.format(rmse))
print('MAE: {:.2f}'.format(mae))
print('MAPE: {:.2f}'.format(mape))
print('MAPE2: {:.2f}'.format(mape_2))
print('R2: {:.2f}'.format(r2))


############ CATBOOST VALIDATION ##############


train_dataset = cb.Pool(X1_train, y1_train) 
test_dataset = cb.Pool(X1_val, y1_val)

# cbr = cb.CatBoostRegressor(iterations=None,
#                         learning_rate=None,
#                         depth=None,
#                         l2_leaf_reg=None,
#                         model_size_reg=None,
#                         rsm=None,
#                         loss_function='RMSE',
#                         border_count=None,
#                         feature_border_type=None,
#                         per_float_feature_quantization=None,
#                         input_borders=None,
#                         output_borders=None,
#                         fold_permutation_block=None,
#                         od_pval=None,
#                         od_wait=None,
#                         od_type=None,
#                         nan_mode=None,
#                         counter_calc_method=None,
#                         leaf_estimation_iterations=None,
#                         leaf_estimation_method=None,
#                         thread_count=None,
#                         random_seed=None,
#                         use_best_model=None,
#                         best_model_min_trees=None,
#                         verbose=None,
#                         silent=None,
#                         logging_level=None,
#                         metric_period=None,
#                         ctr_leaf_count_limit=None,
#                         store_all_simple_ctr=None,
#                         max_ctr_complexity=None,
#                         has_time=None,
#                         allow_const_label=None,
#                         one_hot_max_size=None,
#                         random_strength=None,
#                         name=None,
#                         ignored_features=None,
#                         train_dir=None,
#                         custom_metric=None,
#                         eval_metric=None,
#                         bagging_temperature=None,
#                         save_snapshot=None,
#                         snapshot_file=None,
#                         snapshot_interval=None,
#                         fold_len_multiplier=None,
#                         used_ram_limit=None,
#                         gpu_ram_part=None,
#                         pinned_memory_size=None,
#                         allow_writing_files=None,
#                         final_ctr_computation_mode=None,
#                         approx_on_full_history=None,
#                         boosting_type=None,
#                         simple_ctr=None,
#                         combinations_ctr=None,
#                         per_feature_ctr=None,
#                         ctr_target_border_count=None,
#                         task_type=None,
#                         device_config=None,                        
#                         devices=None,
#                         bootstrap_type=None,
#                         subsample=None,                        
#                         sampling_unit=None,
#                         dev_score_calc_obj_block_size=None,
#                         max_depth=None,
#                         n_estimators=None,
#                         num_boost_round=None,
#                         num_trees=None,
#                         colsample_bylevel=None,
#                         random_state=None,
#                         reg_lambda=None,
#                         objective=None,
#                         eta=None,
#                         max_bin=None,
#                         gpu_cat_features_storage=None,
#                         data_partition=None,
#                         metadata=None,
#                         early_stopping_rounds=None,
#                         cat_features=None,
#                         grow_policy=None,
#                         min_data_in_leaf=None,
#                         min_child_samples=None,
#                         max_leaves=None,
#                         num_leaves=None,
#                         score_function=None,
#                         leaf_estimation_backtracking=None,
#                         ctr_history_unit=None,
#                         monotone_constraints=None,
#                         feature_weights=None,
#                         penalties_coefficient=None,
#                         first_feature_use_penalties=None,
#                         model_shrink_rate=None,
#                         model_shrink_mode=None,
#                         langevin=None,
#                         diffusion_temperature=None,
#                         posterior_sampling=None,
#                         boost_from_average=None)


cbr = cb.CatBoostRegressor(loss_function='RMSE', logging_level='Silent')


grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.01, 0.15],
        # 'depth': [2, 4, 6, 8, 10],
        'depth': [6, 8, 10],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}
cbr.grid_search(grid, train_dataset)

pred = cbr.predict(X1_val)
mse = MSE(y1_val, pred)
rmse = np.sqrt(mse)
mae = MAE(y1_val, pred)
mape = MAPE(y1_val, pred)
y1_pred_shape = pred.reshape(len(pred), 1)
mape_2 = mape2(y1_val, y1_pred_shape)

r2 = r2_score(y1_val, pred)

print("Testing CATBoost performance: ")

score = cbr.score(X1_train, y1_train)  
print("Training score: ", score)

# print("Training prediction: ", pred)

scores2 = model_selection.cross_val_score(cbr, X1_train, y1_train,cv=3)
print("Mean cross-validation score: %.2f" % scores2.mean())

kfold2 = model_selection.KFold(n_splits=3, shuffle=True)
kf_cv_scores2 = model_selection.cross_val_score(cbr, X1_train, y1_train, cv=kfold2)
print("K-fold CV average score: %.2f" % kf_cv_scores2.mean())


print('MSE: {:.2f}'.format(mse))
print('RMSE: {:.2f}'.format(rmse))
print('MAE: {:.2f}'.format(mae))
print('MAPE: {:.2f}'.format(mape))
print("MAPE2: %.2f" % mape_2)
print('R2: {:.2f}'.format(r2))

exit()


#%% training
## Value_seconds_2 for time difference
# X1_train_1 = df_3.iloc[0:train_th,0:df_3.shape[1]]
# print(X1_train_1)
# X1_train = X1_train_1[['geo_lat', 'geo_lon', 'geo_haversine']]
X1_train = X1_train_1[['geo_lat', 'geo_lon', 'geo_haversine', 'speed_value', 'acceleration_value', 'outdoors_temp_value','day_zone', 'time_zone_1', 'time_zone_2', 'time_zone_3']] ## keep for training 
# print(X1_train)
y1_train = X1_train_1['value_seconds_2'] ## training is done with seconds_diff
# print(y1_train)
# X1_test_1 = df_3.iloc[train_th:df_3.shape[0]+1,0:df_3.shape[1]]
# print(X1_test_1)
# X1_test = X1_test_1[['geo_lat', 'geo_lon', 'geo_haversine']]
X1_test = X1_test_1[['geo_lat', 'geo_lon', 'geo_haversine', 'speed_value', 'acceleration_value', 'outdoors_temp_value', 'day_zone', 'time_zone_1', 'time_zone_2', 'time_zone_3']] ## testing
# print(X1_test)
y1_test = X1_test_1[['value_seconds_2']] ## testing
# print(y1_test)


# exit()
def mape2(test, pred):
    mape2 = np.mean(np.abs((np.array(test) - np.array(pred)) / np.array(test))) * 100
    return mape2





############## REGRESSION ###############

########## XGBOOST ###############

# xgbr = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bynode=1, colsample_bytree=1, gamma=0,
#        importance_type='gain', learning_rate=0.1, max_delta_step=0,
#        max_depth=3, min_child_weight=1, missing=1, n_estimators=5,
#        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42,
#        silent=None, subsample=1, verbosity=1) 
# print(xgbr)
xgbr = xgb.XGBRegressor(base_score=0.45, verbosity=1, objective='reg:squarederror') #,base_score=0.2,  learning_rate=0.08, eval_metric='mape', max_depth=5)
xgbr.fit(X1_train, y1_train)

print("Testing XGBoost performance: ")

score = xgbr.score(X1_train, y1_train)
print("Training score: ", score)

# print("Actual values: ", y1_test['value_seconds'])

scores = model_selection.cross_val_score(xgbr, X1_train, y1_train,cv=3)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = model_selection.KFold(n_splits=3, shuffle=True)
kf_cv_scores = model_selection.cross_val_score(xgbr, X1_train, y1_train, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


y1_pred = xgbr.predict(X1_test)
# print(y1_pred)
# # print(y1_pred.ndim)
# print(y1_test)
mse = MSE(y1_test, y1_pred)
rmse = np.sqrt(mse)
mae = MAE(y1_test, y1_pred)
mape = MAPE(y1_test, y1_pred)
# mape_2 = mape2(y1_test, y1_pred)
y1_pred_shape = y1_pred.reshape(len(y1_pred), 1)
mape_2 = mape2(y1_test, y1_pred_shape)
r2 = r2_score(y1_test, y1_pred)

adjusted_r_squared = 1 - (1-r2)*(len(y1_test)-1)/(len(y1_test)-X1_test.shape[1]-1)

# print("Training prediction: ", y1_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (rmse))
print("MAE: %.2f" % mae)
print("MAPE: %.2f" % mape)
# print("MAPE2: %.2f" % mape_2)
print("MAPE2: %.2f" % mape_2)
print("R2: %.2f" % r2)
print("adjusted R2: %.2f" % adjusted_r_squared)
# exit()

############ LIGHTGBM ##############


# lgbr = lgb.LGBMRegressor(learning_rate=0.1,max_depth=-2, objective='regression')
lgbr = lgb.LGBMRegressor(learning_rate=0.15, objective='regression', boosting_type='goss', num_leaves=22, max_depth = -13, n_estimators = 600)

lgbr.fit(X1_train,y1_train,eval_set=[(X1_test,y1_test),(X1_train,y1_train)], eval_metric='rmse')
# lgbr.fit(X1_train,y1_train, eval_metric='rmse')
print("Testing LightGBM performance: ")

score = lgbr.score(X1_train, y1_train)
print("Training score: ", score)

pred = lgbr.predict(X1_test)
# print("Training prediction: ", pred)

scores2 = model_selection.cross_val_score(lgbr, X1_train, y1_train,cv=3)
print("Mean cross-validation score: %.2f" % scores2.mean())

kfold2 = model_selection.KFold(n_splits=3, shuffle=True)
kf_cv_scores2 = model_selection.cross_val_score(lgbr, X1_train, y1_train, cv=kfold2)
print("K-fold CV average score: %.2f" % kf_cv_scores2.mean())

mse = MSE(y1_test, pred)
rmse = np.sqrt(mse)
mae = MAE(y1_test, pred)
mape = MAPE(y1_test, pred)
y1_pred_shape = pred.reshape(len(pred), 1)
mape_2 = mape2(y1_test, y1_pred_shape)
r2 = r2_score(y1_test, pred)

print('MSE: {:.2f}'.format(mse))
print('RMSE: {:.2f}'.format(rmse))
print('MAE: {:.2f}'.format(mae))
print('MAPE: {:.2f}'.format(mape))
print('MAPE2: {:.2f}'.format(mape_2))
print('R2: {:.2f}'.format(r2))

# exit()
############ CATBOOST ##############


train_dataset = cb.Pool(X1_train, y1_train) 
test_dataset = cb.Pool(X1_test, y1_test)

# cbr = cb.CatBoostRegressor(iterations=None,
#                         learning_rate=None,
#                         depth=None,
#                         l2_leaf_reg=None,
#                         model_size_reg=None,
#                         rsm=None,
#                         loss_function='RMSE',
#                         border_count=None,
#                         feature_border_type=None,
#                         per_float_feature_quantization=None,
#                         input_borders=None,
#                         output_borders=None,
#                         fold_permutation_block=None,
#                         od_pval=None,
#                         od_wait=None,
#                         od_type=None,
#                         nan_mode=None,
#                         counter_calc_method=None,
#                         leaf_estimation_iterations=None,
#                         leaf_estimation_method=None,
#                         thread_count=None,
#                         random_seed=None,
#                         use_best_model=None,
#                         best_model_min_trees=None,
#                         verbose=None,
#                         silent=None,
#                         logging_level=None,
#                         metric_period=None,
#                         ctr_leaf_count_limit=None,
#                         store_all_simple_ctr=None,
#                         max_ctr_complexity=None,
#                         has_time=None,
#                         allow_const_label=None,
#                         one_hot_max_size=None,
#                         random_strength=None,
#                         name=None,
#                         ignored_features=None,
#                         train_dir=None,
#                         custom_metric=None,
#                         eval_metric=None,
#                         bagging_temperature=None,
#                         save_snapshot=None,
#                         snapshot_file=None,
#                         snapshot_interval=None,
#                         fold_len_multiplier=None,
#                         used_ram_limit=None,
#                         gpu_ram_part=None,
#                         pinned_memory_size=None,
#                         allow_writing_files=None,
#                         final_ctr_computation_mode=None,
#                         approx_on_full_history=None,
#                         boosting_type=None,
#                         simple_ctr=None,
#                         combinations_ctr=None,
#                         per_feature_ctr=None,
#                         ctr_target_border_count=None,
#                         task_type=None,
#                         device_config=None,                        
#                         devices=None,
#                         bootstrap_type=None,
#                         subsample=None,                        
#                         sampling_unit=None,
#                         dev_score_calc_obj_block_size=None,
#                         max_depth=None,
#                         n_estimators=None,
#                         num_boost_round=None,
#                         num_trees=None,
#                         colsample_bylevel=None,
#                         random_state=None,
#                         reg_lambda=None,
#                         objective=None,
#                         eta=None,
#                         max_bin=None,
#                         gpu_cat_features_storage=None,
#                         data_partition=None,
#                         metadata=None,
#                         early_stopping_rounds=None,
#                         cat_features=None,
#                         grow_policy=None,
#                         min_data_in_leaf=None,
#                         min_child_samples=None,
#                         max_leaves=None,
#                         num_leaves=None,
#                         score_function=None,
#                         leaf_estimation_backtracking=None,
#                         ctr_history_unit=None,
#                         monotone_constraints=None,
#                         feature_weights=None,
#                         penalties_coefficient=None,
#                         first_feature_use_penalties=None,
#                         model_shrink_rate=None,
#                         model_shrink_mode=None,
#                         langevin=None,
#                         diffusion_temperature=None,
#                         posterior_sampling=None,
#                         boost_from_average=None)


cbr = cb.CatBoostRegressor(loss_function='RMSE', logging_level='Silent')


grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.01, 0.15],
        # 'depth': [2, 4, 6, 8, 10],
        'depth': [6, 8, 10],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}
cbr.grid_search(grid, train_dataset)

pred = cbr.predict(X1_test)
mse = MSE(y1_test, pred)
rmse = np.sqrt(mse)
mae = MAE(y1_test, pred)
mape = MAPE(y1_test, pred)
y1_pred_shape = pred.reshape(len(pred), 1)
mape_2 = mape2(y1_test, y1_pred_shape)

r2 = r2_score(y1_test, pred)

print("Testing CATBoost performance: ")

score = cbr.score(X1_train, y1_train)  
print("Training score: ", score)

# print("Training prediction: ", pred)

scores2 = model_selection.cross_val_score(cbr, X1_train, y1_train,cv=3)
print("Mean cross-validation score: %.2f" % scores2.mean())

kfold2 = model_selection.KFold(n_splits=3, shuffle=True)
kf_cv_scores2 = model_selection.cross_val_score(cbr, X1_train, y1_train, cv=kfold2)
print("K-fold CV average score: %.2f" % kf_cv_scores2.mean())


print('MSE: {:.2f}'.format(mse))
print('RMSE: {:.2f}'.format(rmse))
print('MAE: {:.2f}'.format(mae))
print('MAPE: {:.2f}'.format(mape))
print("MAPE2: %.2f" % mape_2)
print('R2: {:.2f}'.format(r2))

# exit()
