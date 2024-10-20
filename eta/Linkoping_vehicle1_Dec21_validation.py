import  numpy as  np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn import model_selection
# from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE, mean_absolute_percentage_error as MAPE, r2_score
import xgboost  as xgb
import catboost as cb
import lightgbm as lgb

def mape2(test, pred):
    mape2 = np.mean(np.abs((np.array(test) - np.array(pred)) / np.array(test))) * 100
    return mape2


df_3 = pd.read_csv('200m_10min_1shift_final_set.csv')

# training for validation
X1_train_1 = df_3[df_3['record_date'].astype(str).str.contains('2021-12-09 | 2021-12-10 | 2021-12-11 | 2021-12-13, 2021-12-16') == False]
# X1_train_1 = X1_train_1[X1_train_1['record_date'].astype(str).str.contains('2021-12-16') == False] #remove this entry from everywhere because it contains only one
# print("x1 train")
# print(X1_train_1)

X1_val_1 = df_3[df_3['record_date'].astype(str).str.contains('2021-12-01 | 2021-12-02 | 2021-12-03 | 2021-12-05 | 2021-12-06 | \
    2021-12-07 | 2021-12-08 | 2021-12-11 | 2021-12-13 | 2021-12-14 | 2021-12-16') == False]
# X1_val_1 = X1_val_1[X1_val_1['record_date'].astype(str).str.contains('2021-12-16') == False] #remove this entry from everywhere because it contains only one

# print("x1 val")
# print(X1_val_1)

X1_train = X1_train_1[['geo_lat', 'geo_lon', 'geo_haversine',\
     'speed_value', 'acceleration_value', 'outdoors_temp_value',\
        'day_zone', 'time_zone_1', 'time_zone_2', 'time_zone_3',\
            'geo_lat_prev1', 'geo_lon_prev1', 'value_seconds_2_prev1']] ## keep for training 
# print(X1_train)
y1_train = X1_train_1['value_seconds_2'] ## training is done with seconds_diff
# print(y1_train)
X1_val = X1_val_1[['geo_lat', 'geo_lon', 'geo_haversine',\
     'speed_value', 'acceleration_value', 'outdoors_temp_value',\
         'day_zone', 'time_zone_1', 'time_zone_2', 'time_zone_3',\
            'geo_lat_prev1', 'geo_lon_prev1', 'value_seconds_2_prev1']] ## validation set
# print(X1_val)
y1_val = X1_val_1[['value_seconds_2']] ## validation
# print(y1_val)

# exit()

#%%

########## XGBOOST VALIDATION ###############

# xgbr = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bynode=1, colsample_bytree=1, gamma=0,
#        importance_type='gain', learning_rate=0.1, max_delta_step=0,
#        max_depth=3, min_child_weight=1, missing=1, n_estimators=5,
#        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42,
#        silent=None, subsample=1, verbosity=1) 
# print(xgbr)


xgbr = xgb.XGBRegressor(learning_rate=0.15, base_score=0.69, verbosity=1, objective='reg:squarederror',\
       eval_metric='mape', max_depth=3, gamma=0.1, eta = 0.1,\
        colsample_bytree=1, subsample=1, min_child_weight=1) #, seed=50
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
lgbr = lgb.LGBMRegressor(learning_rate= 0.12, objective='regression', boosting_type='goss',\
     num_leaves=15, max_depth = 10, n_estimators = 550, subsample= 0.9) 

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


# exit()

############ CATBOOST VALIDATION ##############


train_dataset = cb.Pool(X1_train, y1_train) 
test_dataset = cb.Pool(X1_val, y1_val)

cbr = cb.CatBoostRegressor(loss_function='RMSE',eval_metric='MAPE', grow_policy='Depthwise', \
      learning_rate=0.08, max_depth=8, n_estimators = 80, l2_leaf_reg=6, verbose=1, \
        subsample = 1, min_child_samples = 2, random_strength=0)

# cbr = cb.CatBoostRegressor(loss_function='RMSE',eval_metric='MAPE', grow_policy='Depthwise')
# grid = {'iterations': 80,
#         'learning_rate': 0.08,
#         'depth': 8,
#         'l2_leaf_reg': 6}
# cbr.grid_search(grid, train_dataset)

cbr.fit(train_dataset, eval_set=test_dataset, early_stopping_rounds=50, plot=False, silent=True)


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

