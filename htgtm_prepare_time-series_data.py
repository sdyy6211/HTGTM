
import numpy as np
import pandas as pd
import datetime
import pickle
import os

def read_timeseries_data(path):
    data = pd.read_csv(path,index_col = 0)
    data.task_start_time = pd.to_datetime(data.task_start_time)
    data.task_end_time = pd.to_datetime(data.task_end_time)
    data.loc[:,'task_len'] = (data.loc[:,'task_end_time'] - data.loc[:,'task_start_time']).apply(lambda x: x.total_seconds()/60)
    return data

def user_timeseries_data_extract(x):
    coordinates = x.loc[:,['task_start_date','longitude','latitude']].groupby('task_start_date').mean()
    task_count = x.loc[:,['task_start_date','task_id']].groupby('task_start_date').count()
    task_length = x.loc[:,['task_start_date','task_len']].groupby('task_start_date').sum()
    return pd.concat([coordinates,task_count,task_length],axis=1)

def get_ts_data(data_train,data_test,user_id_remapping,dates,dates_mapping):
    data_total = pd.concat([data_train,data_test]).groupby('user_id')
    index_total = data_total.nth(0).index.sort_values()

    agg_df = data_total.apply(user_timeseries_data_extract)
    tsmean = agg_df.droplevel(1).mean(axis = 0)
    tsstd = agg_df.droplevel(1).std(axis = 0)

    datavalues = np.zeros((len(user_id_remapping),dates.shape[0],4))
    time_stamps = np.zeros((len(user_id_remapping),dates.shape[0],1))
    masks = np.zeros((len(user_id_remapping),dates.shape[0],1))

    for i in index_total:
        onesample = user_timeseries_data_extract(data_total.get_group(i))
        onesample_time_stamp = [dates_mapping[i] for i in onesample.index]
        # padding the time stamp (0 is the padding value)
        onesample_time_stamp = np.pad(onesample_time_stamp,(0,dates.shape[0]-len(onesample_time_stamp)),'constant',constant_values=0).reshape(1,-1,1)

        onesample_datavalues = (onesample.values - tsmean.values)/(tsstd.values)
        # padding the data
        onesample_datavalues = np.pad(onesample_datavalues,((0,dates.shape[0]-onesample_datavalues.shape[0]),(0,0)),'constant',constant_values=0).reshape(1,-1,4)

        # generate a mask for the padding values
        onesample_mask = np.zeros(onesample_time_stamp.shape).reshape(1,-1,1)
        onesample_mask[onesample_time_stamp==0] = 1

        datavalues[user_id_remapping[i],:,:] = onesample_datavalues
        time_stamps[user_id_remapping[i],:,:] = onesample_time_stamp
        masks[user_id_remapping[i],:,:] = onesample_mask

    return datavalues,time_stamps,masks

if __name__ == '__main__':

    data_train = read_timeseries_data(r'.\retention_train_feat.csv')
    data_test = read_timeseries_data(r'.\retention_test_feat.csv')

    #load user_id_remapping from user_id_remapping.pkl using pickle
    with open(r'.\user_id_remapping.pkl','rb') as f:
        user_id_remapping = pickle.load(f)

    lastdate = data_train.task_start_date.sort_values().iloc[-1]
    firstdate = data_train.task_start_date.sort_values().iloc[0]

    dates = pd.date_range(firstdate, lastdate)
    # create a mapping that maps dates to indices ranging from 0 to len(dates)-1
    dates_mapping = dict(zip(dates.astype('str'), range(1,len(dates)+1)))
    date_df = pd.DataFrame(np.zeros((dates.shape[0],4)), index = dates,columns=['longitute','latitude','task_count','task_length'])

    datavalues,time_stamps,masks = get_ts_data(data_train,data_test,user_id_remapping,dates,dates_mapping)

    if not os.path.exists(r'.\temp'):
        os.mkdir(r'.\temp')

    # save the data
    np.save(r'.\temp\datavalues_full_std.npy',datavalues)
    np.save(r'.\temp\time_stamps_full_std.npy',time_stamps)
    np.save(r'.\temp\masks_full_std.npy',masks)

    print('finished processing time-series data')