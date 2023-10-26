
import numpy as np
import pandas as pd
import os

# create a function that counts the number of tasks and number of being organizer in each quarter of the total time span for each user
def count_quarter(x):
    # define 4 quarters throughout the time span
    total_len_days = lastdate-firstdate

    quarter1 = firstdate + pd.Timedelta(total_len_days/4,unit = 'd')
    quarter2 = firstdate + pd.Timedelta(total_len_days/2,unit = 'd')
    quarter3 = firstdate + pd.Timedelta(total_len_days*3/4,unit = 'd')
    # create a dataframe that contains the number of tasks and number of being organizer in each quarter
    quarter_df = pd.DataFrame(columns = ['quarter1_tasks','quarter2_tasks','quarter3_tasks','quarter4_tasks',
                                         'quarter1_organizer','quarter2_organizer','quarter3_organizer','quarter4_organizer'])
    quarter_df.loc[x.name,'quarter1_tasks'] = len(x[x.task_start_time < quarter1])
    quarter_df.loc[x.name,'quarter2_tasks'] = len(x[(x.task_start_time >= quarter1) & (x.task_start_time < quarter2)])
    quarter_df.loc[x.name,'quarter3_tasks'] = len(x[(x.task_start_time >= quarter2) & (x.task_start_time < quarter3)])
    quarter_df.loc[x.name,'quarter4_tasks'] = len(x[x.task_start_time >= quarter3])

    # number of being organizer in each quarter
    quarter_df.loc[x.name,'quarter1_organizer'] = len(x[(x.task_start_time < quarter1) & (x.is_organizer == 1)])
    quarter_df.loc[x.name,'quarter2_organizer'] = len(x[(x.task_start_time >= quarter1) & (x.task_start_time < quarter2) & (x.is_organizer == 1)])
    quarter_df.loc[x.name,'quarter3_organizer'] = len(x[(x.task_start_time >= quarter2) & (x.task_start_time < quarter3) & (x.is_organizer == 1)])
    quarter_df.loc[x.name,'quarter4_organizer'] = len(x[(x.task_start_time >= quarter3) & (x.is_organizer == 1)])
    
    return quarter_df

def get_holiday(x):

    holidays_2020 = [pd.date_range('2020-01-01','2020-01-03',freq='d'),
                     pd.date_range('2020-01-24','2020-02-02',freq='d'),
                     pd.date_range('2020-04-04','2020-04-06',freq='d'),
                     pd.date_range('2020-05-01','2020-05-05',freq='d'),
                     pd.date_range('2020-06-25','2020-01-27',freq='d'),
                     pd.date_range('2020-10-01','2020-10-08',freq='d')
                     ]
    
    holidays_2021 = [pd.date_range('2021-01-01','2021-01-03',freq='d'),
                        pd.date_range('2021-02-11','2021-02-17',freq='d'),
                        pd.date_range('2021-04-03','2021-04-05',freq='d'),
                        pd.date_range('2021-05-01','2021-05-05',freq='d'),
                        pd.date_range('2021-06-12','2021-06-14',freq='d'),
                        pd.date_range('2021-09-19','2021-09-21',freq='d'),
                        pd.date_range('2021-10-01','2021-10-07',freq='d')
                        ]
    
    holidays_2022 = [pd.date_range('2022-01-01','2022-01-03',freq='d'),
                        pd.date_range('2022-01-31','2022-02-06',freq='d'),
                        pd.date_range('2022-04-03','2022-04-05',freq='d'),
                        pd.date_range('2022-04-30','2022-05-04',freq='d'),
                        pd.date_range('2022-06-03','2022-06-05',freq='d'),
                        pd.date_range('2022-09-10','2022-09-12',freq='d'),
                        pd.date_range('2022-10-01','2022-10-07',freq='d')
                        ]
    
    holiday_dates = []
    for i in holidays_2020:
        holiday_dates.extend(i)
    for i in holidays_2021:
        holiday_dates.extend(i)
    for i in holidays_2022:
        holiday_dates.extend(i)
  
    holiday_dates = pd.to_datetime(holiday_dates)

    return x.task_start_date.isin(holiday_dates).astype(int).sum()

def get_covid_corr(x):
    
    user_specific_date = date_range_df.copy()
    user_specific_date.loc[x.task_start_date] = 1
    covid_cases = pd.concat([covid_case_gd_local, covid_case_gd_foreign], axis=1).fillna(0)
    user_specific_date = user_specific_date.merge(covid_cases, left_index=True, right_index=True, how='left')
    corr_local = user_specific_date.loc[:,['count','gdlocal_cases']].corr().iloc[0,1]
    corr_foreign = user_specific_date.loc[:,['count','gdforeign_cases']].corr().iloc[0,1]
    return pd.DataFrame(np.array([[corr_local,corr_foreign]]),columns=['corr_local','corr_foreign'])

# a function that calculates the distance between two geo points
def geo_distance(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 

    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def calculate_distance_moved(x):
    coors = x.loc[:,['longitude','latitude']].values
    unique_coors = np.unique(coors,axis=0)
    distance = 0
    for i in range(len(unique_coors)-1):
        distance+=geo_distance(unique_coors[i,1],unique_coors[i,0],unique_coors[i+1,1],unique_coors[i+1,0])
    return distance

# define 3 time periods into [20,4],[4,12],[12,20], meaning [8pm-4am],[4am-12pm],[12pm-8pm]
def time_period(x):
    if x.hour >= 20 or x.hour < 4:
        return 0
    elif x.hour >= 4 and x.hour < 12:
        return 1
    else:
        return 2
# append 0 if not exist
def append_zero(x):
    for i in range(3):
        if i not in x.index:
            x[i] = 0
    return x.sort_index()

def process_tabular_data(userlevel,user_index,total_len):

    tasks_and_orgs = userlevel.apply(lambda x: count_quarter(x)).droplevel(1).loc[user_index]
    coordinates_center = userlevel[['latitude','longitude']].mean().loc[user_index]
    coordinates_variation = userlevel[['latitude','longitude']].var().loc[user_index]
    coordinates_variation[coordinates_variation.isna()] = 0
    total_time_task = userlevel.apply(lambda x:(x.task_end_time - x.task_start_time).mean().total_seconds()/3600).loc[user_index]
    weekend_time_task = userlevel.apply(lambda x: (x.task_start_time.apply(lambda x:x.weekday() in [5,6])).sum()).loc[user_index]
    participate_length = userlevel.apply(lambda x: (x.task_end_time.max()-x.task_start_time.min()).total_seconds()/total_len).loc[user_index]
    taks_time_period = userlevel.apply(lambda x:append_zero(x.task_start_time.apply(time_period).value_counts())).loc[user_index]

    distance_moved = userlevel.apply(lambda x:calculate_distance_moved(x)).loc[user_index]
    active_days = userlevel.apply(lambda x:x.task_start_date.nunique()).loc[user_index]
    tasks_at_same_day = userlevel.apply(lambda x:(x.groupby('task_start_date').agg('count').task_id>1).sum()).loc[user_index]

    covid_corr = userlevel.apply(lambda x:get_covid_corr(x)).droplevel(1).loc[user_index]

    user_public_holidays = userlevel.apply(lambda x:get_holiday(x)).loc[user_index]

    weekend_time_task = weekend_time_task+user_public_holidays

    processed_data = pd.concat([
                            tasks_and_orgs,
                            coordinates_center.rename({'latitude':'latitude_mean','longitude':'longitude_mean'},axis = 1),
                            coordinates_variation.rename({'latitude':'latitude_var','longitude':'longitude_var'},axis = 1),
                            total_time_task.rename('total_time_task'),
                            taks_time_period.rename(columns={0:'time_period_0',1:'time_period_1',2:'time_period_2'}),
                            weekend_time_task.rename('weekend_time_task'),
                            participate_length.rename('participate_length'),
                            distance_moved.rename('distance_moved'),
                            active_days.rename('active_days'),
                            tasks_at_same_day.rename('tasks_at_same_day'),
                            covid_corr.rename(columns={'corr_local':'covid_corr_local','corr_foreign':'covid_corr_foreign'}),
                            ],axis = 1)
    
    return processed_data

if __name__ == '__main__':

    data_train = pd.read_csv(r'.\retention_train_feat.csv',index_col = 0)
    data_test = pd.read_csv(r'.\retention_test_feat.csv',index_col = 0)

    data_train.task_start_time = pd.to_datetime(data_train.task_start_time)
    data_train.task_end_time = pd.to_datetime(data_train.task_end_time)
    data_train.task_start_date = pd.to_datetime(data_train.task_start_date)
    data_train.task_end_date = pd.to_datetime(data_train.task_end_date)

    covid_case_gd_local = pd.read_csv(r'.\gdlocal_ma.csv',index_col = 0)
    covid_case_gd_local.index = pd.to_datetime(covid_case_gd_local.index)
    covid_case_gd_local.rename(columns = {'0':'gdlocal_cases'},inplace = True)
    covid_case_gd_foreign = pd.read_csv(r'.\gdforeign_ma.csv',index_col = 0)
    covid_case_gd_foreign.index = pd.to_datetime(covid_case_gd_foreign.index)
    covid_case_gd_foreign.rename(columns = {'0':'gdforeign_cases'},inplace = True)

    #create a pandas series that contains each date between the first and last date
    date_range = pd.date_range(data_train.task_start_date.sort_values().iloc[0],
                            data_train.task_end_date.sort_values().iloc[-1],freq='d')
    date_range_df = pd.DataFrame(zip(date_range,np.zeros(len(date_range))),columns = ['date','count']).set_index('date')

    lastdate = data_train.task_start_time.sort_values().iloc[-1]
    firstdate = data_train.task_start_time.sort_values().iloc[0]
    total_len = (lastdate-firstdate).total_seconds()

    userlevel = data_train.groupby('user_id')
    user_index = userlevel.nth(0).index

    # check if there is a temp folder, if not, create one
    if not os.path.exists(r'.\temp'):
        os.mkdir(r'.\temp')

    processed_tabular_data = process_tabular_data(userlevel,user_index,total_len)

    processed_tabular_data.to_csv(r'.\temp\processed_tabular_data_train.csv')

    data_test.task_start_time = pd.to_datetime(data_test.task_start_time)
    data_test.task_end_time = pd.to_datetime(data_test.task_end_time)

    total_len_test = total_len

    userlevel_test = data_test.groupby('user_id')
    user_index_test = userlevel_test.nth(0).index

    processed_tabular_data_test = process_tabular_data(userlevel_test,user_index_test,total_len_test)

    processed_tabular_data_test.to_csv(r'.\temp\processed_tabular_data_test.csv')

    print('finished processing tabular data')