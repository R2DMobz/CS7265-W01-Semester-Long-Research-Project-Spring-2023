import pandas as pd
import haversine as hs
from haversine import Unit
import asyncio
import os

THRESHOLD = 0.2

basedir = os.path.dirname(__file__)

def get_distance(loc1, loc2):
    return hs.haversine(loc1, loc2, unit=Unit.MILES)


async def step1_add_closest_airport_to_covid_data():
    covid_data = pd.read_csv(os.path.join(basedir, 'files', 'grouped_covid_data_lat_lang.csv'))
    airline_data = pd.read_csv(os.path.join(basedir, 'files', 'grouped_airport_data_lat_lang.csv'))
    #                        df.drop(          df[(          df['age']       >= 30) & (           df['gender'] == 'M')].index)
    airline_data = airline_data.drop(airline_data[(airline_data['Scheduled']  == 0) & (airline_data['Charter'] >= 0)].index)

    covid_data['closest_airport'] = pd.Series()
    covid_data['closest_airport_distance'] = pd.Series()
    covid_data['closest_airport_lat'] = pd.Series()
    covid_data['closest_airport_lng'] = pd.Series()

    print(covid_data.head())
    print(airline_data.head())

    for i, row in covid_data.iterrows():
        cur_lat = row['lat']
        cur_lng = row['lng']
        airline_data['result'] = airline_data.apply(
            lambda airRow: get_distance([cur_lat, cur_lng], [airRow['lat'], airRow['lng']]), axis=1)

        nearest_airport_min = airline_data['result'].idxmin()
        temp_airlineName = airline_data.loc[nearest_airport_min, 'Airport Name']
        temp_airlineDistance = airline_data.loc[nearest_airport_min, 'result']
        temp_airlineLat = airline_data.loc[nearest_airport_min, 'lat']
        temp_airlineLng = airline_data.loc[nearest_airport_min, 'lng']

        covid_data.loc[i, 'closest_airport'] = temp_airlineName
        covid_data.loc[i, 'closest_airport_distance'] = temp_airlineDistance
        covid_data.loc[i, 'closest_airport_lat'] = temp_airlineLat
        covid_data.loc[i, 'closest_airport_lng'] = temp_airlineLng

    covid_data.to_csv(os.path.join(basedir, 'files', 'covid_data_withAirport_Data_v2.csv'), index=False)


async def step2_transform_airport_to_spike_by_passengers():
    airline_data = pd.read_csv(os.path.join(basedir, 'files', 'grouped_airport_data_lat_lang.csv'))
    airline_data = airline_data.drop(airline_data[(airline_data['Scheduled']  == 0) & (airline_data['Charter'] >= 0)].index)


    airline_data['data_dte'] = airline_data['data_dte'].apply(lambda x: "20" + x.split('/')[2] + "-" + (
        "0" + x.split('/')[0] if len(x.split('/')[0]) == 1 else x.split('/')[0]))
    airline_data_pivot = airline_data.pivot(index=['Airport Name', 'STATE', 'COUNTRY'], columns=['data_dte'],
                                            values='Total')
    airline_data_pivot = airline_data_pivot.reset_index()
    airline_data_pivot.fillna(0, inplace=True)

    numeric_cols = airline_data_pivot.select_dtypes(include=['int64', 'float64']).columns
    airline_data_pivot[numeric_cols] = airline_data_pivot[numeric_cols].astype(int)

    # airline_data_transposed = airline_data.T

    print(airline_data_pivot.head())
    airline_data_pivot.to_csv(os.path.join(basedir, 'files', 'AirportData_FinalFormatted_Before_Normal.csv'))

    airline_data_Mod = airline_data_pivot.copy()

    airline_data_Mod.to_csv(os.path.join(basedir, 'files', 'AirportData_for_spike.csv'), index=False)


async def step3_transform_covid_to_spike_by_airport():
    covid_data = pd.read_csv(os.path.join(basedir, 'files', 'covid_data_withAirport_Data_v2.csv'))
    covid_data = covid_data.groupby(['closest_airport', 'closest_airport_lat', 'closest_airport_lng'], axis=0,
                                    as_index=False).sum()
    covid_data.drop(['lat', 'lng', 'closest_airport_distance'], axis=1, inplace=True)

    covid_data_Mod = covid_data.copy()

    # normalizing the spikes, setting threshold to 50 percent from previous month
    # for i, row in covid_data.iterrows():
    #     for j in range(3, covid_data_Mod.shape[1]):
    #         covid_data_Mod.iloc[i, j] = 1 if (row[j - 1] != 0) and ((row[j] - row[j - 1]) / row[j - 1]) > THRESHOLD else 0

    covid_data_Mod.to_csv(os.path.join(basedir, 'files', 'Airports_covid_data_for_spike.csv'), index=False)

async def step4_merge_covid_airport_data():
    airport = pd.read_csv(os.path.join(basedir, 'files', 'AirportData_for_spike.csv'))
    covid = pd.read_csv(os.path.join(basedir, 'files', 'Airports_covid_data_for_spike.csv'))

    covid.drop(['closest_airport_lat', 'closest_airport_lng'], axis=1, inplace=True)

    # using pandas melt function

    # formatting covid data to be airport name, month, covid spike
    covid_pivot = pd.melt(covid, id_vars=['closest_airport'], value_vars=covid.columns[1:])
    covid_pivot.columns = ['AIRPORT_NAME', 'MONTH', 'COVID_SPIKE']

    # formatting airport data
    airport_pivot = pd.melt(airport, id_vars=['Airport Name', 'STATE', 'COUNTRY'], value_vars=airport.columns[3:])
    airport_pivot.columns = ['AIRPORT_NAME', 'STATE', 'COUNTRY', 'MONTH', 'AIRPORT_SPIKE']

    final_data_frame = airport_pivot.copy()
    final_data_frame['COVID_MONTH'] = final_data_frame['MONTH'].apply(lambda x: (x.split('-')[0] if int(x.split('-')[1]) < 12 else str(int(x.split('-')[0]) + 1)) + '-' + (str(int(int(x.split('-')[1]) % 12) + 1) if len(str(int(int(x.split('-')[1]) % 12) + 1)) > 1 else "0" + str(int(int(x.split('-')[1]) % 12) + 1)))

    final_data_frame_2 = pd.merge(final_data_frame, covid_pivot, left_on=['AIRPORT_NAME', 'COVID_MONTH'], right_on=['AIRPORT_NAME', 'MONTH'], how='left')
    final_data_frame_2.dropna(axis=0, inplace=True)
    final_data_frame_2.drop(['MONTH_y'], axis=1, inplace=True)

    final_data_frame_2.to_csv(os.path.join(basedir, 'files', 'FinalTransformedData_ReadyForClassification_v2.csv'), index=False)

    print(final_data_frame_2.head())

async def main():
    tasks = [
        asyncio.create_task(step1_add_closest_airport_to_covid_data()),
        asyncio.create_task(step2_transform_airport_to_spike_by_passengers()),
        asyncio.create_task(step3_transform_covid_to_spike_by_airport()),
        asyncio.create_task(step4_merge_covid_airport_data())
    ]
    await asyncio.gather(*tasks)
    # step1_add_closest_airport_to_covid_data()
    # step2_transform_airport_to_spike_by_passengers()
    # step3_transform_covid_to_spike_by_airport()
    # step4_merge_covid_airport_data()

if __name__ == '__main__':
    asyncio.run(main())

