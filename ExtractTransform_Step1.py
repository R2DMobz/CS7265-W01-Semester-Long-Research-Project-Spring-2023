import pandas as pd
import os
import asyncio

basedir = os.path.dirname(__file__)


async def step4_add_lat_lng_toAirPortData():
    city_state_data = pd.read_csv(os.path.join(basedir, 'files', 'source_data', 'county_city_state.csv'))
    city_state_data['city'] = city_state_data['city'].str.upper()
    city_state_data['state_name'] = city_state_data['state_name'].str.upper()

    airline_data = pd.read_csv(os.path.join(basedir, 'files', 'grouped_airport_data_byMonth.csv'))
    airline_data['CITY'] = airline_data['CITY'].str.upper()
    airline_data['STATE'] = airline_data['STATE'].str.upper()
    airline_data = airline_data[airline_data.STATE != '#VALUE!']
    airline_data['CITY'] = airline_data['CITY'].str.split('/').str[0]
    airline_data['CITY'] = airline_data['CITY'].str.replace('FT.', 'FORT')
    airline_data['STATE'] = airline_data['STATE'].str.replace('D.C.', 'DISTRICT OF COLUMBIA')

    airport_data_grouped_lat_lng = pd.merge(airline_data,
                                            city_state_data[['lat', 'lng', 'city', 'state_name', 'state_id']],
                                            left_on=['CITY', 'STATE'], right_on=['city', 'state_name'], how='left')
    airport_data_grouped_lat_lng = airport_data_grouped_lat_lng[airport_data_grouped_lat_lng['lat'].notna()]
    print('grouped data - merged')
    print(airport_data_grouped_lat_lng)

    airport_data_grouped_lat_lng.to_csv(os.path.join(basedir, 'files', 'grouped_airport_data_lat_lang.csv'), index=False)


async def step3_add_lat_lng_toCovidData():
    city_state_data = pd.read_csv(os.path.join(basedir, 'files', 'source_data', 'county_city_state.csv'))
    covid_data = pd.read_csv(os.path.join(basedir, 'files', 'grouped_covid_data.csv'))

    city_state_data['city'] = city_state_data['city'].str.upper()
    city_state_data['state_id'] = city_state_data['state_id'].str.upper()

    covid_data['CITY'] = covid_data['CITY'].str.upper()
    covid_data['State'] = covid_data['State'].str.upper()
    print('Covid Data')
    print(covid_data.head())

    print('City State Data')
    print(city_state_data)

    covid_data_grouped_lat_lng = pd.merge(covid_data, city_state_data[['lat', 'lng', 'city', 'state_id']],
                                          left_on=['CITY', 'State'], right_on=['city', 'state_id'], how='left')
    covid_data_grouped_lat_lng = covid_data_grouped_lat_lng[covid_data_grouped_lat_lng['lat'].notna()]
    print('grouped data - merged')
    print(covid_data_grouped_lat_lng)

    covid_data_grouped_lat_lng.to_csv(os.path.join(basedir, 'files', 'grouped_covid_data_lat_lang.csv'), index=False)


async def step1_parse_covid_data():
    covid_data = pd.read_csv(os.path.join(basedir, 'files', 'source_data', 'covid_confirmed_usafacts.csv'))
    print(covid_data.head())
    # first thing have to do is group by city
    covid_data.drop(['County Name', 'countyFIPS', 'StateFIPS'], axis=1, inplace=True)
    covid_data_ = covid_data.groupby(['State', 'CITY'], axis=0, as_index=False).sum()
    print(covid_data_.columns)
    grouped_county_data = covid_data.groupby(['State', 'CITY'], axis=0).sum()

    grouped_county_data.columns = [pd.to_datetime(i) if pd.to_datetime(i) else i for i in grouped_county_data.columns]

    print(grouped_county_data.columns)

    grouped_df = grouped_county_data.groupby(grouped_county_data.columns.to_series().dt.to_period('M'), axis=1).sum()

    print(grouped_df.columns)

    grouped_df.to_csv(os.path.join(basedir, 'files', 'grouped_covid_data.csv'), sep=',', encoding='utf-8')


async def step2_parse_airport_data():
    airline_data = pd.read_csv(os.path.join(basedir, 'files', 'source_data', 'Airline_Data_By_Month.csv'))
    airline_data.drop(['Year', 'Month', 'Airport ID', 'airlineid', 'carriergroup'], axis=1, inplace=True)
    print(airline_data.columns)

    data_grouped = airline_data.groupby(['data_dte', 'Airport Name', 'CITY', 'STATE', 'COUNTRY'],
                                        axis=0).sum()

    print(data_grouped.head())
    data_grouped.sort_values(by=['data_dte', 'COUNTRY', 'STATE', 'CITY']).to_csv(
        os.path.join(basedir, 'files', 'grouped_airport_data_byMonth.csv'), sep=',', encoding='utf-8')


async def main():
    tasks = [
        asyncio.create_task(step1_parse_covid_data()),
        asyncio.create_task(step2_parse_airport_data()),
        asyncio.create_task(step3_add_lat_lng_toCovidData()),
        asyncio.create_task(step4_add_lat_lng_toAirPortData())
    ]
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    asyncio.run(main())
