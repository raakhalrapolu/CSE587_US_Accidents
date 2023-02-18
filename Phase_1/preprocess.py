""""We first install kaggle using pip inorder to load the dataset from kaggle
! pip install kaggle
use the above command to install kaggle from pip
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
kaggle.json is the authentication details which can be downloaded from your kaggle account

! kaggle datasets download -d sobhanmoosavi/us-accidents --unzip

The above command downloads the dataset from the source
"""""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from pickle import dump

# accidents = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec21_updated.csv')
accidents = pd.read_csv('/content/US_Accidents_Dec21_updated.csv')
print(accidents.shape)  # this will print the shape of the dataframe loaded with the data

print(accidents.info())  # this will give the dataset information columns details


def preprocessing_input(accidents):
    """"    
    Data Cleaning/Processing
    **1. Handling missing values**

    Here, we will remove any incorrect or missing values from the dataset. Action on Side column:
    

    # Take care of incorrect and missing values"""""

    accidents = accidents.drop(accidents[accidents.Side == 'N'].index)
    accidents["Side"].value_counts()

    """Here we observed that there is one row without side value as R/L therefore we can drop it.

    Also dropped ID and Description column as they are not providing information on the accident which can be feeded 
    into the models in the future. """
    accidents.drop(['ID', 'Description'], axis=1, inplace=True)
    # Handling data with a high cardinality
    print(accidents.Weather_Condition.unique())
    """Data can be combined or binned. By assembling them into coherent sets, the goal is to lower the number of 
    unique values. As long as the groupings do not significantly affect model performance and business stakeholder 
    explainability, this approach is excellent. """

    accidents.loc[
        accidents["Weather_Condition"].str.contains("N/A Precipitation", na=False), "Weather_Condition"] = np.nan
    accidents.loc[
        accidents["Weather_Condition"].str.contains("Snow|Sleet|Wintry", na=False), "Weather_Condition"] = "Snow"
    accidents.loc[
        accidents["Weather_Condition"].str.contains("Smoke|Volcanic Ash", na=False), "Weather_Condition"] = "Smoke"
    accidents.loc[
        accidents["Weather_Condition"].str.contains("Cloud|Overcast", na=False), "Weather_Condition"] = "Cloudy"
    accidents.loc[accidents["Weather_Condition"].str.contains("Sand|Dust", na=False), "Weather_Condition"] = "Sand"
    accidents.loc[accidents["Weather_Condition"].str.contains("Wind|Squalls", na=False), "Weather_Condition"] = "Windy"
    accidents.loc[accidents["Weather_Condition"].str.contains("Mist|Haze|Fog", na=False), "Weather_Condition"] = "Fog"
    accidents.loc[
        accidents["Weather_Condition"].str.contains("Thunder|T-Storm", na=False), "Weather_Condition"] = "Thunderstorm"
    accidents.loc[
        accidents["Weather_Condition"].str.contains("Rain|Drizzle|Shower", na=False), "Weather_Condition"] = "Rain"
    accidents.loc[accidents["Weather_Condition"].str.contains("Hail|Pellets", na=False), "Weather_Condition"] = "Hail"
    accidents.loc[accidents["Weather_Condition"].str.contains("Fair", na=False), "Weather_Condition"] = "Clear"

    print('Unique values:', accidents["Weather_Condition"].unique())

    """ One-hot Encoding** 

    We converted bool values to 0s and 1s as will be easier to feed the models we make in phase2.
    """

    # One-hot Encoding

    accidents = accidents.replace([True, False], [1, 0])

    """Remove Duplicate Data**

    We did not have any duplicates but this was fail safe cleaning as Duplicate data sets have the potential to 
    contaminate the training data with the test data, or the other way around. This can make the model bias towards 
    one feature. """

    Duplicate_Data = accidents[accidents.duplicated()]
    print("Duplicate Rows :{}".format(Duplicate_Data))
    accidents.drop_duplicates(inplace=True)

    """NA value handling (Imputation by mean value)**

    Since filling them with zero doesn't make much sense, we populated the following columns with their average value.
    """

    accidents['Wind_Speed(mph)'] = accidents['Wind_Speed(mph)'].fillna(accidents['Wind_Speed(mph)'].mean())
    accidents['Humidity(%)'] = accidents['Humidity(%)'].fillna(accidents['Humidity(%)'].mean())
    accidents['Temperature(F)'] = accidents['Temperature(F)'].fillna(accidents['Temperature(F)'].mean())
    accidents['Pressure(in)'] = accidents['Pressure(in)'].fillna(accidents['Pressure(in)'].mean())

    """NA value handling (Imputation by median value)**

    When the data is skewed, it is wise to think about replacing the missing values with the median value. Keep in 
    mind that only numerical data can be used to impute missing data using the median value. Because it lessens the 
    impact of outliers, using the median to impute is more reliable. """

    # NA value handling (Imputation by median value)

    accidents['Visibility(mi)'] = accidents['Visibility(mi)'].fillna(accidents['Visibility(mi)'].median())
    accidents['Wind_Chill(F)'] = accidents['Wind_Chill(F)'].fillna(accidents['Wind_Chill(F)'].median())
    accidents['Precipitation(in)'] = accidents['Precipitation(in)'].fillna(accidents['Precipitation(in)'].median())

    """Impute empty value with most occurring value (or mode)**

    The mode value, or most frequent value, of the full feature column is substituted for the missing values in mode 
    imputation, a further approach. It is wise to think about utilizing mode values to replace missing values when 
    the data is skewed. You might think about substituting values for data points like the city field with mode. It 
    should be noted that both numerical and categorical data can be used to impute missing data using mode values. """

    # Impute empty value with most occurring value (or mode)

    accidents['City'] = accidents.groupby('State')['City'].transform(
        lambda data: data.fillna(data.value_counts().index[0]))

    """**8. Scaling of feature**

    We will scale and normalize the characteristics in this part. We normalized the values of the continuous features 
    to enhance the performance of our models. """

    # Scaling of feature

    ss = MinMaxScaler()
    attributes = ['Wind_Speed(mph)', 'Pressure(in)', 'Humidity(%)', 'Visibility(mi)', 'Temperature(F)']
    accidents[attributes] = ss.fit_transform(accidents[attributes])
    dump(ss, open('scaler.pkl', 'wb'))

    """**9. Adding of feature**

    We made the decision to divide the Start Time feature (so that we can feed it to the model later) into the following components: year, month, day, weekday, hour, and minute.
    """

    accidents['Start_Time'] = pd.to_datetime(accidents['Start_Time'], errors='coerce')
    accidents['End_Time'] = pd.to_datetime(accidents['End_Time'], errors='coerce')

    # Extract year, month, day, hour and weekday
    accidents['Year'] = accidents['Start_Time'].dt.year
    accidents['Month'] = accidents['Start_Time'].dt.strftime('%b')
    accidents['Day'] = accidents['Start_Time'].dt.day
    accidents['Hour'] = accidents['Start_Time'].dt.hour
    accidents['Weekday'] = accidents['Start_Time'].dt.strftime('%a')
    # Extract the amount of time in the unit of minutes for each accident, round to the nearest integer
    td = 'Time_Duration(min)'
    accidents[td] = round((accidents['End_Time'] - accidents['Start_Time']) / np.timedelta64(1, 'm'))

    return accidents


accidents_data = preprocessing_input(accidents)
