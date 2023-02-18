import matplotlib.pyplot as plt
from Phase_1.preprocess import accidents_data
import seaborn as sns
import pandas as pd

accidents = accidents_data

# Plot for bool feature attributes
fig = plt.figure(figsize=(11, 11))
fig_dims = (3, 2)

plt.subplot2grid(fig_dims, (0, 0))
accidents['Amenity'].value_counts().plot(kind='bar',
                                         title='Amenity')
plt.subplot2grid(fig_dims, (0, 1))
accidents['Bump'].value_counts().plot(kind='bar',
                                      title='Bump')
plt.subplot2grid(fig_dims, (1, 0))
accidents['Crossing'].value_counts().plot(kind='bar',
                                          title='Crossing')
plt.subplot2grid(fig_dims, (1, 1))
accidents['Junction'].value_counts().plot(kind='bar',
                                          title='Junction')
plt.subplot2grid(fig_dims, (2, 0))
accidents['Sunrise_Sunset'].value_counts().plot(kind='bar',
                                                title='Sunrise_Sunset')
plt.subplot2grid(fig_dims, (2, 1))
accidents['Traffic_Signal'].value_counts().plot(kind='bar',
                                                title='Traffic_Signal')
plt.show()
plt.close()

# voilin plot to study the distribution of the data


plt.figure(figsize=(12, 8))
sns.violinplot(x='Severity', y='Wind_Chill(F)', data=accidents)
plt.xlabel('Severity', fontsize=12)
plt.ylabel('Wind_Chill(F)', fontsize=12)
plt.show()
plt.close()

# box plot for mean and quartile distribution
plt.figure(figsize=(12, 8))
sns.boxplot(x="Severity", y="Wind_Chill(F)", data=accidents)
plt.ylabel('Wind_Chill(F)', fontsize=12)
plt.xlabel('Severity', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
plt.close()

# Plot for Weather condition
fig, ax = plt.subplots(figsize=(20, 10))
accidents['Weather_Condition'].value_counts().sort_values(ascending=False).head(5).plot.bar(width=0.8, edgecolor='red',
                                                                                            align='center', linewidth=2)
plt.xlabel('Weather_Condition', fontsize=18)
plt.ylabel('Number of Accidents', fontsize=18)
ax.tick_params(labelsize=20)
plt.title('Weather Condition for accidents - Top 5', fontsize=20)
plt.grid()
plt.ioff()
plt.show()
plt.close()

# bar plot for state wise number of accidents
plt.figure(figsize=(12, 8))
accidents.State.value_counts().sort_values(ascending=False).head(18).plot.bar(width=0.3, edgecolor='k', align='center',
                                                                              linewidth=2)
plt.xlabel('State_name', fontsize=18)
plt.ylabel('Number of Accidents', fontsize=18)
ax.tick_params(labelsize=20)
plt.title('State wise accidents', fontsize=20)
plt.grid()
plt.ioff()
plt.show()
plt.close()

# bar plot for timezone vs number of accidents
plt.figure(figsize=(12, 8))
accidents.Timezone.value_counts().sort_values(ascending=False).head(18).plot.bar(width=0.3, edgecolor='k',
                                                                                 align='center', linewidth=2)
plt.xlabel('Time Zone', fontsize=18)
plt.ylabel('Number of Accidents', fontsize=18)
ax.tick_params(labelsize=30)
plt.title('Accident cases for different timezones in US (2016-2020)', fontsize=20)
plt.grid()
plt.ioff()

# hour wise plot wrt accidents
plt.figure(figsize=(12, 8))
accidents.H.value_counts().sort_values(ascending=True).head(24).reset_index().plot.bar(width=0.3, edgecolor='k',
                                                                                       align='center', linewidth=2)
plt.xlabel('Hour', fontsize=18)
plt.ylabel('Number of Accidents', fontsize=18)
ax.tick_params(labelsize=30)
plt.title('Accident cases for different Hours', fontsize=20)
plt.grid()
plt.ioff()

# Correlation matrix


corr = accidents.corr()
corr.style.background_gradient(cmap='YlOrRd')

sns.heatmap(accidents[['Severity', 'Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                       'Visibility(mi)', 'Precipitation(in)', 'Weather_Condition']].corr())

day = pd.DataFrame(accidents.D.value_counts()).reset_index().rename(columns={'index': 'Day', 'Start_Time': 'Cases'})

# Week wise vs number of accidents
plt.figure(figsize=(9, 4))
plt.title('\n Accident cases for different days of a week\n', size=20, color='grey')
plt.xlabel('\n Day \n', fontsize=15, color='grey')
plt.ylabel('\nAccident Cases\n', fontsize=15, color='grey')
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
a = sns.barplot(x=day.Day, y=day.D, palette="muted")
plt.show()
plt.close()

# wind direction vs accidents
fig = plt.figure(figsize=(12, 10))
sns.countplot(y='Wind_Direction', data=accidents,
              order=accidents['Wind_Direction'].value_counts()[:15].index).set_title("Top 15 Wind_Direction",
                                                                                     fontsize=18)
plt.show()

# Sunset Sunrise Day vs Night

plt.figure(figsize=(10, 8))
accidents.Sunrise_Sunset.value_counts().sort_values(ascending=False).head(5).plot.bar(width=0.3, edgecolor='k',
                                                                                      align='center', linewidth=2)
plt.xlabel('Sunrise_Sunset', fontsize=18)
plt.ylabel('Number of Accidents', fontsize=18)
ax.tick_params(labelsize=30)
plt.title('Accident cases for Sunrise_Sunset (2016-2020)', fontsize=20)
plt.grid()
plt.ioff()
plt.show()
plt.close()

# hour wise plot wrt accidents
plt.figure(figsize=(12, 8))
accidents.M.value_counts().sort_values(ascending=True).head(24).reset_index().plot.bar(width=0.3, edgecolor='k',
                                                                                       align='center', linewidth=2)
plt.xlabel('Month', fontsize=18)
plt.ylabel('Number of Accidents', fontsize=18)
ax.tick_params(labelsize=30)
plt.title('Accident cases Month Wise', fontsize=20)
plt.grid()
plt.ioff()
plt.show()
plt.close()
