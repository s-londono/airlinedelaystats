# Analysis of Airline Disruption Data

In this project we use Python and libraries such as Pandas and Scikitlearn to analyze the performance of flights 
in the U.S. We follow the CRISP-DM process to examine the data.

The dataset contains of information about domestic flights operated by large airlines in the U.S. throughout 2008. 
It provides details about the schedule and delays of each flight. The data was originally compiled and made available 
to the public by the Bureau of Transportation Statistics, but for our purposes we will use the version 
available at Kaggle:

[https://www.kaggle.com/giovamata/airlinedelaycauses](https://www.kaggle.com/giovamata/airlinedelaycauses)

Each record comprises details about a specific flight, such as its number, origin, destination, airline, 
scheduled timeline and actual timeline. 

Our analysis focuses on flight disruptions. Delays (deviations from the scheduled timeline) are particularly 
intersting, as well as their potential causes. The results will hopefully provide insights useful to passengers.

Throughout the analysis we aimed at answering the following questions:

- Which were the best and the worst carriers (airlines) regarding delays and cancellations in 2008?

- Can we identify months of the year 2008 when delays were particularly long?

- What were the major causes of delays affecting domestic flights in the U.S. in 2008?

- Can we build a model to estimate how much delay could be expected in a flight?

The answers we arrive at will hopefully help passengers make more informed decisions in their travels and avoid 
the inconveniences of getting into a delayed flight.

### Results

As a result of the Analysis, we conclude that by making use of descriptive statistics and plots, we can dig a lot of 
useful information about the data. We were able to answer the first three questions using these techniques and 
actually, there are many more interesting facts about the problem at hand that could be uncovered using 
traditional statistics.

The data shows that in the year of study, the carrier with the shortest mean delays was AQ and the carrier with 
the longest mean delays was YV. 

Regarding cancelled and diverted flights, we can see from the bar plot that in general, the percentage of cancelled 
and diverged flights is pretty low for all carriers (all less than 1%). We can also notice that results vary much 
more between carriers when compared to mean delays.

We didn't find any months of the year when average departure or arrival delays are particularly long. As expected, 
December has slightly longer delays than the rest of the months.

We found that shows that departure delays were mostly caused by the aircraft arriving late at the origin airport 
and by problems related to the carrier. As expected, arrival delays are mostly associated with those same causes.

Sadly, we did not succeed in our attempt to build a regression model for flight delays based on this dataset. 
It could be that there are no deterministic relations among the independent variables and the 
dependent variable. There could also be many more variables that affect flight delays, but 
are not included in the dataset.

Nevertheless, more advanced machine learning techniques and further analysis could prove otherwise.

## Project Structure

The project consists of two files:

- *AirlineDelayAnalysis.ipynb:*
   
   Jupyter notebook implementing the analysis. Here we pre-process the data and perform computations step by step, 
   by following the CRISP-DM process. Contains documentation about each operation and explains the decisions 
   we made throughout the analysis.
   
- *Utilities.py:*

   Contains utilitarian functions used in the analysis. Imported as a module by AirlineDelayAnalysis.ipynb.

## Dependencies

This project requires Python 3.6 and Jupyter Notebook 6.0. It requires the following libraries at the specified version 
or above:

- [Numpy v1.16](https://numpy.org/)
- [Pandas v0.24](https://pandas.pydata.org/)
- [Matplotlib v3.1](https://matplotlib.org/users/installing.html)
- [Seaborn v0.9](https://seaborn.pydata.org/)
- [Scikit-learn v0.21](https://seaborn.pydata.org/)

Although not required, we recommend using [Anaconda](https://www.anaconda.com/) to manage both, dependencies 
and Python environments.
   
## References

- The dataset was originally published by the U.S. Bureau of Transportation Statistics:

   [http://stat-computing.org/dataexpo/2009/the-data.html](http://stat-computing.org/dataexpo/2009/the-data.html)
   
- In this project we use the version of the Arline on-time dataset available at Kaggle:

   [https://www.kaggle.com/giovamata/airlinedelaycauses](https://www.kaggle.com/giovamata/airlinedelaycauses)

- The dataset identifies carriers (i.e. airlines) by IATA code:

   [https://en.wikipedia.org/wiki/List_of_airlines_of_the_United_States](https://en.wikipedia.org/wiki/List_of_airlines_of_the_United_States)

- The function bi_bar_plot defined in Utilities.py, is based on an example available at the Matplotlib documentation:

  [https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py](https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py)
