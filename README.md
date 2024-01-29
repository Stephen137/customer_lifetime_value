# Google Merchandise Store - customer analytics dashboard

![dashboard](https://github.com/Stephen137/google_merchandise_store/assets/97410145/99e27f12-eae8-453d-8c2a-fae0421c269b)


## 1. Project Overview

My starting point for this project is to frame two common business problems faced by the [Google Merchandise Store](https://shop.googlemerchandisestore.com/) and ecommerce businesses in general.

### Frame the business problem

- What will customers spend in the next 90-days? (Regression)
- What is the probability that a customer will make a purchase in the next 90-days? (Classification)

In order to provide insights to these questions, we first need to :

### Translate the raw data into meaningful visualizations and insights

- explore athe dataset to extract meaningful trends, patterns and anomalies regarding customer behaviour
- visualize using plotly and plotnine
- create additional RMF (Recency, Monetary, Frequency) to unlock hidden insights 

### Use the insights to make future predictions

- train a machine learnign regression mode to predict the `$` customer spend in the next-90 days
- train a machine learning classification model to predict the `likelihood` of customer spend in the next-90 days


### Communicate using an interactive dashboard

- present the findings of my analysis through an interactive dashboard
- share my findings through deployment of a Dash [ecommerce customer analytics app](http://stephen137.pythonanywhere.com/)


### Move from insights to actionable outcomes

We can use these predictions to identify :

- Which customers have the highest spend `probability` in next 90-days?

> **target for new products similar to what they have purchased in the past**


- Which customers have recently purchased (within 90 days) but are `unlikely` to buy (probability less than 30%)? 

> **incentivize actions to increase probability**\
> **provide discounts, encourage referring a friend, nurture by letting them know what's coming**


- Missed opportunities: Big spenders that could be unlocked?

> **Send bundle offers encouraging volume purchases**\
> **Focus on missed opportunities**

By adopting this approach ecommerce businesses can make informed decisions to improve customer satisfaction, increase sales, and enhance overall operational efficiency.

## 2. Customer Lifetime Value

Companies use this metric to gauge profitability and to identify which customers to focus on. In short, CLV is the estimated profit from the future relationship with a customer. There are many different approaches to modeling CLV.

### 2.1 `Economic / cashflow approach`  

![cashflow](https://github.com/Stephen137/google_merchandise_store/assets/97410145/f6fff81d-0b8f-4776-8745-c444bab3b510)

Challenges with this approach:

- Lifetime value is great, but more important is a pre-defined period of time, for example the next 90-days

### 2.2 `Machine learning approach` 

- Step 1: Subset a cohort
- Step 2: Temoral splitting - use future information to develop targets
- Step 3: Create RFM features (Recency, Frequency, Monetary)

### 3. Dataset, schema, packages

The dataset I will be analysing is the [Google Analytics sample dataset for BigQuery](https://support.google.com/analytics/answer/7586738?hl=en&ref_topic=3416089&sjid=15552203920864108300-EU#zippy=%2Cin-this-article). 

The sample dataset contains obfuscated Google Analytics 360 data from the [Google Merchandise Store](https://www.googlemerchandisestore.com/shop.axd/Home?utm_source=Partners&utm_medium=affiliate&utm_campaign=Data%20Share%20Promo), a real ecommerce store. The Google Merchandise Store sells Google branded merchandise. The data is typical of what you would see for an ecommerce website. It includes the following kinds of information:

- `Traffic source data:` information about where website visitors originate. This includes data about organic traffic, paid search traffic, display traffic, etc.
- `Content data:` information about the behavior of users on the site. This includes the URLs of pages that visitors look at, how they interact with content, etc.
- `Transactional data:` information about the transactions that occur on the Google Merchandise Store website.

The dataset covers the 366 day period 1 August 2016 to 1 August 2017. 

#### 3.1 Schema

Before diving into our analysis, it is good practice to explore the dataset schema. 

If you take a close look at [the schema](https://support.google.com/analytics/answer/3437719?hl=en) you will see that the structure involves nested fields. It can be extremely cost-effective (both in terms of storage and in terms of query time) to use nested fields rather than flatten out all your data. Nested, repeated fields are very powerful, but the SQL required to query them looks a bit unfamiliar. So, it’s worth spending a little time with `STRUCT`, `UNNEST` and `ARRAY_AGG.` 


Fields which are of type INTEGER or STRING can be accessed directly. However things a bit trickier as we go deeper into the structure. 

`totals` for example is of Type `RECORD` and includes sub-variables. In order to access these we need to use `dot` notation:

    totals.visits
    
Things get even trickier with fields of Type `RECORD` and Mode `Repeated`, for example `hits`. In order to access the subvariable `hitNumber` we need to use the `UNNEST()` function, give it an alias, and then we can use `dot` notation :

    SELECT
        h.hitNumber
    FROM
    `bigquery-public-data.google_analytics_sample.ga_sessions_*`,
    UNNEST(hits) AS h
    
We can go even deeper. To access `hits.product.productRevenue` we need to first UNNEST hits and then because product is also of Type `RECORD` and Mode `Repeated` we have to UNNEST(hits.product). Expanding on the above example :

    SELECT
        h.hitNumber,
        p.productRevenue
    FROM
    `bigquery-public-data.google_analytics_sample.ga_sessions_*`,
    UNNEST(hits) AS h,
    UNNEST(hits.product) AS p

Google cloud have [learning resources](https://cloud.google.com/bigquery/docs/arrays) to help you.

OK, now that I've gotten to know the dataset a bit better it's time to dive into some exploratory analysis.

### 3.2 A Kind of Magic
![magic](https://github.com/Stephen137/google_merchandise_store/assets/97410145/dbfd28e9-d166-4fb1-a318-ae71cadf5acc)

Jupyter notebooks (which i am using for this project) provide a convenient interactive computing environment for various programming languages, including Python.  You might be aware of so called `magic` commands, denoted by a % or %% symbol. These are special commands that enhance the functionality and provide additional features within Jupyter notebooks. Here's a brief summary of some commonly used magic commands:

#### Line Magic Commands (%):

- `%run:` Execute a Python script as if it were a program.
- `%load:` Load code into a cell from an external script.
- `%pwd:` Print the current working directory.
- `%cd:` Change the current working directory.
- `%ls:` List the contents of the current directory.
- `%matplotlib:` Enable interactive Matplotlib plots.

#### Cell Magic Commands (%%):

- `%%time:` Measure the time it takes for the code in the cell to run.
- `%%html:` Render the cell content as HTML.


#### Interactive Shell Commands:

- `!:` Run shell commands directly from a Jupyter cell.

### 3.3 Import packages and libraries


```python
%load_ext google.cloud.bigquery 
from google.cloud import bigquery
import google.cloud.bigquery
from google.oauth2 import service_account

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import pytimetk as tk
import plotly.express as px
import joblib 
from missingno import matrix # missing data

#!pip install plydata
import plydata.cat_tools as cat
import plotnine as pn
import pytimetk as tk
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

pn.options.dpi = 300
```

## 4. Exploratory Data Analysis

### 4.1 Using %%bigquery to run SQL within Jupyter

The **%%bigquery** magic which allows you to run SQL queries on data held in BigQuery from the comfort of your own local Jupyter Notebook. 

The data is separated into daily tables but you can use `_*` to access everything, and if you specify a name after %%big query, it will return a pandas DataFrame!


```python
%%bigquery total_days
SELECT COUNT(DISTINCT date) AS number_days
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
```


    Query is running:   0%|          |



    Downloading:   0%|          |



```python
total_days
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>366</td>
    </tr>
  </tbody>
</table>
</div>




```python
type(total_days)
```




    pandas.core.frame.DataFrame




```python
%%bigquery
SELECT COUNT(*)
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
```


    Query is running:   0%|          |



    Downloading:   0%|          |





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f0_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>903653</td>
    </tr>
  </tbody>
</table>
</div>



Ok, so the dataset covers a period of 366 days and contains 903,653 rows. Let's preview one day :


```python
%%bigquery sample_day
SELECT *
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170713`
```


    Query is running:   0%|          |



    Downloading:   0%|          |



```python
sample_day
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>visitorId</th>
      <th>visitNumber</th>
      <th>visitId</th>
      <th>visitStartTime</th>
      <th>date</th>
      <th>totals</th>
      <th>trafficSource</th>
      <th>device</th>
      <th>geoNetwork</th>
      <th>customDimensions</th>
      <th>hits</th>
      <th>fullVisitorId</th>
      <th>userId</th>
      <th>clientId</th>
      <th>channelGrouping</th>
      <th>socialEngagementType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;NA&gt;</td>
      <td>2</td>
      <td>1500008750</td>
      <td>1500008750</td>
      <td>20170713</td>
      <td>{'visits': 1, 'hits': 1, 'pageviews': 1, 'time...</td>
      <td>{'referralPath': None, 'campaign': '(not set)'...</td>
      <td>{'browser': 'Firefox', 'browserVersion': 'not ...</td>
      <td>{'continent': 'Americas', 'subContinent': 'Nor...</td>
      <td>[{'index': 4, 'value': 'North America'}]</td>
      <td>[{'hitNumber': 1, 'time': 0, 'hour': 22, 'minu...</td>
      <td>137294517588272857</td>
      <td>None</td>
      <td>None</td>
      <td>Direct</td>
      <td>Not Socially Engaged</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>1499957243</td>
      <td>1499957243</td>
      <td>20170713</td>
      <td>{'visits': 1, 'hits': 1, 'pageviews': 1, 'time...</td>
      <td>{'referralPath': None, 'campaign': '(not set)'...</td>
      <td>{'browser': 'Chrome', 'browserVersion': 'not a...</td>
      <td>{'continent': 'Americas', 'subContinent': 'Nor...</td>
      <td>[{'index': 4, 'value': 'North America'}]</td>
      <td>[{'hitNumber': 1, 'time': 0, 'hour': 7, 'minut...</td>
      <td>4373106646092857768</td>
      <td>None</td>
      <td>None</td>
      <td>Direct</td>
      <td>Not Socially Engaged</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>1499968083</td>
      <td>1499968083</td>
      <td>20170713</td>
      <td>{'visits': 1, 'hits': 1, 'pageviews': 1, 'time...</td>
      <td>{'referralPath': None, 'campaign': '(not set)'...</td>
      <td>{'browser': 'Safari', 'browserVersion': 'not a...</td>
      <td>{'continent': 'Americas', 'subContinent': 'Nor...</td>
      <td>[{'index': 4, 'value': 'North America'}]</td>
      <td>[{'hitNumber': 1, 'time': 0, 'hour': 10, 'minu...</td>
      <td>160773093174680026</td>
      <td>None</td>
      <td>None</td>
      <td>Direct</td>
      <td>Not Socially Engaged</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;NA&gt;</td>
      <td>2</td>
      <td>1499952856</td>
      <td>1499952856</td>
      <td>20170713</td>
      <td>{'visits': 1, 'hits': 1, 'pageviews': 1, 'time...</td>
      <td>{'referralPath': None, 'campaign': '(not set)'...</td>
      <td>{'browser': 'Chrome', 'browserVersion': 'not a...</td>
      <td>{'continent': 'Americas', 'subContinent': 'Nor...</td>
      <td>[{'index': 4, 'value': 'North America'}]</td>
      <td>[{'hitNumber': 1, 'time': 0, 'hour': 6, 'minut...</td>
      <td>1117853031731048699</td>
      <td>None</td>
      <td>None</td>
      <td>Direct</td>
      <td>Not Socially Engaged</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;NA&gt;</td>
      <td>4</td>
      <td>1499982847</td>
      <td>1499982847</td>
      <td>20170713</td>
      <td>{'visits': 1, 'hits': 1, 'pageviews': 1, 'time...</td>
      <td>{'referralPath': None, 'campaign': '(not set)'...</td>
      <td>{'browser': 'Chrome', 'browserVersion': 'not a...</td>
      <td>{'continent': 'Americas', 'subContinent': 'Nor...</td>
      <td>[{'index': 4, 'value': 'North America'}]</td>
      <td>[{'hitNumber': 1, 'time': 0, 'hour': 14, 'minu...</td>
      <td>1319757127869798182</td>
      <td>None</td>
      <td>None</td>
      <td>Display</td>
      <td>Not Socially Engaged</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2736</th>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>1499986111</td>
      <td>1499986111</td>
      <td>20170713</td>
      <td>{'visits': 1, 'hits': 73, 'pageviews': 57, 'ti...</td>
      <td>{'referralPath': None, 'campaign': '(not set)'...</td>
      <td>{'browser': 'Safari', 'browserVersion': 'not a...</td>
      <td>{'continent': 'Americas', 'subContinent': 'Nor...</td>
      <td>[{'index': 4, 'value': 'North America'}]</td>
      <td>[{'hitNumber': 1, 'time': 0, 'hour': 15, 'minu...</td>
      <td>4100305414080508541</td>
      <td>None</td>
      <td>None</td>
      <td>Organic Search</td>
      <td>Not Socially Engaged</td>
    </tr>
    <tr>
      <th>2737</th>
      <td>&lt;NA&gt;</td>
      <td>5</td>
      <td>1499969354</td>
      <td>1499969354</td>
      <td>20170713</td>
      <td>{'visits': 1, 'hits': 86, 'pageviews': 57, 'ti...</td>
      <td>{'referralPath': '/', 'campaign': '(not set)',...</td>
      <td>{'browser': 'Chrome', 'browserVersion': 'not a...</td>
      <td>{'continent': 'Americas', 'subContinent': 'Nor...</td>
      <td>[{'index': 4, 'value': 'North America'}]</td>
      <td>[{'hitNumber': 1, 'time': 0, 'hour': 11, 'minu...</td>
      <td>806992249032686650</td>
      <td>None</td>
      <td>None</td>
      <td>Referral</td>
      <td>Not Socially Engaged</td>
    </tr>
    <tr>
      <th>2738</th>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>1499931174</td>
      <td>1499931174</td>
      <td>20170713</td>
      <td>{'visits': 1, 'hits': 132, 'pageviews': 85, 't...</td>
      <td>{'referralPath': None, 'campaign': '(not set)'...</td>
      <td>{'browser': 'Safari', 'browserVersion': 'not a...</td>
      <td>{'continent': 'Americas', 'subContinent': 'Nor...</td>
      <td>[{'index': 4, 'value': 'North America'}]</td>
      <td>[{'hitNumber': 1, 'time': 0, 'hour': 0, 'minut...</td>
      <td>3917496719101325275</td>
      <td>None</td>
      <td>None</td>
      <td>Direct</td>
      <td>Not Socially Engaged</td>
    </tr>
    <tr>
      <th>2739</th>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>1499951313</td>
      <td>1499951315</td>
      <td>20170713</td>
      <td>{'visits': 1, 'hits': 156, 'pageviews': 109, '...</td>
      <td>{'referralPath': None, 'campaign': '(not set)'...</td>
      <td>{'browser': 'Chrome', 'browserVersion': 'not a...</td>
      <td>{'continent': 'Americas', 'subContinent': 'Nor...</td>
      <td>[{'index': 4, 'value': 'North America'}]</td>
      <td>[{'hitNumber': 2, 'time': 0, 'hour': 6, 'minut...</td>
      <td>9417857471295131045</td>
      <td>None</td>
      <td>None</td>
      <td>Direct</td>
      <td>Not Socially Engaged</td>
    </tr>
    <tr>
      <th>2740</th>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>1499982367</td>
      <td>1499982367</td>
      <td>20170713</td>
      <td>{'visits': 1, 'hits': 169, 'pageviews': 86, 't...</td>
      <td>{'referralPath': '/intl/sr/yt/about/copyright/...</td>
      <td>{'browser': 'Chrome', 'browserVersion': 'not a...</td>
      <td>{'continent': 'Europe', 'subContinent': 'South...</td>
      <td>[]</td>
      <td>[{'hitNumber': 1, 'time': 0, 'hour': 14, 'minu...</td>
      <td>0707769484819212214</td>
      <td>None</td>
      <td>None</td>
      <td>Social</td>
      <td>Not Socially Engaged</td>
    </tr>
  </tbody>
</table>
<p>2741 rows × 16 columns</p>
</div>



As previously discussed, a lot of nested fields to unravel. 


```python
sample_day.glimpse()
```

    <class 'pandas.core.frame.DataFrame'>: 2741 rows of 16 columns
    visitorId:             Int64             [<NA>, <NA>, <NA>, <NA>, <NA>,  ...
    visitNumber:           Int64             [2, 1, 1, 2, 4, 1, 3, 2, 1, 1,  ...
    visitId:               Int64             [1500008750, 1499957243, 149996 ...
    visitStartTime:        Int64             [1500008750, 1499957243, 149996 ...
    date:                  object            ['20170713', '20170713', '20170 ...
    totals:                object            [{'visits': 1, 'hits': 1, 'page ...
    trafficSource:         object            [{'referralPath': None, 'campai ...
    device:                object            [{'browser': 'Firefox', 'browse ...
    geoNetwork:            object            [{'continent': 'Americas', 'sub ...
    customDimensions:      object            [array([{'index': 4, 'value': ' ...
    hits:                  object            [array([{'hitNumber': 1, 'time' ...
    fullVisitorId:         object            ['137294517588272857', '4373106 ...
    userId:                object            [None, None, None, None, None,  ...
    clientId:              object            [None, None, None, None, None,  ...
    channelGrouping:       object            ['Direct', 'Direct', 'Direct',  ...
    socialEngagementType:  object            ['Not Socially Engaged', 'Not S ...


### 4.2 Generate our master data for further analysis


```python
%%bigquery google_merch_store

SELECT 
    fullVisitorId,
    date,    
    product.productQuantity AS quantity,
    product.v2ProductName AS product_name,
    product.productPrice / 1000000 AS price, 
    product.productQuantity * product.productPrice / 1000000 AS product_revenue,
    totals.totalTransactionRevenue / 1000000 AS total_transaction_revenue
FROM
    `bigquery-public-data.google_analytics_sample.ga_sessions_*`, 
    UNNEST (hits) AS hits, 
    UNNEST(hits.product) AS product
WHERE 
    hits.eCommerceAction.action_type = '6' # Completed purchase    
    AND product.productQuantity > 0
    AND product.productPrice > 0 
    AND totals.totalTransactionRevenue > 0 # This value is 1 for sessions with interaction events.
ORDER BY
    date
```


    Query is running:   0%|          |



    Downloading:   0%|          |



```python
google_merch_store
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fullVisitorId</th>
      <th>date</th>
      <th>quantity</th>
      <th>product_name</th>
      <th>price</th>
      <th>product_revenue</th>
      <th>total_transaction_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3213840074316400693</td>
      <td>20160801</td>
      <td>20</td>
      <td>Color Changing Grip Pen</td>
      <td>1.20</td>
      <td>24.00</td>
      <td>170.40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3213840074316400693</td>
      <td>20160801</td>
      <td>20</td>
      <td>Kick Ball</td>
      <td>1.59</td>
      <td>31.80</td>
      <td>170.40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3213840074316400693</td>
      <td>20160801</td>
      <td>20</td>
      <td>Electronics Accessory Pouch</td>
      <td>3.99</td>
      <td>79.80</td>
      <td>170.40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3213840074316400693</td>
      <td>20160801</td>
      <td>20</td>
      <td>Badge Holder</td>
      <td>1.59</td>
      <td>31.80</td>
      <td>170.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2976178532750719771</td>
      <td>20160801</td>
      <td>4</td>
      <td>Maze Pen</td>
      <td>0.99</td>
      <td>3.96</td>
      <td>19.93</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36741</th>
      <td>8016003971239765913</td>
      <td>20170801</td>
      <td>2</td>
      <td>Android Men's Short Sleeve Hero Tee White</td>
      <td>6.80</td>
      <td>13.60</td>
      <td>131.96</td>
    </tr>
    <tr>
      <th>36742</th>
      <td>8016003971239765913</td>
      <td>20170801</td>
      <td>1</td>
      <td>Women's Performance Full Zip Jacket Black</td>
      <td>67.19</td>
      <td>67.19</td>
      <td>131.96</td>
    </tr>
    <tr>
      <th>36743</th>
      <td>8016003971239765913</td>
      <td>20170801</td>
      <td>1</td>
      <td>Android Men's Engineer Short Sleeve Tee Charcoal</td>
      <td>15.99</td>
      <td>15.99</td>
      <td>131.96</td>
    </tr>
    <tr>
      <th>36744</th>
      <td>8016003971239765913</td>
      <td>20170801</td>
      <td>1</td>
      <td>Google Infant Short Sleeve Tee Red</td>
      <td>13.59</td>
      <td>13.59</td>
      <td>131.96</td>
    </tr>
    <tr>
      <th>36745</th>
      <td>8016003971239765913</td>
      <td>20170801</td>
      <td>1</td>
      <td>Android Infant Short Sleeve Tee Pewter</td>
      <td>13.59</td>
      <td>13.59</td>
      <td>131.96</td>
    </tr>
  </tbody>
</table>
<p>36746 rows × 7 columns</p>
</div>



### 4.3 PyTimeTK

While the Python ecosystem offers tools like pandas, they sometimes can be verbose and not optimized for all operations, especially for complex time-based aggregations and visualizations. [pytimetk](https://github.com/business-science/pytimetk) significantly simplifies the process of time series manipulation and visualization. By leveraging the polars backend, you can experience speed improvements ranging from 3X to 3500X.

A usefuly feature is `.glimpse()` which does what it says on the tin :


```python
google_merch_store.glimpse()
```

    <class 'pandas.core.frame.DataFrame'>: 36746 rows of 7 columns
    fullVisitorId:              object            ['3213840074316400693', '3 ...
    date:                       object            ['20160801', '20160801', ' ...
    quantity:                   Int64             [20, 20, 20, 20, 4, 1, 1,  ...
    product_name:               object            ['Color Changing Grip Pen' ...
    price:                      float64           [1.2, 1.59, 3.99, 1.59, 0. ...
    product_revenue:            float64           [24.0, 31.8, 79.8, 31.8, 3 ...
    total_transaction_revenue:  float64           [170.4, 170.4, 170.4, 170. ...


### 4.4 Custom profile function

This custom function aids the EDA process and is a pseeudo hybrid of pandas `.info()` and `.describe()` :


```python
# Custom profiling function
def profile_data(data):
    """Panda Profiling Function

    Args:
        data (DataFrame): A data frame to profile

    Returns:
        DataFrame: A data frame with profiled data
    """
    return pd.concat(
        [
            pd.Series(data.dtypes, name = "Dtype"),
            # Counts
            pd.Series(data.count(), name = "Count"),
            pd.Series(data.isnull().sum(), name = "NA Count"),
            pd.Series(data.nunique(), name = "Count Unique"),
            # Stats
            pd.Series(data.min(), name = "Min"),
            pd.Series(data.max(), name = "Max"),
            pd.Series(data.mean(), name = "Mean"),
            pd.Series(data.median(), name = "Median"),
            pd.Series(data.mode().iloc[0], name = "Mode"),
        ],
        axis=1
    )
```


```python
profile_data(google_merch_store)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dtype</th>
      <th>Count</th>
      <th>NA Count</th>
      <th>Count Unique</th>
      <th>Min</th>
      <th>Max</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fullVisitorId</th>
      <td>object</td>
      <td>36746</td>
      <td>0</td>
      <td>9995</td>
      <td>0000213131142648941</td>
      <td>9998996003043230595</td>
      <td>inf</td>
      <td>4.512553e+18</td>
      <td>1957458976293878100</td>
    </tr>
    <tr>
      <th>date</th>
      <td>object</td>
      <td>36746</td>
      <td>0</td>
      <td>365</td>
      <td>20160801</td>
      <td>20170801</td>
      <td>inf</td>
      <td>2.017013e+07</td>
      <td>20161212</td>
    </tr>
    <tr>
      <th>quantity</th>
      <td>Int64</td>
      <td>36746</td>
      <td>0</td>
      <td>123</td>
      <td>1</td>
      <td>1000</td>
      <td>6.423584</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>product_name</th>
      <td>object</td>
      <td>36746</td>
      <td>0</td>
      <td>490</td>
      <td>1 oz Hand Sanitizer</td>
      <td>YouTube Youth Short Sleeve Tee Red</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Google Sunglasses</td>
    </tr>
    <tr>
      <th>price</th>
      <td>float64</td>
      <td>36746</td>
      <td>0</td>
      <td>264</td>
      <td>0.79</td>
      <td>250.0</td>
      <td>16.602405</td>
      <td>1.329000e+01</td>
      <td>13.59</td>
    </tr>
    <tr>
      <th>product_revenue</th>
      <td>float64</td>
      <td>36746</td>
      <td>0</td>
      <td>2001</td>
      <td>0.79</td>
      <td>9495.0</td>
      <td>45.727660</td>
      <td>1.749000e+01</td>
      <td>13.59</td>
    </tr>
    <tr>
      <th>total_transaction_revenue</th>
      <td>float64</td>
      <td>36746</td>
      <td>0</td>
      <td>6199</td>
      <td>1.2</td>
      <td>47082.06</td>
      <td>366.675515</td>
      <td>1.022000e+02</td>
      <td>23.99</td>
    </tr>
  </tbody>
</table>
</div>



So we can very quickly see that our dataset consists of `36,746` purchase transactions made by `9,995` unique customers and the common purchase was Google sunglasses. 

Before we progress, we need to convert date from an object type to datetime: 


```python
# Convert the date column to datetime64 with nanosecond precision
google_merch_store['date'] = pd.to_datetime(google_merch_store['date'], format='%Y%m%d')

google_merch_store.glimpse()
```

    <class 'pandas.core.frame.DataFrame'>: 36746 rows of 7 columns
    fullVisitorId:              object            ['3213840074316400693', '3 ...
    date:                       datetime64[ns]    [Timestamp('2016-08-01 00: ...
    quantity:                   Int64             [20, 20, 20, 20, 4, 1, 1,  ...
    product_name:               object            ['Color Changing Grip Pen' ...
    price:                      float64           [1.2, 1.59, 3.99, 1.59, 0. ...
    product_revenue:            float64           [24.0, 31.8, 79.8, 31.8, 3 ...
    total_transaction_revenue:  float64           [170.4, 170.4, 170.4, 170. ...


## 5. Feature pre-processing

### 5.1.1 Subset a Cohort


```python
# Show only the first transaction made by each unique customer
cust_first_purch = google_merch_store \
    .sort_values(['fullVisitorId', 'date']) \
    .groupby('fullVisitorId') \
    .first()

cust_first_purch
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>quantity</th>
      <th>product_name</th>
      <th>price</th>
      <th>product_revenue</th>
      <th>total_transaction_revenue</th>
    </tr>
    <tr>
      <th>fullVisitorId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000213131142648941</th>
      <td>2017-04-28</td>
      <td>1</td>
      <td>BLM Sweatshirt</td>
      <td>33.59</td>
      <td>33.59</td>
      <td>39.59</td>
    </tr>
    <tr>
      <th>0002871498069867123</th>
      <td>2016-08-23</td>
      <td>15</td>
      <td>Google Metallic Notebook Set</td>
      <td>5.99</td>
      <td>89.85</td>
      <td>97.35</td>
    </tr>
    <tr>
      <th>0003450834640354121</th>
      <td>2016-10-18</td>
      <td>1</td>
      <td>Google Laptop and Cell Phone Stickers</td>
      <td>1.99</td>
      <td>1.99</td>
      <td>59.95</td>
    </tr>
    <tr>
      <th>0003961110741104601</th>
      <td>2017-05-21</td>
      <td>1</td>
      <td>YouTube Custom Decals</td>
      <td>1.99</td>
      <td>1.99</td>
      <td>10.98</td>
    </tr>
    <tr>
      <th>000435324061339869</th>
      <td>2016-10-20</td>
      <td>1</td>
      <td>Google Men's  Zip Hoodie</td>
      <td>44.79</td>
      <td>44.79</td>
      <td>46.79</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9991633376050115277</th>
      <td>2017-02-17</td>
      <td>1</td>
      <td>BLM Sweatshirt</td>
      <td>33.59</td>
      <td>33.59</td>
      <td>35.59</td>
    </tr>
    <tr>
      <th>9994767073213036303</th>
      <td>2016-08-09</td>
      <td>3</td>
      <td>Electronics Accessory Pouch</td>
      <td>4.99</td>
      <td>14.97</td>
      <td>140.32</td>
    </tr>
    <tr>
      <th>9997409246962677759</th>
      <td>2016-12-08</td>
      <td>1</td>
      <td>Crunch Noise Dog Toy</td>
      <td>3.99</td>
      <td>3.99</td>
      <td>40.36</td>
    </tr>
    <tr>
      <th>9998597322098588317</th>
      <td>2016-08-01</td>
      <td>30</td>
      <td>PaperMate Ink Joy Retractable Pen</td>
      <td>1.25</td>
      <td>37.50</td>
      <td>102.20</td>
    </tr>
    <tr>
      <th>9998996003043230595</th>
      <td>2016-11-16</td>
      <td>1</td>
      <td>Google Vintage Henley Grey/Black</td>
      <td>29.99</td>
      <td>29.99</td>
      <td>66.98</td>
    </tr>
  </tbody>
</table>
<p>9995 rows × 6 columns</p>
</div>




```python
cust_first_purch['date'].min()
```




    Timestamp('2016-08-01 00:00:00')




```python
cust_first_purch['date'].max()
```




    Timestamp('2017-08-01 00:00:00')



### 5.1.2 Visualize all purchases within cohort


```python
# grab date values for our y axis, and price for our x axis
google_merch_store \
    .reset_index() \
    .set_index('date')[['price']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-08-01</th>
      <td>1.20</td>
    </tr>
    <tr>
      <th>2016-08-01</th>
      <td>1.59</td>
    </tr>
    <tr>
      <th>2016-08-01</th>
      <td>3.99</td>
    </tr>
    <tr>
      <th>2016-08-01</th>
      <td>1.59</td>
    </tr>
    <tr>
      <th>2016-08-01</th>
      <td>0.99</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-08-01</th>
      <td>6.80</td>
    </tr>
    <tr>
      <th>2017-08-01</th>
      <td>67.19</td>
    </tr>
    <tr>
      <th>2017-08-01</th>
      <td>15.99</td>
    </tr>
    <tr>
      <th>2017-08-01</th>
      <td>13.59</td>
    </tr>
    <tr>
      <th>2017-08-01</th>
      <td>13.59</td>
    </tr>
  </tbody>
</table>
<p>36746 rows × 1 columns</p>
</div>




```python
# Aggregate all customer revenue per month
google_merch_store\
    .reset_index() \
    .set_index('date')[['price']] \
    .resample(
        rule = "MS" # group by month
    ) \
    .sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-08-01</th>
      <td>68604.07</td>
    </tr>
    <tr>
      <th>2016-09-01</th>
      <td>44910.37</td>
    </tr>
    <tr>
      <th>2016-10-01</th>
      <td>45032.57</td>
    </tr>
    <tr>
      <th>2016-11-01</th>
      <td>48906.36</td>
    </tr>
    <tr>
      <th>2016-12-01</th>
      <td>81358.52</td>
    </tr>
    <tr>
      <th>2017-01-01</th>
      <td>36202.85</td>
    </tr>
    <tr>
      <th>2017-02-01</th>
      <td>37402.96</td>
    </tr>
    <tr>
      <th>2017-03-01</th>
      <td>52514.72</td>
    </tr>
    <tr>
      <th>2017-04-01</th>
      <td>48165.02</td>
    </tr>
    <tr>
      <th>2017-05-01</th>
      <td>53689.45</td>
    </tr>
    <tr>
      <th>2017-06-01</th>
      <td>42449.14</td>
    </tr>
    <tr>
      <th>2017-07-01</th>
      <td>48436.32</td>
    </tr>
    <tr>
      <th>2017-08-01</th>
      <td>2399.64</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot the time series
google_merch_store \
    .reset_index() \
    .set_index('date')[['price']] \
    .resample(
        rule = "MS" # group by month
    ) \
    .sum() \
    .plot()
```




    <AxesSubplot: xlabel='date'>




    
![png](images/output_47_1.png)
    


### 5.1.3 Visualize individual customer purchases


```python
ids = google_merch_store['fullVisitorId'].unique()
ids
```




    array(['3213840074316400693', '2976178532750719771',
           '6569605994631186947', ..., '3101662058536674321',
           '9308310352918219134', '8016003971239765913'], dtype=object)




```python
# select a radom slice of 10 customers
ids_selected = ids[989:999]

google_merch_store \
    [google_merch_store['fullVisitorId'].isin(ids_selected)] \
    .groupby(['fullVisitorId', 'date']) \
    .sum() \
    .reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fullVisitorId</th>
      <th>date</th>
      <th>quantity</th>
      <th>price</th>
      <th>product_revenue</th>
      <th>total_transaction_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1293664043695932921</td>
      <td>2016-08-30</td>
      <td>3</td>
      <td>23.17</td>
      <td>23.17</td>
      <td>29.10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>330289116549575054</td>
      <td>2016-08-31</td>
      <td>1</td>
      <td>27.19</td>
      <td>27.19</td>
      <td>47.95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>330289116549575054</td>
      <td>2017-01-12</td>
      <td>2</td>
      <td>73.58</td>
      <td>73.58</td>
      <td>165.16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4019564737576499248</td>
      <td>2016-08-30</td>
      <td>38</td>
      <td>98.33</td>
      <td>335.62</td>
      <td>2878.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4332670999936880007</td>
      <td>2016-08-31</td>
      <td>1</td>
      <td>98.99</td>
      <td>98.99</td>
      <td>123.65</td>
    </tr>
    <tr>
      <th>5</th>
      <td>435373542373805498</td>
      <td>2016-08-30</td>
      <td>8</td>
      <td>127.92</td>
      <td>127.92</td>
      <td>1160.80</td>
    </tr>
    <tr>
      <th>6</th>
      <td>435373542373805498</td>
      <td>2016-11-05</td>
      <td>1</td>
      <td>44.79</td>
      <td>44.79</td>
      <td>45.79</td>
    </tr>
    <tr>
      <th>7</th>
      <td>435373542373805498</td>
      <td>2016-12-01</td>
      <td>5</td>
      <td>227.15</td>
      <td>227.15</td>
      <td>1170.75</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5423659711610895780</td>
      <td>2016-08-30</td>
      <td>16</td>
      <td>260.30</td>
      <td>285.48</td>
      <td>3385.14</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5542047417982345824</td>
      <td>2016-08-30</td>
      <td>16</td>
      <td>63.51</td>
      <td>71.49</td>
      <td>921.90</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6564820894573937867</td>
      <td>2016-08-30</td>
      <td>3</td>
      <td>36.77</td>
      <td>36.77</td>
      <td>167.88</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6564820894573937867</td>
      <td>2016-12-21</td>
      <td>2</td>
      <td>8.79</td>
      <td>17.58</td>
      <td>25.58</td>
    </tr>
    <tr>
      <th>12</th>
      <td>8445777031468826793</td>
      <td>2016-08-31</td>
      <td>30</td>
      <td>0.99</td>
      <td>29.70</td>
      <td>53.56</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9972043774359472649</td>
      <td>2016-08-31</td>
      <td>35</td>
      <td>15.19</td>
      <td>531.65</td>
      <td>680.22</td>
    </tr>
  </tbody>
</table>
</div>




```python
cust_id_subset_df = google_merch_store \
    [google_merch_store['fullVisitorId'].isin(ids_selected)] \
    .groupby(['fullVisitorId', 'date']) \
    .sum() \
    .reset_index()
```

### 5.1.4 Visualize first 10 customers' purchasing 


```python
# plot randomly selected 10 customers using Plotnine
pn.ggplot(data=cust_id_subset_df, mapping=pn.aes('date', 'price', group='fullVisitorId')) \
    + pn.geom_line() \
    + pn.geom_point() \
    + pn.facet_wrap('fullVisitorId') \
    + pn.scale_x_date(date_breaks="1 year", date_labels="%Y")
```

    /home/stephen137/mambaforge/lib/python3.10/site-packages/plotnine/geoms/geom_path.py:111: PlotnineWarning: geom_path: Each group consist of only one observation. Do you need to adjust the group aesthetic?
    /home/stephen137/mambaforge/lib/python3.10/site-packages/plotnine/geoms/geom_path.py:111: PlotnineWarning: geom_path: Each group consist of only one observation. Do you need to adjust the group aesthetic?
    /home/stephen137/mambaforge/lib/python3.10/site-packages/plotnine/geoms/geom_path.py:111: PlotnineWarning: geom_path: Each group consist of only one observation. Do you need to adjust the group aesthetic?
    /home/stephen137/mambaforge/lib/python3.10/site-packages/plotnine/geoms/geom_path.py:111: PlotnineWarning: geom_path: Each group consist of only one observation. Do you need to adjust the group aesthetic?



    
![png](images/output_53_1.png)
    





    <Figure Size: (1920 x 1440)>



This type of plot allows us to get a quick overview of customer purchasing pattern, which could provide a good indication of future business prospects. 

### 5.2 Temporal splitting 

In order to be able to predict purchases in the next 90-days, we need to draw a line, and split our data into two distinct time periods. 


```python
# define cut-off period
n_days   = 90
max_date = google_merch_store['date'].max() 
max_date
```




    Timestamp('2017-08-01 00:00:00')




```python
cutoff = max_date - pd.to_timedelta(n_days, unit = "d")
cutoff
```




    Timestamp('2017-05-03 00:00:00')



The cut-off date is 3 May 2017. The data for the period 1 August 2016 to 3 May 2017 will be used as training data by our machine learning model. Any data after this date will be used to verify the accuracy of predictions made on the prior data and is essentially therefore our held out test set. 


```python
# Create our cohort dataset which covers the period up until the last 90 days
temporal_in_df = google_merch_store \
    [google_merch_store['date'] <= cutoff]
temporal_in_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fullVisitorId</th>
      <th>date</th>
      <th>quantity</th>
      <th>product_name</th>
      <th>price</th>
      <th>product_revenue</th>
      <th>total_transaction_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3213840074316400693</td>
      <td>2016-08-01</td>
      <td>20</td>
      <td>Color Changing Grip Pen</td>
      <td>1.20</td>
      <td>24.00</td>
      <td>170.40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3213840074316400693</td>
      <td>2016-08-01</td>
      <td>20</td>
      <td>Kick Ball</td>
      <td>1.59</td>
      <td>31.80</td>
      <td>170.40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3213840074316400693</td>
      <td>2016-08-01</td>
      <td>20</td>
      <td>Electronics Accessory Pouch</td>
      <td>3.99</td>
      <td>79.80</td>
      <td>170.40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3213840074316400693</td>
      <td>2016-08-01</td>
      <td>20</td>
      <td>Badge Holder</td>
      <td>1.59</td>
      <td>31.80</td>
      <td>170.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2976178532750719771</td>
      <td>2016-08-01</td>
      <td>4</td>
      <td>Maze Pen</td>
      <td>0.99</td>
      <td>3.96</td>
      <td>19.93</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27320</th>
      <td>2346359290191771618</td>
      <td>2017-05-03</td>
      <td>1</td>
      <td>Google Tote Bag</td>
      <td>9.99</td>
      <td>9.99</td>
      <td>173.44</td>
    </tr>
    <tr>
      <th>27321</th>
      <td>2346359290191771618</td>
      <td>2017-05-03</td>
      <td>10</td>
      <td>Maze Pen</td>
      <td>0.99</td>
      <td>9.90</td>
      <td>173.44</td>
    </tr>
    <tr>
      <th>27322</th>
      <td>2346359290191771618</td>
      <td>2017-05-03</td>
      <td>1</td>
      <td>Badge Holder</td>
      <td>1.99</td>
      <td>1.99</td>
      <td>173.44</td>
    </tr>
    <tr>
      <th>27323</th>
      <td>2346359290191771618</td>
      <td>2017-05-03</td>
      <td>10</td>
      <td>Galaxy Screen Cleaning Cloth</td>
      <td>1.99</td>
      <td>19.90</td>
      <td>173.44</td>
    </tr>
    <tr>
      <th>27324</th>
      <td>2346359290191771618</td>
      <td>2017-05-03</td>
      <td>1</td>
      <td>Google Tube Power Bank</td>
      <td>16.99</td>
      <td>16.99</td>
      <td>173.44</td>
    </tr>
  </tbody>
</table>
<p>27325 rows × 7 columns</p>
</div>




```python
# Create our 90-day dataset
temporal_out_df = google_merch_store \
    [google_merch_store['date'] > cutoff]
temporal_out_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fullVisitorId</th>
      <th>date</th>
      <th>quantity</th>
      <th>product_name</th>
      <th>price</th>
      <th>product_revenue</th>
      <th>total_transaction_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27325</th>
      <td>2618320847776659905</td>
      <td>2017-05-04</td>
      <td>1</td>
      <td>Google Women's Vintage Hero Tee Black</td>
      <td>10.63</td>
      <td>10.63</td>
      <td>17.63</td>
    </tr>
    <tr>
      <th>27326</th>
      <td>6720643639676411949</td>
      <td>2017-05-04</td>
      <td>1</td>
      <td>Google Onesie Green</td>
      <td>19.19</td>
      <td>19.19</td>
      <td>27.19</td>
    </tr>
    <tr>
      <th>27327</th>
      <td>7412836405745272778</td>
      <td>2017-05-04</td>
      <td>5</td>
      <td>Google Collapsible Duffel Black</td>
      <td>17.59</td>
      <td>87.95</td>
      <td>96.95</td>
    </tr>
    <tr>
      <th>27328</th>
      <td>372706149688864468</td>
      <td>2017-05-04</td>
      <td>4</td>
      <td>BLM Sweatshirt</td>
      <td>33.59</td>
      <td>134.36</td>
      <td>236.13</td>
    </tr>
    <tr>
      <th>27329</th>
      <td>372706149688864468</td>
      <td>2017-05-04</td>
      <td>3</td>
      <td>BLM Sweatshirt</td>
      <td>33.59</td>
      <td>100.77</td>
      <td>236.13</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36741</th>
      <td>8016003971239765913</td>
      <td>2017-08-01</td>
      <td>2</td>
      <td>Android Men's Short Sleeve Hero Tee White</td>
      <td>6.80</td>
      <td>13.60</td>
      <td>131.96</td>
    </tr>
    <tr>
      <th>36742</th>
      <td>8016003971239765913</td>
      <td>2017-08-01</td>
      <td>1</td>
      <td>Women's Performance Full Zip Jacket Black</td>
      <td>67.19</td>
      <td>67.19</td>
      <td>131.96</td>
    </tr>
    <tr>
      <th>36743</th>
      <td>8016003971239765913</td>
      <td>2017-08-01</td>
      <td>1</td>
      <td>Android Men's Engineer Short Sleeve Tee Charcoal</td>
      <td>15.99</td>
      <td>15.99</td>
      <td>131.96</td>
    </tr>
    <tr>
      <th>36744</th>
      <td>8016003971239765913</td>
      <td>2017-08-01</td>
      <td>1</td>
      <td>Google Infant Short Sleeve Tee Red</td>
      <td>13.59</td>
      <td>13.59</td>
      <td>131.96</td>
    </tr>
    <tr>
      <th>36745</th>
      <td>8016003971239765913</td>
      <td>2017-08-01</td>
      <td>1</td>
      <td>Android Infant Short Sleeve Tee Pewter</td>
      <td>13.59</td>
      <td>13.59</td>
      <td>131.96</td>
    </tr>
  </tbody>
</table>
<p>9421 rows × 7 columns</p>
</div>



### 5.3 RFM analysis

RFM analysis is a customer segmentation technique commonly used in marketing and analytics to categorize customers based on their behavior and interactions with a business. The acronym "RFM" stands for Recency, Frequency, and Monetary Value, which are three key dimensions used to evaluate and understand customer behavior. Here's a brief overview of each component:

#### Recency (R):

`Definition:` Recency refers to how recently a customer has interacted or made a purchase.\
`Calculation:` It is typically measured by the time elapsed since the customer's last purchase, activity, or interaction.\
`Objective:` Recent customers are often more engaged, so businesses may want to identify and target those who have interacted with the company recently.

#### Frequency (F):

`Definition:` Frequency measures how often a customer interacts or makes purchases.\
`Calculation:` It involves counting the number of transactions or interactions within a specific period.\
`Objective:` Higher frequency may indicate more loyal and engaged customers. Businesses may want to reward or incentivize customers with high frequency.

#### Monetary Value (M):

`Definition:` Monetary Value represents the total value of a customer's transactions.\
`Calculation:` It involves summing up the monetary value of all transactions made by a customer.\
`Objective:` Identifying customers with high monetary value helps businesses focus on their most valuable segments and tailor marketing strategies accordingly.

RFM analysis provides actionable insights into customer behavior, allowing businesses to develop targeted strategies for customer retention, acquisition, and overall business growth.

### 5.3.1 Create target variables from our 90-day data


```python
# Make Targets from out data 
targets_df = temporal_out_df \
    .drop('quantity', axis=1) \
    .drop('product_name', axis=1) \
    .drop('product_revenue', axis=1) \
    .drop('total_transaction_revenue', axis=1) \
    .groupby('fullVisitorId') \
    .sum() \
    .rename({'price': 'spend_90_total'}, axis = 1) \
    .assign(spend_90_flag = 1)

targets_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spend_90_total</th>
      <th>spend_90_flag</th>
    </tr>
    <tr>
      <th>fullVisitorId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0003961110741104601</th>
      <td>4.98</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0006911334202687206</th>
      <td>58.50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0010295111715775250</th>
      <td>2.39</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0012561433643490595</th>
      <td>4.39</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0014262055593378383</th>
      <td>77.30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9973195165804180005</th>
      <td>0.99</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9973665079624172058</th>
      <td>78.95</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9974351919673138742</th>
      <td>53.60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9982700667464896535</th>
      <td>33.59</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99879093370825436</th>
      <td>3.50</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2639 rows × 2 columns</p>
</div>



So we can see that out of a total of 9,995 customers, only 2,639 made a purchase in the final 90 day period.

### 5.3.2 Create recency (date) features 


```python
max_date = temporal_in_df['date'].max()
max_date
```




    Timestamp('2017-05-03 00:00:00')




```python
recency_features_df = temporal_in_df \
    [['fullVisitorId', 'date']] \
    .groupby('fullVisitorId') \
    .apply(
        lambda x: (x['date'].max() - max_date) / pd.to_timedelta(1, "day")
    ) \
    .to_frame() \
    .set_axis(["recency"], axis=1)

recency_features_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recency</th>
    </tr>
    <tr>
      <th>fullVisitorId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000213131142648941</th>
      <td>-5.0</td>
    </tr>
    <tr>
      <th>0002871498069867123</th>
      <td>-253.0</td>
    </tr>
    <tr>
      <th>0003450834640354121</th>
      <td>-197.0</td>
    </tr>
    <tr>
      <th>000435324061339869</th>
      <td>-195.0</td>
    </tr>
    <tr>
      <th>0007617910709180468</th>
      <td>-142.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>9991633376050115277</th>
      <td>-75.0</td>
    </tr>
    <tr>
      <th>9994767073213036303</th>
      <td>-267.0</td>
    </tr>
    <tr>
      <th>9997409246962677759</th>
      <td>-146.0</td>
    </tr>
    <tr>
      <th>9998597322098588317</th>
      <td>-275.0</td>
    </tr>
    <tr>
      <th>9998996003043230595</th>
      <td>-168.0</td>
    </tr>
  </tbody>
</table>
<p>7478 rows × 1 columns</p>
</div>



The above table shows, for each customer, how recent their most recent purchase was, with reference to the final purchase date of our recency dataset, 3 May 2017.


```python
min_date = temporal_in_df['date'].min()
min_date
```




    Timestamp('2016-08-01 00:00:00')




```python
date_range = max_date - min_date
date_range
```




    Timedelta('275 days 00:00:00')



### 5.3.3 Create frequency (count) features


```python
frequency_features_df = temporal_in_df \
    [['fullVisitorId', 'date']] \
    .groupby('fullVisitorId') \
    .count() \
    .set_axis(['frequency'], axis=1)

frequency_features_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>frequency</th>
    </tr>
    <tr>
      <th>fullVisitorId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000213131142648941</th>
      <td>1</td>
    </tr>
    <tr>
      <th>0002871498069867123</th>
      <td>1</td>
    </tr>
    <tr>
      <th>0003450834640354121</th>
      <td>5</td>
    </tr>
    <tr>
      <th>000435324061339869</th>
      <td>1</td>
    </tr>
    <tr>
      <th>0007617910709180468</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>9991633376050115277</th>
      <td>1</td>
    </tr>
    <tr>
      <th>9994767073213036303</th>
      <td>6</td>
    </tr>
    <tr>
      <th>9997409246962677759</th>
      <td>5</td>
    </tr>
    <tr>
      <th>9998597322098588317</th>
      <td>2</td>
    </tr>
    <tr>
      <th>9998996003043230595</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>7478 rows × 1 columns</p>
</div>



This shows how many purchase transactions each customer made.

### 5.3.4 Create monetary (price) features


```python
# average spend per transaction
price_features_df = temporal_in_df \
    .groupby('fullVisitorId') \
    .aggregate(
        {
            'price': ["sum", "mean"]
        }
    ) \
    .set_axis(['price_sum', 'price_mean'], axis = 1)

price_features_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_sum</th>
      <th>price_mean</th>
    </tr>
    <tr>
      <th>fullVisitorId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000213131142648941</th>
      <td>33.59</td>
      <td>33.590000</td>
    </tr>
    <tr>
      <th>0002871498069867123</th>
      <td>5.99</td>
      <td>5.990000</td>
    </tr>
    <tr>
      <th>0003450834640354121</th>
      <td>57.95</td>
      <td>11.590000</td>
    </tr>
    <tr>
      <th>000435324061339869</th>
      <td>44.79</td>
      <td>44.790000</td>
    </tr>
    <tr>
      <th>0007617910709180468</th>
      <td>18.99</td>
      <td>18.990000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9991633376050115277</th>
      <td>33.59</td>
      <td>33.590000</td>
    </tr>
    <tr>
      <th>9994767073213036303</th>
      <td>51.94</td>
      <td>8.656667</td>
    </tr>
    <tr>
      <th>9997409246962677759</th>
      <td>32.36</td>
      <td>6.472000</td>
    </tr>
    <tr>
      <th>9998597322098588317</th>
      <td>3.24</td>
      <td>1.620000</td>
    </tr>
    <tr>
      <th>9998996003043230595</th>
      <td>59.98</td>
      <td>29.990000</td>
    </tr>
  </tbody>
</table>
<p>7478 rows × 2 columns</p>
</div>



This gives us the average spend per transaction for each customer.

### 5.3.5 Combine all features


```python
features_df = pd.concat(
    [recency_features_df, frequency_features_df, price_features_df], axis = 1
) \
    .merge(
        targets_df, 
        left_index  = True, 
        right_index = True, 
        how         = "left"
    ) \
    .fillna(0) # where no spend populate with 0

features_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recency</th>
      <th>frequency</th>
      <th>price_sum</th>
      <th>price_mean</th>
      <th>spend_90_total</th>
      <th>spend_90_flag</th>
    </tr>
    <tr>
      <th>fullVisitorId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000213131142648941</th>
      <td>-5.0</td>
      <td>1</td>
      <td>33.59</td>
      <td>33.590000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>0002871498069867123</th>
      <td>-253.0</td>
      <td>1</td>
      <td>5.99</td>
      <td>5.990000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>0003450834640354121</th>
      <td>-197.0</td>
      <td>5</td>
      <td>57.95</td>
      <td>11.590000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>000435324061339869</th>
      <td>-195.0</td>
      <td>1</td>
      <td>44.79</td>
      <td>44.790000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>0007617910709180468</th>
      <td>-142.0</td>
      <td>1</td>
      <td>18.99</td>
      <td>18.990000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9991633376050115277</th>
      <td>-75.0</td>
      <td>1</td>
      <td>33.59</td>
      <td>33.590000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9994767073213036303</th>
      <td>-267.0</td>
      <td>6</td>
      <td>51.94</td>
      <td>8.656667</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9997409246962677759</th>
      <td>-146.0</td>
      <td>5</td>
      <td>32.36</td>
      <td>6.472000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9998597322098588317</th>
      <td>-275.0</td>
      <td>2</td>
      <td>3.24</td>
      <td>1.620000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9998996003043230595</th>
      <td>-168.0</td>
      <td>2</td>
      <td>59.98</td>
      <td>29.990000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>7478 rows × 6 columns</p>
</div>



## 6. Machine learning

Now that we have our recency, frequency and monetary features prepared, the next step is to leverage machine learning `regression models` to help us predict future customer spend based on these features.

In an ecommerce context, these regression models can provide valuable insights for:

- `Demand Forecasting:` Predicting future sales to optimize inventory levels and prevent stockouts or overstock situations.

- `Pricing Strategies:` Analyzing the impact of different pricing models and discounts on sales to determine the most effective pricing strategy.

- `Customer Segmentation:` Understanding customer segments based on purchasing behavior to tailor marketing messages and promotions.

- `Product Recommendations:` Enhancing personalized product recommendations based on individual customer preferences and historical data.

- `Marketing Optimization:` Allocating marketing budgets effectively by identifying the most influential factors driving sales.



### 6.1 XGBoost

[XGBoost (Extreme Gradient Boosting)](https://xgboost.readthedocs.io/en/stable/) is a powerful and efficient machine learning algorithm that belongs to the gradient boosting family. It has gained popularity for its high performance and effectiveness in various predictive modeling tasks.


```python
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

X = features_df[['recency', 'frequency', 'price_sum', 'price_mean']]
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recency</th>
      <th>frequency</th>
      <th>price_sum</th>
      <th>price_mean</th>
    </tr>
    <tr>
      <th>fullVisitorId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000213131142648941</th>
      <td>-5.0</td>
      <td>1</td>
      <td>33.59</td>
      <td>33.590000</td>
    </tr>
    <tr>
      <th>0002871498069867123</th>
      <td>-253.0</td>
      <td>1</td>
      <td>5.99</td>
      <td>5.990000</td>
    </tr>
    <tr>
      <th>0003450834640354121</th>
      <td>-197.0</td>
      <td>5</td>
      <td>57.95</td>
      <td>11.590000</td>
    </tr>
    <tr>
      <th>000435324061339869</th>
      <td>-195.0</td>
      <td>1</td>
      <td>44.79</td>
      <td>44.790000</td>
    </tr>
    <tr>
      <th>0007617910709180468</th>
      <td>-142.0</td>
      <td>1</td>
      <td>18.99</td>
      <td>18.990000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9991633376050115277</th>
      <td>-75.0</td>
      <td>1</td>
      <td>33.59</td>
      <td>33.590000</td>
    </tr>
    <tr>
      <th>9994767073213036303</th>
      <td>-267.0</td>
      <td>6</td>
      <td>51.94</td>
      <td>8.656667</td>
    </tr>
    <tr>
      <th>9997409246962677759</th>
      <td>-146.0</td>
      <td>5</td>
      <td>32.36</td>
      <td>6.472000</td>
    </tr>
    <tr>
      <th>9998597322098588317</th>
      <td>-275.0</td>
      <td>2</td>
      <td>3.24</td>
      <td>1.620000</td>
    </tr>
    <tr>
      <th>9998996003043230595</th>
      <td>-168.0</td>
      <td>2</td>
      <td>59.98</td>
      <td>29.990000</td>
    </tr>
  </tbody>
</table>
<p>7478 rows × 4 columns</p>
</div>



### 6.2 Regression model

#### 6.2.1 Build and train a model


```python
# define our target variable
y_spend = features_df['spend_90_total']

# Regression model as we are trying to predict customer spend in next-90 days
xgb_reg_spec = XGBRegressor(
    objective="reg:squarederror",   
    random_state=123
)

xgb_reg_model = GridSearchCV(
    estimator=xgb_reg_spec, 
    param_grid=dict(
        learning_rate = [0.01, 0.1, 0.3, 0.5]
    ),
    scoring = 'neg_mean_absolute_error',
    refit   = True, # creates a 6th model which used for pred in production based on best of 5 models
    cv      = 5 # 5 fold cross-validation
)
```


```python
# Train our model on the test day
xgb_reg_model.fit(X, y_spend)
```

### 6.2.2 Model evaluation


```python
xgb_reg_model.best_score_
```




    -2.2946875595942826



The interpretation is that are model is out by around $2 per transaction, which seems pretty reasonable.


```python
xgb_reg_model.best_params_
```




    {'learning_rate': 0.01}




```python
xgb_reg_model.best_estimator_
```

### 6.2.3 Predicted customer spend ($) in next 90 days


```python
predictions_reg = xgb_reg_model.predict(X)
predictions_reg
```




    array([1.1863576 , 0.26471087, 0.26471087, ..., 0.26471087, 0.26471087,
           0.26471087], dtype=float32)




```python
len(predictions_reg)
```




    7478



### 6.3 Classification model

### 6.3.1 Probability of customer spend in the next 90 days

We will once again leverage XGBoost but this time a `classification` model is required as we are not trying to predict a $ value, but whether they will spend or not. There are a wide range of machine learning classification models, some of which are outlined below: 
    
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Neural Networks
- Ensemble Methods
- Association Rule Mining (e.g., Apriori Algorithm)

By leveraging these models, ecommerce businesses can enhance customer experiences, optimize marketing strategies, and improve overall operational efficiency.

### 6.3.2 Build and train a model


```python
# # define our target variable
y_prob = features_df['spend_90_flag']
```


```python
# Classification model as we are trying to predict whether customer spends in next-90 days or not
xgb_clf_spec = XGBClassifier(
    objective    = "binary:logistic",   
    random_state = 123
)

xgb_clf_model = GridSearchCV(
    estimator=xgb_clf_spec, 
    param_grid=dict(
        learning_rate = [0.01, 0.1, 0.3, 0.5]
    ),
    scoring = 'roc_auc',
    refit   = True,
    cv      = 5
)
```


```python
# Train our model on the test data
xgb_clf_model.fit(X, y_prob)
```

### 6.3.3 Model evaluation


```python
xgb_clf_model.best_score_
```




    0.837403308391585




```python
xgb_clf_model.best_params_
```




    {'learning_rate': 0.1}




```python
xgb_clf_model.best_estimator_
```

### 6.3.4 Predict probability of customer spend next-90 days


```python
predictions_clf = xgb_clf_model.predict_proba(X) # predict prob rather than score
predictions_clf
```




    array([[9.7599518e-01, 2.4004841e-02],
           [9.9948597e-01, 5.1402434e-04],
           [9.9950981e-01, 4.9017661e-04],
           ...,
           [9.9914837e-01, 8.5162191e-04],
           [9.9901974e-01, 9.8024833e-04],
           [9.9941909e-01, 5.8090949e-04]], dtype=float32)



The first value in the outputed array represents the probabilty of not making a purchase, with the second value representing the reciprocal, i.e. the probability of making a purchase. To illustrate, the first customer has a 2.4% probability of making a purchase in the next 90-days, the second customer, 5.1%, the third 4.9% and so on.

### 6.4 Feature importance

So which customer features can give us an insight into their future spending behaviour? If we can profile our customers we can better understand their unique preferences and enhance user experience which will ultimately translate into recurring future revenue streams.

### 6.4.1 Regression model


```python
# dictionary of relative importance of features
imp_spend_amount_dict = xgb_reg_model \
    .best_estimator_ \
    .get_booster() \
    .get_score(importance_type = 'gain') 
imp_spend_amount_dict
```




    {'recency': 21329.5546875,
     'frequency': 26971.447265625,
     'price_sum': 24244.16796875,
     'price_mean': 13376.1591796875}




```python
# create a DataFrame from the dictionary
imp_spend_amount_df = pd.DataFrame(
    data  = {
        'feature':list(imp_spend_amount_dict.keys()),
        'value':list(imp_spend_amount_dict.values())
    }
) \
    .assign(
        feature = lambda x: cat.cat_reorder(x['feature'] , x['value'])
    )
imp_spend_amount_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>recency</td>
      <td>21329.554688</td>
    </tr>
    <tr>
      <th>1</th>
      <td>frequency</td>
      <td>26971.447266</td>
    </tr>
    <tr>
      <th>2</th>
      <td>price_sum</td>
      <td>24244.167969</td>
    </tr>
    <tr>
      <th>3</th>
      <td>price_mean</td>
      <td>13376.159180</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize feature importance
pn.ggplot(data=imp_spend_amount_df) + \
    pn.aes(x='feature', y='value') + \
    pn.geom_col() + \
    pn.coord_flip()
```


    
![png](images/output_117_0.png)
    





    <Figure Size: (1920 x 1440)>



Intuitively, frequent spenders are generally likely to be a good indicator of future spend. This is corroborated in the above feature importance plot.

### 6.4.2 Classification model


```python
# dictionary of relative importance of features
imp_spend_prob_dict = xgb_clf_model \
    .best_estimator_ \
    .get_booster() \
    .get_score(importance_type = 'gain') 
imp_spend_prob_dict
```




    {'recency': 2.1909291744232178,
     'frequency': 1.2872744798660278,
     'price_sum': 1.2381023168563843,
     'price_mean': 1.1211071014404297}




```python
# create a DataFrame from the dictionary
imp_spend_prob_df = pd.DataFrame(
    data  = {
        'feature':list(imp_spend_prob_dict.keys()),
        'value':list(imp_spend_prob_dict.values())
    }
) \
    .assign(
        feature = lambda x: cat.cat_reorder(x['feature'] , x['value'])
    )
imp_spend_prob_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>recency</td>
      <td>2.190929</td>
    </tr>
    <tr>
      <th>1</th>
      <td>frequency</td>
      <td>1.287274</td>
    </tr>
    <tr>
      <th>2</th>
      <td>price_sum</td>
      <td>1.238102</td>
    </tr>
    <tr>
      <th>3</th>
      <td>price_mean</td>
      <td>1.121107</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize feature importance
pn.ggplot(data = imp_spend_prob_df) + \
    pn.aes(x='feature', y='value') + \
    pn.geom_col() + \
    pn.coord_flip() 
```


    
![png](images/output_122_0.png)
    





    <Figure Size: (1920 x 1440)>



In terms of the probability of future spend, if a customer has spent recently and frequently, then these are strong indicators. The amount spent and average spend are not considered to be important by the model.

## 7. Pickle (save) our predictions, features, and models


```python
# Combine our predictions
predictions_df = pd.concat(
    [
        pd.DataFrame(predictions_reg).set_axis(['pred_spend'], axis=1),
        pd.DataFrame(predictions_clf)[[1]].set_axis(['pred_prob'], axis=1),
        features_df.reset_index()
    ], 
    axis=1
)
predictions_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_spend</th>
      <th>pred_prob</th>
      <th>fullVisitorId</th>
      <th>recency</th>
      <th>frequency</th>
      <th>price_sum</th>
      <th>price_mean</th>
      <th>spend_90_total</th>
      <th>spend_90_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.186358</td>
      <td>0.024005</td>
      <td>0000213131142648941</td>
      <td>-5.0</td>
      <td>1</td>
      <td>33.59</td>
      <td>33.590000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.264711</td>
      <td>0.000514</td>
      <td>0002871498069867123</td>
      <td>-253.0</td>
      <td>1</td>
      <td>5.99</td>
      <td>5.990000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.264711</td>
      <td>0.000490</td>
      <td>0003450834640354121</td>
      <td>-197.0</td>
      <td>5</td>
      <td>57.95</td>
      <td>11.590000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.264711</td>
      <td>0.000253</td>
      <td>000435324061339869</td>
      <td>-195.0</td>
      <td>1</td>
      <td>44.79</td>
      <td>44.790000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.264711</td>
      <td>0.000409</td>
      <td>0007617910709180468</td>
      <td>-142.0</td>
      <td>1</td>
      <td>18.99</td>
      <td>18.990000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7473</th>
      <td>0.361555</td>
      <td>0.012704</td>
      <td>9991633376050115277</td>
      <td>-75.0</td>
      <td>1</td>
      <td>33.59</td>
      <td>33.590000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7474</th>
      <td>0.264711</td>
      <td>0.000526</td>
      <td>9994767073213036303</td>
      <td>-267.0</td>
      <td>6</td>
      <td>51.94</td>
      <td>8.656667</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7475</th>
      <td>0.264711</td>
      <td>0.000852</td>
      <td>9997409246962677759</td>
      <td>-146.0</td>
      <td>5</td>
      <td>32.36</td>
      <td>6.472000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7476</th>
      <td>0.264711</td>
      <td>0.000980</td>
      <td>9998597322098588317</td>
      <td>-275.0</td>
      <td>2</td>
      <td>3.24</td>
      <td>1.620000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7477</th>
      <td>0.264711</td>
      <td>0.000581</td>
      <td>9998996003043230595</td>
      <td>-168.0</td>
      <td>2</td>
      <td>59.98</td>
      <td>29.990000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>7478 rows × 9 columns</p>
</div>




```python
import pickle
```


```python
with open("artifacts/predictions_df.pkl", "wb") as file:
    pickle.dump(predictions_df, file, protocol=4)
```


```python
# Pickle our predictions
predictions_df.to_pickle("artifacts/predictions_df.pkl")
```


```python
with open("artifacts/imp_spend_amount_df.pkl", "wb") as file:
    pickle.dump(imp_spend_amount_df, file, protocol=4)
```


```python
with open("artifacts/imp_spend_prob_df.pkl", "wb") as file:
    pickle.dump(imp_spend_prob_df, file, protocol=4)
```


```python
### Save our Feature Importances
imp_spend_amount_df.to_pickle("artifacts/imp_spend_amount_df.pkl")
imp_spend_prob_df.to_pickle("artifacts/imp_spend_prob_df.pkl")
```


```python
### Save Models
joblib.dump(xgb_reg_model, 'artifacts/xgb_reg_model.pkl')
joblib.dump(xgb_clf_model, 'artifacts/xgb_clf_model.pkl')
```


```python
model = joblib.load('artifacts/xgb_reg_model.pkl')
model.predict(X)
```




    array([1.1863576 , 0.26471087, 0.26471087, ..., 0.26471087, 0.26471087,
           0.26471087], dtype=float32)



## 8. Actionable outcomes

### Q1. Which customers have the highest spend `probability` in next 90-days? 

- Target for new products similar to what they have purchased in the past.


```python
predictions_df \
    .sort_values('pred_prob', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_spend</th>
      <th>pred_prob</th>
      <th>fullVisitorId</th>
      <th>recency</th>
      <th>frequency</th>
      <th>price_sum</th>
      <th>price_mean</th>
      <th>spend_90_total</th>
      <th>spend_90_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3747</th>
      <td>103.428444</td>
      <td>0.849751</td>
      <td>4984366501121503466</td>
      <td>-23.0</td>
      <td>55</td>
      <td>320.57</td>
      <td>5.828545</td>
      <td>258.50</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6156</th>
      <td>258.405396</td>
      <td>0.836090</td>
      <td>8197879643797712877</td>
      <td>-7.0</td>
      <td>65</td>
      <td>2052.37</td>
      <td>31.574923</td>
      <td>657.03</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1430</th>
      <td>225.969971</td>
      <td>0.827522</td>
      <td>1957458976293878100</td>
      <td>-14.0</td>
      <td>109</td>
      <td>2333.35</td>
      <td>21.406881</td>
      <td>338.26</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3724</th>
      <td>34.220909</td>
      <td>0.806135</td>
      <td>4950411203281265700</td>
      <td>0.0</td>
      <td>23</td>
      <td>120.62</td>
      <td>5.244348</td>
      <td>92.35</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5888</th>
      <td>69.604485</td>
      <td>0.782023</td>
      <td>7813149961404844386</td>
      <td>0.0</td>
      <td>48</td>
      <td>1906.50</td>
      <td>39.718750</td>
      <td>175.79</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1307</th>
      <td>0.264711</td>
      <td>0.000214</td>
      <td>1799218307967476916</td>
      <td>-202.0</td>
      <td>1</td>
      <td>16.99</td>
      <td>16.990000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2573</th>
      <td>0.264711</td>
      <td>0.000214</td>
      <td>3461504855909388246</td>
      <td>-206.0</td>
      <td>1</td>
      <td>16.99</td>
      <td>16.990000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4435</th>
      <td>0.264711</td>
      <td>0.000214</td>
      <td>5844997119511169482</td>
      <td>-216.0</td>
      <td>1</td>
      <td>18.99</td>
      <td>18.990000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6228</th>
      <td>0.264711</td>
      <td>0.000214</td>
      <td>8292732469404938023</td>
      <td>-227.0</td>
      <td>1</td>
      <td>16.99</td>
      <td>16.990000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3206</th>
      <td>0.264711</td>
      <td>0.000214</td>
      <td>4259785355346404932</td>
      <td>-210.0</td>
      <td>1</td>
      <td>18.99</td>
      <td>18.990000</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>7478 rows × 9 columns</p>
</div>



We can see that the model seems to be working well in terms of identifying those customers that are likely to spend in the next-90 days. The top 5 customers above ranked by probability of spend all actually spent. Conversely, the bottom 5 customers didn't spend.

### Q2. Which customers have recently purchased (within 90 days) but are unlikely to buy (probability less than 30%)? 

- Incentivize actions to increase probability.
- Provide discounts, encourage referring a friend, nurture by letting them know what's coming.


```python
predictions_df \
    [
        predictions_df['recency'] > -90
    ] \
    [
        predictions_df['pred_prob'] < 0.30
    ] \
    .sort_values('pred_prob', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_spend</th>
      <th>pred_prob</th>
      <th>fullVisitorId</th>
      <th>recency</th>
      <th>frequency</th>
      <th>price_sum</th>
      <th>price_mean</th>
      <th>spend_90_total</th>
      <th>spend_90_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2384</th>
      <td>2.970100</td>
      <td>0.299618</td>
      <td>3197533100947860058</td>
      <td>-33.0</td>
      <td>7</td>
      <td>153.44</td>
      <td>21.920</td>
      <td>86.54</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>419</th>
      <td>4.334545</td>
      <td>0.291470</td>
      <td>0554420125524525961</td>
      <td>-10.0</td>
      <td>3</td>
      <td>52.77</td>
      <td>17.590</td>
      <td>172.37</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3406</th>
      <td>2.242671</td>
      <td>0.291250</td>
      <td>4515197722749947898</td>
      <td>-3.0</td>
      <td>2</td>
      <td>28.78</td>
      <td>14.390</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1861</th>
      <td>2.242671</td>
      <td>0.291250</td>
      <td>2518379317255532090</td>
      <td>-3.0</td>
      <td>2</td>
      <td>28.78</td>
      <td>14.390</td>
      <td>28.78</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5688</th>
      <td>3.153815</td>
      <td>0.290945</td>
      <td>7548511681521037018</td>
      <td>-2.0</td>
      <td>4</td>
      <td>33.98</td>
      <td>8.495</td>
      <td>42.97</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6446</th>
      <td>0.594005</td>
      <td>0.001943</td>
      <td>8590369633898567459</td>
      <td>-35.0</td>
      <td>1</td>
      <td>2.39</td>
      <td>2.390</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5561</th>
      <td>0.594005</td>
      <td>0.001943</td>
      <td>7390358029425621068</td>
      <td>-35.0</td>
      <td>1</td>
      <td>1.99</td>
      <td>1.990</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2165</th>
      <td>0.361555</td>
      <td>0.001671</td>
      <td>293387037477216156</td>
      <td>-72.0</td>
      <td>1</td>
      <td>1.99</td>
      <td>1.990</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2666</th>
      <td>0.361555</td>
      <td>0.001671</td>
      <td>3573113591289892546</td>
      <td>-74.0</td>
      <td>1</td>
      <td>1.99</td>
      <td>1.990</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1490</th>
      <td>0.361555</td>
      <td>0.001671</td>
      <td>2037170738057013329</td>
      <td>-73.0</td>
      <td>1</td>
      <td>1.99</td>
      <td>1.990</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2324 rows × 9 columns</p>
</div>



### Q3. Missed opportunities: Big spenders that could be unlocked 

- Send bundle offers encouraging volume purchases
- Focus on missed opportunities


```python
# identify those customers predicted to spend but did not
predictions_df \
    [
        predictions_df['spend_90_total'] == 0.0
    ] \
    .sort_values('pred_spend', ascending=False) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_spend</th>
      <th>pred_prob</th>
      <th>fullVisitorId</th>
      <th>recency</th>
      <th>frequency</th>
      <th>price_sum</th>
      <th>price_mean</th>
      <th>spend_90_total</th>
      <th>spend_90_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2955</th>
      <td>81.598740</td>
      <td>0.126212</td>
      <td>3955127543379144640</td>
      <td>-2.0</td>
      <td>8</td>
      <td>358.32</td>
      <td>44.790000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2565</th>
      <td>33.736187</td>
      <td>0.063039</td>
      <td>3449924104971285851</td>
      <td>-91.0</td>
      <td>14</td>
      <td>483.86</td>
      <td>34.561429</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5696</th>
      <td>33.736187</td>
      <td>0.085885</td>
      <td>7561014297963838461</td>
      <td>-58.0</td>
      <td>18</td>
      <td>522.82</td>
      <td>29.045556</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>30.029305</td>
      <td>0.017011</td>
      <td>3916992730920009646</td>
      <td>-64.0</td>
      <td>7</td>
      <td>416.73</td>
      <td>59.532857</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4026</th>
      <td>30.029305</td>
      <td>0.009925</td>
      <td>5349155616428631188</td>
      <td>-76.0</td>
      <td>13</td>
      <td>391.87</td>
      <td>30.143846</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1767</th>
      <td>-0.646184</td>
      <td>0.025690</td>
      <td>2396848817613598114</td>
      <td>-59.0</td>
      <td>44</td>
      <td>1416.56</td>
      <td>32.194545</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5808</th>
      <td>-0.646184</td>
      <td>0.023305</td>
      <td>7713012430069756739</td>
      <td>-62.0</td>
      <td>18</td>
      <td>783.82</td>
      <td>43.545556</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>751</th>
      <td>-0.646184</td>
      <td>0.017749</td>
      <td>1045033759778661078</td>
      <td>-86.0</td>
      <td>22</td>
      <td>657.38</td>
      <td>29.880909</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7001</th>
      <td>-0.646184</td>
      <td>0.013225</td>
      <td>9349161317881541522</td>
      <td>-50.0</td>
      <td>29</td>
      <td>896.71</td>
      <td>30.921034</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3873</th>
      <td>-0.925417</td>
      <td>0.007162</td>
      <td>5149788969578895545</td>
      <td>-66.0</td>
      <td>9</td>
      <td>340.71</td>
      <td>37.856667</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>7356 rows × 9 columns</p>
</div>



## Communication of findings

### 9.1 e-commerce analytics dashboard using Dash for viewing locally

We can create an interactive analytics dashboard which can be shared with the Marketing Department to enable targeted campaigns to be launched, by following these 3 steps:

1. Create a virtual environment from the command line :

        conda env create -f environment.yml
    
This minimizes the risk of conflicting package dependencies or dependency hell! If you do end up down there then you can refer to [the documentation](https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts). Good luck!

2. Activate the environment:

        conda activate google_merch_clv (this is the name specified in the yaml file below - can change this as desired)
        

3. Launch the app:

        python app.py    
        
The dashboard can then be viewed by visiting http://127.0.0.1:8050/ in your web browser.

### 9.2 Deployment of Dash e-commerce app

Now that we have tested our app and it runs successfully locally, it's time to deploy it! There are a variety of platforms around, most have some kind of pay wall. [Python Anywhere](https://www.pythonanywhere.com/) offer a basic free hosting service but tou might find you need a bit more resource. At the time of writing $7/$8 per month allows you to have two apps hosted and gives you 2000s of CPU usage and 2.0GB of file storage.

It has a very user-friendly interface and after reviewing some online video tutorials and debugging using the support forums and stackoverflow, i was able to get [my app](http://stephen137.pythonanywhere.com/) running.

## 10. Key takeaways

This project demonstrated my proficiency in:

- `business analytics`: the project was framed by two business problems/questions which are relevant to ecommerce businesses
- `SQL:` transform a complex real-life BigData datset with nested fields, into meaningful insights with regard to customer behaviour
- `pandas`: data cleansing and wrangling
- `machine learning:` created a BASELINE regression and classification model and performed feature engineering to enhance model performance
- `visualization`: using the plotly and plotnine libraries
- `app creation` : deployment of an interactive Dash app

At the time of writing I am currently working on a custoomer segmentation and recommendation engine project which makes use of the same dataset.
