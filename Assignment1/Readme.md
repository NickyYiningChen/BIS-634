# Assignment 1
## Problem 1
Below is the body of function temp_tester
```python
def temp_tester(normal_temp):
    def judge_normal(temp):
        if abs(temp - normal_temp) <= 1:
            print('True -- i.e. not a fever')
            return True

        if abs(temp - normal_temp) > 1:
            temp_f = normal_temp * 1.8 + 32
            if abs(temp - temp_f) <= 1:
                print('False -- normal in degrees F but our reference temp was in degrees C')
            elif temp - normal_temp > 0:
                print('False -- this would be a severe fever')
            else:
                print('False -- too low')
            return False
    return judge_normal
```
Below are the tests and answers
```python
>>>human_tester = temp_tester(37)
>>>chicken_tester = temp_tester(41.1)

>>>chicken_tester(42)
>>>human_tester(42)   
>>>chicken_tester(43) 
>>>human_tester(35)   
>>human_tester(98.6) 

True -- i.e. not a fever
False -- this would be a severe fever
False -- this would be a severe fever
False -- too low
False -- normal in degrees F but our reference temp was in degrees C
```
## Problem 2
```python
import pandas as pd
import numpy as np
import sqlite3
with sqlite3.connect("hw1-population.db") as db:
    data = pd.read_sql_query("SELECT * FROM population", db)
```
>What columns does it have? 

There are four columns: name, age, weight, eyecolor
```python
# check columns
data.head()
```
>How many rows (think: people) does it have? 

It has 152361 people
```python
# check row number
len(data)
```
> Examine the distribution of the ages in the dataset

Mean = 39.510527927396524

Standard Deviation = 24.152760068601445

Min = 0.0007476719217636152

Max = 99.99154733076972

```python
data.age.mean()
data.age.std()
data.age.min()
data.age.max()
```

>Plot a histogram of the distribution with an appropriate number of bins for the size of the dataset

In my code, I used the Sturges' Rule to calculate the appropriate number of bins:

    bins = ceil(1+log2N) = ceil(18.21713) = 19 


```python
data.hist(column = 'age', bins = 19)
```
Below is the histogram with age 
![](agehist.png)
>Comment on any outliers or patterns you notice in the distribution of ages

The histogram with age as the column is skewed to the right. The number of people drop sharply when the age comes to 60.

> Examine the distribution of the weight in the dataset

Mean = 60.88413415993031

Standard Deviation = 18.411824265661494

Min = 3.3820836824389326

Max = 100.43579300336947

```python
data.weight.mean()
data.weight.std()
data.weight.min()
data.weight.max()
```
>Plot a histogram of the distribution 
```python
data.plot.hist(column = 'weight', bins = 19)
```
Below is the histogram with age 
![](weighthist.png)
>Comment on any outliers or patterns you notice in the distribution of weights

The graph is slightly skewed to the left. The number of people increase sharply and reaches a peak when the weight comes to 60 

>Make a scatterplot of the weights vs the ages
```python
data.plot.scatter(x = 'age', y = 'weight')
```
Below is the scatterplot
![](scatterplot.png)
>Describe the general relationship between the two variables

In this dataset, weight is increasing linearly as age increases when the age interval is (0,20). In general, the weight of people reaches a peak when people are at the age of 20. After the age of 20, as the age of people increase, the weight of people stays amost the same which appears a line with slope 0 in the scattered plot.

>You should notice at least one outlier that does not follow the general relationship. What is the name of the person?

In the scatterplot, we could notice that there is an individual lying as an extreme outlier with age slightly greater than 40 while weight less than 30.
So in the code, I set (weight<30 && age>40) as the critiria to locate the outlier
```python
print(np.where((data['age'] > 40) & (data['weight'] < 30)))
```
The output of the above code is:
```
(array([537]),)
```

> What is the name of the person? 

So now we successfully locate the row number of the outlier. I used the following code to print all the relative information of this person.
```python
data.iloc[537]
```
The output of above code is: (the name of this person is Anthony Freeman)
```
name        Anthony Freeman
age                    41.3
weight                 21.7
eyecolor              green
Name: 537, dtype: object
```

