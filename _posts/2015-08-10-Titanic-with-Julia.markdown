---
layout:     post
title:      "Titanic with Julia"
subtitle:   ""
date:       2015-08-10 12:00:00
author:     ""
header-img: "img/decision_tree.jpg"
---

# Julia on Titanic

This is an introduction to Data Analysis and Decision Trees using Julia. In this tutorial we will explore how to tackle [Kaggle's Titanic competition](https://www.kaggle.com/c/titanic) using Julia and Machine Learning. This tutorial is adopted from the [Kaggle R tutorial on Machine Learning](https://campus.datacamp.com/courses/kaggle-r-tutorial-on-machine-learning) on [Datacamp](https://www.datacamp.com/) In case you're new to Julia, you can read more about its awesomeness on [julialang.org](http://julialang.org/).  
Again, the point of this tutorial is not to teach machine learning but to provide a starting point to get your hands dirty with Julia code.
The benchmark numbers on the Julia website look pretty impressive. So get ready to embrace Julia with a warm hug!

![benchmark](https://raw.githubusercontent.com/ajkl/ajkl.github.io/master/img/Julia_benchmark.png)

Lets get started. We will mostly be using 3 main packages from Julia ecosystem   
* [DataFrames](https://github.com/JuliaStats/DataFrames.jl)
* [Gadfly](http://gadflyjl.org/)
* [DecisionTree](https://github.com/bensadeghi/DecisionTree.jl)

We start with loading the dataset from the Titanic Competition from kaggle. We will use readtable for that and inspect the first few data points with head() on the loaded DataFrame. 
We will use the already split train and test sets from DataCamp
training set: http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv
test set: http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv
I have them downloaded in the data directory.


    using DataFrames


    train = readtable("data/train.csv")
    head(train)




<table class="data-frame"><tr><th></th><th>PassengerId</th><th>Survived</th><th>Pclass</th><th>Name</th><th>Sex</th><th>Age</th><th>SibSp</th><th>Parch</th><th>Ticket</th><th>Fare</th><th>Cabin</th><th>Embarked</th></tr><tr><th>1</th><td>1</td><td>0</td><td>3</td><td>Braund, Mr. Owen Harris</td><td>male</td><td>22.0</td><td>1</td><td>0</td><td>A/5 21171</td><td>7.25</td><td>NA</td><td>S</td></tr><tr><th>2</th><td>2</td><td>1</td><td>1</td><td>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td><td>female</td><td>38.0</td><td>1</td><td>0</td><td>PC 17599</td><td>71.2833</td><td>C85</td><td>C</td></tr><tr><th>3</th><td>3</td><td>1</td><td>3</td><td>Heikkinen, Miss. Laina</td><td>female</td><td>26.0</td><td>0</td><td>0</td><td>STON/O2. 3101282</td><td>7.925</td><td>NA</td><td>S</td></tr><tr><th>4</th><td>4</td><td>1</td><td>1</td><td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td><td>female</td><td>35.0</td><td>1</td><td>0</td><td>113803</td><td>53.1</td><td>C123</td><td>S</td></tr><tr><th>5</th><td>5</td><td>0</td><td>3</td><td>Allen, Mr. William Henry</td><td>male</td><td>35.0</td><td>0</td><td>0</td><td>373450</td><td>8.05</td><td>NA</td><td>S</td></tr><tr><th>6</th><td>6</td><td>0</td><td>3</td><td>Moran, Mr. James</td><td>male</td><td>NA</td><td>0</td><td>0</td><td>330877</td><td>8.4583</td><td>NA</td><td>Q</td></tr></table>




    test = readtable("data/test.csv")
    head(test)




<table class="data-frame"><tr><th></th><th>PassengerId</th><th>Pclass</th><th>Name</th><th>Sex</th><th>Age</th><th>SibSp</th><th>Parch</th><th>Ticket</th><th>Fare</th><th>Cabin</th><th>Embarked</th></tr><tr><th>1</th><td>892</td><td>3</td><td>Kelly, Mr. James</td><td>male</td><td>34.5</td><td>0</td><td>0</td><td>330911</td><td>7.8292</td><td>NA</td><td>Q</td></tr><tr><th>2</th><td>893</td><td>3</td><td>Wilkes, Mrs. James (Ellen Needs)</td><td>female</td><td>47.0</td><td>1</td><td>0</td><td>363272</td><td>7.0</td><td>NA</td><td>S</td></tr><tr><th>3</th><td>894</td><td>2</td><td>Myles, Mr. Thomas Francis</td><td>male</td><td>62.0</td><td>0</td><td>0</td><td>240276</td><td>9.6875</td><td>NA</td><td>Q</td></tr><tr><th>4</th><td>895</td><td>3</td><td>Wirz, Mr. Albert</td><td>male</td><td>27.0</td><td>0</td><td>0</td><td>315154</td><td>8.6625</td><td>NA</td><td>S</td></tr><tr><th>5</th><td>896</td><td>3</td><td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td><td>female</td><td>22.0</td><td>1</td><td>1</td><td>3101298</td><td>12.2875</td><td>NA</td><td>S</td></tr><tr><th>6</th><td>897</td><td>3</td><td>Svensson, Mr. Johan Cervin</td><td>male</td><td>14.0</td><td>0</td><td>0</td><td>7538</td><td>9.225</td><td>NA</td><td>S</td></tr></table>




    size(train, 1)




    891




Lets take a closer look at our datasets. `describe()` helps us to summarize the entire dataset.


    describe(train)

    PassengerId
    Min      1.0
    1st Qu.  223.5
    Median   446.0
    Mean     446.0
    3rd Qu.  668.5
    Max      891.0
    NAs      0
    NA%      0.0%
    
    Survived
    Min      0.0
    1st Qu.  0.0
    Median   0.0
    Mean     0.3838383838383838
    3rd Qu.  1.0
    Max      1.0
    NAs      0
    NA%      0.0%
    
    Pclass
    Min      1.0
    1st Qu.  2.0
    Median   3.0
    Mean     2.308641975308642
    3rd Qu.  3.0
    Max      3.0
    NAs      0
    NA%      0.0%
    
    Name
    Length  891
    Type    UTF8String
    NAs     0
    NA%     0.0%
    Unique  891
    
    Sex
    Length  891
    Type    UTF8String
    NAs     0
    NA%     0.0%
    Unique  2
    
    Age
    Min      0.42
    1st Qu.  20.125
    Median   28.0
    Mean     29.69911764705882
    3rd Qu.  38.0
    Max      80.0
    NAs      177
    NA%      19.87%
    
    SibSp
    Min      0.0
    1st Qu.  0.0
    Median   0.0
    Mean     0.5230078563411896
    3rd Qu.  1.0
    Max      8.0
    NAs      0
    NA%      0.0%
    
    Parch
    Min      0.0
    1st Qu.  0.0
    Median   0.0
    Mean     0.38159371492704824
    3rd Qu.  0.0
    Max      6.0
    NAs      0
    NA%      0.0%
    
    Ticket
    Length  891
    Type    UTF8String
    NAs     0
    NA%     0.0%
    Unique  681
    
    Fare
    Min      0.0
    1st Qu.  7.9104
    Median   14.4542
    Mean     32.20420796857464
    3rd Qu.  31.0
    Max      512.3292
    NAs      0
    NA%      0.0%
    
    Cabin
    Length  891
    Type    UTF8String
    NAs     687
    NA%     77.1%
    Unique  148
    
    Embarked
    Length  891
    Type    UTF8String
    NAs     2
    NA%     0.22%
    Unique  4
    


If you want to just check the datatypes of the columns you can use `eltypes()`


    eltypes(train)




    12-element Array{Type{T<:Top},1}:
     Int64     
     Int64     
     Int64     
     UTF8String
     UTF8String
     Float64   
     Int64     
     Int64     
     UTF8String
     Float64   
     UTF8String
     UTF8String



## Data Analysis
Lets get our hand real dirty now. We want to check how many people survived the disaster. Its good to know how the distribution looks like on the classification classes.  

`counts()` will give you a frequency table, but it does not tell you how the split is.
It is the equivalent of `table()` in R.


    counts(train[:Survived])




    2-element Array{Int64,1}:
     549
     342



`countmap()` for rescue! Countmap gives you a dictionary of `value => frequency`


    countmap(train[:Survived])




    Dict{Union(NAtype,Int64),Int64} with 2 entries:
      0 => 549
      1 => 342



If you want proportions like `prop.table()` in R, you can use `proportions()` or `proportionmap()`


    proportions(train[:Survived])




    2-element Array{Float64,1}:
     0.616162
     0.383838




    proportionmap(train[:Survived])




    Dict{Union(NAtype,Int64),Float64} with 2 entries:
      0 => 0.6161616161616161
      1 => 0.3838383838383838



`Counts` does not work for categorical variables, so you can use `countmap()` there.


    countmap(train[:Sex])

    Dict{Union(NAtype,UTF8String),Int64} with 2 entries:
      "male"   => 577
      "female" => 314



Now that we know the split between male and female population on Titanic and the proportions that survived, let check if the sex of a person had an impact on the chances of surviving.  
We will examine this using a stacked histogram of Sex vs Survived using Gadfly package.


    using Gadfly


    plot(train, x="Sex", y="Survived", color="Survived", Geom.histogram(position=:stack), Scale.color_discrete_manual("red","green"))




<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:gadfly="http://www.gadflyjl.org/ns"
     version="1.2"
     width="141.42mm" height="100mm" viewBox="0 0 141.42 100"
     stroke="none"
     fill="#000000"
     stroke-width="0.3"
     font-size="3.88"

     id="fig-8cb17422ee6242b29bb8c9357ec0f6f7">
<g class="plotroot yscalable" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-1">
  <g font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" fill="#564A55" stroke="#000000" stroke-opacity="0.000" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-2">
    <text x="70.87" y="88.39" text-anchor="middle" dy="0.6em">Sex</text>
  </g>
  <g class="guide xlabels" font-size="2.82" font-family="'PT Sans Caption','Helvetica Neue','Helvetica',sans-serif" fill="#6C606B" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-3">
    <text x="45.25" y="84.39" text-anchor="middle" visibility="visible" gadfly:scale="1.0">male</text>
    <text x="96.49" y="84.39" text-anchor="middle" visibility="visible" gadfly:scale="1.0">female</text>
  </g>
  <g class="guide colorkey" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-4">
    <g fill="#4C404B" font-size="2.82" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-5">
      <text x="125.93" y="42.86" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-6" class="color_0">0</text>
      <text x="125.93" y="46.48" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-7" class="color_1">1</text>
    </g>
    <g stroke="#000000" stroke-opacity="0.000" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-8">
      <rect x="123.11" y="41.95" width="1.81" height="1.81" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-9" class="color_0" fill="#FF0000"/>
      <rect x="123.11" y="45.58" width="1.81" height="1.81" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-10" class="color_1" fill="#008000"/>
    </g>
    <g fill="#362A35" font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" stroke="#000000" stroke-opacity="0.000" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-11">
      <text x="123.11" y="39.04" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-12">Survived</text>
    </g>
  </g>
  <g clip-path="url(#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-14)" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-13">
    <g pointer-events="visible" opacity="1" fill="#000000" fill-opacity="0.000" stroke="#000000" stroke-opacity="0.000" class="guide background" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-15">
      <rect x="19.63" y="5" width="102.48" height="75.72" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-16"/>
    </g>
    <g class="guide ygridlines xfixed" stroke-dasharray="0.5,0.5" stroke-width="0.2" stroke="#D0D0E0" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-17">
      <path fill="none" d="M19.63,162.38 L 122.11 162.38" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-18" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,150.43 L 122.11 150.43" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-19" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,138.48 L 122.11 138.48" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-20" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,126.52 L 122.11 126.52" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-21" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,114.57 L 122.11 114.57" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-22" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,102.62 L 122.11 102.62" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-23" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,90.67 L 122.11 90.67" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-24" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,78.71 L 122.11 78.71" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-25" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,66.76 L 122.11 66.76" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-26" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,54.81 L 122.11 54.81" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-27" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,42.86 L 122.11 42.86" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-28" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,30.9 L 122.11 30.9" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-29" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,18.95 L 122.11 18.95" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-30" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,7 L 122.11 7" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-31" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,-4.95 L 122.11 -4.95" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-32" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,-16.9 L 122.11 -16.9" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-33" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,-28.86 L 122.11 -28.86" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-34" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,-40.81 L 122.11 -40.81" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-35" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,-52.76 L 122.11 -52.76" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-36" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,-64.71 L 122.11 -64.71" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-37" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,-76.67 L 122.11 -76.67" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-38" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M19.63,150.43 L 122.11 150.43" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-39" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,148.04 L 122.11 148.04" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-40" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,145.65 L 122.11 145.65" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-41" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,143.26 L 122.11 143.26" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-42" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,140.87 L 122.11 140.87" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-43" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,138.48 L 122.11 138.48" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-44" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,136.09 L 122.11 136.09" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-45" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,133.7 L 122.11 133.7" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-46" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,131.31 L 122.11 131.31" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-47" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,128.92 L 122.11 128.92" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-48" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,126.52 L 122.11 126.52" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-49" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,124.13 L 122.11 124.13" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-50" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,121.74 L 122.11 121.74" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-51" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,119.35 L 122.11 119.35" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-52" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,116.96 L 122.11 116.96" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-53" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,114.57 L 122.11 114.57" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-54" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,112.18 L 122.11 112.18" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-55" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,109.79 L 122.11 109.79" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-56" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,107.4 L 122.11 107.4" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-57" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,105.01 L 122.11 105.01" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-58" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,102.62 L 122.11 102.62" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-59" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,100.23 L 122.11 100.23" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-60" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,97.84 L 122.11 97.84" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-61" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,95.45 L 122.11 95.45" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-62" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,93.06 L 122.11 93.06" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-63" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,90.67 L 122.11 90.67" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-64" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,88.28 L 122.11 88.28" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-65" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,85.89 L 122.11 85.89" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-66" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,83.5 L 122.11 83.5" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-67" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,81.11 L 122.11 81.11" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-68" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,78.71 L 122.11 78.71" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-69" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,76.32 L 122.11 76.32" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-70" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,73.93 L 122.11 73.93" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-71" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,71.54 L 122.11 71.54" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,69.15 L 122.11 69.15" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-73" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,66.76 L 122.11 66.76" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-74" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,64.37 L 122.11 64.37" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-75" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,61.98 L 122.11 61.98" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-76" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,59.59 L 122.11 59.59" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-77" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,57.2 L 122.11 57.2" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-78" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,54.81 L 122.11 54.81" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-79" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,52.42 L 122.11 52.42" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-80" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,50.03 L 122.11 50.03" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-81" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,47.64 L 122.11 47.64" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-82" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,45.25 L 122.11 45.25" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-83" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,42.86 L 122.11 42.86" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-84" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,40.47 L 122.11 40.47" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-85" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,38.08 L 122.11 38.08" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-86" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,35.69 L 122.11 35.69" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-87" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,33.3 L 122.11 33.3" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-88" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,30.9 L 122.11 30.9" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-89" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,28.51 L 122.11 28.51" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-90" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,26.12 L 122.11 26.12" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-91" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,23.73 L 122.11 23.73" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-92" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,21.34 L 122.11 21.34" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-93" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,18.95 L 122.11 18.95" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-94" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,16.56 L 122.11 16.56" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-95" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,14.17 L 122.11 14.17" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-96" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,11.78 L 122.11 11.78" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-97" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,9.39 L 122.11 9.39" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-98" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,7 L 122.11 7" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-99" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,4.61 L 122.11 4.61" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-100" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,2.22 L 122.11 2.22" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-101" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-0.17 L 122.11 -0.17" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-102" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-2.56 L 122.11 -2.56" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-103" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-4.95 L 122.11 -4.95" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-104" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-7.34 L 122.11 -7.34" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-105" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-9.73 L 122.11 -9.73" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-106" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-12.12 L 122.11 -12.12" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-107" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-14.51 L 122.11 -14.51" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-108" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-16.9 L 122.11 -16.9" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-109" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-19.3 L 122.11 -19.3" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-110" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-21.69 L 122.11 -21.69" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-111" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-24.08 L 122.11 -24.08" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-112" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-26.47 L 122.11 -26.47" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-113" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-28.86 L 122.11 -28.86" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-114" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-31.25 L 122.11 -31.25" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-115" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-33.64 L 122.11 -33.64" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-116" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-36.03 L 122.11 -36.03" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-117" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-38.42 L 122.11 -38.42" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-118" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-40.81 L 122.11 -40.81" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-119" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-43.2 L 122.11 -43.2" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-120" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-45.59 L 122.11 -45.59" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-121" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-47.98 L 122.11 -47.98" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-122" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-50.37 L 122.11 -50.37" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-123" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-52.76 L 122.11 -52.76" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-124" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-55.15 L 122.11 -55.15" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-125" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-57.54 L 122.11 -57.54" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-126" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-59.93 L 122.11 -59.93" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-127" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-62.32 L 122.11 -62.32" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-128" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,-64.71 L 122.11 -64.71" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-129" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M19.63,198.24 L 122.11 198.24" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-130" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M19.63,78.71 L 122.11 78.71" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-131" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M19.63,-40.81 L 122.11 -40.81" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-132" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M19.63,-160.33 L 122.11 -160.33" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-133" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M19.63,150.43 L 122.11 150.43" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-134" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,144.45 L 122.11 144.45" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-135" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,138.48 L 122.11 138.48" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-136" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,132.5 L 122.11 132.5" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-137" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,126.52 L 122.11 126.52" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-138" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,120.55 L 122.11 120.55" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-139" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,114.57 L 122.11 114.57" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-140" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,108.6 L 122.11 108.6" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-141" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,102.62 L 122.11 102.62" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-142" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,96.64 L 122.11 96.64" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-143" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,90.67 L 122.11 90.67" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-144" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,84.69 L 122.11 84.69" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-145" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,78.71 L 122.11 78.71" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-146" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,72.74 L 122.11 72.74" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-147" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,66.76 L 122.11 66.76" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-148" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,60.79 L 122.11 60.79" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-149" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,54.81 L 122.11 54.81" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-150" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,48.83 L 122.11 48.83" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-151" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,42.86 L 122.11 42.86" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-152" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,36.88 L 122.11 36.88" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-153" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,30.9 L 122.11 30.9" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-154" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,24.93 L 122.11 24.93" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-155" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,18.95 L 122.11 18.95" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-156" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,12.98 L 122.11 12.98" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-157" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,7 L 122.11 7" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-158" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,1.02 L 122.11 1.02" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-159" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,-4.95 L 122.11 -4.95" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-160" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,-10.93 L 122.11 -10.93" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-161" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,-16.9 L 122.11 -16.9" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-162" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,-22.88 L 122.11 -22.88" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-163" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,-28.86 L 122.11 -28.86" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-164" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,-34.83 L 122.11 -34.83" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-165" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,-40.81 L 122.11 -40.81" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-166" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,-46.79 L 122.11 -46.79" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-167" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,-52.76 L 122.11 -52.76" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-168" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,-58.74 L 122.11 -58.74" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-169" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M19.63,-64.71 L 122.11 -64.71" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-170" visibility="hidden" gadfly:scale="5.0"/>
    </g>
    <g class="guide xgridlines yfixed" stroke-dasharray="0.5,0.5" stroke-width="0.2" stroke="#D0D0E0" visibility="visible" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-171">
      <path fill="none" d="M70.87,5 L 70.87 80.72" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-172" gadfly:scale="1.0"/>
    </g>
    <g class="plotpanel" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-173">
      <g shape-rendering="crispEdges" stroke-width="0.3" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-174">
        <g stroke="#000000" stroke-opacity="0.000" class="geometry" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-175">
          <rect x="19.61" y="65.69" width="51.29" height="13.03" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-176" fill="#008000"/>
          <rect x="70.85" y="50.87" width="51.29" height="27.85" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-177" fill="#008000"/>
          <rect x="19.61" y="9.75" width="51.29" height="55.94" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-178" fill="#FF0000"/>
          <rect x="70.85" y="41.18" width="51.29" height="9.68" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-179" fill="#FF0000"/>
        </g>
      </g>
    </g>
    <g opacity="0" class="guide zoomslider" stroke="#000000" stroke-opacity="0.000" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-180">
      <g fill="#EAEAEA" stroke-width="0.3" stroke-opacity="0" stroke="#6A6A6A" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-181">
        <rect x="115.11" y="8" width="4" height="4" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-182"/>
        <g class="button_logo" fill="#6A6A6A" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-183">
          <path d="M115.91,9.6 L 116.71 9.6 116.71 8.8 117.51 8.8 117.51 9.6 118.31 9.6 118.31 10.4 117.51 10.4 117.51 11.2 116.71 11.2 116.71 10.4 115.91 10.4 z" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-184"/>
        </g>
      </g>
      <g fill="#EAEAEA" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-185">
        <rect x="95.61" y="8" width="19" height="4" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-186"/>
      </g>
      <g class="zoomslider_thumb" fill="#6A6A6A" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-187">
        <rect x="104.11" y="8" width="2" height="4" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-188"/>
      </g>
      <g fill="#EAEAEA" stroke-width="0.3" stroke-opacity="0" stroke="#6A6A6A" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-189">
        <rect x="91.11" y="8" width="4" height="4" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-190"/>
        <g class="button_logo" fill="#6A6A6A" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-191">
          <path d="M91.91,9.6 L 94.31 9.6 94.31 10.4 91.91 10.4 z" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-192"/>
        </g>
      </g>
    </g>
  </g>
  <g class="guide ylabels" font-size="2.82" font-family="'PT Sans Caption','Helvetica Neue','Helvetica',sans-serif" fill="#6C606B" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-193">
    <text x="18.63" y="162.38" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-194" visibility="hidden" gadfly:scale="1.0">-700</text>
    <text x="18.63" y="150.43" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-195" visibility="hidden" gadfly:scale="1.0">-600</text>
    <text x="18.63" y="138.48" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-196" visibility="hidden" gadfly:scale="1.0">-500</text>
    <text x="18.63" y="126.52" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-197" visibility="hidden" gadfly:scale="1.0">-400</text>
    <text x="18.63" y="114.57" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-198" visibility="hidden" gadfly:scale="1.0">-300</text>
    <text x="18.63" y="102.62" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-199" visibility="hidden" gadfly:scale="1.0">-200</text>
    <text x="18.63" y="90.67" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-200" visibility="hidden" gadfly:scale="1.0">-100</text>
    <text x="18.63" y="78.71" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-201" visibility="visible" gadfly:scale="1.0">0</text>
    <text x="18.63" y="66.76" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-202" visibility="visible" gadfly:scale="1.0">100</text>
    <text x="18.63" y="54.81" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-203" visibility="visible" gadfly:scale="1.0">200</text>
    <text x="18.63" y="42.86" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-204" visibility="visible" gadfly:scale="1.0">300</text>
    <text x="18.63" y="30.9" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-205" visibility="visible" gadfly:scale="1.0">400</text>
    <text x="18.63" y="18.95" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-206" visibility="visible" gadfly:scale="1.0">500</text>
    <text x="18.63" y="7" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-207" visibility="visible" gadfly:scale="1.0">600</text>
    <text x="18.63" y="-4.95" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-208" visibility="hidden" gadfly:scale="1.0">700</text>
    <text x="18.63" y="-16.9" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-209" visibility="hidden" gadfly:scale="1.0">800</text>
    <text x="18.63" y="-28.86" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-210" visibility="hidden" gadfly:scale="1.0">900</text>
    <text x="18.63" y="-40.81" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-211" visibility="hidden" gadfly:scale="1.0">1000</text>
    <text x="18.63" y="-52.76" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-212" visibility="hidden" gadfly:scale="1.0">1100</text>
    <text x="18.63" y="-64.71" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-213" visibility="hidden" gadfly:scale="1.0">1200</text>
    <text x="18.63" y="-76.67" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-214" visibility="hidden" gadfly:scale="1.0">1300</text>
    <text x="18.63" y="150.43" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-215" visibility="hidden" gadfly:scale="10.0">-600</text>
    <text x="18.63" y="148.04" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-216" visibility="hidden" gadfly:scale="10.0">-580</text>
    <text x="18.63" y="145.65" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-217" visibility="hidden" gadfly:scale="10.0">-560</text>
    <text x="18.63" y="143.26" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-218" visibility="hidden" gadfly:scale="10.0">-540</text>
    <text x="18.63" y="140.87" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-219" visibility="hidden" gadfly:scale="10.0">-520</text>
    <text x="18.63" y="138.48" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-220" visibility="hidden" gadfly:scale="10.0">-500</text>
    <text x="18.63" y="136.09" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-221" visibility="hidden" gadfly:scale="10.0">-480</text>
    <text x="18.63" y="133.7" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-222" visibility="hidden" gadfly:scale="10.0">-460</text>
    <text x="18.63" y="131.31" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-223" visibility="hidden" gadfly:scale="10.0">-440</text>
    <text x="18.63" y="128.92" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-224" visibility="hidden" gadfly:scale="10.0">-420</text>
    <text x="18.63" y="126.52" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-225" visibility="hidden" gadfly:scale="10.0">-400</text>
    <text x="18.63" y="124.13" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-226" visibility="hidden" gadfly:scale="10.0">-380</text>
    <text x="18.63" y="121.74" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-227" visibility="hidden" gadfly:scale="10.0">-360</text>
    <text x="18.63" y="119.35" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-228" visibility="hidden" gadfly:scale="10.0">-340</text>
    <text x="18.63" y="116.96" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-229" visibility="hidden" gadfly:scale="10.0">-320</text>
    <text x="18.63" y="114.57" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-230" visibility="hidden" gadfly:scale="10.0">-300</text>
    <text x="18.63" y="112.18" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-231" visibility="hidden" gadfly:scale="10.0">-280</text>
    <text x="18.63" y="109.79" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-232" visibility="hidden" gadfly:scale="10.0">-260</text>
    <text x="18.63" y="107.4" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-233" visibility="hidden" gadfly:scale="10.0">-240</text>
    <text x="18.63" y="105.01" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-234" visibility="hidden" gadfly:scale="10.0">-220</text>
    <text x="18.63" y="102.62" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-235" visibility="hidden" gadfly:scale="10.0">-200</text>
    <text x="18.63" y="100.23" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-236" visibility="hidden" gadfly:scale="10.0">-180</text>
    <text x="18.63" y="97.84" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-237" visibility="hidden" gadfly:scale="10.0">-160</text>
    <text x="18.63" y="95.45" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-238" visibility="hidden" gadfly:scale="10.0">-140</text>
    <text x="18.63" y="93.06" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-239" visibility="hidden" gadfly:scale="10.0">-120</text>
    <text x="18.63" y="90.67" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-240" visibility="hidden" gadfly:scale="10.0">-100</text>
    <text x="18.63" y="88.28" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-241" visibility="hidden" gadfly:scale="10.0">-80</text>
    <text x="18.63" y="85.89" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-242" visibility="hidden" gadfly:scale="10.0">-60</text>
    <text x="18.63" y="83.5" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-243" visibility="hidden" gadfly:scale="10.0">-40</text>
    <text x="18.63" y="81.11" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-244" visibility="hidden" gadfly:scale="10.0">-20</text>
    <text x="18.63" y="78.71" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-245" visibility="hidden" gadfly:scale="10.0">0</text>
    <text x="18.63" y="76.32" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-246" visibility="hidden" gadfly:scale="10.0">20</text>
    <text x="18.63" y="73.93" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-247" visibility="hidden" gadfly:scale="10.0">40</text>
    <text x="18.63" y="71.54" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-248" visibility="hidden" gadfly:scale="10.0">60</text>
    <text x="18.63" y="69.15" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-249" visibility="hidden" gadfly:scale="10.0">80</text>
    <text x="18.63" y="66.76" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-250" visibility="hidden" gadfly:scale="10.0">100</text>
    <text x="18.63" y="64.37" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-251" visibility="hidden" gadfly:scale="10.0">120</text>
    <text x="18.63" y="61.98" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-252" visibility="hidden" gadfly:scale="10.0">140</text>
    <text x="18.63" y="59.59" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-253" visibility="hidden" gadfly:scale="10.0">160</text>
    <text x="18.63" y="57.2" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-254" visibility="hidden" gadfly:scale="10.0">180</text>
    <text x="18.63" y="54.81" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-255" visibility="hidden" gadfly:scale="10.0">200</text>
    <text x="18.63" y="52.42" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-256" visibility="hidden" gadfly:scale="10.0">220</text>
    <text x="18.63" y="50.03" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-257" visibility="hidden" gadfly:scale="10.0">240</text>
    <text x="18.63" y="47.64" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-258" visibility="hidden" gadfly:scale="10.0">260</text>
    <text x="18.63" y="45.25" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-259" visibility="hidden" gadfly:scale="10.0">280</text>
    <text x="18.63" y="42.86" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-260" visibility="hidden" gadfly:scale="10.0">300</text>
    <text x="18.63" y="40.47" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-261" visibility="hidden" gadfly:scale="10.0">320</text>
    <text x="18.63" y="38.08" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-262" visibility="hidden" gadfly:scale="10.0">340</text>
    <text x="18.63" y="35.69" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-263" visibility="hidden" gadfly:scale="10.0">360</text>
    <text x="18.63" y="33.3" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-264" visibility="hidden" gadfly:scale="10.0">380</text>
    <text x="18.63" y="30.9" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-265" visibility="hidden" gadfly:scale="10.0">400</text>
    <text x="18.63" y="28.51" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-266" visibility="hidden" gadfly:scale="10.0">420</text>
    <text x="18.63" y="26.12" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-267" visibility="hidden" gadfly:scale="10.0">440</text>
    <text x="18.63" y="23.73" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-268" visibility="hidden" gadfly:scale="10.0">460</text>
    <text x="18.63" y="21.34" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-269" visibility="hidden" gadfly:scale="10.0">480</text>
    <text x="18.63" y="18.95" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-270" visibility="hidden" gadfly:scale="10.0">500</text>
    <text x="18.63" y="16.56" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-271" visibility="hidden" gadfly:scale="10.0">520</text>
    <text x="18.63" y="14.17" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-272" visibility="hidden" gadfly:scale="10.0">540</text>
    <text x="18.63" y="11.78" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-273" visibility="hidden" gadfly:scale="10.0">560</text>
    <text x="18.63" y="9.39" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-274" visibility="hidden" gadfly:scale="10.0">580</text>
    <text x="18.63" y="7" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-275" visibility="hidden" gadfly:scale="10.0">600</text>
    <text x="18.63" y="4.61" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-276" visibility="hidden" gadfly:scale="10.0">620</text>
    <text x="18.63" y="2.22" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-277" visibility="hidden" gadfly:scale="10.0">640</text>
    <text x="18.63" y="-0.17" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-278" visibility="hidden" gadfly:scale="10.0">660</text>
    <text x="18.63" y="-2.56" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-279" visibility="hidden" gadfly:scale="10.0">680</text>
    <text x="18.63" y="-4.95" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-280" visibility="hidden" gadfly:scale="10.0">700</text>
    <text x="18.63" y="-7.34" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-281" visibility="hidden" gadfly:scale="10.0">720</text>
    <text x="18.63" y="-9.73" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-282" visibility="hidden" gadfly:scale="10.0">740</text>
    <text x="18.63" y="-12.12" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-283" visibility="hidden" gadfly:scale="10.0">760</text>
    <text x="18.63" y="-14.51" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-284" visibility="hidden" gadfly:scale="10.0">780</text>
    <text x="18.63" y="-16.9" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-285" visibility="hidden" gadfly:scale="10.0">800</text>
    <text x="18.63" y="-19.3" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-286" visibility="hidden" gadfly:scale="10.0">820</text>
    <text x="18.63" y="-21.69" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-287" visibility="hidden" gadfly:scale="10.0">840</text>
    <text x="18.63" y="-24.08" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-288" visibility="hidden" gadfly:scale="10.0">860</text>
    <text x="18.63" y="-26.47" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-289" visibility="hidden" gadfly:scale="10.0">880</text>
    <text x="18.63" y="-28.86" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-290" visibility="hidden" gadfly:scale="10.0">900</text>
    <text x="18.63" y="-31.25" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-291" visibility="hidden" gadfly:scale="10.0">920</text>
    <text x="18.63" y="-33.64" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-292" visibility="hidden" gadfly:scale="10.0">940</text>
    <text x="18.63" y="-36.03" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-293" visibility="hidden" gadfly:scale="10.0">960</text>
    <text x="18.63" y="-38.42" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-294" visibility="hidden" gadfly:scale="10.0">980</text>
    <text x="18.63" y="-40.81" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-295" visibility="hidden" gadfly:scale="10.0">1000</text>
    <text x="18.63" y="-43.2" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-296" visibility="hidden" gadfly:scale="10.0">1020</text>
    <text x="18.63" y="-45.59" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-297" visibility="hidden" gadfly:scale="10.0">1040</text>
    <text x="18.63" y="-47.98" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-298" visibility="hidden" gadfly:scale="10.0">1060</text>
    <text x="18.63" y="-50.37" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-299" visibility="hidden" gadfly:scale="10.0">1080</text>
    <text x="18.63" y="-52.76" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-300" visibility="hidden" gadfly:scale="10.0">1100</text>
    <text x="18.63" y="-55.15" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-301" visibility="hidden" gadfly:scale="10.0">1120</text>
    <text x="18.63" y="-57.54" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-302" visibility="hidden" gadfly:scale="10.0">1140</text>
    <text x="18.63" y="-59.93" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-303" visibility="hidden" gadfly:scale="10.0">1160</text>
    <text x="18.63" y="-62.32" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-304" visibility="hidden" gadfly:scale="10.0">1180</text>
    <text x="18.63" y="-64.71" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-305" visibility="hidden" gadfly:scale="10.0">1200</text>
    <text x="18.63" y="198.24" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-306" visibility="hidden" gadfly:scale="0.5">-1000</text>
    <text x="18.63" y="78.71" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-307" visibility="hidden" gadfly:scale="0.5">0</text>
    <text x="18.63" y="-40.81" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-308" visibility="hidden" gadfly:scale="0.5">1000</text>
    <text x="18.63" y="-160.33" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-309" visibility="hidden" gadfly:scale="0.5">2000</text>
    <text x="18.63" y="150.43" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-310" visibility="hidden" gadfly:scale="5.0">-600</text>
    <text x="18.63" y="144.45" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-311" visibility="hidden" gadfly:scale="5.0">-550</text>
    <text x="18.63" y="138.48" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-312" visibility="hidden" gadfly:scale="5.0">-500</text>
    <text x="18.63" y="132.5" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-313" visibility="hidden" gadfly:scale="5.0">-450</text>
    <text x="18.63" y="126.52" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-314" visibility="hidden" gadfly:scale="5.0">-400</text>
    <text x="18.63" y="120.55" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-315" visibility="hidden" gadfly:scale="5.0">-350</text>
    <text x="18.63" y="114.57" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-316" visibility="hidden" gadfly:scale="5.0">-300</text>
    <text x="18.63" y="108.6" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-317" visibility="hidden" gadfly:scale="5.0">-250</text>
    <text x="18.63" y="102.62" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-318" visibility="hidden" gadfly:scale="5.0">-200</text>
    <text x="18.63" y="96.64" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-319" visibility="hidden" gadfly:scale="5.0">-150</text>
    <text x="18.63" y="90.67" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-320" visibility="hidden" gadfly:scale="5.0">-100</text>
    <text x="18.63" y="84.69" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-321" visibility="hidden" gadfly:scale="5.0">-50</text>
    <text x="18.63" y="78.71" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-322" visibility="hidden" gadfly:scale="5.0">0</text>
    <text x="18.63" y="72.74" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-323" visibility="hidden" gadfly:scale="5.0">50</text>
    <text x="18.63" y="66.76" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-324" visibility="hidden" gadfly:scale="5.0">100</text>
    <text x="18.63" y="60.79" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-325" visibility="hidden" gadfly:scale="5.0">150</text>
    <text x="18.63" y="54.81" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-326" visibility="hidden" gadfly:scale="5.0">200</text>
    <text x="18.63" y="48.83" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-327" visibility="hidden" gadfly:scale="5.0">250</text>
    <text x="18.63" y="42.86" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-328" visibility="hidden" gadfly:scale="5.0">300</text>
    <text x="18.63" y="36.88" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-329" visibility="hidden" gadfly:scale="5.0">350</text>
    <text x="18.63" y="30.9" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-330" visibility="hidden" gadfly:scale="5.0">400</text>
    <text x="18.63" y="24.93" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-331" visibility="hidden" gadfly:scale="5.0">450</text>
    <text x="18.63" y="18.95" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-332" visibility="hidden" gadfly:scale="5.0">500</text>
    <text x="18.63" y="12.98" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-333" visibility="hidden" gadfly:scale="5.0">550</text>
    <text x="18.63" y="7" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-334" visibility="hidden" gadfly:scale="5.0">600</text>
    <text x="18.63" y="1.02" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-335" visibility="hidden" gadfly:scale="5.0">650</text>
    <text x="18.63" y="-4.95" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-336" visibility="hidden" gadfly:scale="5.0">700</text>
    <text x="18.63" y="-10.93" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-337" visibility="hidden" gadfly:scale="5.0">750</text>
    <text x="18.63" y="-16.9" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-338" visibility="hidden" gadfly:scale="5.0">800</text>
    <text x="18.63" y="-22.88" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-339" visibility="hidden" gadfly:scale="5.0">850</text>
    <text x="18.63" y="-28.86" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-340" visibility="hidden" gadfly:scale="5.0">900</text>
    <text x="18.63" y="-34.83" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-341" visibility="hidden" gadfly:scale="5.0">950</text>
    <text x="18.63" y="-40.81" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-342" visibility="hidden" gadfly:scale="5.0">1000</text>
    <text x="18.63" y="-46.79" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-343" visibility="hidden" gadfly:scale="5.0">1050</text>
    <text x="18.63" y="-52.76" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-344" visibility="hidden" gadfly:scale="5.0">1100</text>
    <text x="18.63" y="-58.74" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-345" visibility="hidden" gadfly:scale="5.0">1150</text>
    <text x="18.63" y="-64.71" text-anchor="end" dy="0.35em" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-346" visibility="hidden" gadfly:scale="5.0">1200</text>
  </g>
  <g font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" fill="#564A55" stroke="#000000" stroke-opacity="0.000" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-347">
    <text x="8.81" y="40.86" text-anchor="middle" dy="0.35em" transform="rotate(-90, 8.81, 42.86)" id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-348">Survived</text>
  </g>
</g>
<defs>
<clipPath id="fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-14">
  <path d="M19.63,5 L 122.11 5 122.11 80.72 19.63 80.72" />
</clipPath
></defs>
<script> <![CDATA[
(function(N){var k=/[\.\/]/,L=/\s*,\s*/,C=function(a,d){return a-d},a,v,y={n:{}},M=function(){for(var a=0,d=this.length;a<d;a++)if("undefined"!=typeof this[a])return this[a]},A=function(){for(var a=this.length;--a;)if("undefined"!=typeof this[a])return this[a]},w=function(k,d){k=String(k);var f=v,n=Array.prototype.slice.call(arguments,2),u=w.listeners(k),p=0,b,q=[],e={},l=[],r=a;l.firstDefined=M;l.lastDefined=A;a=k;for(var s=v=0,x=u.length;s<x;s++)"zIndex"in u[s]&&(q.push(u[s].zIndex),0>u[s].zIndex&&
(e[u[s].zIndex]=u[s]));for(q.sort(C);0>q[p];)if(b=e[q[p++] ],l.push(b.apply(d,n)),v)return v=f,l;for(s=0;s<x;s++)if(b=u[s],"zIndex"in b)if(b.zIndex==q[p]){l.push(b.apply(d,n));if(v)break;do if(p++,(b=e[q[p] ])&&l.push(b.apply(d,n)),v)break;while(b)}else e[b.zIndex]=b;else if(l.push(b.apply(d,n)),v)break;v=f;a=r;return l};w._events=y;w.listeners=function(a){a=a.split(k);var d=y,f,n,u,p,b,q,e,l=[d],r=[];u=0;for(p=a.length;u<p;u++){e=[];b=0;for(q=l.length;b<q;b++)for(d=l[b].n,f=[d[a[u] ],d["*"] ],n=2;n--;)if(d=
f[n])e.push(d),r=r.concat(d.f||[]);l=e}return r};w.on=function(a,d){a=String(a);if("function"!=typeof d)return function(){};for(var f=a.split(L),n=0,u=f.length;n<u;n++)(function(a){a=a.split(k);for(var b=y,f,e=0,l=a.length;e<l;e++)b=b.n,b=b.hasOwnProperty(a[e])&&b[a[e] ]||(b[a[e] ]={n:{}});b.f=b.f||[];e=0;for(l=b.f.length;e<l;e++)if(b.f[e]==d){f=!0;break}!f&&b.f.push(d)})(f[n]);return function(a){+a==+a&&(d.zIndex=+a)}};w.f=function(a){var d=[].slice.call(arguments,1);return function(){w.apply(null,
[a,null].concat(d).concat([].slice.call(arguments,0)))}};w.stop=function(){v=1};w.nt=function(k){return k?(new RegExp("(?:\\.|\\/|^)"+k+"(?:\\.|\\/|$)")).test(a):a};w.nts=function(){return a.split(k)};w.off=w.unbind=function(a,d){if(a){var f=a.split(L);if(1<f.length)for(var n=0,u=f.length;n<u;n++)w.off(f[n],d);else{for(var f=a.split(k),p,b,q,e,l=[y],n=0,u=f.length;n<u;n++)for(e=0;e<l.length;e+=q.length-2){q=[e,1];p=l[e].n;if("*"!=f[n])p[f[n] ]&&q.push(p[f[n] ]);else for(b in p)p.hasOwnProperty(b)&&
q.push(p[b]);l.splice.apply(l,q)}n=0;for(u=l.length;n<u;n++)for(p=l[n];p.n;){if(d){if(p.f){e=0;for(f=p.f.length;e<f;e++)if(p.f[e]==d){p.f.splice(e,1);break}!p.f.length&&delete p.f}for(b in p.n)if(p.n.hasOwnProperty(b)&&p.n[b].f){q=p.n[b].f;e=0;for(f=q.length;e<f;e++)if(q[e]==d){q.splice(e,1);break}!q.length&&delete p.n[b].f}}else for(b in delete p.f,p.n)p.n.hasOwnProperty(b)&&p.n[b].f&&delete p.n[b].f;p=p.n}}}else w._events=y={n:{}}};w.once=function(a,d){var f=function(){w.unbind(a,f);return d.apply(this,
arguments)};return w.on(a,f)};w.version="0.4.2";w.toString=function(){return"You are running Eve 0.4.2"};"undefined"!=typeof module&&module.exports?module.exports=w:"function"===typeof define&&define.amd?define("eve",[],function(){return w}):N.eve=w})(this);
(function(N,k){"function"===typeof define&&define.amd?define("Snap.svg",["eve"],function(L){return k(N,L)}):k(N,N.eve)})(this,function(N,k){var L=function(a){var k={},y=N.requestAnimationFrame||N.webkitRequestAnimationFrame||N.mozRequestAnimationFrame||N.oRequestAnimationFrame||N.msRequestAnimationFrame||function(a){setTimeout(a,16)},M=Array.isArray||function(a){return a instanceof Array||"[object Array]"==Object.prototype.toString.call(a)},A=0,w="M"+(+new Date).toString(36),z=function(a){if(null==
a)return this.s;var b=this.s-a;this.b+=this.dur*b;this.B+=this.dur*b;this.s=a},d=function(a){if(null==a)return this.spd;this.spd=a},f=function(a){if(null==a)return this.dur;this.s=this.s*a/this.dur;this.dur=a},n=function(){delete k[this.id];this.update();a("mina.stop."+this.id,this)},u=function(){this.pdif||(delete k[this.id],this.update(),this.pdif=this.get()-this.b)},p=function(){this.pdif&&(this.b=this.get()-this.pdif,delete this.pdif,k[this.id]=this)},b=function(){var a;if(M(this.start)){a=[];
for(var b=0,e=this.start.length;b<e;b++)a[b]=+this.start[b]+(this.end[b]-this.start[b])*this.easing(this.s)}else a=+this.start+(this.end-this.start)*this.easing(this.s);this.set(a)},q=function(){var l=0,b;for(b in k)if(k.hasOwnProperty(b)){var e=k[b],f=e.get();l++;e.s=(f-e.b)/(e.dur/e.spd);1<=e.s&&(delete k[b],e.s=1,l--,function(b){setTimeout(function(){a("mina.finish."+b.id,b)})}(e));e.update()}l&&y(q)},e=function(a,r,s,x,G,h,J){a={id:w+(A++).toString(36),start:a,end:r,b:s,s:0,dur:x-s,spd:1,get:G,
set:h,easing:J||e.linear,status:z,speed:d,duration:f,stop:n,pause:u,resume:p,update:b};k[a.id]=a;r=0;for(var K in k)if(k.hasOwnProperty(K)&&(r++,2==r))break;1==r&&y(q);return a};e.time=Date.now||function(){return+new Date};e.getById=function(a){return k[a]||null};e.linear=function(a){return a};e.easeout=function(a){return Math.pow(a,1.7)};e.easein=function(a){return Math.pow(a,0.48)};e.easeinout=function(a){if(1==a)return 1;if(0==a)return 0;var b=0.48-a/1.04,e=Math.sqrt(0.1734+b*b);a=e-b;a=Math.pow(Math.abs(a),
1/3)*(0>a?-1:1);b=-e-b;b=Math.pow(Math.abs(b),1/3)*(0>b?-1:1);a=a+b+0.5;return 3*(1-a)*a*a+a*a*a};e.backin=function(a){return 1==a?1:a*a*(2.70158*a-1.70158)};e.backout=function(a){if(0==a)return 0;a-=1;return a*a*(2.70158*a+1.70158)+1};e.elastic=function(a){return a==!!a?a:Math.pow(2,-10*a)*Math.sin(2*(a-0.075)*Math.PI/0.3)+1};e.bounce=function(a){a<1/2.75?a*=7.5625*a:a<2/2.75?(a-=1.5/2.75,a=7.5625*a*a+0.75):a<2.5/2.75?(a-=2.25/2.75,a=7.5625*a*a+0.9375):(a-=2.625/2.75,a=7.5625*a*a+0.984375);return a};
return N.mina=e}("undefined"==typeof k?function(){}:k),C=function(){function a(c,t){if(c){if(c.tagName)return x(c);if(y(c,"array")&&a.set)return a.set.apply(a,c);if(c instanceof e)return c;if(null==t)return c=G.doc.querySelector(c),x(c)}return new s(null==c?"100%":c,null==t?"100%":t)}function v(c,a){if(a){"#text"==c&&(c=G.doc.createTextNode(a.text||""));"string"==typeof c&&(c=v(c));if("string"==typeof a)return"xlink:"==a.substring(0,6)?c.getAttributeNS(m,a.substring(6)):"xml:"==a.substring(0,4)?c.getAttributeNS(la,
a.substring(4)):c.getAttribute(a);for(var da in a)if(a[h](da)){var b=J(a[da]);b?"xlink:"==da.substring(0,6)?c.setAttributeNS(m,da.substring(6),b):"xml:"==da.substring(0,4)?c.setAttributeNS(la,da.substring(4),b):c.setAttribute(da,b):c.removeAttribute(da)}}else c=G.doc.createElementNS(la,c);return c}function y(c,a){a=J.prototype.toLowerCase.call(a);return"finite"==a?isFinite(c):"array"==a&&(c instanceof Array||Array.isArray&&Array.isArray(c))?!0:"null"==a&&null===c||a==typeof c&&null!==c||"object"==
a&&c===Object(c)||$.call(c).slice(8,-1).toLowerCase()==a}function M(c){if("function"==typeof c||Object(c)!==c)return c;var a=new c.constructor,b;for(b in c)c[h](b)&&(a[b]=M(c[b]));return a}function A(c,a,b){function m(){var e=Array.prototype.slice.call(arguments,0),f=e.join("\u2400"),d=m.cache=m.cache||{},l=m.count=m.count||[];if(d[h](f)){a:for(var e=l,l=f,B=0,H=e.length;B<H;B++)if(e[B]===l){e.push(e.splice(B,1)[0]);break a}return b?b(d[f]):d[f]}1E3<=l.length&&delete d[l.shift()];l.push(f);d[f]=c.apply(a,
e);return b?b(d[f]):d[f]}return m}function w(c,a,b,m,e,f){return null==e?(c-=b,a-=m,c||a?(180*I.atan2(-a,-c)/C+540)%360:0):w(c,a,e,f)-w(b,m,e,f)}function z(c){return c%360*C/180}function d(c){var a=[];c=c.replace(/(?:^|\s)(\w+)\(([^)]+)\)/g,function(c,b,m){m=m.split(/\s*,\s*|\s+/);"rotate"==b&&1==m.length&&m.push(0,0);"scale"==b&&(2<m.length?m=m.slice(0,2):2==m.length&&m.push(0,0),1==m.length&&m.push(m[0],0,0));"skewX"==b?a.push(["m",1,0,I.tan(z(m[0])),1,0,0]):"skewY"==b?a.push(["m",1,I.tan(z(m[0])),
0,1,0,0]):a.push([b.charAt(0)].concat(m));return c});return a}function f(c,t){var b=O(c),m=new a.Matrix;if(b)for(var e=0,f=b.length;e<f;e++){var h=b[e],d=h.length,B=J(h[0]).toLowerCase(),H=h[0]!=B,l=H?m.invert():0,E;"t"==B&&2==d?m.translate(h[1],0):"t"==B&&3==d?H?(d=l.x(0,0),B=l.y(0,0),H=l.x(h[1],h[2]),l=l.y(h[1],h[2]),m.translate(H-d,l-B)):m.translate(h[1],h[2]):"r"==B?2==d?(E=E||t,m.rotate(h[1],E.x+E.width/2,E.y+E.height/2)):4==d&&(H?(H=l.x(h[2],h[3]),l=l.y(h[2],h[3]),m.rotate(h[1],H,l)):m.rotate(h[1],
h[2],h[3])):"s"==B?2==d||3==d?(E=E||t,m.scale(h[1],h[d-1],E.x+E.width/2,E.y+E.height/2)):4==d?H?(H=l.x(h[2],h[3]),l=l.y(h[2],h[3]),m.scale(h[1],h[1],H,l)):m.scale(h[1],h[1],h[2],h[3]):5==d&&(H?(H=l.x(h[3],h[4]),l=l.y(h[3],h[4]),m.scale(h[1],h[2],H,l)):m.scale(h[1],h[2],h[3],h[4])):"m"==B&&7==d&&m.add(h[1],h[2],h[3],h[4],h[5],h[6])}return m}function n(c,t){if(null==t){var m=!0;t="linearGradient"==c.type||"radialGradient"==c.type?c.node.getAttribute("gradientTransform"):"pattern"==c.type?c.node.getAttribute("patternTransform"):
c.node.getAttribute("transform");if(!t)return new a.Matrix;t=d(t)}else t=a._.rgTransform.test(t)?J(t).replace(/\.{3}|\u2026/g,c._.transform||aa):d(t),y(t,"array")&&(t=a.path?a.path.toString.call(t):J(t)),c._.transform=t;var b=f(t,c.getBBox(1));if(m)return b;c.matrix=b}function u(c){c=c.node.ownerSVGElement&&x(c.node.ownerSVGElement)||c.node.parentNode&&x(c.node.parentNode)||a.select("svg")||a(0,0);var t=c.select("defs"),t=null==t?!1:t.node;t||(t=r("defs",c.node).node);return t}function p(c){return c.node.ownerSVGElement&&
x(c.node.ownerSVGElement)||a.select("svg")}function b(c,a,m){function b(c){if(null==c)return aa;if(c==+c)return c;v(B,{width:c});try{return B.getBBox().width}catch(a){return 0}}function h(c){if(null==c)return aa;if(c==+c)return c;v(B,{height:c});try{return B.getBBox().height}catch(a){return 0}}function e(b,B){null==a?d[b]=B(c.attr(b)||0):b==a&&(d=B(null==m?c.attr(b)||0:m))}var f=p(c).node,d={},B=f.querySelector(".svg---mgr");B||(B=v("rect"),v(B,{x:-9E9,y:-9E9,width:10,height:10,"class":"svg---mgr",
fill:"none"}),f.appendChild(B));switch(c.type){case "rect":e("rx",b),e("ry",h);case "image":e("width",b),e("height",h);case "text":e("x",b);e("y",h);break;case "circle":e("cx",b);e("cy",h);e("r",b);break;case "ellipse":e("cx",b);e("cy",h);e("rx",b);e("ry",h);break;case "line":e("x1",b);e("x2",b);e("y1",h);e("y2",h);break;case "marker":e("refX",b);e("markerWidth",b);e("refY",h);e("markerHeight",h);break;case "radialGradient":e("fx",b);e("fy",h);break;case "tspan":e("dx",b);e("dy",h);break;default:e(a,
b)}f.removeChild(B);return d}function q(c){y(c,"array")||(c=Array.prototype.slice.call(arguments,0));for(var a=0,b=0,m=this.node;this[a];)delete this[a++];for(a=0;a<c.length;a++)"set"==c[a].type?c[a].forEach(function(c){m.appendChild(c.node)}):m.appendChild(c[a].node);for(var h=m.childNodes,a=0;a<h.length;a++)this[b++]=x(h[a]);return this}function e(c){if(c.snap in E)return E[c.snap];var a=this.id=V(),b;try{b=c.ownerSVGElement}catch(m){}this.node=c;b&&(this.paper=new s(b));this.type=c.tagName;this.anims=
{};this._={transform:[]};c.snap=a;E[a]=this;"g"==this.type&&(this.add=q);if(this.type in{g:1,mask:1,pattern:1})for(var e in s.prototype)s.prototype[h](e)&&(this[e]=s.prototype[e])}function l(c){this.node=c}function r(c,a){var b=v(c);a.appendChild(b);return x(b)}function s(c,a){var b,m,f,d=s.prototype;if(c&&"svg"==c.tagName){if(c.snap in E)return E[c.snap];var l=c.ownerDocument;b=new e(c);m=c.getElementsByTagName("desc")[0];f=c.getElementsByTagName("defs")[0];m||(m=v("desc"),m.appendChild(l.createTextNode("Created with Snap")),
b.node.appendChild(m));f||(f=v("defs"),b.node.appendChild(f));b.defs=f;for(var ca in d)d[h](ca)&&(b[ca]=d[ca]);b.paper=b.root=b}else b=r("svg",G.doc.body),v(b.node,{height:a,version:1.1,width:c,xmlns:la});return b}function x(c){return!c||c instanceof e||c instanceof l?c:c.tagName&&"svg"==c.tagName.toLowerCase()?new s(c):c.tagName&&"object"==c.tagName.toLowerCase()&&"image/svg+xml"==c.type?new s(c.contentDocument.getElementsByTagName("svg")[0]):new e(c)}a.version="0.3.0";a.toString=function(){return"Snap v"+
this.version};a._={};var G={win:N,doc:N.document};a._.glob=G;var h="hasOwnProperty",J=String,K=parseFloat,U=parseInt,I=Math,P=I.max,Q=I.min,Y=I.abs,C=I.PI,aa="",$=Object.prototype.toString,F=/^\s*((#[a-f\d]{6})|(#[a-f\d]{3})|rgba?\(\s*([\d\.]+%?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+%?(?:\s*,\s*[\d\.]+%?)?)\s*\)|hsba?\(\s*([\d\.]+(?:deg|\xb0|%)?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+(?:%?\s*,\s*[\d\.]+)?%?)\s*\)|hsla?\(\s*([\d\.]+(?:deg|\xb0|%)?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+(?:%?\s*,\s*[\d\.]+)?%?)\s*\))\s*$/i;a._.separator=
RegExp("[,\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]+");var S=RegExp("[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*"),X={hs:1,rg:1},W=RegExp("([a-z])[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029,]*((-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*)+)",
"ig"),ma=RegExp("([rstm])[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029,]*((-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*)+)","ig"),Z=RegExp("(-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?)[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*",
"ig"),na=0,ba="S"+(+new Date).toString(36),V=function(){return ba+(na++).toString(36)},m="http://www.w3.org/1999/xlink",la="http://www.w3.org/2000/svg",E={},ca=a.url=function(c){return"url('#"+c+"')"};a._.$=v;a._.id=V;a.format=function(){var c=/\{([^\}]+)\}/g,a=/(?:(?:^|\.)(.+?)(?=\[|\.|$|\()|\[('|")(.+?)\2\])(\(\))?/g,b=function(c,b,m){var h=m;b.replace(a,function(c,a,b,m,t){a=a||m;h&&(a in h&&(h=h[a]),"function"==typeof h&&t&&(h=h()))});return h=(null==h||h==m?c:h)+""};return function(a,m){return J(a).replace(c,
function(c,a){return b(c,a,m)})}}();a._.clone=M;a._.cacher=A;a.rad=z;a.deg=function(c){return 180*c/C%360};a.angle=w;a.is=y;a.snapTo=function(c,a,b){b=y(b,"finite")?b:10;if(y(c,"array"))for(var m=c.length;m--;){if(Y(c[m]-a)<=b)return c[m]}else{c=+c;m=a%c;if(m<b)return a-m;if(m>c-b)return a-m+c}return a};a.getRGB=A(function(c){if(!c||(c=J(c)).indexOf("-")+1)return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka};if("none"==c)return{r:-1,g:-1,b:-1,hex:"none",toString:ka};!X[h](c.toLowerCase().substring(0,
2))&&"#"!=c.charAt()&&(c=T(c));if(!c)return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka};var b,m,e,f,d;if(c=c.match(F)){c[2]&&(e=U(c[2].substring(5),16),m=U(c[2].substring(3,5),16),b=U(c[2].substring(1,3),16));c[3]&&(e=U((d=c[3].charAt(3))+d,16),m=U((d=c[3].charAt(2))+d,16),b=U((d=c[3].charAt(1))+d,16));c[4]&&(d=c[4].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b*=2.55),m=K(d[1]),"%"==d[1].slice(-1)&&(m*=2.55),e=K(d[2]),"%"==d[2].slice(-1)&&(e*=2.55),"rgba"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),
d[3]&&"%"==d[3].slice(-1)&&(f/=100));if(c[5])return d=c[5].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b/=100),m=K(d[1]),"%"==d[1].slice(-1)&&(m/=100),e=K(d[2]),"%"==d[2].slice(-1)&&(e/=100),"deg"!=d[0].slice(-3)&&"\u00b0"!=d[0].slice(-1)||(b/=360),"hsba"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),d[3]&&"%"==d[3].slice(-1)&&(f/=100),a.hsb2rgb(b,m,e,f);if(c[6])return d=c[6].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b/=100),m=K(d[1]),"%"==d[1].slice(-1)&&(m/=100),e=K(d[2]),"%"==d[2].slice(-1)&&(e/=100),
"deg"!=d[0].slice(-3)&&"\u00b0"!=d[0].slice(-1)||(b/=360),"hsla"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),d[3]&&"%"==d[3].slice(-1)&&(f/=100),a.hsl2rgb(b,m,e,f);b=Q(I.round(b),255);m=Q(I.round(m),255);e=Q(I.round(e),255);f=Q(P(f,0),1);c={r:b,g:m,b:e,toString:ka};c.hex="#"+(16777216|e|m<<8|b<<16).toString(16).slice(1);c.opacity=y(f,"finite")?f:1;return c}return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka}},a);a.hsb=A(function(c,b,m){return a.hsb2rgb(c,b,m).hex});a.hsl=A(function(c,b,m){return a.hsl2rgb(c,
b,m).hex});a.rgb=A(function(c,a,b,m){if(y(m,"finite")){var e=I.round;return"rgba("+[e(c),e(a),e(b),+m.toFixed(2)]+")"}return"#"+(16777216|b|a<<8|c<<16).toString(16).slice(1)});var T=function(c){var a=G.doc.getElementsByTagName("head")[0]||G.doc.getElementsByTagName("svg")[0];T=A(function(c){if("red"==c.toLowerCase())return"rgb(255, 0, 0)";a.style.color="rgb(255, 0, 0)";a.style.color=c;c=G.doc.defaultView.getComputedStyle(a,aa).getPropertyValue("color");return"rgb(255, 0, 0)"==c?null:c});return T(c)},
qa=function(){return"hsb("+[this.h,this.s,this.b]+")"},ra=function(){return"hsl("+[this.h,this.s,this.l]+")"},ka=function(){return 1==this.opacity||null==this.opacity?this.hex:"rgba("+[this.r,this.g,this.b,this.opacity]+")"},D=function(c,b,m){null==b&&y(c,"object")&&"r"in c&&"g"in c&&"b"in c&&(m=c.b,b=c.g,c=c.r);null==b&&y(c,string)&&(m=a.getRGB(c),c=m.r,b=m.g,m=m.b);if(1<c||1<b||1<m)c/=255,b/=255,m/=255;return[c,b,m]},oa=function(c,b,m,e){c=I.round(255*c);b=I.round(255*b);m=I.round(255*m);c={r:c,
g:b,b:m,opacity:y(e,"finite")?e:1,hex:a.rgb(c,b,m),toString:ka};y(e,"finite")&&(c.opacity=e);return c};a.color=function(c){var b;y(c,"object")&&"h"in c&&"s"in c&&"b"in c?(b=a.hsb2rgb(c),c.r=b.r,c.g=b.g,c.b=b.b,c.opacity=1,c.hex=b.hex):y(c,"object")&&"h"in c&&"s"in c&&"l"in c?(b=a.hsl2rgb(c),c.r=b.r,c.g=b.g,c.b=b.b,c.opacity=1,c.hex=b.hex):(y(c,"string")&&(c=a.getRGB(c)),y(c,"object")&&"r"in c&&"g"in c&&"b"in c&&!("error"in c)?(b=a.rgb2hsl(c),c.h=b.h,c.s=b.s,c.l=b.l,b=a.rgb2hsb(c),c.v=b.b):(c={hex:"none"},
c.r=c.g=c.b=c.h=c.s=c.v=c.l=-1,c.error=1));c.toString=ka;return c};a.hsb2rgb=function(c,a,b,m){y(c,"object")&&"h"in c&&"s"in c&&"b"in c&&(b=c.b,a=c.s,c=c.h,m=c.o);var e,h,d;c=360*c%360/60;d=b*a;a=d*(1-Y(c%2-1));b=e=h=b-d;c=~~c;b+=[d,a,0,0,a,d][c];e+=[a,d,d,a,0,0][c];h+=[0,0,a,d,d,a][c];return oa(b,e,h,m)};a.hsl2rgb=function(c,a,b,m){y(c,"object")&&"h"in c&&"s"in c&&"l"in c&&(b=c.l,a=c.s,c=c.h);if(1<c||1<a||1<b)c/=360,a/=100,b/=100;var e,h,d;c=360*c%360/60;d=2*a*(0.5>b?b:1-b);a=d*(1-Y(c%2-1));b=e=
h=b-d/2;c=~~c;b+=[d,a,0,0,a,d][c];e+=[a,d,d,a,0,0][c];h+=[0,0,a,d,d,a][c];return oa(b,e,h,m)};a.rgb2hsb=function(c,a,b){b=D(c,a,b);c=b[0];a=b[1];b=b[2];var m,e;m=P(c,a,b);e=m-Q(c,a,b);c=((0==e?0:m==c?(a-b)/e:m==a?(b-c)/e+2:(c-a)/e+4)+360)%6*60/360;return{h:c,s:0==e?0:e/m,b:m,toString:qa}};a.rgb2hsl=function(c,a,b){b=D(c,a,b);c=b[0];a=b[1];b=b[2];var m,e,h;m=P(c,a,b);e=Q(c,a,b);h=m-e;c=((0==h?0:m==c?(a-b)/h:m==a?(b-c)/h+2:(c-a)/h+4)+360)%6*60/360;m=(m+e)/2;return{h:c,s:0==h?0:0.5>m?h/(2*m):h/(2-2*
m),l:m,toString:ra}};a.parsePathString=function(c){if(!c)return null;var b=a.path(c);if(b.arr)return a.path.clone(b.arr);var m={a:7,c:6,o:2,h:1,l:2,m:2,r:4,q:4,s:4,t:2,v:1,u:3,z:0},e=[];y(c,"array")&&y(c[0],"array")&&(e=a.path.clone(c));e.length||J(c).replace(W,function(c,a,b){var h=[];c=a.toLowerCase();b.replace(Z,function(c,a){a&&h.push(+a)});"m"==c&&2<h.length&&(e.push([a].concat(h.splice(0,2))),c="l",a="m"==a?"l":"L");"o"==c&&1==h.length&&e.push([a,h[0] ]);if("r"==c)e.push([a].concat(h));else for(;h.length>=
m[c]&&(e.push([a].concat(h.splice(0,m[c]))),m[c]););});e.toString=a.path.toString;b.arr=a.path.clone(e);return e};var O=a.parseTransformString=function(c){if(!c)return null;var b=[];y(c,"array")&&y(c[0],"array")&&(b=a.path.clone(c));b.length||J(c).replace(ma,function(c,a,m){var e=[];a.toLowerCase();m.replace(Z,function(c,a){a&&e.push(+a)});b.push([a].concat(e))});b.toString=a.path.toString;return b};a._.svgTransform2string=d;a._.rgTransform=RegExp("^[a-z][\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*-?\\.?\\d",
"i");a._.transform2matrix=f;a._unit2px=b;a._.getSomeDefs=u;a._.getSomeSVG=p;a.select=function(c){return x(G.doc.querySelector(c))};a.selectAll=function(c){c=G.doc.querySelectorAll(c);for(var b=(a.set||Array)(),m=0;m<c.length;m++)b.push(x(c[m]));return b};setInterval(function(){for(var c in E)if(E[h](c)){var a=E[c],b=a.node;("svg"!=a.type&&!b.ownerSVGElement||"svg"==a.type&&(!b.parentNode||"ownerSVGElement"in b.parentNode&&!b.ownerSVGElement))&&delete E[c]}},1E4);(function(c){function m(c){function a(c,
b){var m=v(c.node,b);(m=(m=m&&m.match(d))&&m[2])&&"#"==m.charAt()&&(m=m.substring(1))&&(f[m]=(f[m]||[]).concat(function(a){var m={};m[b]=ca(a);v(c.node,m)}))}function b(c){var a=v(c.node,"xlink:href");a&&"#"==a.charAt()&&(a=a.substring(1))&&(f[a]=(f[a]||[]).concat(function(a){c.attr("xlink:href","#"+a)}))}var e=c.selectAll("*"),h,d=/^\s*url\(("|'|)(.*)\1\)\s*$/;c=[];for(var f={},l=0,E=e.length;l<E;l++){h=e[l];a(h,"fill");a(h,"stroke");a(h,"filter");a(h,"mask");a(h,"clip-path");b(h);var t=v(h.node,
"id");t&&(v(h.node,{id:h.id}),c.push({old:t,id:h.id}))}l=0;for(E=c.length;l<E;l++)if(e=f[c[l].old])for(h=0,t=e.length;h<t;h++)e[h](c[l].id)}function e(c,a,b){return function(m){m=m.slice(c,a);1==m.length&&(m=m[0]);return b?b(m):m}}function d(c){return function(){var a=c?"<"+this.type:"",b=this.node.attributes,m=this.node.childNodes;if(c)for(var e=0,h=b.length;e<h;e++)a+=" "+b[e].name+'="'+b[e].value.replace(/"/g,'\\"')+'"';if(m.length){c&&(a+=">");e=0;for(h=m.length;e<h;e++)3==m[e].nodeType?a+=m[e].nodeValue:
1==m[e].nodeType&&(a+=x(m[e]).toString());c&&(a+="</"+this.type+">")}else c&&(a+="/>");return a}}c.attr=function(c,a){if(!c)return this;if(y(c,"string"))if(1<arguments.length){var b={};b[c]=a;c=b}else return k("snap.util.getattr."+c,this).firstDefined();for(var m in c)c[h](m)&&k("snap.util.attr."+m,this,c[m]);return this};c.getBBox=function(c){if(!a.Matrix||!a.path)return this.node.getBBox();var b=this,m=new a.Matrix;if(b.removed)return a._.box();for(;"use"==b.type;)if(c||(m=m.add(b.transform().localMatrix.translate(b.attr("x")||
0,b.attr("y")||0))),b.original)b=b.original;else var e=b.attr("xlink:href"),b=b.original=b.node.ownerDocument.getElementById(e.substring(e.indexOf("#")+1));var e=b._,h=a.path.get[b.type]||a.path.get.deflt;try{if(c)return e.bboxwt=h?a.path.getBBox(b.realPath=h(b)):a._.box(b.node.getBBox()),a._.box(e.bboxwt);b.realPath=h(b);b.matrix=b.transform().localMatrix;e.bbox=a.path.getBBox(a.path.map(b.realPath,m.add(b.matrix)));return a._.box(e.bbox)}catch(d){return a._.box()}};var f=function(){return this.string};
c.transform=function(c){var b=this._;if(null==c){var m=this;c=new a.Matrix(this.node.getCTM());for(var e=n(this),h=[e],d=new a.Matrix,l=e.toTransformString(),b=J(e)==J(this.matrix)?J(b.transform):l;"svg"!=m.type&&(m=m.parent());)h.push(n(m));for(m=h.length;m--;)d.add(h[m]);return{string:b,globalMatrix:c,totalMatrix:d,localMatrix:e,diffMatrix:c.clone().add(e.invert()),global:c.toTransformString(),total:d.toTransformString(),local:l,toString:f}}c instanceof a.Matrix?this.matrix=c:n(this,c);this.node&&
("linearGradient"==this.type||"radialGradient"==this.type?v(this.node,{gradientTransform:this.matrix}):"pattern"==this.type?v(this.node,{patternTransform:this.matrix}):v(this.node,{transform:this.matrix}));return this};c.parent=function(){return x(this.node.parentNode)};c.append=c.add=function(c){if(c){if("set"==c.type){var a=this;c.forEach(function(c){a.add(c)});return this}c=x(c);this.node.appendChild(c.node);c.paper=this.paper}return this};c.appendTo=function(c){c&&(c=x(c),c.append(this));return this};
c.prepend=function(c){if(c){if("set"==c.type){var a=this,b;c.forEach(function(c){b?b.after(c):a.prepend(c);b=c});return this}c=x(c);var m=c.parent();this.node.insertBefore(c.node,this.node.firstChild);this.add&&this.add();c.paper=this.paper;this.parent()&&this.parent().add();m&&m.add()}return this};c.prependTo=function(c){c=x(c);c.prepend(this);return this};c.before=function(c){if("set"==c.type){var a=this;c.forEach(function(c){var b=c.parent();a.node.parentNode.insertBefore(c.node,a.node);b&&b.add()});
this.parent().add();return this}c=x(c);var b=c.parent();this.node.parentNode.insertBefore(c.node,this.node);this.parent()&&this.parent().add();b&&b.add();c.paper=this.paper;return this};c.after=function(c){c=x(c);var a=c.parent();this.node.nextSibling?this.node.parentNode.insertBefore(c.node,this.node.nextSibling):this.node.parentNode.appendChild(c.node);this.parent()&&this.parent().add();a&&a.add();c.paper=this.paper;return this};c.insertBefore=function(c){c=x(c);var a=this.parent();c.node.parentNode.insertBefore(this.node,
c.node);this.paper=c.paper;a&&a.add();c.parent()&&c.parent().add();return this};c.insertAfter=function(c){c=x(c);var a=this.parent();c.node.parentNode.insertBefore(this.node,c.node.nextSibling);this.paper=c.paper;a&&a.add();c.parent()&&c.parent().add();return this};c.remove=function(){var c=this.parent();this.node.parentNode&&this.node.parentNode.removeChild(this.node);delete this.paper;this.removed=!0;c&&c.add();return this};c.select=function(c){return x(this.node.querySelector(c))};c.selectAll=
function(c){c=this.node.querySelectorAll(c);for(var b=(a.set||Array)(),m=0;m<c.length;m++)b.push(x(c[m]));return b};c.asPX=function(c,a){null==a&&(a=this.attr(c));return+b(this,c,a)};c.use=function(){var c,a=this.node.id;a||(a=this.id,v(this.node,{id:a}));c="linearGradient"==this.type||"radialGradient"==this.type||"pattern"==this.type?r(this.type,this.node.parentNode):r("use",this.node.parentNode);v(c.node,{"xlink:href":"#"+a});c.original=this;return c};var l=/\S+/g;c.addClass=function(c){var a=(c||
"").match(l)||[];c=this.node;var b=c.className.baseVal,m=b.match(l)||[],e,h,d;if(a.length){for(e=0;d=a[e++];)h=m.indexOf(d),~h||m.push(d);a=m.join(" ");b!=a&&(c.className.baseVal=a)}return this};c.removeClass=function(c){var a=(c||"").match(l)||[];c=this.node;var b=c.className.baseVal,m=b.match(l)||[],e,h;if(m.length){for(e=0;h=a[e++];)h=m.indexOf(h),~h&&m.splice(h,1);a=m.join(" ");b!=a&&(c.className.baseVal=a)}return this};c.hasClass=function(c){return!!~(this.node.className.baseVal.match(l)||[]).indexOf(c)};
c.toggleClass=function(c,a){if(null!=a)return a?this.addClass(c):this.removeClass(c);var b=(c||"").match(l)||[],m=this.node,e=m.className.baseVal,h=e.match(l)||[],d,f,E;for(d=0;E=b[d++];)f=h.indexOf(E),~f?h.splice(f,1):h.push(E);b=h.join(" ");e!=b&&(m.className.baseVal=b);return this};c.clone=function(){var c=x(this.node.cloneNode(!0));v(c.node,"id")&&v(c.node,{id:c.id});m(c);c.insertAfter(this);return c};c.toDefs=function(){u(this).appendChild(this.node);return this};c.pattern=c.toPattern=function(c,
a,b,m){var e=r("pattern",u(this));null==c&&(c=this.getBBox());y(c,"object")&&"x"in c&&(a=c.y,b=c.width,m=c.height,c=c.x);v(e.node,{x:c,y:a,width:b,height:m,patternUnits:"userSpaceOnUse",id:e.id,viewBox:[c,a,b,m].join(" ")});e.node.appendChild(this.node);return e};c.marker=function(c,a,b,m,e,h){var d=r("marker",u(this));null==c&&(c=this.getBBox());y(c,"object")&&"x"in c&&(a=c.y,b=c.width,m=c.height,e=c.refX||c.cx,h=c.refY||c.cy,c=c.x);v(d.node,{viewBox:[c,a,b,m].join(" "),markerWidth:b,markerHeight:m,
orient:"auto",refX:e||0,refY:h||0,id:d.id});d.node.appendChild(this.node);return d};var E=function(c,a,b,m){"function"!=typeof b||b.length||(m=b,b=L.linear);this.attr=c;this.dur=a;b&&(this.easing=b);m&&(this.callback=m)};a._.Animation=E;a.animation=function(c,a,b,m){return new E(c,a,b,m)};c.inAnim=function(){var c=[],a;for(a in this.anims)this.anims[h](a)&&function(a){c.push({anim:new E(a._attrs,a.dur,a.easing,a._callback),mina:a,curStatus:a.status(),status:function(c){return a.status(c)},stop:function(){a.stop()}})}(this.anims[a]);
return c};a.animate=function(c,a,b,m,e,h){"function"!=typeof e||e.length||(h=e,e=L.linear);var d=L.time();c=L(c,a,d,d+m,L.time,b,e);h&&k.once("mina.finish."+c.id,h);return c};c.stop=function(){for(var c=this.inAnim(),a=0,b=c.length;a<b;a++)c[a].stop();return this};c.animate=function(c,a,b,m){"function"!=typeof b||b.length||(m=b,b=L.linear);c instanceof E&&(m=c.callback,b=c.easing,a=b.dur,c=c.attr);var d=[],f=[],l={},t,ca,n,T=this,q;for(q in c)if(c[h](q)){T.equal?(n=T.equal(q,J(c[q])),t=n.from,ca=
n.to,n=n.f):(t=+T.attr(q),ca=+c[q]);var la=y(t,"array")?t.length:1;l[q]=e(d.length,d.length+la,n);d=d.concat(t);f=f.concat(ca)}t=L.time();var p=L(d,f,t,t+a,L.time,function(c){var a={},b;for(b in l)l[h](b)&&(a[b]=l[b](c));T.attr(a)},b);T.anims[p.id]=p;p._attrs=c;p._callback=m;k("snap.animcreated."+T.id,p);k.once("mina.finish."+p.id,function(){delete T.anims[p.id];m&&m.call(T)});k.once("mina.stop."+p.id,function(){delete T.anims[p.id]});return T};var T={};c.data=function(c,b){var m=T[this.id]=T[this.id]||
{};if(0==arguments.length)return k("snap.data.get."+this.id,this,m,null),m;if(1==arguments.length){if(a.is(c,"object")){for(var e in c)c[h](e)&&this.data(e,c[e]);return this}k("snap.data.get."+this.id,this,m[c],c);return m[c]}m[c]=b;k("snap.data.set."+this.id,this,b,c);return this};c.removeData=function(c){null==c?T[this.id]={}:T[this.id]&&delete T[this.id][c];return this};c.outerSVG=c.toString=d(1);c.innerSVG=d()})(e.prototype);a.parse=function(c){var a=G.doc.createDocumentFragment(),b=!0,m=G.doc.createElement("div");
c=J(c);c.match(/^\s*<\s*svg(?:\s|>)/)||(c="<svg>"+c+"</svg>",b=!1);m.innerHTML=c;if(c=m.getElementsByTagName("svg")[0])if(b)a=c;else for(;c.firstChild;)a.appendChild(c.firstChild);m.innerHTML=aa;return new l(a)};l.prototype.select=e.prototype.select;l.prototype.selectAll=e.prototype.selectAll;a.fragment=function(){for(var c=Array.prototype.slice.call(arguments,0),b=G.doc.createDocumentFragment(),m=0,e=c.length;m<e;m++){var h=c[m];h.node&&h.node.nodeType&&b.appendChild(h.node);h.nodeType&&b.appendChild(h);
"string"==typeof h&&b.appendChild(a.parse(h).node)}return new l(b)};a._.make=r;a._.wrap=x;s.prototype.el=function(c,a){var b=r(c,this.node);a&&b.attr(a);return b};k.on("snap.util.getattr",function(){var c=k.nt(),c=c.substring(c.lastIndexOf(".")+1),a=c.replace(/[A-Z]/g,function(c){return"-"+c.toLowerCase()});return pa[h](a)?this.node.ownerDocument.defaultView.getComputedStyle(this.node,null).getPropertyValue(a):v(this.node,c)});var pa={"alignment-baseline":0,"baseline-shift":0,clip:0,"clip-path":0,
"clip-rule":0,color:0,"color-interpolation":0,"color-interpolation-filters":0,"color-profile":0,"color-rendering":0,cursor:0,direction:0,display:0,"dominant-baseline":0,"enable-background":0,fill:0,"fill-opacity":0,"fill-rule":0,filter:0,"flood-color":0,"flood-opacity":0,font:0,"font-family":0,"font-size":0,"font-size-adjust":0,"font-stretch":0,"font-style":0,"font-variant":0,"font-weight":0,"glyph-orientation-horizontal":0,"glyph-orientation-vertical":0,"image-rendering":0,kerning:0,"letter-spacing":0,
"lighting-color":0,marker:0,"marker-end":0,"marker-mid":0,"marker-start":0,mask:0,opacity:0,overflow:0,"pointer-events":0,"shape-rendering":0,"stop-color":0,"stop-opacity":0,stroke:0,"stroke-dasharray":0,"stroke-dashoffset":0,"stroke-linecap":0,"stroke-linejoin":0,"stroke-miterlimit":0,"stroke-opacity":0,"stroke-width":0,"text-anchor":0,"text-decoration":0,"text-rendering":0,"unicode-bidi":0,visibility:0,"word-spacing":0,"writing-mode":0};k.on("snap.util.attr",function(c){var a=k.nt(),b={},a=a.substring(a.lastIndexOf(".")+
1);b[a]=c;var m=a.replace(/-(\w)/gi,function(c,a){return a.toUpperCase()}),a=a.replace(/[A-Z]/g,function(c){return"-"+c.toLowerCase()});pa[h](a)?this.node.style[m]=null==c?aa:c:v(this.node,b)});a.ajax=function(c,a,b,m){var e=new XMLHttpRequest,h=V();if(e){if(y(a,"function"))m=b,b=a,a=null;else if(y(a,"object")){var d=[],f;for(f in a)a.hasOwnProperty(f)&&d.push(encodeURIComponent(f)+"="+encodeURIComponent(a[f]));a=d.join("&")}e.open(a?"POST":"GET",c,!0);a&&(e.setRequestHeader("X-Requested-With","XMLHttpRequest"),
e.setRequestHeader("Content-type","application/x-www-form-urlencoded"));b&&(k.once("snap.ajax."+h+".0",b),k.once("snap.ajax."+h+".200",b),k.once("snap.ajax."+h+".304",b));e.onreadystatechange=function(){4==e.readyState&&k("snap.ajax."+h+"."+e.status,m,e)};if(4==e.readyState)return e;e.send(a);return e}};a.load=function(c,b,m){a.ajax(c,function(c){c=a.parse(c.responseText);m?b.call(m,c):b(c)})};a.getElementByPoint=function(c,a){var b,m,e=G.doc.elementFromPoint(c,a);if(G.win.opera&&"svg"==e.tagName){b=
e;m=b.getBoundingClientRect();b=b.ownerDocument;var h=b.body,d=b.documentElement;b=m.top+(g.win.pageYOffset||d.scrollTop||h.scrollTop)-(d.clientTop||h.clientTop||0);m=m.left+(g.win.pageXOffset||d.scrollLeft||h.scrollLeft)-(d.clientLeft||h.clientLeft||0);h=e.createSVGRect();h.x=c-m;h.y=a-b;h.width=h.height=1;b=e.getIntersectionList(h,null);b.length&&(e=b[b.length-1])}return e?x(e):null};a.plugin=function(c){c(a,e,s,G,l)};return G.win.Snap=a}();C.plugin(function(a,k,y,M,A){function w(a,d,f,b,q,e){null==
d&&"[object SVGMatrix]"==z.call(a)?(this.a=a.a,this.b=a.b,this.c=a.c,this.d=a.d,this.e=a.e,this.f=a.f):null!=a?(this.a=+a,this.b=+d,this.c=+f,this.d=+b,this.e=+q,this.f=+e):(this.a=1,this.c=this.b=0,this.d=1,this.f=this.e=0)}var z=Object.prototype.toString,d=String,f=Math;(function(n){function k(a){return a[0]*a[0]+a[1]*a[1]}function p(a){var d=f.sqrt(k(a));a[0]&&(a[0]/=d);a[1]&&(a[1]/=d)}n.add=function(a,d,e,f,n,p){var k=[[],[],[] ],u=[[this.a,this.c,this.e],[this.b,this.d,this.f],[0,0,1] ];d=[[a,
e,n],[d,f,p],[0,0,1] ];a&&a instanceof w&&(d=[[a.a,a.c,a.e],[a.b,a.d,a.f],[0,0,1] ]);for(a=0;3>a;a++)for(e=0;3>e;e++){for(f=n=0;3>f;f++)n+=u[a][f]*d[f][e];k[a][e]=n}this.a=k[0][0];this.b=k[1][0];this.c=k[0][1];this.d=k[1][1];this.e=k[0][2];this.f=k[1][2];return this};n.invert=function(){var a=this.a*this.d-this.b*this.c;return new w(this.d/a,-this.b/a,-this.c/a,this.a/a,(this.c*this.f-this.d*this.e)/a,(this.b*this.e-this.a*this.f)/a)};n.clone=function(){return new w(this.a,this.b,this.c,this.d,this.e,
this.f)};n.translate=function(a,d){return this.add(1,0,0,1,a,d)};n.scale=function(a,d,e,f){null==d&&(d=a);(e||f)&&this.add(1,0,0,1,e,f);this.add(a,0,0,d,0,0);(e||f)&&this.add(1,0,0,1,-e,-f);return this};n.rotate=function(b,d,e){b=a.rad(b);d=d||0;e=e||0;var l=+f.cos(b).toFixed(9);b=+f.sin(b).toFixed(9);this.add(l,b,-b,l,d,e);return this.add(1,0,0,1,-d,-e)};n.x=function(a,d){return a*this.a+d*this.c+this.e};n.y=function(a,d){return a*this.b+d*this.d+this.f};n.get=function(a){return+this[d.fromCharCode(97+
a)].toFixed(4)};n.toString=function(){return"matrix("+[this.get(0),this.get(1),this.get(2),this.get(3),this.get(4),this.get(5)].join()+")"};n.offset=function(){return[this.e.toFixed(4),this.f.toFixed(4)]};n.determinant=function(){return this.a*this.d-this.b*this.c};n.split=function(){var b={};b.dx=this.e;b.dy=this.f;var d=[[this.a,this.c],[this.b,this.d] ];b.scalex=f.sqrt(k(d[0]));p(d[0]);b.shear=d[0][0]*d[1][0]+d[0][1]*d[1][1];d[1]=[d[1][0]-d[0][0]*b.shear,d[1][1]-d[0][1]*b.shear];b.scaley=f.sqrt(k(d[1]));
p(d[1]);b.shear/=b.scaley;0>this.determinant()&&(b.scalex=-b.scalex);var e=-d[0][1],d=d[1][1];0>d?(b.rotate=a.deg(f.acos(d)),0>e&&(b.rotate=360-b.rotate)):b.rotate=a.deg(f.asin(e));b.isSimple=!+b.shear.toFixed(9)&&(b.scalex.toFixed(9)==b.scaley.toFixed(9)||!b.rotate);b.isSuperSimple=!+b.shear.toFixed(9)&&b.scalex.toFixed(9)==b.scaley.toFixed(9)&&!b.rotate;b.noRotation=!+b.shear.toFixed(9)&&!b.rotate;return b};n.toTransformString=function(a){a=a||this.split();if(+a.shear.toFixed(9))return"m"+[this.get(0),
this.get(1),this.get(2),this.get(3),this.get(4),this.get(5)];a.scalex=+a.scalex.toFixed(4);a.scaley=+a.scaley.toFixed(4);a.rotate=+a.rotate.toFixed(4);return(a.dx||a.dy?"t"+[+a.dx.toFixed(4),+a.dy.toFixed(4)]:"")+(1!=a.scalex||1!=a.scaley?"s"+[a.scalex,a.scaley,0,0]:"")+(a.rotate?"r"+[+a.rotate.toFixed(4),0,0]:"")}})(w.prototype);a.Matrix=w;a.matrix=function(a,d,f,b,k,e){return new w(a,d,f,b,k,e)}});C.plugin(function(a,v,y,M,A){function w(h){return function(d){k.stop();d instanceof A&&1==d.node.childNodes.length&&
("radialGradient"==d.node.firstChild.tagName||"linearGradient"==d.node.firstChild.tagName||"pattern"==d.node.firstChild.tagName)&&(d=d.node.firstChild,b(this).appendChild(d),d=u(d));if(d instanceof v)if("radialGradient"==d.type||"linearGradient"==d.type||"pattern"==d.type){d.node.id||e(d.node,{id:d.id});var f=l(d.node.id)}else f=d.attr(h);else f=a.color(d),f.error?(f=a(b(this).ownerSVGElement).gradient(d))?(f.node.id||e(f.node,{id:f.id}),f=l(f.node.id)):f=d:f=r(f);d={};d[h]=f;e(this.node,d);this.node.style[h]=
x}}function z(a){k.stop();a==+a&&(a+="px");this.node.style.fontSize=a}function d(a){var b=[];a=a.childNodes;for(var e=0,f=a.length;e<f;e++){var l=a[e];3==l.nodeType&&b.push(l.nodeValue);"tspan"==l.tagName&&(1==l.childNodes.length&&3==l.firstChild.nodeType?b.push(l.firstChild.nodeValue):b.push(d(l)))}return b}function f(){k.stop();return this.node.style.fontSize}var n=a._.make,u=a._.wrap,p=a.is,b=a._.getSomeDefs,q=/^url\(#?([^)]+)\)$/,e=a._.$,l=a.url,r=String,s=a._.separator,x="";k.on("snap.util.attr.mask",
function(a){if(a instanceof v||a instanceof A){k.stop();a instanceof A&&1==a.node.childNodes.length&&(a=a.node.firstChild,b(this).appendChild(a),a=u(a));if("mask"==a.type)var d=a;else d=n("mask",b(this)),d.node.appendChild(a.node);!d.node.id&&e(d.node,{id:d.id});e(this.node,{mask:l(d.id)})}});(function(a){k.on("snap.util.attr.clip",a);k.on("snap.util.attr.clip-path",a);k.on("snap.util.attr.clipPath",a)})(function(a){if(a instanceof v||a instanceof A){k.stop();if("clipPath"==a.type)var d=a;else d=
n("clipPath",b(this)),d.node.appendChild(a.node),!d.node.id&&e(d.node,{id:d.id});e(this.node,{"clip-path":l(d.id)})}});k.on("snap.util.attr.fill",w("fill"));k.on("snap.util.attr.stroke",w("stroke"));var G=/^([lr])(?:\(([^)]*)\))?(.*)$/i;k.on("snap.util.grad.parse",function(a){a=r(a);var b=a.match(G);if(!b)return null;a=b[1];var e=b[2],b=b[3],e=e.split(/\s*,\s*/).map(function(a){return+a==a?+a:a});1==e.length&&0==e[0]&&(e=[]);b=b.split("-");b=b.map(function(a){a=a.split(":");var b={color:a[0]};a[1]&&
(b.offset=parseFloat(a[1]));return b});return{type:a,params:e,stops:b}});k.on("snap.util.attr.d",function(b){k.stop();p(b,"array")&&p(b[0],"array")&&(b=a.path.toString.call(b));b=r(b);b.match(/[ruo]/i)&&(b=a.path.toAbsolute(b));e(this.node,{d:b})})(-1);k.on("snap.util.attr.#text",function(a){k.stop();a=r(a);for(a=M.doc.createTextNode(a);this.node.firstChild;)this.node.removeChild(this.node.firstChild);this.node.appendChild(a)})(-1);k.on("snap.util.attr.path",function(a){k.stop();this.attr({d:a})})(-1);
k.on("snap.util.attr.class",function(a){k.stop();this.node.className.baseVal=a})(-1);k.on("snap.util.attr.viewBox",function(a){a=p(a,"object")&&"x"in a?[a.x,a.y,a.width,a.height].join(" "):p(a,"array")?a.join(" "):a;e(this.node,{viewBox:a});k.stop()})(-1);k.on("snap.util.attr.transform",function(a){this.transform(a);k.stop()})(-1);k.on("snap.util.attr.r",function(a){"rect"==this.type&&(k.stop(),e(this.node,{rx:a,ry:a}))})(-1);k.on("snap.util.attr.textpath",function(a){k.stop();if("text"==this.type){var d,
f;if(!a&&this.textPath){for(a=this.textPath;a.node.firstChild;)this.node.appendChild(a.node.firstChild);a.remove();delete this.textPath}else if(p(a,"string")?(d=b(this),a=u(d.parentNode).path(a),d.appendChild(a.node),d=a.id,a.attr({id:d})):(a=u(a),a instanceof v&&(d=a.attr("id"),d||(d=a.id,a.attr({id:d})))),d)if(a=this.textPath,f=this.node,a)a.attr({"xlink:href":"#"+d});else{for(a=e("textPath",{"xlink:href":"#"+d});f.firstChild;)a.appendChild(f.firstChild);f.appendChild(a);this.textPath=u(a)}}})(-1);
k.on("snap.util.attr.text",function(a){if("text"==this.type){for(var b=this.node,d=function(a){var b=e("tspan");if(p(a,"array"))for(var f=0;f<a.length;f++)b.appendChild(d(a[f]));else b.appendChild(M.doc.createTextNode(a));b.normalize&&b.normalize();return b};b.firstChild;)b.removeChild(b.firstChild);for(a=d(a);a.firstChild;)b.appendChild(a.firstChild)}k.stop()})(-1);k.on("snap.util.attr.fontSize",z)(-1);k.on("snap.util.attr.font-size",z)(-1);k.on("snap.util.getattr.transform",function(){k.stop();
return this.transform()})(-1);k.on("snap.util.getattr.textpath",function(){k.stop();return this.textPath})(-1);(function(){function b(d){return function(){k.stop();var b=M.doc.defaultView.getComputedStyle(this.node,null).getPropertyValue("marker-"+d);return"none"==b?b:a(M.doc.getElementById(b.match(q)[1]))}}function d(a){return function(b){k.stop();var d="marker"+a.charAt(0).toUpperCase()+a.substring(1);if(""==b||!b)this.node.style[d]="none";else if("marker"==b.type){var f=b.node.id;f||e(b.node,{id:b.id});
this.node.style[d]=l(f)}}}k.on("snap.util.getattr.marker-end",b("end"))(-1);k.on("snap.util.getattr.markerEnd",b("end"))(-1);k.on("snap.util.getattr.marker-start",b("start"))(-1);k.on("snap.util.getattr.markerStart",b("start"))(-1);k.on("snap.util.getattr.marker-mid",b("mid"))(-1);k.on("snap.util.getattr.markerMid",b("mid"))(-1);k.on("snap.util.attr.marker-end",d("end"))(-1);k.on("snap.util.attr.markerEnd",d("end"))(-1);k.on("snap.util.attr.marker-start",d("start"))(-1);k.on("snap.util.attr.markerStart",
d("start"))(-1);k.on("snap.util.attr.marker-mid",d("mid"))(-1);k.on("snap.util.attr.markerMid",d("mid"))(-1)})();k.on("snap.util.getattr.r",function(){if("rect"==this.type&&e(this.node,"rx")==e(this.node,"ry"))return k.stop(),e(this.node,"rx")})(-1);k.on("snap.util.getattr.text",function(){if("text"==this.type||"tspan"==this.type){k.stop();var a=d(this.node);return 1==a.length?a[0]:a}})(-1);k.on("snap.util.getattr.#text",function(){return this.node.textContent})(-1);k.on("snap.util.getattr.viewBox",
function(){k.stop();var b=e(this.node,"viewBox");if(b)return b=b.split(s),a._.box(+b[0],+b[1],+b[2],+b[3])})(-1);k.on("snap.util.getattr.points",function(){var a=e(this.node,"points");k.stop();if(a)return a.split(s)})(-1);k.on("snap.util.getattr.path",function(){var a=e(this.node,"d");k.stop();return a})(-1);k.on("snap.util.getattr.class",function(){return this.node.className.baseVal})(-1);k.on("snap.util.getattr.fontSize",f)(-1);k.on("snap.util.getattr.font-size",f)(-1)});C.plugin(function(a,v,y,
M,A){function w(a){return a}function z(a){return function(b){return+b.toFixed(3)+a}}var d={"+":function(a,b){return a+b},"-":function(a,b){return a-b},"/":function(a,b){return a/b},"*":function(a,b){return a*b}},f=String,n=/[a-z]+$/i,u=/^\s*([+\-\/*])\s*=\s*([\d.eE+\-]+)\s*([^\d\s]+)?\s*$/;k.on("snap.util.attr",function(a){if(a=f(a).match(u)){var b=k.nt(),b=b.substring(b.lastIndexOf(".")+1),q=this.attr(b),e={};k.stop();var l=a[3]||"",r=q.match(n),s=d[a[1] ];r&&r==l?a=s(parseFloat(q),+a[2]):(q=this.asPX(b),
a=s(this.asPX(b),this.asPX(b,a[2]+l)));isNaN(q)||isNaN(a)||(e[b]=a,this.attr(e))}})(-10);k.on("snap.util.equal",function(a,b){var q=f(this.attr(a)||""),e=f(b).match(u);if(e){k.stop();var l=e[3]||"",r=q.match(n),s=d[e[1] ];if(r&&r==l)return{from:parseFloat(q),to:s(parseFloat(q),+e[2]),f:z(r)};q=this.asPX(a);return{from:q,to:s(q,this.asPX(a,e[2]+l)),f:w}}})(-10)});C.plugin(function(a,v,y,M,A){var w=y.prototype,z=a.is;w.rect=function(a,d,k,p,b,q){var e;null==q&&(q=b);z(a,"object")&&"[object Object]"==
a?e=a:null!=a&&(e={x:a,y:d,width:k,height:p},null!=b&&(e.rx=b,e.ry=q));return this.el("rect",e)};w.circle=function(a,d,k){var p;z(a,"object")&&"[object Object]"==a?p=a:null!=a&&(p={cx:a,cy:d,r:k});return this.el("circle",p)};var d=function(){function a(){this.parentNode.removeChild(this)}return function(d,k){var p=M.doc.createElement("img"),b=M.doc.body;p.style.cssText="position:absolute;left:-9999em;top:-9999em";p.onload=function(){k.call(p);p.onload=p.onerror=null;b.removeChild(p)};p.onerror=a;
b.appendChild(p);p.src=d}}();w.image=function(f,n,k,p,b){var q=this.el("image");if(z(f,"object")&&"src"in f)q.attr(f);else if(null!=f){var e={"xlink:href":f,preserveAspectRatio:"none"};null!=n&&null!=k&&(e.x=n,e.y=k);null!=p&&null!=b?(e.width=p,e.height=b):d(f,function(){a._.$(q.node,{width:this.offsetWidth,height:this.offsetHeight})});a._.$(q.node,e)}return q};w.ellipse=function(a,d,k,p){var b;z(a,"object")&&"[object Object]"==a?b=a:null!=a&&(b={cx:a,cy:d,rx:k,ry:p});return this.el("ellipse",b)};
w.path=function(a){var d;z(a,"object")&&!z(a,"array")?d=a:a&&(d={d:a});return this.el("path",d)};w.group=w.g=function(a){var d=this.el("g");1==arguments.length&&a&&!a.type?d.attr(a):arguments.length&&d.add(Array.prototype.slice.call(arguments,0));return d};w.svg=function(a,d,k,p,b,q,e,l){var r={};z(a,"object")&&null==d?r=a:(null!=a&&(r.x=a),null!=d&&(r.y=d),null!=k&&(r.width=k),null!=p&&(r.height=p),null!=b&&null!=q&&null!=e&&null!=l&&(r.viewBox=[b,q,e,l]));return this.el("svg",r)};w.mask=function(a){var d=
this.el("mask");1==arguments.length&&a&&!a.type?d.attr(a):arguments.length&&d.add(Array.prototype.slice.call(arguments,0));return d};w.ptrn=function(a,d,k,p,b,q,e,l){if(z(a,"object"))var r=a;else arguments.length?(r={},null!=a&&(r.x=a),null!=d&&(r.y=d),null!=k&&(r.width=k),null!=p&&(r.height=p),null!=b&&null!=q&&null!=e&&null!=l&&(r.viewBox=[b,q,e,l])):r={patternUnits:"userSpaceOnUse"};return this.el("pattern",r)};w.use=function(a){return null!=a?(make("use",this.node),a instanceof v&&(a.attr("id")||
a.attr({id:ID()}),a=a.attr("id")),this.el("use",{"xlink:href":a})):v.prototype.use.call(this)};w.text=function(a,d,k){var p={};z(a,"object")?p=a:null!=a&&(p={x:a,y:d,text:k||""});return this.el("text",p)};w.line=function(a,d,k,p){var b={};z(a,"object")?b=a:null!=a&&(b={x1:a,x2:k,y1:d,y2:p});return this.el("line",b)};w.polyline=function(a){1<arguments.length&&(a=Array.prototype.slice.call(arguments,0));var d={};z(a,"object")&&!z(a,"array")?d=a:null!=a&&(d={points:a});return this.el("polyline",d)};
w.polygon=function(a){1<arguments.length&&(a=Array.prototype.slice.call(arguments,0));var d={};z(a,"object")&&!z(a,"array")?d=a:null!=a&&(d={points:a});return this.el("polygon",d)};(function(){function d(){return this.selectAll("stop")}function n(b,d){var f=e("stop"),k={offset:+d+"%"};b=a.color(b);k["stop-color"]=b.hex;1>b.opacity&&(k["stop-opacity"]=b.opacity);e(f,k);this.node.appendChild(f);return this}function u(){if("linearGradient"==this.type){var b=e(this.node,"x1")||0,d=e(this.node,"x2")||
1,f=e(this.node,"y1")||0,k=e(this.node,"y2")||0;return a._.box(b,f,math.abs(d-b),math.abs(k-f))}b=this.node.r||0;return a._.box((this.node.cx||0.5)-b,(this.node.cy||0.5)-b,2*b,2*b)}function p(a,d){function f(a,b){for(var d=(b-u)/(a-w),e=w;e<a;e++)h[e].offset=+(+u+d*(e-w)).toFixed(2);w=a;u=b}var n=k("snap.util.grad.parse",null,d).firstDefined(),p;if(!n)return null;n.params.unshift(a);p="l"==n.type.toLowerCase()?b.apply(0,n.params):q.apply(0,n.params);n.type!=n.type.toLowerCase()&&e(p.node,{gradientUnits:"userSpaceOnUse"});
var h=n.stops,n=h.length,u=0,w=0;n--;for(var v=0;v<n;v++)"offset"in h[v]&&f(v,h[v].offset);h[n].offset=h[n].offset||100;f(n,h[n].offset);for(v=0;v<=n;v++){var y=h[v];p.addStop(y.color,y.offset)}return p}function b(b,k,p,q,w){b=a._.make("linearGradient",b);b.stops=d;b.addStop=n;b.getBBox=u;null!=k&&e(b.node,{x1:k,y1:p,x2:q,y2:w});return b}function q(b,k,p,q,w,h){b=a._.make("radialGradient",b);b.stops=d;b.addStop=n;b.getBBox=u;null!=k&&e(b.node,{cx:k,cy:p,r:q});null!=w&&null!=h&&e(b.node,{fx:w,fy:h});
return b}var e=a._.$;w.gradient=function(a){return p(this.defs,a)};w.gradientLinear=function(a,d,e,f){return b(this.defs,a,d,e,f)};w.gradientRadial=function(a,b,d,e,f){return q(this.defs,a,b,d,e,f)};w.toString=function(){var b=this.node.ownerDocument,d=b.createDocumentFragment(),b=b.createElement("div"),e=this.node.cloneNode(!0);d.appendChild(b);b.appendChild(e);a._.$(e,{xmlns:"http://www.w3.org/2000/svg"});b=b.innerHTML;d.removeChild(d.firstChild);return b};w.clear=function(){for(var a=this.node.firstChild,
b;a;)b=a.nextSibling,"defs"!=a.tagName?a.parentNode.removeChild(a):w.clear.call({node:a}),a=b}})()});C.plugin(function(a,k,y,M){function A(a){var b=A.ps=A.ps||{};b[a]?b[a].sleep=100:b[a]={sleep:100};setTimeout(function(){for(var d in b)b[L](d)&&d!=a&&(b[d].sleep--,!b[d].sleep&&delete b[d])});return b[a]}function w(a,b,d,e){null==a&&(a=b=d=e=0);null==b&&(b=a.y,d=a.width,e=a.height,a=a.x);return{x:a,y:b,width:d,w:d,height:e,h:e,x2:a+d,y2:b+e,cx:a+d/2,cy:b+e/2,r1:F.min(d,e)/2,r2:F.max(d,e)/2,r0:F.sqrt(d*
d+e*e)/2,path:s(a,b,d,e),vb:[a,b,d,e].join(" ")}}function z(){return this.join(",").replace(N,"$1")}function d(a){a=C(a);a.toString=z;return a}function f(a,b,d,h,f,k,l,n,p){if(null==p)return e(a,b,d,h,f,k,l,n);if(0>p||e(a,b,d,h,f,k,l,n)<p)p=void 0;else{var q=0.5,O=1-q,s;for(s=e(a,b,d,h,f,k,l,n,O);0.01<Z(s-p);)q/=2,O+=(s<p?1:-1)*q,s=e(a,b,d,h,f,k,l,n,O);p=O}return u(a,b,d,h,f,k,l,n,p)}function n(b,d){function e(a){return+(+a).toFixed(3)}return a._.cacher(function(a,h,l){a instanceof k&&(a=a.attr("d"));
a=I(a);for(var n,p,D,q,O="",s={},c=0,t=0,r=a.length;t<r;t++){D=a[t];if("M"==D[0])n=+D[1],p=+D[2];else{q=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6]);if(c+q>h){if(d&&!s.start){n=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6],h-c);O+=["C"+e(n.start.x),e(n.start.y),e(n.m.x),e(n.m.y),e(n.x),e(n.y)];if(l)return O;s.start=O;O=["M"+e(n.x),e(n.y)+"C"+e(n.n.x),e(n.n.y),e(n.end.x),e(n.end.y),e(D[5]),e(D[6])].join();c+=q;n=+D[5];p=+D[6];continue}if(!b&&!d)return n=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6],h-c)}c+=q;n=+D[5];p=+D[6]}O+=
D.shift()+D}s.end=O;return n=b?c:d?s:u(n,p,D[0],D[1],D[2],D[3],D[4],D[5],1)},null,a._.clone)}function u(a,b,d,e,h,f,k,l,n){var p=1-n,q=ma(p,3),s=ma(p,2),c=n*n,t=c*n,r=q*a+3*s*n*d+3*p*n*n*h+t*k,q=q*b+3*s*n*e+3*p*n*n*f+t*l,s=a+2*n*(d-a)+c*(h-2*d+a),t=b+2*n*(e-b)+c*(f-2*e+b),x=d+2*n*(h-d)+c*(k-2*h+d),c=e+2*n*(f-e)+c*(l-2*f+e);a=p*a+n*d;b=p*b+n*e;h=p*h+n*k;f=p*f+n*l;l=90-180*F.atan2(s-x,t-c)/S;return{x:r,y:q,m:{x:s,y:t},n:{x:x,y:c},start:{x:a,y:b},end:{x:h,y:f},alpha:l}}function p(b,d,e,h,f,n,k,l){a.is(b,
"array")||(b=[b,d,e,h,f,n,k,l]);b=U.apply(null,b);return w(b.min.x,b.min.y,b.max.x-b.min.x,b.max.y-b.min.y)}function b(a,b,d){return b>=a.x&&b<=a.x+a.width&&d>=a.y&&d<=a.y+a.height}function q(a,d){a=w(a);d=w(d);return b(d,a.x,a.y)||b(d,a.x2,a.y)||b(d,a.x,a.y2)||b(d,a.x2,a.y2)||b(a,d.x,d.y)||b(a,d.x2,d.y)||b(a,d.x,d.y2)||b(a,d.x2,d.y2)||(a.x<d.x2&&a.x>d.x||d.x<a.x2&&d.x>a.x)&&(a.y<d.y2&&a.y>d.y||d.y<a.y2&&d.y>a.y)}function e(a,b,d,e,h,f,n,k,l){null==l&&(l=1);l=(1<l?1:0>l?0:l)/2;for(var p=[-0.1252,
0.1252,-0.3678,0.3678,-0.5873,0.5873,-0.7699,0.7699,-0.9041,0.9041,-0.9816,0.9816],q=[0.2491,0.2491,0.2335,0.2335,0.2032,0.2032,0.1601,0.1601,0.1069,0.1069,0.0472,0.0472],s=0,c=0;12>c;c++)var t=l*p[c]+l,r=t*(t*(-3*a+9*d-9*h+3*n)+6*a-12*d+6*h)-3*a+3*d,t=t*(t*(-3*b+9*e-9*f+3*k)+6*b-12*e+6*f)-3*b+3*e,s=s+q[c]*F.sqrt(r*r+t*t);return l*s}function l(a,b,d){a=I(a);b=I(b);for(var h,f,l,n,k,s,r,O,x,c,t=d?0:[],w=0,v=a.length;w<v;w++)if(x=a[w],"M"==x[0])h=k=x[1],f=s=x[2];else{"C"==x[0]?(x=[h,f].concat(x.slice(1)),
h=x[6],f=x[7]):(x=[h,f,h,f,k,s,k,s],h=k,f=s);for(var G=0,y=b.length;G<y;G++)if(c=b[G],"M"==c[0])l=r=c[1],n=O=c[2];else{"C"==c[0]?(c=[l,n].concat(c.slice(1)),l=c[6],n=c[7]):(c=[l,n,l,n,r,O,r,O],l=r,n=O);var z;var K=x,B=c;z=d;var H=p(K),J=p(B);if(q(H,J)){for(var H=e.apply(0,K),J=e.apply(0,B),H=~~(H/8),J=~~(J/8),U=[],A=[],F={},M=z?0:[],P=0;P<H+1;P++){var C=u.apply(0,K.concat(P/H));U.push({x:C.x,y:C.y,t:P/H})}for(P=0;P<J+1;P++)C=u.apply(0,B.concat(P/J)),A.push({x:C.x,y:C.y,t:P/J});for(P=0;P<H;P++)for(K=
0;K<J;K++){var Q=U[P],L=U[P+1],B=A[K],C=A[K+1],N=0.001>Z(L.x-Q.x)?"y":"x",S=0.001>Z(C.x-B.x)?"y":"x",R;R=Q.x;var Y=Q.y,V=L.x,ea=L.y,fa=B.x,ga=B.y,ha=C.x,ia=C.y;if(W(R,V)<X(fa,ha)||X(R,V)>W(fa,ha)||W(Y,ea)<X(ga,ia)||X(Y,ea)>W(ga,ia))R=void 0;else{var $=(R*ea-Y*V)*(fa-ha)-(R-V)*(fa*ia-ga*ha),aa=(R*ea-Y*V)*(ga-ia)-(Y-ea)*(fa*ia-ga*ha),ja=(R-V)*(ga-ia)-(Y-ea)*(fa-ha);if(ja){var $=$/ja,aa=aa/ja,ja=+$.toFixed(2),ba=+aa.toFixed(2);R=ja<+X(R,V).toFixed(2)||ja>+W(R,V).toFixed(2)||ja<+X(fa,ha).toFixed(2)||
ja>+W(fa,ha).toFixed(2)||ba<+X(Y,ea).toFixed(2)||ba>+W(Y,ea).toFixed(2)||ba<+X(ga,ia).toFixed(2)||ba>+W(ga,ia).toFixed(2)?void 0:{x:$,y:aa}}else R=void 0}R&&F[R.x.toFixed(4)]!=R.y.toFixed(4)&&(F[R.x.toFixed(4)]=R.y.toFixed(4),Q=Q.t+Z((R[N]-Q[N])/(L[N]-Q[N]))*(L.t-Q.t),B=B.t+Z((R[S]-B[S])/(C[S]-B[S]))*(C.t-B.t),0<=Q&&1>=Q&&0<=B&&1>=B&&(z?M++:M.push({x:R.x,y:R.y,t1:Q,t2:B})))}z=M}else z=z?0:[];if(d)t+=z;else{H=0;for(J=z.length;H<J;H++)z[H].segment1=w,z[H].segment2=G,z[H].bez1=x,z[H].bez2=c;t=t.concat(z)}}}return t}
function r(a){var b=A(a);if(b.bbox)return C(b.bbox);if(!a)return w();a=I(a);for(var d=0,e=0,h=[],f=[],l,n=0,k=a.length;n<k;n++)l=a[n],"M"==l[0]?(d=l[1],e=l[2],h.push(d),f.push(e)):(d=U(d,e,l[1],l[2],l[3],l[4],l[5],l[6]),h=h.concat(d.min.x,d.max.x),f=f.concat(d.min.y,d.max.y),d=l[5],e=l[6]);a=X.apply(0,h);l=X.apply(0,f);h=W.apply(0,h);f=W.apply(0,f);f=w(a,l,h-a,f-l);b.bbox=C(f);return f}function s(a,b,d,e,h){if(h)return[["M",+a+ +h,b],["l",d-2*h,0],["a",h,h,0,0,1,h,h],["l",0,e-2*h],["a",h,h,0,0,1,
-h,h],["l",2*h-d,0],["a",h,h,0,0,1,-h,-h],["l",0,2*h-e],["a",h,h,0,0,1,h,-h],["z"] ];a=[["M",a,b],["l",d,0],["l",0,e],["l",-d,0],["z"] ];a.toString=z;return a}function x(a,b,d,e,h){null==h&&null==e&&(e=d);a=+a;b=+b;d=+d;e=+e;if(null!=h){var f=Math.PI/180,l=a+d*Math.cos(-e*f);a+=d*Math.cos(-h*f);var n=b+d*Math.sin(-e*f);b+=d*Math.sin(-h*f);d=[["M",l,n],["A",d,d,0,+(180<h-e),0,a,b] ]}else d=[["M",a,b],["m",0,-e],["a",d,e,0,1,1,0,2*e],["a",d,e,0,1,1,0,-2*e],["z"] ];d.toString=z;return d}function G(b){var e=
A(b);if(e.abs)return d(e.abs);Q(b,"array")&&Q(b&&b[0],"array")||(b=a.parsePathString(b));if(!b||!b.length)return[["M",0,0] ];var h=[],f=0,l=0,n=0,k=0,p=0;"M"==b[0][0]&&(f=+b[0][1],l=+b[0][2],n=f,k=l,p++,h[0]=["M",f,l]);for(var q=3==b.length&&"M"==b[0][0]&&"R"==b[1][0].toUpperCase()&&"Z"==b[2][0].toUpperCase(),s,r,w=p,c=b.length;w<c;w++){h.push(s=[]);r=b[w];p=r[0];if(p!=p.toUpperCase())switch(s[0]=p.toUpperCase(),s[0]){case "A":s[1]=r[1];s[2]=r[2];s[3]=r[3];s[4]=r[4];s[5]=r[5];s[6]=+r[6]+f;s[7]=+r[7]+
l;break;case "V":s[1]=+r[1]+l;break;case "H":s[1]=+r[1]+f;break;case "R":for(var t=[f,l].concat(r.slice(1)),u=2,v=t.length;u<v;u++)t[u]=+t[u]+f,t[++u]=+t[u]+l;h.pop();h=h.concat(P(t,q));break;case "O":h.pop();t=x(f,l,r[1],r[2]);t.push(t[0]);h=h.concat(t);break;case "U":h.pop();h=h.concat(x(f,l,r[1],r[2],r[3]));s=["U"].concat(h[h.length-1].slice(-2));break;case "M":n=+r[1]+f,k=+r[2]+l;default:for(u=1,v=r.length;u<v;u++)s[u]=+r[u]+(u%2?f:l)}else if("R"==p)t=[f,l].concat(r.slice(1)),h.pop(),h=h.concat(P(t,
q)),s=["R"].concat(r.slice(-2));else if("O"==p)h.pop(),t=x(f,l,r[1],r[2]),t.push(t[0]),h=h.concat(t);else if("U"==p)h.pop(),h=h.concat(x(f,l,r[1],r[2],r[3])),s=["U"].concat(h[h.length-1].slice(-2));else for(t=0,u=r.length;t<u;t++)s[t]=r[t];p=p.toUpperCase();if("O"!=p)switch(s[0]){case "Z":f=+n;l=+k;break;case "H":f=s[1];break;case "V":l=s[1];break;case "M":n=s[s.length-2],k=s[s.length-1];default:f=s[s.length-2],l=s[s.length-1]}}h.toString=z;e.abs=d(h);return h}function h(a,b,d,e){return[a,b,d,e,d,
e]}function J(a,b,d,e,h,f){var l=1/3,n=2/3;return[l*a+n*d,l*b+n*e,l*h+n*d,l*f+n*e,h,f]}function K(b,d,e,h,f,l,n,k,p,s){var r=120*S/180,q=S/180*(+f||0),c=[],t,x=a._.cacher(function(a,b,c){var d=a*F.cos(c)-b*F.sin(c);a=a*F.sin(c)+b*F.cos(c);return{x:d,y:a}});if(s)v=s[0],t=s[1],l=s[2],u=s[3];else{t=x(b,d,-q);b=t.x;d=t.y;t=x(k,p,-q);k=t.x;p=t.y;F.cos(S/180*f);F.sin(S/180*f);t=(b-k)/2;v=(d-p)/2;u=t*t/(e*e)+v*v/(h*h);1<u&&(u=F.sqrt(u),e*=u,h*=u);var u=e*e,w=h*h,u=(l==n?-1:1)*F.sqrt(Z((u*w-u*v*v-w*t*t)/
(u*v*v+w*t*t)));l=u*e*v/h+(b+k)/2;var u=u*-h*t/e+(d+p)/2,v=F.asin(((d-u)/h).toFixed(9));t=F.asin(((p-u)/h).toFixed(9));v=b<l?S-v:v;t=k<l?S-t:t;0>v&&(v=2*S+v);0>t&&(t=2*S+t);n&&v>t&&(v-=2*S);!n&&t>v&&(t-=2*S)}if(Z(t-v)>r){var c=t,w=k,G=p;t=v+r*(n&&t>v?1:-1);k=l+e*F.cos(t);p=u+h*F.sin(t);c=K(k,p,e,h,f,0,n,w,G,[t,c,l,u])}l=t-v;f=F.cos(v);r=F.sin(v);n=F.cos(t);t=F.sin(t);l=F.tan(l/4);e=4/3*e*l;l*=4/3*h;h=[b,d];b=[b+e*r,d-l*f];d=[k+e*t,p-l*n];k=[k,p];b[0]=2*h[0]-b[0];b[1]=2*h[1]-b[1];if(s)return[b,d,k].concat(c);
c=[b,d,k].concat(c).join().split(",");s=[];k=0;for(p=c.length;k<p;k++)s[k]=k%2?x(c[k-1],c[k],q).y:x(c[k],c[k+1],q).x;return s}function U(a,b,d,e,h,f,l,k){for(var n=[],p=[[],[] ],s,r,c,t,q=0;2>q;++q)0==q?(r=6*a-12*d+6*h,s=-3*a+9*d-9*h+3*l,c=3*d-3*a):(r=6*b-12*e+6*f,s=-3*b+9*e-9*f+3*k,c=3*e-3*b),1E-12>Z(s)?1E-12>Z(r)||(s=-c/r,0<s&&1>s&&n.push(s)):(t=r*r-4*c*s,c=F.sqrt(t),0>t||(t=(-r+c)/(2*s),0<t&&1>t&&n.push(t),s=(-r-c)/(2*s),0<s&&1>s&&n.push(s)));for(r=q=n.length;q--;)s=n[q],c=1-s,p[0][q]=c*c*c*a+3*
c*c*s*d+3*c*s*s*h+s*s*s*l,p[1][q]=c*c*c*b+3*c*c*s*e+3*c*s*s*f+s*s*s*k;p[0][r]=a;p[1][r]=b;p[0][r+1]=l;p[1][r+1]=k;p[0].length=p[1].length=r+2;return{min:{x:X.apply(0,p[0]),y:X.apply(0,p[1])},max:{x:W.apply(0,p[0]),y:W.apply(0,p[1])}}}function I(a,b){var e=!b&&A(a);if(!b&&e.curve)return d(e.curve);var f=G(a),l=b&&G(b),n={x:0,y:0,bx:0,by:0,X:0,Y:0,qx:null,qy:null},k={x:0,y:0,bx:0,by:0,X:0,Y:0,qx:null,qy:null},p=function(a,b,c){if(!a)return["C",b.x,b.y,b.x,b.y,b.x,b.y];a[0]in{T:1,Q:1}||(b.qx=b.qy=null);
switch(a[0]){case "M":b.X=a[1];b.Y=a[2];break;case "A":a=["C"].concat(K.apply(0,[b.x,b.y].concat(a.slice(1))));break;case "S":"C"==c||"S"==c?(c=2*b.x-b.bx,b=2*b.y-b.by):(c=b.x,b=b.y);a=["C",c,b].concat(a.slice(1));break;case "T":"Q"==c||"T"==c?(b.qx=2*b.x-b.qx,b.qy=2*b.y-b.qy):(b.qx=b.x,b.qy=b.y);a=["C"].concat(J(b.x,b.y,b.qx,b.qy,a[1],a[2]));break;case "Q":b.qx=a[1];b.qy=a[2];a=["C"].concat(J(b.x,b.y,a[1],a[2],a[3],a[4]));break;case "L":a=["C"].concat(h(b.x,b.y,a[1],a[2]));break;case "H":a=["C"].concat(h(b.x,
b.y,a[1],b.y));break;case "V":a=["C"].concat(h(b.x,b.y,b.x,a[1]));break;case "Z":a=["C"].concat(h(b.x,b.y,b.X,b.Y))}return a},s=function(a,b){if(7<a[b].length){a[b].shift();for(var c=a[b];c.length;)q[b]="A",l&&(u[b]="A"),a.splice(b++,0,["C"].concat(c.splice(0,6)));a.splice(b,1);v=W(f.length,l&&l.length||0)}},r=function(a,b,c,d,e){a&&b&&"M"==a[e][0]&&"M"!=b[e][0]&&(b.splice(e,0,["M",d.x,d.y]),c.bx=0,c.by=0,c.x=a[e][1],c.y=a[e][2],v=W(f.length,l&&l.length||0))},q=[],u=[],c="",t="",x=0,v=W(f.length,
l&&l.length||0);for(;x<v;x++){f[x]&&(c=f[x][0]);"C"!=c&&(q[x]=c,x&&(t=q[x-1]));f[x]=p(f[x],n,t);"A"!=q[x]&&"C"==c&&(q[x]="C");s(f,x);l&&(l[x]&&(c=l[x][0]),"C"!=c&&(u[x]=c,x&&(t=u[x-1])),l[x]=p(l[x],k,t),"A"!=u[x]&&"C"==c&&(u[x]="C"),s(l,x));r(f,l,n,k,x);r(l,f,k,n,x);var w=f[x],z=l&&l[x],y=w.length,U=l&&z.length;n.x=w[y-2];n.y=w[y-1];n.bx=$(w[y-4])||n.x;n.by=$(w[y-3])||n.y;k.bx=l&&($(z[U-4])||k.x);k.by=l&&($(z[U-3])||k.y);k.x=l&&z[U-2];k.y=l&&z[U-1]}l||(e.curve=d(f));return l?[f,l]:f}function P(a,
b){for(var d=[],e=0,h=a.length;h-2*!b>e;e+=2){var f=[{x:+a[e-2],y:+a[e-1]},{x:+a[e],y:+a[e+1]},{x:+a[e+2],y:+a[e+3]},{x:+a[e+4],y:+a[e+5]}];b?e?h-4==e?f[3]={x:+a[0],y:+a[1]}:h-2==e&&(f[2]={x:+a[0],y:+a[1]},f[3]={x:+a[2],y:+a[3]}):f[0]={x:+a[h-2],y:+a[h-1]}:h-4==e?f[3]=f[2]:e||(f[0]={x:+a[e],y:+a[e+1]});d.push(["C",(-f[0].x+6*f[1].x+f[2].x)/6,(-f[0].y+6*f[1].y+f[2].y)/6,(f[1].x+6*f[2].x-f[3].x)/6,(f[1].y+6*f[2].y-f[3].y)/6,f[2].x,f[2].y])}return d}y=k.prototype;var Q=a.is,C=a._.clone,L="hasOwnProperty",
N=/,?([a-z]),?/gi,$=parseFloat,F=Math,S=F.PI,X=F.min,W=F.max,ma=F.pow,Z=F.abs;M=n(1);var na=n(),ba=n(0,1),V=a._unit2px;a.path=A;a.path.getTotalLength=M;a.path.getPointAtLength=na;a.path.getSubpath=function(a,b,d){if(1E-6>this.getTotalLength(a)-d)return ba(a,b).end;a=ba(a,d,1);return b?ba(a,b).end:a};y.getTotalLength=function(){if(this.node.getTotalLength)return this.node.getTotalLength()};y.getPointAtLength=function(a){return na(this.attr("d"),a)};y.getSubpath=function(b,d){return a.path.getSubpath(this.attr("d"),
b,d)};a._.box=w;a.path.findDotsAtSegment=u;a.path.bezierBBox=p;a.path.isPointInsideBBox=b;a.path.isBBoxIntersect=q;a.path.intersection=function(a,b){return l(a,b)};a.path.intersectionNumber=function(a,b){return l(a,b,1)};a.path.isPointInside=function(a,d,e){var h=r(a);return b(h,d,e)&&1==l(a,[["M",d,e],["H",h.x2+10] ],1)%2};a.path.getBBox=r;a.path.get={path:function(a){return a.attr("path")},circle:function(a){a=V(a);return x(a.cx,a.cy,a.r)},ellipse:function(a){a=V(a);return x(a.cx||0,a.cy||0,a.rx,
a.ry)},rect:function(a){a=V(a);return s(a.x||0,a.y||0,a.width,a.height,a.rx,a.ry)},image:function(a){a=V(a);return s(a.x||0,a.y||0,a.width,a.height)},line:function(a){return"M"+[a.attr("x1")||0,a.attr("y1")||0,a.attr("x2"),a.attr("y2")]},polyline:function(a){return"M"+a.attr("points")},polygon:function(a){return"M"+a.attr("points")+"z"},deflt:function(a){a=a.node.getBBox();return s(a.x,a.y,a.width,a.height)}};a.path.toRelative=function(b){var e=A(b),h=String.prototype.toLowerCase;if(e.rel)return d(e.rel);
a.is(b,"array")&&a.is(b&&b[0],"array")||(b=a.parsePathString(b));var f=[],l=0,n=0,k=0,p=0,s=0;"M"==b[0][0]&&(l=b[0][1],n=b[0][2],k=l,p=n,s++,f.push(["M",l,n]));for(var r=b.length;s<r;s++){var q=f[s]=[],x=b[s];if(x[0]!=h.call(x[0]))switch(q[0]=h.call(x[0]),q[0]){case "a":q[1]=x[1];q[2]=x[2];q[3]=x[3];q[4]=x[4];q[5]=x[5];q[6]=+(x[6]-l).toFixed(3);q[7]=+(x[7]-n).toFixed(3);break;case "v":q[1]=+(x[1]-n).toFixed(3);break;case "m":k=x[1],p=x[2];default:for(var c=1,t=x.length;c<t;c++)q[c]=+(x[c]-(c%2?l:
n)).toFixed(3)}else for(f[s]=[],"m"==x[0]&&(k=x[1]+l,p=x[2]+n),q=0,c=x.length;q<c;q++)f[s][q]=x[q];x=f[s].length;switch(f[s][0]){case "z":l=k;n=p;break;case "h":l+=+f[s][x-1];break;case "v":n+=+f[s][x-1];break;default:l+=+f[s][x-2],n+=+f[s][x-1]}}f.toString=z;e.rel=d(f);return f};a.path.toAbsolute=G;a.path.toCubic=I;a.path.map=function(a,b){if(!b)return a;var d,e,h,f,l,n,k;a=I(a);h=0;for(l=a.length;h<l;h++)for(k=a[h],f=1,n=k.length;f<n;f+=2)d=b.x(k[f],k[f+1]),e=b.y(k[f],k[f+1]),k[f]=d,k[f+1]=e;return a};
a.path.toString=z;a.path.clone=d});C.plugin(function(a,v,y,C){var A=Math.max,w=Math.min,z=function(a){this.items=[];this.bindings={};this.length=0;this.type="set";if(a)for(var f=0,n=a.length;f<n;f++)a[f]&&(this[this.items.length]=this.items[this.items.length]=a[f],this.length++)};v=z.prototype;v.push=function(){for(var a,f,n=0,k=arguments.length;n<k;n++)if(a=arguments[n])f=this.items.length,this[f]=this.items[f]=a,this.length++;return this};v.pop=function(){this.length&&delete this[this.length--];
return this.items.pop()};v.forEach=function(a,f){for(var n=0,k=this.items.length;n<k&&!1!==a.call(f,this.items[n],n);n++);return this};v.animate=function(d,f,n,u){"function"!=typeof n||n.length||(u=n,n=L.linear);d instanceof a._.Animation&&(u=d.callback,n=d.easing,f=n.dur,d=d.attr);var p=arguments;if(a.is(d,"array")&&a.is(p[p.length-1],"array"))var b=!0;var q,e=function(){q?this.b=q:q=this.b},l=0,r=u&&function(){l++==this.length&&u.call(this)};return this.forEach(function(a,l){k.once("snap.animcreated."+
a.id,e);b?p[l]&&a.animate.apply(a,p[l]):a.animate(d,f,n,r)})};v.remove=function(){for(;this.length;)this.pop().remove();return this};v.bind=function(a,f,k){var u={};if("function"==typeof f)this.bindings[a]=f;else{var p=k||a;this.bindings[a]=function(a){u[p]=a;f.attr(u)}}return this};v.attr=function(a){var f={},k;for(k in a)if(this.bindings[k])this.bindings[k](a[k]);else f[k]=a[k];a=0;for(k=this.items.length;a<k;a++)this.items[a].attr(f);return this};v.clear=function(){for(;this.length;)this.pop()};
v.splice=function(a,f,k){a=0>a?A(this.length+a,0):a;f=A(0,w(this.length-a,f));var u=[],p=[],b=[],q;for(q=2;q<arguments.length;q++)b.push(arguments[q]);for(q=0;q<f;q++)p.push(this[a+q]);for(;q<this.length-a;q++)u.push(this[a+q]);var e=b.length;for(q=0;q<e+u.length;q++)this.items[a+q]=this[a+q]=q<e?b[q]:u[q-e];for(q=this.items.length=this.length-=f-e;this[q];)delete this[q++];return new z(p)};v.exclude=function(a){for(var f=0,k=this.length;f<k;f++)if(this[f]==a)return this.splice(f,1),!0;return!1};
v.insertAfter=function(a){for(var f=this.items.length;f--;)this.items[f].insertAfter(a);return this};v.getBBox=function(){for(var a=[],f=[],k=[],u=[],p=this.items.length;p--;)if(!this.items[p].removed){var b=this.items[p].getBBox();a.push(b.x);f.push(b.y);k.push(b.x+b.width);u.push(b.y+b.height)}a=w.apply(0,a);f=w.apply(0,f);k=A.apply(0,k);u=A.apply(0,u);return{x:a,y:f,x2:k,y2:u,width:k-a,height:u-f,cx:a+(k-a)/2,cy:f+(u-f)/2}};v.clone=function(a){a=new z;for(var f=0,k=this.items.length;f<k;f++)a.push(this.items[f].clone());
return a};v.toString=function(){return"Snap\u2018s set"};v.type="set";a.set=function(){var a=new z;arguments.length&&a.push.apply(a,Array.prototype.slice.call(arguments,0));return a}});C.plugin(function(a,v,y,C){function A(a){var b=a[0];switch(b.toLowerCase()){case "t":return[b,0,0];case "m":return[b,1,0,0,1,0,0];case "r":return 4==a.length?[b,0,a[2],a[3] ]:[b,0];case "s":return 5==a.length?[b,1,1,a[3],a[4] ]:3==a.length?[b,1,1]:[b,1]}}function w(b,d,f){d=q(d).replace(/\.{3}|\u2026/g,b);b=a.parseTransformString(b)||
[];d=a.parseTransformString(d)||[];for(var k=Math.max(b.length,d.length),p=[],v=[],h=0,w,z,y,I;h<k;h++){y=b[h]||A(d[h]);I=d[h]||A(y);if(y[0]!=I[0]||"r"==y[0].toLowerCase()&&(y[2]!=I[2]||y[3]!=I[3])||"s"==y[0].toLowerCase()&&(y[3]!=I[3]||y[4]!=I[4])){b=a._.transform2matrix(b,f());d=a._.transform2matrix(d,f());p=[["m",b.a,b.b,b.c,b.d,b.e,b.f] ];v=[["m",d.a,d.b,d.c,d.d,d.e,d.f] ];break}p[h]=[];v[h]=[];w=0;for(z=Math.max(y.length,I.length);w<z;w++)w in y&&(p[h][w]=y[w]),w in I&&(v[h][w]=I[w])}return{from:u(p),
to:u(v),f:n(p)}}function z(a){return a}function d(a){return function(b){return+b.toFixed(3)+a}}function f(b){return a.rgb(b[0],b[1],b[2])}function n(a){var b=0,d,f,k,n,h,p,q=[];d=0;for(f=a.length;d<f;d++){h="[";p=['"'+a[d][0]+'"'];k=1;for(n=a[d].length;k<n;k++)p[k]="val["+b++ +"]";h+=p+"]";q[d]=h}return Function("val","return Snap.path.toString.call(["+q+"])")}function u(a){for(var b=[],d=0,f=a.length;d<f;d++)for(var k=1,n=a[d].length;k<n;k++)b.push(a[d][k]);return b}var p={},b=/[a-z]+$/i,q=String;
p.stroke=p.fill="colour";v.prototype.equal=function(a,b){return k("snap.util.equal",this,a,b).firstDefined()};k.on("snap.util.equal",function(e,k){var r,s;r=q(this.attr(e)||"");var x=this;if(r==+r&&k==+k)return{from:+r,to:+k,f:z};if("colour"==p[e])return r=a.color(r),s=a.color(k),{from:[r.r,r.g,r.b,r.opacity],to:[s.r,s.g,s.b,s.opacity],f:f};if("transform"==e||"gradientTransform"==e||"patternTransform"==e)return k instanceof a.Matrix&&(k=k.toTransformString()),a._.rgTransform.test(k)||(k=a._.svgTransform2string(k)),
w(r,k,function(){return x.getBBox(1)});if("d"==e||"path"==e)return r=a.path.toCubic(r,k),{from:u(r[0]),to:u(r[1]),f:n(r[0])};if("points"==e)return r=q(r).split(a._.separator),s=q(k).split(a._.separator),{from:r,to:s,f:function(a){return a}};aUnit=r.match(b);s=q(k).match(b);return aUnit&&aUnit==s?{from:parseFloat(r),to:parseFloat(k),f:d(aUnit)}:{from:this.asPX(e),to:this.asPX(e,k),f:z}})});C.plugin(function(a,v,y,C){var A=v.prototype,w="createTouch"in C.doc;v="click dblclick mousedown mousemove mouseout mouseover mouseup touchstart touchmove touchend touchcancel".split(" ");
var z={mousedown:"touchstart",mousemove:"touchmove",mouseup:"touchend"},d=function(a,b){var d="y"==a?"scrollTop":"scrollLeft",e=b&&b.node?b.node.ownerDocument:C.doc;return e[d in e.documentElement?"documentElement":"body"][d]},f=function(){this.returnValue=!1},n=function(){return this.originalEvent.preventDefault()},u=function(){this.cancelBubble=!0},p=function(){return this.originalEvent.stopPropagation()},b=function(){if(C.doc.addEventListener)return function(a,b,e,f){var k=w&&z[b]?z[b]:b,l=function(k){var l=
d("y",f),q=d("x",f);if(w&&z.hasOwnProperty(b))for(var r=0,u=k.targetTouches&&k.targetTouches.length;r<u;r++)if(k.targetTouches[r].target==a||a.contains(k.targetTouches[r].target)){u=k;k=k.targetTouches[r];k.originalEvent=u;k.preventDefault=n;k.stopPropagation=p;break}return e.call(f,k,k.clientX+q,k.clientY+l)};b!==k&&a.addEventListener(b,l,!1);a.addEventListener(k,l,!1);return function(){b!==k&&a.removeEventListener(b,l,!1);a.removeEventListener(k,l,!1);return!0}};if(C.doc.attachEvent)return function(a,
b,e,h){var k=function(a){a=a||h.node.ownerDocument.window.event;var b=d("y",h),k=d("x",h),k=a.clientX+k,b=a.clientY+b;a.preventDefault=a.preventDefault||f;a.stopPropagation=a.stopPropagation||u;return e.call(h,a,k,b)};a.attachEvent("on"+b,k);return function(){a.detachEvent("on"+b,k);return!0}}}(),q=[],e=function(a){for(var b=a.clientX,e=a.clientY,f=d("y"),l=d("x"),n,p=q.length;p--;){n=q[p];if(w)for(var r=a.touches&&a.touches.length,u;r--;){if(u=a.touches[r],u.identifier==n.el._drag.id||n.el.node.contains(u.target)){b=
u.clientX;e=u.clientY;(a.originalEvent?a.originalEvent:a).preventDefault();break}}else a.preventDefault();b+=l;e+=f;k("snap.drag.move."+n.el.id,n.move_scope||n.el,b-n.el._drag.x,e-n.el._drag.y,b,e,a)}},l=function(b){a.unmousemove(e).unmouseup(l);for(var d=q.length,f;d--;)f=q[d],f.el._drag={},k("snap.drag.end."+f.el.id,f.end_scope||f.start_scope||f.move_scope||f.el,b);q=[]};for(y=v.length;y--;)(function(d){a[d]=A[d]=function(e,f){a.is(e,"function")&&(this.events=this.events||[],this.events.push({name:d,
f:e,unbind:b(this.node||document,d,e,f||this)}));return this};a["un"+d]=A["un"+d]=function(a){for(var b=this.events||[],e=b.length;e--;)if(b[e].name==d&&(b[e].f==a||!a)){b[e].unbind();b.splice(e,1);!b.length&&delete this.events;break}return this}})(v[y]);A.hover=function(a,b,d,e){return this.mouseover(a,d).mouseout(b,e||d)};A.unhover=function(a,b){return this.unmouseover(a).unmouseout(b)};var r=[];A.drag=function(b,d,f,h,n,p){function u(r,v,w){(r.originalEvent||r).preventDefault();this._drag.x=v;
this._drag.y=w;this._drag.id=r.identifier;!q.length&&a.mousemove(e).mouseup(l);q.push({el:this,move_scope:h,start_scope:n,end_scope:p});d&&k.on("snap.drag.start."+this.id,d);b&&k.on("snap.drag.move."+this.id,b);f&&k.on("snap.drag.end."+this.id,f);k("snap.drag.start."+this.id,n||h||this,v,w,r)}if(!arguments.length){var v;return this.drag(function(a,b){this.attr({transform:v+(v?"T":"t")+[a,b]})},function(){v=this.transform().local})}this._drag={};r.push({el:this,start:u});this.mousedown(u);return this};
A.undrag=function(){for(var b=r.length;b--;)r[b].el==this&&(this.unmousedown(r[b].start),r.splice(b,1),k.unbind("snap.drag.*."+this.id));!r.length&&a.unmousemove(e).unmouseup(l);return this}});C.plugin(function(a,v,y,C){y=y.prototype;var A=/^\s*url\((.+)\)/,w=String,z=a._.$;a.filter={};y.filter=function(d){var f=this;"svg"!=f.type&&(f=f.paper);d=a.parse(w(d));var k=a._.id(),u=z("filter");z(u,{id:k,filterUnits:"userSpaceOnUse"});u.appendChild(d.node);f.defs.appendChild(u);return new v(u)};k.on("snap.util.getattr.filter",
function(){k.stop();var d=z(this.node,"filter");if(d)return(d=w(d).match(A))&&a.select(d[1])});k.on("snap.util.attr.filter",function(d){if(d instanceof v&&"filter"==d.type){k.stop();var f=d.node.id;f||(z(d.node,{id:d.id}),f=d.id);z(this.node,{filter:a.url(f)})}d&&"none"!=d||(k.stop(),this.node.removeAttribute("filter"))});a.filter.blur=function(d,f){null==d&&(d=2);return a.format('<feGaussianBlur stdDeviation="{def}"/>',{def:null==f?d:[d,f]})};a.filter.blur.toString=function(){return this()};a.filter.shadow=
function(d,f,k,u,p){"string"==typeof k&&(p=u=k,k=4);"string"!=typeof u&&(p=u,u="#000");null==k&&(k=4);null==p&&(p=1);null==d&&(d=0,f=2);null==f&&(f=d);u=a.color(u||"#000");return a.format('<feGaussianBlur in="SourceAlpha" stdDeviation="{blur}"/><feOffset dx="{dx}" dy="{dy}" result="offsetblur"/><feFlood flood-color="{color}"/><feComposite in2="offsetblur" operator="in"/><feComponentTransfer><feFuncA type="linear" slope="{opacity}"/></feComponentTransfer><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>',
{color:u,dx:d,dy:f,blur:k,opacity:p})};a.filter.shadow.toString=function(){return this()};a.filter.grayscale=function(d){null==d&&(d=1);return a.format('<feColorMatrix type="matrix" values="{a} {b} {c} 0 0 {d} {e} {f} 0 0 {g} {b} {h} 0 0 0 0 0 1 0"/>',{a:0.2126+0.7874*(1-d),b:0.7152-0.7152*(1-d),c:0.0722-0.0722*(1-d),d:0.2126-0.2126*(1-d),e:0.7152+0.2848*(1-d),f:0.0722-0.0722*(1-d),g:0.2126-0.2126*(1-d),h:0.0722+0.9278*(1-d)})};a.filter.grayscale.toString=function(){return this()};a.filter.sepia=
function(d){null==d&&(d=1);return a.format('<feColorMatrix type="matrix" values="{a} {b} {c} 0 0 {d} {e} {f} 0 0 {g} {h} {i} 0 0 0 0 0 1 0"/>',{a:0.393+0.607*(1-d),b:0.769-0.769*(1-d),c:0.189-0.189*(1-d),d:0.349-0.349*(1-d),e:0.686+0.314*(1-d),f:0.168-0.168*(1-d),g:0.272-0.272*(1-d),h:0.534-0.534*(1-d),i:0.131+0.869*(1-d)})};a.filter.sepia.toString=function(){return this()};a.filter.saturate=function(d){null==d&&(d=1);return a.format('<feColorMatrix type="saturate" values="{amount}"/>',{amount:1-
d})};a.filter.saturate.toString=function(){return this()};a.filter.hueRotate=function(d){return a.format('<feColorMatrix type="hueRotate" values="{angle}"/>',{angle:d||0})};a.filter.hueRotate.toString=function(){return this()};a.filter.invert=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="table" tableValues="{amount} {amount2}"/><feFuncG type="table" tableValues="{amount} {amount2}"/><feFuncB type="table" tableValues="{amount} {amount2}"/></feComponentTransfer>',{amount:d,
amount2:1-d})};a.filter.invert.toString=function(){return this()};a.filter.brightness=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="linear" slope="{amount}"/><feFuncG type="linear" slope="{amount}"/><feFuncB type="linear" slope="{amount}"/></feComponentTransfer>',{amount:d})};a.filter.brightness.toString=function(){return this()};a.filter.contrast=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="linear" slope="{amount}" intercept="{amount2}"/><feFuncG type="linear" slope="{amount}" intercept="{amount2}"/><feFuncB type="linear" slope="{amount}" intercept="{amount2}"/></feComponentTransfer>',
{amount:d,amount2:0.5-d/2})};a.filter.contrast.toString=function(){return this()}});return C});

]]> </script>
<script> <![CDATA[

(function (glob, factory) {
    // AMD support
    if (typeof define === "function" && define.amd) {
        // Define as an anonymous module
        define("Gadfly", ["Snap.svg"], function (Snap) {
            return factory(Snap);
        });
    } else {
        // Browser globals (glob is window)
        // Snap adds itself to window
        glob.Gadfly = factory(glob.Snap);
    }
}(this, function (Snap) {

var Gadfly = {};

// Get an x/y coordinate value in pixels
var xPX = function(fig, x) {
    var client_box = fig.node.getBoundingClientRect();
    return x * fig.node.viewBox.baseVal.width / client_box.width;
};

var yPX = function(fig, y) {
    var client_box = fig.node.getBoundingClientRect();
    return y * fig.node.viewBox.baseVal.height / client_box.height;
};


Snap.plugin(function (Snap, Element, Paper, global) {
    // Traverse upwards from a snap element to find and return the first
    // note with the "plotroot" class.
    Element.prototype.plotroot = function () {
        var element = this;
        while (!element.hasClass("plotroot") && element.parent() != null) {
            element = element.parent();
        }
        return element;
    };

    Element.prototype.svgroot = function () {
        var element = this;
        while (element.node.nodeName != "svg" && element.parent() != null) {
            element = element.parent();
        }
        return element;
    };

    Element.prototype.plotbounds = function () {
        var root = this.plotroot()
        var bbox = root.select(".guide.background").node.getBBox();
        return {
            x0: bbox.x,
            x1: bbox.x + bbox.width,
            y0: bbox.y,
            y1: bbox.y + bbox.height
        };
    };

    Element.prototype.plotcenter = function () {
        var root = this.plotroot()
        var bbox = root.select(".guide.background").node.getBBox();
        return {
            x: bbox.x + bbox.width / 2,
            y: bbox.y + bbox.height / 2
        };
    };

    // Emulate IE style mouseenter/mouseleave events, since Microsoft always
    // does everything right.
    // See: http://www.dynamic-tools.net/toolbox/isMouseLeaveOrEnter/
    var events = ["mouseenter", "mouseleave"];

    for (i in events) {
        (function (event_name) {
            var event_name = events[i];
            Element.prototype[event_name] = function (fn, scope) {
                if (Snap.is(fn, "function")) {
                    var fn2 = function (event) {
                        if (event.type != "mouseover" && event.type != "mouseout") {
                            return;
                        }

                        var reltg = event.relatedTarget ? event.relatedTarget :
                            event.type == "mouseout" ? event.toElement : event.fromElement;
                        while (reltg && reltg != this.node) reltg = reltg.parentNode;

                        if (reltg != this.node) {
                            return fn.apply(this, event);
                        }
                    };

                    if (event_name == "mouseenter") {
                        this.mouseover(fn2, scope);
                    } else {
                        this.mouseout(fn2, scope);
                    }
                }
                return this;
            };
        })(events[i]);
    }


    Element.prototype.mousewheel = function (fn, scope) {
        if (Snap.is(fn, "function")) {
            var el = this;
            var fn2 = function (event) {
                fn.apply(el, [event]);
            };
        }

        this.node.addEventListener(
            /Firefox/i.test(navigator.userAgent) ? "DOMMouseScroll" : "mousewheel",
            fn2);

        return this;
    };


    // Snap's attr function can be too slow for things like panning/zooming.
    // This is a function to directly update element attributes without going
    // through eve.
    Element.prototype.attribute = function(key, val) {
        if (val === undefined) {
            return this.node.getAttribute(key);
        } else {
            this.node.setAttribute(key, val);
            return this;
        }
    };
});


// When the plot is moused over, emphasize the grid lines.
Gadfly.plot_mouseover = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);

    var xgridlines = root.select(".xgridlines"),
        ygridlines = root.select(".ygridlines");

    xgridlines.data("unfocused_strokedash",
                    xgridlines.attribute("stroke-dasharray").replace(/(\d)(,|$)/g, "$1mm$2"));
    ygridlines.data("unfocused_strokedash",
                    ygridlines.attribute("stroke-dasharray").replace(/(\d)(,|$)/g, "$1mm$2"));

    // emphasize grid lines
    var destcolor = root.data("focused_xgrid_color");
    xgridlines.attribute("stroke-dasharray", "none")
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    destcolor = root.data("focused_ygrid_color");
    ygridlines.attribute("stroke-dasharray", "none")
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    // reveal zoom slider
    root.select(".zoomslider")
        .animate({opacity: 1.0}, 250);
};


// Unemphasize grid lines on mouse out.
Gadfly.plot_mouseout = function(event) {
    var root = this.plotroot();
    var xgridlines = root.select(".xgridlines"),
        ygridlines = root.select(".ygridlines");

    var destcolor = root.data("unfocused_xgrid_color");

    xgridlines.attribute("stroke-dasharray", xgridlines.data("unfocused_strokedash"))
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    destcolor = root.data("unfocused_ygrid_color");
    ygridlines.attribute("stroke-dasharray", ygridlines.data("unfocused_strokedash"))
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    // hide zoom slider
    root.select(".zoomslider")
        .animate({opacity: 0.0}, 250);
};


var set_geometry_transform = function(root, tx, ty, scale) {
    var xscalable = root.hasClass("xscalable"),
        yscalable = root.hasClass("yscalable");

    var old_scale = root.data("scale");

    var xscale = xscalable ? scale : 1.0,
        yscale = yscalable ? scale : 1.0;

    tx = xscalable ? tx : 0.0;
    ty = yscalable ? ty : 0.0;

    var t = new Snap.Matrix().translate(tx, ty).scale(xscale, yscale);

    root.selectAll(".geometry, image")
        .forEach(function (element, i) {
            element.transform(t);
        });

    bounds = root.plotbounds();

    if (yscalable) {
        var xfixed_t = new Snap.Matrix().translate(0, ty).scale(1.0, yscale);
        root.selectAll(".xfixed")
            .forEach(function (element, i) {
                element.transform(xfixed_t);
            });

        root.select(".ylabels")
            .transform(xfixed_t)
            .selectAll("text")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var cx = element.asPX("x"),
                        cy = element.asPX("y");
                    var st = element.data("static_transform");
                    unscale_t = new Snap.Matrix();
                    unscale_t.scale(1, 1/scale, cx, cy).add(st);
                    element.transform(unscale_t);

                    var y = cy * scale + ty;
                    element.attr("visibility",
                        bounds.y0 <= y && y <= bounds.y1 ? "visible" : "hidden");
                }
            });
    }

    if (xscalable) {
        var yfixed_t = new Snap.Matrix().translate(tx, 0).scale(xscale, 1.0);
        var xtrans = new Snap.Matrix().translate(tx, 0);
        root.selectAll(".yfixed")
            .forEach(function (element, i) {
                element.transform(yfixed_t);
            });

        root.select(".xlabels")
            .transform(yfixed_t)
            .selectAll("text")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var cx = element.asPX("x"),
                        cy = element.asPX("y");
                    var st = element.data("static_transform");
                    unscale_t = new Snap.Matrix();
                    unscale_t.scale(1/scale, 1, cx, cy).add(st);

                    element.transform(unscale_t);

                    var x = cx * scale + tx;
                    element.attr("visibility",
                        bounds.x0 <= x && x <= bounds.x1 ? "visible" : "hidden");
                    }
            });
    }

    // we must unscale anything that is scale invariance: widths, raiduses, etc.
    var size_attribs = ["font-size"];
    var unscaled_selection = ".geometry, .geometry *";
    if (xscalable) {
        size_attribs.push("rx");
        unscaled_selection += ", .xgridlines";
    }
    if (yscalable) {
        size_attribs.push("ry");
        unscaled_selection += ", .ygridlines";
    }

    root.selectAll(unscaled_selection)
        .forEach(function (element, i) {
            // circle need special help
            if (element.node.nodeName == "circle") {
                var cx = element.attribute("cx"),
                    cy = element.attribute("cy");
                unscale_t = new Snap.Matrix().scale(1/xscale, 1/yscale,
                                                        cx, cy);
                element.transform(unscale_t);
                return;
            }

            for (i in size_attribs) {
                var key = size_attribs[i];
                var val = parseFloat(element.attribute(key));
                if (val !== undefined && val != 0 && !isNaN(val)) {
                    element.attribute(key, val * old_scale / scale);
                }
            }
        });
};


// Find the most appropriate tick scale and update label visibility.
var update_tickscale = function(root, scale, axis) {
    if (!root.hasClass(axis + "scalable")) return;

    var tickscales = root.data(axis + "tickscales");
    var best_tickscale = 1.0;
    var best_tickscale_dist = Infinity;
    for (tickscale in tickscales) {
        var dist = Math.abs(Math.log(tickscale) - Math.log(scale));
        if (dist < best_tickscale_dist) {
            best_tickscale_dist = dist;
            best_tickscale = tickscale;
        }
    }

    if (best_tickscale != root.data(axis + "tickscale")) {
        root.data(axis + "tickscale", best_tickscale);
        var mark_inscale_gridlines = function (element, i) {
            var inscale = element.attr("gadfly:scale") == best_tickscale;
            element.attribute("gadfly:inscale", inscale);
            element.attr("visibility", inscale ? "visible" : "hidden");
        };

        var mark_inscale_labels = function (element, i) {
            var inscale = element.attr("gadfly:scale") == best_tickscale;
            element.attribute("gadfly:inscale", inscale);
            element.attr("visibility", inscale ? "visible" : "hidden");
        };

        root.select("." + axis + "gridlines").selectAll("path").forEach(mark_inscale_gridlines);
        root.select("." + axis + "labels").selectAll("text").forEach(mark_inscale_labels);
    }
};


var set_plot_pan_zoom = function(root, tx, ty, scale) {
    var old_scale = root.data("scale");
    var bounds = root.plotbounds();

    var width = bounds.x1 - bounds.x0,
        height = bounds.y1 - bounds.y0;

    // compute the viewport derived from tx, ty, and scale
    var x_min = -width * scale - (scale * width - width),
        x_max = width * scale,
        y_min = -height * scale - (scale * height - height),
        y_max = height * scale;

    var x0 = bounds.x0 - scale * bounds.x0,
        y0 = bounds.y0 - scale * bounds.y0;

    var tx = Math.max(Math.min(tx - x0, x_max), x_min),
        ty = Math.max(Math.min(ty - y0, y_max), y_min);

    tx += x0;
    ty += y0;

    // when the scale change, we may need to alter which set of
    // ticks is being displayed
    if (scale != old_scale) {
        update_tickscale(root, scale, "x");
        update_tickscale(root, scale, "y");
    }

    set_geometry_transform(root, tx, ty, scale);

    root.data("scale", scale);
    root.data("tx", tx);
    root.data("ty", ty);
};


var scale_centered_translation = function(root, scale) {
    var bounds = root.plotbounds();

    var width = bounds.x1 - bounds.x0,
        height = bounds.y1 - bounds.y0;

    var tx0 = root.data("tx"),
        ty0 = root.data("ty");

    var scale0 = root.data("scale");

    // how off from center the current view is
    var xoff = tx0 - (bounds.x0 * (1 - scale0) + (width * (1 - scale0)) / 2),
        yoff = ty0 - (bounds.y0 * (1 - scale0) + (height * (1 - scale0)) / 2);

    // rescale offsets
    xoff = xoff * scale / scale0;
    yoff = yoff * scale / scale0;

    // adjust for the panel position being scaled
    var x_edge_adjust = bounds.x0 * (1 - scale),
        y_edge_adjust = bounds.y0 * (1 - scale);

    return {
        x: xoff + x_edge_adjust + (width - width * scale) / 2,
        y: yoff + y_edge_adjust + (height - height * scale) / 2
    };
};


// Initialize data for panning zooming if it isn't already.
var init_pan_zoom = function(root) {
    if (root.data("zoompan-ready")) {
        return;
    }

    // The non-scaling-stroke trick. Rather than try to correct for the
    // stroke-width when zooming, we force it to a fixed value.
    var px_per_mm = root.node.getCTM().a;

    // Drag events report deltas in pixels, which we'd like to convert to
    // millimeters.
    root.data("px_per_mm", px_per_mm);

    root.selectAll("path")
        .forEach(function (element, i) {
        sw = element.asPX("stroke-width") * px_per_mm;
        if (sw > 0) {
            element.attribute("stroke-width", sw);
            element.attribute("vector-effect", "non-scaling-stroke");
        }
    });

    // Store ticks labels original tranformation
    root.selectAll(".xlabels > text, .ylabels > text")
        .forEach(function (element, i) {
            var lm = element.transform().localMatrix;
            element.data("static_transform",
                new Snap.Matrix(lm.a, lm.b, lm.c, lm.d, lm.e, lm.f));
        });

    var xgridlines = root.select(".xgridlines");
    var ygridlines = root.select(".ygridlines");
    var xlabels = root.select(".xlabels");
    var ylabels = root.select(".ylabels");

    if (root.data("tx") === undefined) root.data("tx", 0);
    if (root.data("ty") === undefined) root.data("ty", 0);
    if (root.data("scale") === undefined) root.data("scale", 1.0);
    if (root.data("xtickscales") === undefined) {

        // index all the tick scales that are listed
        var xtickscales = {};
        var ytickscales = {};
        var add_x_tick_scales = function (element, i) {
            xtickscales[element.attribute("gadfly:scale")] = true;
        };
        var add_y_tick_scales = function (element, i) {
            ytickscales[element.attribute("gadfly:scale")] = true;
        };

        if (xgridlines) xgridlines.selectAll("path").forEach(add_x_tick_scales);
        if (ygridlines) ygridlines.selectAll("path").forEach(add_y_tick_scales);
        if (xlabels) xlabels.selectAll("text").forEach(add_x_tick_scales);
        if (ylabels) ylabels.selectAll("text").forEach(add_y_tick_scales);

        root.data("xtickscales", xtickscales);
        root.data("ytickscales", ytickscales);
        root.data("xtickscale", 1.0);
    }

    var min_scale = 1.0, max_scale = 1.0;
    for (scale in xtickscales) {
        min_scale = Math.min(min_scale, scale);
        max_scale = Math.max(max_scale, scale);
    }
    for (scale in ytickscales) {
        min_scale = Math.min(min_scale, scale);
        max_scale = Math.max(max_scale, scale);
    }
    root.data("min_scale", min_scale);
    root.data("max_scale", max_scale);

    // store the original positions of labels
    if (xlabels) {
        xlabels.selectAll("text")
               .forEach(function (element, i) {
                   element.data("x", element.asPX("x"));
               });
    }

    if (ylabels) {
        ylabels.selectAll("text")
               .forEach(function (element, i) {
                   element.data("y", element.asPX("y"));
               });
    }

    // mark grid lines and ticks as in or out of scale.
    var mark_inscale = function (element, i) {
        element.attribute("gadfly:inscale", element.attribute("gadfly:scale") == 1.0);
    };

    if (xgridlines) xgridlines.selectAll("path").forEach(mark_inscale);
    if (ygridlines) ygridlines.selectAll("path").forEach(mark_inscale);
    if (xlabels) xlabels.selectAll("text").forEach(mark_inscale);
    if (ylabels) ylabels.selectAll("text").forEach(mark_inscale);

    // figure out the upper ond lower bounds on panning using the maximum
    // and minum grid lines
    var bounds = root.plotbounds();
    var pan_bounds = {
        x0: 0.0,
        y0: 0.0,
        x1: 0.0,
        y1: 0.0
    };

    if (xgridlines) {
        xgridlines
            .selectAll("path")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var bbox = element.node.getBBox();
                    if (bounds.x1 - bbox.x < pan_bounds.x0) {
                        pan_bounds.x0 = bounds.x1 - bbox.x;
                    }
                    if (bounds.x0 - bbox.x > pan_bounds.x1) {
                        pan_bounds.x1 = bounds.x0 - bbox.x;
                    }
                    element.attr("visibility", "visible");
                }
            });
    }

    if (ygridlines) {
        ygridlines
            .selectAll("path")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var bbox = element.node.getBBox();
                    if (bounds.y1 - bbox.y < pan_bounds.y0) {
                        pan_bounds.y0 = bounds.y1 - bbox.y;
                    }
                    if (bounds.y0 - bbox.y > pan_bounds.y1) {
                        pan_bounds.y1 = bounds.y0 - bbox.y;
                    }
                    element.attr("visibility", "visible");
                }
            });
    }

    // nudge these values a little
    pan_bounds.x0 -= 5;
    pan_bounds.x1 += 5;
    pan_bounds.y0 -= 5;
    pan_bounds.y1 += 5;
    root.data("pan_bounds", pan_bounds);

    root.data("zoompan-ready", true)
};


// Panning
Gadfly.guide_background_drag_onmove = function(dx, dy, x, y, event) {
    var root = this.plotroot();
    var px_per_mm = root.data("px_per_mm");
    dx /= px_per_mm;
    dy /= px_per_mm;

    var tx0 = root.data("tx"),
        ty0 = root.data("ty");

    var dx0 = root.data("dx"),
        dy0 = root.data("dy");

    root.data("dx", dx);
    root.data("dy", dy);

    dx = dx - dx0;
    dy = dy - dy0;

    var tx = tx0 + dx,
        ty = ty0 + dy;

    set_plot_pan_zoom(root, tx, ty, root.data("scale"));
};


Gadfly.guide_background_drag_onstart = function(x, y, event) {
    var root = this.plotroot();
    root.data("dx", 0);
    root.data("dy", 0);
    init_pan_zoom(root);
};


Gadfly.guide_background_drag_onend = function(event) {
    var root = this.plotroot();
};


Gadfly.guide_background_scroll = function(event) {
    if (event.shiftKey) {
        var root = this.plotroot();
        init_pan_zoom(root);
        var new_scale = root.data("scale") * Math.pow(2, 0.002 * event.wheelDelta);
        new_scale = Math.max(
            root.data("min_scale"),
            Math.min(root.data("max_scale"), new_scale))
        update_plot_scale(root, new_scale);
        event.stopPropagation();
    }
};


Gadfly.zoomslider_button_mouseover = function(event) {
    this.select(".button_logo")
         .animate({fill: this.data("mouseover_color")}, 100);
};


Gadfly.zoomslider_button_mouseout = function(event) {
     this.select(".button_logo")
         .animate({fill: this.data("mouseout_color")}, 100);
};


Gadfly.zoomslider_zoomout_click = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);
    var min_scale = root.data("min_scale"),
        scale = root.data("scale");
    Snap.animate(
        scale,
        Math.max(min_scale, scale / 1.5),
        function (new_scale) {
            update_plot_scale(root, new_scale);
        },
        200);
};


Gadfly.zoomslider_zoomin_click = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);
    var max_scale = root.data("max_scale"),
        scale = root.data("scale");

    Snap.animate(
        scale,
        Math.min(max_scale, scale * 1.5),
        function (new_scale) {
            update_plot_scale(root, new_scale);
        },
        200);
};


Gadfly.zoomslider_track_click = function(event) {
    // TODO
};


Gadfly.zoomslider_thumb_mousedown = function(event) {
    this.animate({fill: this.data("mouseover_color")}, 100);
};


Gadfly.zoomslider_thumb_mouseup = function(event) {
    this.animate({fill: this.data("mouseout_color")}, 100);
};


// compute the position in [0, 1] of the zoom slider thumb from the current scale
var slider_position_from_scale = function(scale, min_scale, max_scale) {
    if (scale >= 1.0) {
        return 0.5 + 0.5 * (Math.log(scale) / Math.log(max_scale));
    }
    else {
        return 0.5 * (Math.log(scale) - Math.log(min_scale)) / (0 - Math.log(min_scale));
    }
}


var update_plot_scale = function(root, new_scale) {
    var trans = scale_centered_translation(root, new_scale);
    set_plot_pan_zoom(root, trans.x, trans.y, new_scale);

    root.selectAll(".zoomslider_thumb")
        .forEach(function (element, i) {
            var min_pos = element.data("min_pos"),
                max_pos = element.data("max_pos"),
                min_scale = root.data("min_scale"),
                max_scale = root.data("max_scale");
            var xmid = (min_pos + max_pos) / 2;
            var xpos = slider_position_from_scale(new_scale, min_scale, max_scale);
            element.transform(new Snap.Matrix().translate(
                Math.max(min_pos, Math.min(
                         max_pos, min_pos + (max_pos - min_pos) * xpos)) - xmid, 0));
    });
};


Gadfly.zoomslider_thumb_dragmove = function(dx, dy, x, y) {
    var root = this.plotroot();
    var min_pos = this.data("min_pos"),
        max_pos = this.data("max_pos"),
        min_scale = root.data("min_scale"),
        max_scale = root.data("max_scale"),
        old_scale = root.data("old_scale");

    var px_per_mm = root.data("px_per_mm");
    dx /= px_per_mm;
    dy /= px_per_mm;

    var xmid = (min_pos + max_pos) / 2;
    var xpos = slider_position_from_scale(old_scale, min_scale, max_scale) +
                   dx / (max_pos - min_pos);

    // compute the new scale
    var new_scale;
    if (xpos >= 0.5) {
        new_scale = Math.exp(2.0 * (xpos - 0.5) * Math.log(max_scale));
    }
    else {
        new_scale = Math.exp(2.0 * xpos * (0 - Math.log(min_scale)) +
                        Math.log(min_scale));
    }
    new_scale = Math.min(max_scale, Math.max(min_scale, new_scale));

    update_plot_scale(root, new_scale);
};


Gadfly.zoomslider_thumb_dragstart = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);

    // keep track of what the scale was when we started dragging
    root.data("old_scale", root.data("scale"));
};


Gadfly.zoomslider_thumb_dragend = function(event) {
};


var toggle_color_class = function(root, color_class, ison) {
    var guides = root.selectAll(".guide." + color_class + ",.guide ." + color_class);
    var geoms = root.selectAll(".geometry." + color_class + ",.geometry ." + color_class);
    if (ison) {
        guides.animate({opacity: 0.5}, 250);
        geoms.animate({opacity: 0.0}, 250);
    } else {
        guides.animate({opacity: 1.0}, 250);
        geoms.animate({opacity: 1.0}, 250);
    }
};


Gadfly.colorkey_swatch_click = function(event) {
    var root = this.plotroot();
    var color_class = this.data("color_class");

    if (event.shiftKey) {
        root.selectAll(".colorkey text")
            .forEach(function (element) {
                var other_color_class = element.data("color_class");
                if (other_color_class != color_class) {
                    toggle_color_class(root, other_color_class,
                                       element.attr("opacity") == 1.0);
                }
            });
    } else {
        toggle_color_class(root, color_class, this.attr("opacity") == 1.0);
    }
};


return Gadfly;

}));


//@ sourceURL=gadfly.js

(function (glob, factory) {
    // AMD support
      if (typeof require === "function" && typeof define === "function" && define.amd) {
        require(["Snap.svg", "Gadfly"], function (Snap, Gadfly) {
            factory(Snap, Gadfly);
        });
      } else {
          factory(glob.Snap, glob.Gadfly);
      }
})(window, function (Snap, Gadfly) {
    var fig = Snap("#fig-8cb17422ee6242b29bb8c9357ec0f6f7");
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-4")
   .drag(function() {}, function() {}, function() {});
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-6")
   .data("color_class", "color_0")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-7")
   .data("color_class", "color_1")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-9")
   .data("color_class", "color_0")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-10")
   .data("color_class", "color_1")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-13")
   .mouseenter(Gadfly.plot_mouseover)
.mouseleave(Gadfly.plot_mouseout)
.mousewheel(Gadfly.guide_background_scroll)
.drag(Gadfly.guide_background_drag_onmove,
      Gadfly.guide_background_drag_onstart,
      Gadfly.guide_background_drag_onend)
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-17")
   .plotroot().data("unfocused_ygrid_color", "#D0D0E0")
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-17")
   .plotroot().data("focused_ygrid_color", "#A0A0A0")
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-171")
   .plotroot().data("unfocused_xgrid_color", "#D0D0E0")
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-171")
   .plotroot().data("focused_xgrid_color", "#A0A0A0")
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-181")
   .data("mouseover_color", "#cd5c5c")
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-181")
   .data("mouseout_color", "#6a6a6a")
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-181")
   .click(Gadfly.zoomslider_zoomin_click)
.mouseenter(Gadfly.zoomslider_button_mouseover)
.mouseleave(Gadfly.zoomslider_button_mouseout)
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-185")
   .data("max_pos", 106.11)
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-185")
   .data("min_pos", 89.11)
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-185")
   .click(Gadfly.zoomslider_track_click);
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-187")
   .data("max_pos", 106.11)
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-187")
   .data("min_pos", 89.11)
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-187")
   .data("mouseover_color", "#cd5c5c")
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-187")
   .data("mouseout_color", "#6a6a6a")
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-187")
   .drag(Gadfly.zoomslider_thumb_dragmove,
     Gadfly.zoomslider_thumb_dragstart,
     Gadfly.zoomslider_thumb_dragend)
.mousedown(Gadfly.zoomslider_thumb_mousedown)
.mouseup(Gadfly.zoomslider_thumb_mouseup)
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-189")
   .data("mouseover_color", "#cd5c5c")
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-189")
   .data("mouseout_color", "#6a6a6a")
;
fig.select("#fig-8cb17422ee6242b29bb8c9357ec0f6f7-element-189")
   .click(Gadfly.zoomslider_zoomout_click)
.mouseenter(Gadfly.zoomslider_button_mouseover)
.mouseleave(Gadfly.zoomslider_button_mouseout)
;
    });
]]> </script>
</svg>




Its pretty clear that more females survived over males.  
So lets predict Survived=1 if Sex=female else Survived=0. With that you have your first predictive model ready!!

But lets not stop there, we have lot more dimensions to the data. Lets see if we can make use of those to enhance the model. We have Age as one of the columns, using which we can create another dimension to the data - a variable indicating whether the person was a child or not.


    train[:Child] = 1
    train[isna(train[:Age]), :Child] = 1 #this can be avoided. Its just an explicit way to demonstrate the rule
    train[train[:Age] .< 18, :Child] = 1 #same applies for this, as all Child fields are 1 by default except when Age >= 18
    train[train[:Age] .>= 18, :Child] = 0
    head(train)




<table class="data-frame"><tr><th></th><th>PassengerId</th><th>Survived</th><th>Pclass</th><th>Name</th><th>Sex</th><th>Age</th><th>SibSp</th><th>Parch</th><th>Ticket</th><th>Fare</th><th>Cabin</th><th>Embarked</th><th>Child</th></tr><tr><th>1</th><td>1</td><td>0</td><td>3</td><td>Braund, Mr. Owen Harris</td><td>male</td><td>22.0</td><td>1</td><td>0</td><td>A/5 21171</td><td>7.25</td><td>NA</td><td>S</td><td>0</td></tr><tr><th>2</th><td>2</td><td>1</td><td>1</td><td>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td><td>female</td><td>38.0</td><td>1</td><td>0</td><td>PC 17599</td><td>71.2833</td><td>C85</td><td>C</td><td>0</td></tr><tr><th>3</th><td>3</td><td>1</td><td>3</td><td>Heikkinen, Miss. Laina</td><td>female</td><td>26.0</td><td>0</td><td>0</td><td>STON/O2. 3101282</td><td>7.925</td><td>NA</td><td>S</td><td>0</td></tr><tr><th>4</th><td>4</td><td>1</td><td>1</td><td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td><td>female</td><td>35.0</td><td>1</td><td>0</td><td>113803</td><td>53.1</td><td>C123</td><td>S</td><td>0</td></tr><tr><th>5</th><td>5</td><td>0</td><td>3</td><td>Allen, Mr. William Henry</td><td>male</td><td>35.0</td><td>0</td><td>0</td><td>373450</td><td>8.05</td><td>NA</td><td>S</td><td>0</td></tr><tr><th>6</th><td>6</td><td>0</td><td>3</td><td>Moran, Mr. James</td><td>male</td><td>NA</td><td>0</td><td>0</td><td>330877</td><td>8.4583</td><td>NA</td><td>Q</td><td>1</td></tr></table>



Not that we have our Child indicator variable, lets try to plot the survival rate of children on Titanic.


    plot(train, x="Child", y="Survived", color="Survived", Geom.histogram(position=:stack), Scale.color_discrete_manual("red","green"))




<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:gadfly="http://www.gadflyjl.org/ns"
     version="1.2"
     width="141.42mm" height="100mm" viewBox="0 0 141.42 100"
     stroke="none"
     fill="#000000"
     stroke-width="0.3"
     font-size="3.88"

     id="fig-f3ea444778014fbf80c9634001e4f6c2">
<g class="plotroot xscalable yscalable" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-1">
  <g font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" fill="#564A55" stroke="#000000" stroke-opacity="0.000" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-2">
    <text x="71.71" y="88.39" text-anchor="middle" dy="0.6em">Child</text>
  </g>
  <g class="guide xlabels" font-size="2.82" font-family="'PT Sans Caption','Helvetica Neue','Helvetica',sans-serif" fill="#6C606B" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-3">
    <text x="-105.77" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">-4</text>
    <text x="-73.5" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">-3</text>
    <text x="-41.23" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">-2</text>
    <text x="-8.96" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">-1</text>
    <text x="23.31" y="84.39" text-anchor="middle" visibility="visible" gadfly:scale="1.0">0</text>
    <text x="55.57" y="84.39" text-anchor="middle" visibility="visible" gadfly:scale="1.0">1</text>
    <text x="87.84" y="84.39" text-anchor="middle" visibility="visible" gadfly:scale="1.0">2</text>
    <text x="120.11" y="84.39" text-anchor="middle" visibility="visible" gadfly:scale="1.0">3</text>
    <text x="152.38" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">4</text>
    <text x="184.65" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">5</text>
    <text x="216.92" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">6</text>
    <text x="249.19" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">7</text>
    <text x="-73.5" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-3.0</text>
    <text x="-70.28" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-2.9</text>
    <text x="-67.05" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-2.8</text>
    <text x="-63.82" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-2.7</text>
    <text x="-60.59" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-2.6</text>
    <text x="-57.37" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-2.5</text>
    <text x="-54.14" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-2.4</text>
    <text x="-50.91" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-2.3</text>
    <text x="-47.69" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-2.2</text>
    <text x="-44.46" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-2.1</text>
    <text x="-41.23" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-2.0</text>
    <text x="-38.01" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-1.9</text>
    <text x="-34.78" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-1.8</text>
    <text x="-31.55" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-1.7</text>
    <text x="-28.33" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-1.6</text>
    <text x="-25.1" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-1.5</text>
    <text x="-21.87" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-1.4</text>
    <text x="-18.64" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-1.3</text>
    <text x="-15.42" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-1.2</text>
    <text x="-12.19" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-1.1</text>
    <text x="-8.96" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-1.0</text>
    <text x="-5.74" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.9</text>
    <text x="-2.51" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.8</text>
    <text x="0.72" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.7</text>
    <text x="3.94" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.6</text>
    <text x="7.17" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.5</text>
    <text x="10.4" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.4</text>
    <text x="13.62" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.3</text>
    <text x="16.85" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.2</text>
    <text x="20.08" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.1</text>
    <text x="23.31" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.0</text>
    <text x="26.53" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.1</text>
    <text x="29.76" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.2</text>
    <text x="32.99" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.3</text>
    <text x="36.21" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.4</text>
    <text x="39.44" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.5</text>
    <text x="42.67" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.6</text>
    <text x="45.89" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.7</text>
    <text x="49.12" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.8</text>
    <text x="52.35" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.9</text>
    <text x="55.57" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.0</text>
    <text x="58.8" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.1</text>
    <text x="62.03" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.2</text>
    <text x="65.25" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.3</text>
    <text x="68.48" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.4</text>
    <text x="71.71" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.5</text>
    <text x="74.94" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.6</text>
    <text x="78.16" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.7</text>
    <text x="81.39" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.8</text>
    <text x="84.62" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.9</text>
    <text x="87.84" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">2.0</text>
    <text x="91.07" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">2.1</text>
    <text x="94.3" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">2.2</text>
    <text x="97.52" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">2.3</text>
    <text x="100.75" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">2.4</text>
    <text x="103.98" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">2.5</text>
    <text x="107.2" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">2.6</text>
    <text x="110.43" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">2.7</text>
    <text x="113.66" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">2.8</text>
    <text x="116.89" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">2.9</text>
    <text x="120.11" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">3.0</text>
    <text x="123.34" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">3.1</text>
    <text x="126.57" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">3.2</text>
    <text x="129.79" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">3.3</text>
    <text x="133.02" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">3.4</text>
    <text x="136.25" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">3.5</text>
    <text x="139.47" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">3.6</text>
    <text x="142.7" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">3.7</text>
    <text x="145.93" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">3.8</text>
    <text x="149.15" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">3.9</text>
    <text x="152.38" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">4.0</text>
    <text x="155.61" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">4.1</text>
    <text x="158.84" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">4.2</text>
    <text x="162.06" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">4.3</text>
    <text x="165.29" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">4.4</text>
    <text x="168.52" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">4.5</text>
    <text x="171.74" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">4.6</text>
    <text x="174.97" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">4.7</text>
    <text x="178.2" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">4.8</text>
    <text x="181.42" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">4.9</text>
    <text x="184.65" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">5.0</text>
    <text x="187.88" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">5.1</text>
    <text x="191.1" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">5.2</text>
    <text x="194.33" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">5.3</text>
    <text x="197.56" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">5.4</text>
    <text x="200.78" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">5.5</text>
    <text x="204.01" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">5.6</text>
    <text x="207.24" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">5.7</text>
    <text x="210.47" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">5.8</text>
    <text x="213.69" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">5.9</text>
    <text x="216.92" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">6.0</text>
    <text x="-73.5" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="0.5">-3</text>
    <text x="23.31" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="0.5">0</text>
    <text x="120.11" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="0.5">3</text>
    <text x="216.92" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="0.5">6</text>
    <text x="-73.5" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-3.0</text>
    <text x="-67.05" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-2.8</text>
    <text x="-60.59" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-2.6</text>
    <text x="-54.14" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-2.4</text>
    <text x="-47.69" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-2.2</text>
    <text x="-41.23" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-2.0</text>
    <text x="-34.78" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-1.8</text>
    <text x="-28.33" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-1.6</text>
    <text x="-21.87" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-1.4</text>
    <text x="-15.42" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-1.2</text>
    <text x="-8.96" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-1.0</text>
    <text x="-2.51" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.8</text>
    <text x="3.94" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.6</text>
    <text x="10.4" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.4</text>
    <text x="16.85" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.2</text>
    <text x="23.31" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.0</text>
    <text x="29.76" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.2</text>
    <text x="36.21" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.4</text>
    <text x="42.67" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.6</text>
    <text x="49.12" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.8</text>
    <text x="55.57" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.0</text>
    <text x="62.03" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.2</text>
    <text x="68.48" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.4</text>
    <text x="74.94" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.6</text>
    <text x="81.39" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.8</text>
    <text x="87.84" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">2.0</text>
    <text x="94.3" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">2.2</text>
    <text x="100.75" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">2.4</text>
    <text x="107.2" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">2.6</text>
    <text x="113.66" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">2.8</text>
    <text x="120.11" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">3.0</text>
    <text x="126.57" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">3.2</text>
    <text x="133.02" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">3.4</text>
    <text x="139.47" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">3.6</text>
    <text x="145.93" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">3.8</text>
    <text x="152.38" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">4.0</text>
    <text x="158.84" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">4.2</text>
    <text x="165.29" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">4.4</text>
    <text x="171.74" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">4.6</text>
    <text x="178.2" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">4.8</text>
    <text x="184.65" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">5.0</text>
    <text x="191.1" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">5.2</text>
    <text x="197.56" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">5.4</text>
    <text x="204.01" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">5.6</text>
    <text x="210.47" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">5.8</text>
    <text x="216.92" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">6.0</text>
  </g>
  <g class="guide colorkey" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-4">
    <g fill="#4C404B" font-size="2.82" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-5">
      <text x="125.93" y="42.86" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-6" class="color_0">0</text>
      <text x="125.93" y="46.48" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-7" class="color_1">1</text>
    </g>
    <g stroke="#000000" stroke-opacity="0.000" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-8">
      <rect x="123.11" y="41.95" width="1.81" height="1.81" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-9" class="color_0" fill="#FF0000"/>
      <rect x="123.11" y="45.58" width="1.81" height="1.81" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-10" class="color_1" fill="#008000"/>
    </g>
    <g fill="#362A35" font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" stroke="#000000" stroke-opacity="0.000" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-11">
      <text x="123.11" y="39.04" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-12">Survived</text>
    </g>
  </g>
  <g clip-path="url(#fig-f3ea444778014fbf80c9634001e4f6c2-element-14)" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-13">
    <g pointer-events="visible" opacity="1" fill="#000000" fill-opacity="0.000" stroke="#000000" stroke-opacity="0.000" class="guide background" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-15">
      <rect x="21.31" y="5" width="100.81" height="75.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-16"/>
    </g>
    <g class="guide ygridlines xfixed" stroke-dasharray="0.5,0.5" stroke-width="0.2" stroke="#D0D0E0" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-17">
      <path fill="none" d="M21.31,164.77 L 122.11 164.77" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-18" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,150.43 L 122.11 150.43" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-19" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,136.09 L 122.11 136.09" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-20" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,121.74 L 122.11 121.74" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-21" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,107.4 L 122.11 107.4" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-22" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,93.06 L 122.11 93.06" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-23" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,78.72 L 122.11 78.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-24" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,64.37 L 122.11 64.37" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-25" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,50.03 L 122.11 50.03" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-26" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,35.69 L 122.11 35.69" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-27" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,21.34 L 122.11 21.34" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-28" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,7 L 122.11 7" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-29" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,-7.34 L 122.11 -7.34" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-30" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,-21.69 L 122.11 -21.69" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-31" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,-36.03 L 122.11 -36.03" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-32" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,-50.37 L 122.11 -50.37" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-33" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,-64.72 L 122.11 -64.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-34" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,-79.06 L 122.11 -79.06" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-35" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M21.31,150.43 L 122.11 150.43" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-36" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,146.84 L 122.11 146.84" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-37" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,143.26 L 122.11 143.26" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-38" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,139.67 L 122.11 139.67" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-39" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,136.09 L 122.11 136.09" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-40" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,132.5 L 122.11 132.5" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-41" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,128.92 L 122.11 128.92" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-42" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,125.33 L 122.11 125.33" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-43" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,121.74 L 122.11 121.74" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-44" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,118.16 L 122.11 118.16" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-45" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,114.57 L 122.11 114.57" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-46" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,110.99 L 122.11 110.99" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-47" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,107.4 L 122.11 107.4" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-48" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,103.82 L 122.11 103.82" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-49" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,100.23 L 122.11 100.23" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-50" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,96.64 L 122.11 96.64" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-51" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,93.06 L 122.11 93.06" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-52" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,89.47 L 122.11 89.47" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-53" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,85.89 L 122.11 85.89" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-54" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,82.3 L 122.11 82.3" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-55" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,78.72 L 122.11 78.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-56" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,75.13 L 122.11 75.13" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-57" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,71.54 L 122.11 71.54" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-58" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,67.96 L 122.11 67.96" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-59" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,64.37 L 122.11 64.37" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-60" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,60.79 L 122.11 60.79" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-61" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,57.2 L 122.11 57.2" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-62" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,53.61 L 122.11 53.61" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-63" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,50.03 L 122.11 50.03" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-64" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,46.44 L 122.11 46.44" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-65" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,42.86 L 122.11 42.86" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-66" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,39.27 L 122.11 39.27" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-67" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,35.69 L 122.11 35.69" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-68" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,32.1 L 122.11 32.1" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-69" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,28.51 L 122.11 28.51" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-70" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,24.93 L 122.11 24.93" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-71" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,21.34 L 122.11 21.34" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,17.76 L 122.11 17.76" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-73" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,14.17 L 122.11 14.17" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-74" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,10.59 L 122.11 10.59" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-75" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,7 L 122.11 7" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-76" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,3.41 L 122.11 3.41" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-77" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-0.17 L 122.11 -0.17" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-78" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-3.76 L 122.11 -3.76" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-79" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-7.34 L 122.11 -7.34" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-80" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-10.93 L 122.11 -10.93" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-81" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-14.51 L 122.11 -14.51" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-82" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-18.1 L 122.11 -18.1" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-83" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-21.69 L 122.11 -21.69" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-84" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-25.27 L 122.11 -25.27" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-85" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-28.86 L 122.11 -28.86" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-86" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-32.44 L 122.11 -32.44" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-87" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-36.03 L 122.11 -36.03" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-88" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-39.61 L 122.11 -39.61" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-89" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-43.2 L 122.11 -43.2" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-90" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-46.79 L 122.11 -46.79" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-91" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-50.37 L 122.11 -50.37" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-92" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-53.96 L 122.11 -53.96" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-93" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-57.54 L 122.11 -57.54" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-94" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-61.13 L 122.11 -61.13" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-95" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,-64.72 L 122.11 -64.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-96" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M21.31,150.43 L 122.11 150.43" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-97" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M21.31,78.72 L 122.11 78.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-98" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M21.31,7 L 122.11 7" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-99" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M21.31,-64.72 L 122.11 -64.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-100" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M21.31,150.43 L 122.11 150.43" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-101" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,143.26 L 122.11 143.26" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-102" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,136.09 L 122.11 136.09" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-103" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,128.92 L 122.11 128.92" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-104" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,121.74 L 122.11 121.74" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-105" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,114.57 L 122.11 114.57" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-106" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,107.4 L 122.11 107.4" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-107" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,100.23 L 122.11 100.23" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-108" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,93.06 L 122.11 93.06" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-109" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,85.89 L 122.11 85.89" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-110" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,78.72 L 122.11 78.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-111" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,71.54 L 122.11 71.54" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-112" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,64.37 L 122.11 64.37" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-113" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,57.2 L 122.11 57.2" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-114" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,50.03 L 122.11 50.03" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-115" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,42.86 L 122.11 42.86" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-116" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,35.69 L 122.11 35.69" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-117" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,28.51 L 122.11 28.51" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-118" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,21.34 L 122.11 21.34" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-119" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,14.17 L 122.11 14.17" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-120" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,7 L 122.11 7" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-121" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,-0.17 L 122.11 -0.17" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-122" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,-7.34 L 122.11 -7.34" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-123" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,-14.51 L 122.11 -14.51" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-124" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,-21.69 L 122.11 -21.69" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-125" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,-28.86 L 122.11 -28.86" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-126" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,-36.03 L 122.11 -36.03" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-127" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,-43.2 L 122.11 -43.2" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-128" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,-50.37 L 122.11 -50.37" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-129" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,-57.54 L 122.11 -57.54" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-130" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M21.31,-64.72 L 122.11 -64.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-131" visibility="hidden" gadfly:scale="5.0"/>
    </g>
    <g class="guide xgridlines yfixed" stroke-dasharray="0.5,0.5" stroke-width="0.2" stroke="#D0D0E0" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-132">
      <path fill="none" d="M-105.77,5 L -105.77 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-133" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M-73.5,5 L -73.5 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-134" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M-41.23,5 L -41.23 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-135" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M-8.96,5 L -8.96 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-136" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M23.31,5 L 23.31 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-137" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M55.57,5 L 55.57 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-138" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M87.84,5 L 87.84 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-139" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M120.11,5 L 120.11 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-140" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M152.38,5 L 152.38 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-141" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M184.65,5 L 184.65 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-142" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M216.92,5 L 216.92 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-143" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M249.19,5 L 249.19 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-144" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M-73.5,5 L -73.5 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-145" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-70.28,5 L -70.28 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-146" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-67.05,5 L -67.05 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-147" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-63.82,5 L -63.82 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-148" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-60.59,5 L -60.59 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-149" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-57.37,5 L -57.37 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-150" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-54.14,5 L -54.14 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-151" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-50.91,5 L -50.91 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-152" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-47.69,5 L -47.69 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-153" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-44.46,5 L -44.46 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-154" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-41.23,5 L -41.23 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-155" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-38.01,5 L -38.01 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-156" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-34.78,5 L -34.78 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-157" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-31.55,5 L -31.55 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-158" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-28.33,5 L -28.33 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-159" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-25.1,5 L -25.1 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-160" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-21.87,5 L -21.87 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-161" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-18.64,5 L -18.64 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-162" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-15.42,5 L -15.42 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-163" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-12.19,5 L -12.19 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-164" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-8.96,5 L -8.96 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-165" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-5.74,5 L -5.74 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-166" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-2.51,5 L -2.51 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-167" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M0.72,5 L 0.72 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-168" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M3.94,5 L 3.94 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-169" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M7.17,5 L 7.17 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-170" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M10.4,5 L 10.4 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-171" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M13.62,5 L 13.62 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-172" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M16.85,5 L 16.85 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-173" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M20.08,5 L 20.08 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-174" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M23.31,5 L 23.31 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-175" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M26.53,5 L 26.53 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-176" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M29.76,5 L 29.76 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-177" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M32.99,5 L 32.99 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-178" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M36.21,5 L 36.21 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-179" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M39.44,5 L 39.44 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-180" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M42.67,5 L 42.67 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-181" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M45.89,5 L 45.89 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-182" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M49.12,5 L 49.12 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-183" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M52.35,5 L 52.35 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-184" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M55.57,5 L 55.57 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-185" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M58.8,5 L 58.8 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-186" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M62.03,5 L 62.03 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-187" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M65.25,5 L 65.25 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-188" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M68.48,5 L 68.48 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-189" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M71.71,5 L 71.71 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-190" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M74.94,5 L 74.94 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-191" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M78.16,5 L 78.16 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-192" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M81.39,5 L 81.39 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-193" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M84.62,5 L 84.62 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-194" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M87.84,5 L 87.84 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-195" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M91.07,5 L 91.07 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-196" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M94.3,5 L 94.3 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-197" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M97.52,5 L 97.52 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-198" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M100.75,5 L 100.75 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-199" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M103.98,5 L 103.98 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-200" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M107.2,5 L 107.2 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-201" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M110.43,5 L 110.43 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-202" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M113.66,5 L 113.66 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-203" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M116.89,5 L 116.89 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-204" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M120.11,5 L 120.11 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-205" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M123.34,5 L 123.34 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-206" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M126.57,5 L 126.57 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-207" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M129.79,5 L 129.79 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-208" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M133.02,5 L 133.02 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-209" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M136.25,5 L 136.25 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-210" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M139.47,5 L 139.47 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-211" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M142.7,5 L 142.7 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-212" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M145.93,5 L 145.93 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-213" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M149.15,5 L 149.15 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-214" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M152.38,5 L 152.38 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-215" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M155.61,5 L 155.61 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-216" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M158.84,5 L 158.84 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-217" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M162.06,5 L 162.06 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-218" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M165.29,5 L 165.29 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-219" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M168.52,5 L 168.52 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-220" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M171.74,5 L 171.74 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-221" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M174.97,5 L 174.97 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-222" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M178.2,5 L 178.2 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-223" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M181.42,5 L 181.42 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-224" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M184.65,5 L 184.65 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-225" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M187.88,5 L 187.88 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-226" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M191.1,5 L 191.1 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-227" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M194.33,5 L 194.33 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-228" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M197.56,5 L 197.56 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-229" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M200.78,5 L 200.78 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-230" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M204.01,5 L 204.01 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-231" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M207.24,5 L 207.24 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-232" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M210.47,5 L 210.47 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-233" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M213.69,5 L 213.69 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-234" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M216.92,5 L 216.92 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-235" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-73.5,5 L -73.5 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-236" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M23.31,5 L 23.31 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-237" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M120.11,5 L 120.11 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-238" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M216.92,5 L 216.92 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-239" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M-73.5,5 L -73.5 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-240" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-67.05,5 L -67.05 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-241" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-60.59,5 L -60.59 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-242" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-54.14,5 L -54.14 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-243" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-47.69,5 L -47.69 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-244" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-41.23,5 L -41.23 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-245" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-34.78,5 L -34.78 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-246" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-28.33,5 L -28.33 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-247" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-21.87,5 L -21.87 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-248" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-15.42,5 L -15.42 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-249" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-8.96,5 L -8.96 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-250" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-2.51,5 L -2.51 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-251" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M3.94,5 L 3.94 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-252" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M10.4,5 L 10.4 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-253" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M16.85,5 L 16.85 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-254" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M23.31,5 L 23.31 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-255" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M29.76,5 L 29.76 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-256" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M36.21,5 L 36.21 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-257" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M42.67,5 L 42.67 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-258" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M49.12,5 L 49.12 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-259" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M55.57,5 L 55.57 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-260" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M62.03,5 L 62.03 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-261" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M68.48,5 L 68.48 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-262" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M74.94,5 L 74.94 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-263" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M81.39,5 L 81.39 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-264" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M87.84,5 L 87.84 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-265" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M94.3,5 L 94.3 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-266" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M100.75,5 L 100.75 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-267" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M107.2,5 L 107.2 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-268" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M113.66,5 L 113.66 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-269" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M120.11,5 L 120.11 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-270" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M126.57,5 L 126.57 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-271" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M133.02,5 L 133.02 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-272" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M139.47,5 L 139.47 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-273" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M145.93,5 L 145.93 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-274" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M152.38,5 L 152.38 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-275" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M158.84,5 L 158.84 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-276" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M165.29,5 L 165.29 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-277" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M171.74,5 L 171.74 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-278" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M178.2,5 L 178.2 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-279" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M184.65,5 L 184.65 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-280" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M191.1,5 L 191.1 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-281" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M197.56,5 L 197.56 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-282" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M204.01,5 L 204.01 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-283" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M210.47,5 L 210.47 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-284" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M216.92,5 L 216.92 80.72" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-285" visibility="hidden" gadfly:scale="5.0"/>
    </g>
    <g class="plotpanel" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-286">
      <g shape-rendering="crispEdges" stroke-width="0.3" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-287">
        <g stroke="#000000" stroke-opacity="0.000" class="geometry" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-288">
          <rect x="23.28" y="54.19" width="32.32" height="24.53" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-289" fill="#008000"/>
          <rect x="55.55" y="NaN" width="32.32" height="0.01" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-290" fill="#008000"/>
          <rect x="87.82" y="NaN" width="32.32" height="0.01" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-291" fill="#008000"/>
          <rect x="23.28" y="14.82" width="32.32" height="39.37" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-292" fill="#FF0000"/>
          <rect x="55.55" y="NaN" width="32.32" height="0.01" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-293" fill="#FF0000"/>
          <rect x="87.82" y="NaN" width="32.32" height="0.01" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-294" fill="#FF0000"/>
        </g>
      </g>
    </g>
    <g opacity="0" class="guide zoomslider" stroke="#000000" stroke-opacity="0.000" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-295">
      <g fill="#EAEAEA" stroke-width="0.3" stroke-opacity="0" stroke="#6A6A6A" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-296">
        <rect x="115.11" y="8" width="4" height="4" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-297"/>
        <g class="button_logo" fill="#6A6A6A" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-298">
          <path d="M115.91,9.6 L 116.71 9.6 116.71 8.8 117.51 8.8 117.51 9.6 118.31 9.6 118.31 10.4 117.51 10.4 117.51 11.2 116.71 11.2 116.71 10.4 115.91 10.4 z" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-299"/>
        </g>
      </g>
      <g fill="#EAEAEA" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-300">
        <rect x="95.61" y="8" width="19" height="4" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-301"/>
      </g>
      <g class="zoomslider_thumb" fill="#6A6A6A" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-302">
        <rect x="104.11" y="8" width="2" height="4" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-303"/>
      </g>
      <g fill="#EAEAEA" stroke-width="0.3" stroke-opacity="0" stroke="#6A6A6A" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-304">
        <rect x="91.11" y="8" width="4" height="4" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-305"/>
        <g class="button_logo" fill="#6A6A6A" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-306">
          <path d="M91.91,9.6 L 94.31 9.6 94.31 10.4 91.91 10.4 z" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-307"/>
        </g>
      </g>
    </g>
  </g>
  <g class="guide ylabels" font-size="2.82" font-family="'PT Sans Caption','Helvetica Neue','Helvetica',sans-serif" fill="#6C606B" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-308">
    <text x="20.31" y="164.77" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-309" visibility="hidden" gadfly:scale="1.0">-1200</text>
    <text x="20.31" y="150.43" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-310" visibility="hidden" gadfly:scale="1.0">-1000</text>
    <text x="20.31" y="136.09" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-311" visibility="hidden" gadfly:scale="1.0">-800</text>
    <text x="20.31" y="121.74" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-312" visibility="hidden" gadfly:scale="1.0">-600</text>
    <text x="20.31" y="107.4" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-313" visibility="hidden" gadfly:scale="1.0">-400</text>
    <text x="20.31" y="93.06" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-314" visibility="hidden" gadfly:scale="1.0">-200</text>
    <text x="20.31" y="78.72" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-315" visibility="visible" gadfly:scale="1.0">0</text>
    <text x="20.31" y="64.37" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-316" visibility="visible" gadfly:scale="1.0">200</text>
    <text x="20.31" y="50.03" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-317" visibility="visible" gadfly:scale="1.0">400</text>
    <text x="20.31" y="35.69" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-318" visibility="visible" gadfly:scale="1.0">600</text>
    <text x="20.31" y="21.34" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-319" visibility="visible" gadfly:scale="1.0">800</text>
    <text x="20.31" y="7" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-320" visibility="visible" gadfly:scale="1.0">1000</text>
    <text x="20.31" y="-7.34" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-321" visibility="hidden" gadfly:scale="1.0">1200</text>
    <text x="20.31" y="-21.69" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-322" visibility="hidden" gadfly:scale="1.0">1400</text>
    <text x="20.31" y="-36.03" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-323" visibility="hidden" gadfly:scale="1.0">1600</text>
    <text x="20.31" y="-50.37" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-324" visibility="hidden" gadfly:scale="1.0">1800</text>
    <text x="20.31" y="-64.72" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-325" visibility="hidden" gadfly:scale="1.0">2000</text>
    <text x="20.31" y="-79.06" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-326" visibility="hidden" gadfly:scale="1.0">2200</text>
    <text x="20.31" y="150.43" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-327" visibility="hidden" gadfly:scale="10.0">-1000</text>
    <text x="20.31" y="146.84" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-328" visibility="hidden" gadfly:scale="10.0">-950</text>
    <text x="20.31" y="143.26" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-329" visibility="hidden" gadfly:scale="10.0">-900</text>
    <text x="20.31" y="139.67" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-330" visibility="hidden" gadfly:scale="10.0">-850</text>
    <text x="20.31" y="136.09" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-331" visibility="hidden" gadfly:scale="10.0">-800</text>
    <text x="20.31" y="132.5" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-332" visibility="hidden" gadfly:scale="10.0">-750</text>
    <text x="20.31" y="128.92" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-333" visibility="hidden" gadfly:scale="10.0">-700</text>
    <text x="20.31" y="125.33" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-334" visibility="hidden" gadfly:scale="10.0">-650</text>
    <text x="20.31" y="121.74" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-335" visibility="hidden" gadfly:scale="10.0">-600</text>
    <text x="20.31" y="118.16" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-336" visibility="hidden" gadfly:scale="10.0">-550</text>
    <text x="20.31" y="114.57" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-337" visibility="hidden" gadfly:scale="10.0">-500</text>
    <text x="20.31" y="110.99" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-338" visibility="hidden" gadfly:scale="10.0">-450</text>
    <text x="20.31" y="107.4" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-339" visibility="hidden" gadfly:scale="10.0">-400</text>
    <text x="20.31" y="103.82" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-340" visibility="hidden" gadfly:scale="10.0">-350</text>
    <text x="20.31" y="100.23" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-341" visibility="hidden" gadfly:scale="10.0">-300</text>
    <text x="20.31" y="96.64" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-342" visibility="hidden" gadfly:scale="10.0">-250</text>
    <text x="20.31" y="93.06" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-343" visibility="hidden" gadfly:scale="10.0">-200</text>
    <text x="20.31" y="89.47" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-344" visibility="hidden" gadfly:scale="10.0">-150</text>
    <text x="20.31" y="85.89" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-345" visibility="hidden" gadfly:scale="10.0">-100</text>
    <text x="20.31" y="82.3" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-346" visibility="hidden" gadfly:scale="10.0">-50</text>
    <text x="20.31" y="78.72" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-347" visibility="hidden" gadfly:scale="10.0">0</text>
    <text x="20.31" y="75.13" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-348" visibility="hidden" gadfly:scale="10.0">50</text>
    <text x="20.31" y="71.54" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-349" visibility="hidden" gadfly:scale="10.0">100</text>
    <text x="20.31" y="67.96" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-350" visibility="hidden" gadfly:scale="10.0">150</text>
    <text x="20.31" y="64.37" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-351" visibility="hidden" gadfly:scale="10.0">200</text>
    <text x="20.31" y="60.79" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-352" visibility="hidden" gadfly:scale="10.0">250</text>
    <text x="20.31" y="57.2" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-353" visibility="hidden" gadfly:scale="10.0">300</text>
    <text x="20.31" y="53.61" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-354" visibility="hidden" gadfly:scale="10.0">350</text>
    <text x="20.31" y="50.03" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-355" visibility="hidden" gadfly:scale="10.0">400</text>
    <text x="20.31" y="46.44" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-356" visibility="hidden" gadfly:scale="10.0">450</text>
    <text x="20.31" y="42.86" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-357" visibility="hidden" gadfly:scale="10.0">500</text>
    <text x="20.31" y="39.27" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-358" visibility="hidden" gadfly:scale="10.0">550</text>
    <text x="20.31" y="35.69" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-359" visibility="hidden" gadfly:scale="10.0">600</text>
    <text x="20.31" y="32.1" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-360" visibility="hidden" gadfly:scale="10.0">650</text>
    <text x="20.31" y="28.51" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-361" visibility="hidden" gadfly:scale="10.0">700</text>
    <text x="20.31" y="24.93" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-362" visibility="hidden" gadfly:scale="10.0">750</text>
    <text x="20.31" y="21.34" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-363" visibility="hidden" gadfly:scale="10.0">800</text>
    <text x="20.31" y="17.76" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-364" visibility="hidden" gadfly:scale="10.0">850</text>
    <text x="20.31" y="14.17" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-365" visibility="hidden" gadfly:scale="10.0">900</text>
    <text x="20.31" y="10.59" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-366" visibility="hidden" gadfly:scale="10.0">950</text>
    <text x="20.31" y="7" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-367" visibility="hidden" gadfly:scale="10.0">1000</text>
    <text x="20.31" y="3.41" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-368" visibility="hidden" gadfly:scale="10.0">1050</text>
    <text x="20.31" y="-0.17" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-369" visibility="hidden" gadfly:scale="10.0">1100</text>
    <text x="20.31" y="-3.76" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-370" visibility="hidden" gadfly:scale="10.0">1150</text>
    <text x="20.31" y="-7.34" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-371" visibility="hidden" gadfly:scale="10.0">1200</text>
    <text x="20.31" y="-10.93" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-372" visibility="hidden" gadfly:scale="10.0">1250</text>
    <text x="20.31" y="-14.51" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-373" visibility="hidden" gadfly:scale="10.0">1300</text>
    <text x="20.31" y="-18.1" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-374" visibility="hidden" gadfly:scale="10.0">1350</text>
    <text x="20.31" y="-21.69" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-375" visibility="hidden" gadfly:scale="10.0">1400</text>
    <text x="20.31" y="-25.27" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-376" visibility="hidden" gadfly:scale="10.0">1450</text>
    <text x="20.31" y="-28.86" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-377" visibility="hidden" gadfly:scale="10.0">1500</text>
    <text x="20.31" y="-32.44" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-378" visibility="hidden" gadfly:scale="10.0">1550</text>
    <text x="20.31" y="-36.03" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-379" visibility="hidden" gadfly:scale="10.0">1600</text>
    <text x="20.31" y="-39.61" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-380" visibility="hidden" gadfly:scale="10.0">1650</text>
    <text x="20.31" y="-43.2" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-381" visibility="hidden" gadfly:scale="10.0">1700</text>
    <text x="20.31" y="-46.79" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-382" visibility="hidden" gadfly:scale="10.0">1750</text>
    <text x="20.31" y="-50.37" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-383" visibility="hidden" gadfly:scale="10.0">1800</text>
    <text x="20.31" y="-53.96" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-384" visibility="hidden" gadfly:scale="10.0">1850</text>
    <text x="20.31" y="-57.54" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-385" visibility="hidden" gadfly:scale="10.0">1900</text>
    <text x="20.31" y="-61.13" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-386" visibility="hidden" gadfly:scale="10.0">1950</text>
    <text x="20.31" y="-64.72" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-387" visibility="hidden" gadfly:scale="10.0">2000</text>
    <text x="20.31" y="150.43" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-388" visibility="hidden" gadfly:scale="0.5">-1000</text>
    <text x="20.31" y="78.72" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-389" visibility="hidden" gadfly:scale="0.5">0</text>
    <text x="20.31" y="7" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-390" visibility="hidden" gadfly:scale="0.5">1000</text>
    <text x="20.31" y="-64.72" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-391" visibility="hidden" gadfly:scale="0.5">2000</text>
    <text x="20.31" y="150.43" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-392" visibility="hidden" gadfly:scale="5.0">-1000</text>
    <text x="20.31" y="143.26" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-393" visibility="hidden" gadfly:scale="5.0">-900</text>
    <text x="20.31" y="136.09" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-394" visibility="hidden" gadfly:scale="5.0">-800</text>
    <text x="20.31" y="128.92" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-395" visibility="hidden" gadfly:scale="5.0">-700</text>
    <text x="20.31" y="121.74" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-396" visibility="hidden" gadfly:scale="5.0">-600</text>
    <text x="20.31" y="114.57" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-397" visibility="hidden" gadfly:scale="5.0">-500</text>
    <text x="20.31" y="107.4" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-398" visibility="hidden" gadfly:scale="5.0">-400</text>
    <text x="20.31" y="100.23" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-399" visibility="hidden" gadfly:scale="5.0">-300</text>
    <text x="20.31" y="93.06" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-400" visibility="hidden" gadfly:scale="5.0">-200</text>
    <text x="20.31" y="85.89" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-401" visibility="hidden" gadfly:scale="5.0">-100</text>
    <text x="20.31" y="78.72" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-402" visibility="hidden" gadfly:scale="5.0">0</text>
    <text x="20.31" y="71.54" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-403" visibility="hidden" gadfly:scale="5.0">100</text>
    <text x="20.31" y="64.37" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-404" visibility="hidden" gadfly:scale="5.0">200</text>
    <text x="20.31" y="57.2" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-405" visibility="hidden" gadfly:scale="5.0">300</text>
    <text x="20.31" y="50.03" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-406" visibility="hidden" gadfly:scale="5.0">400</text>
    <text x="20.31" y="42.86" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-407" visibility="hidden" gadfly:scale="5.0">500</text>
    <text x="20.31" y="35.69" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-408" visibility="hidden" gadfly:scale="5.0">600</text>
    <text x="20.31" y="28.51" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-409" visibility="hidden" gadfly:scale="5.0">700</text>
    <text x="20.31" y="21.34" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-410" visibility="hidden" gadfly:scale="5.0">800</text>
    <text x="20.31" y="14.17" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-411" visibility="hidden" gadfly:scale="5.0">900</text>
    <text x="20.31" y="7" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-412" visibility="hidden" gadfly:scale="5.0">1000</text>
    <text x="20.31" y="-0.17" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-413" visibility="hidden" gadfly:scale="5.0">1100</text>
    <text x="20.31" y="-7.34" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-414" visibility="hidden" gadfly:scale="5.0">1200</text>
    <text x="20.31" y="-14.51" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-415" visibility="hidden" gadfly:scale="5.0">1300</text>
    <text x="20.31" y="-21.69" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-416" visibility="hidden" gadfly:scale="5.0">1400</text>
    <text x="20.31" y="-28.86" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-417" visibility="hidden" gadfly:scale="5.0">1500</text>
    <text x="20.31" y="-36.03" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-418" visibility="hidden" gadfly:scale="5.0">1600</text>
    <text x="20.31" y="-43.2" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-419" visibility="hidden" gadfly:scale="5.0">1700</text>
    <text x="20.31" y="-50.37" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-420" visibility="hidden" gadfly:scale="5.0">1800</text>
    <text x="20.31" y="-57.54" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-421" visibility="hidden" gadfly:scale="5.0">1900</text>
    <text x="20.31" y="-64.72" text-anchor="end" dy="0.35em" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-422" visibility="hidden" gadfly:scale="5.0">2000</text>
  </g>
  <g font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" fill="#564A55" stroke="#000000" stroke-opacity="0.000" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-423">
    <text x="8.81" y="40.86" text-anchor="middle" dy="0.35em" transform="rotate(-90, 8.81, 42.86)" id="fig-f3ea444778014fbf80c9634001e4f6c2-element-424">Survived</text>
  </g>
</g>
<defs>
<clipPath id="fig-f3ea444778014fbf80c9634001e4f6c2-element-14">
  <path d="M21.31,5 L 122.11 5 122.11 80.72 21.31 80.72" />
</clipPath
></defs>
<script> <![CDATA[
(function(N){var k=/[\.\/]/,L=/\s*,\s*/,C=function(a,d){return a-d},a,v,y={n:{}},M=function(){for(var a=0,d=this.length;a<d;a++)if("undefined"!=typeof this[a])return this[a]},A=function(){for(var a=this.length;--a;)if("undefined"!=typeof this[a])return this[a]},w=function(k,d){k=String(k);var f=v,n=Array.prototype.slice.call(arguments,2),u=w.listeners(k),p=0,b,q=[],e={},l=[],r=a;l.firstDefined=M;l.lastDefined=A;a=k;for(var s=v=0,x=u.length;s<x;s++)"zIndex"in u[s]&&(q.push(u[s].zIndex),0>u[s].zIndex&&
(e[u[s].zIndex]=u[s]));for(q.sort(C);0>q[p];)if(b=e[q[p++] ],l.push(b.apply(d,n)),v)return v=f,l;for(s=0;s<x;s++)if(b=u[s],"zIndex"in b)if(b.zIndex==q[p]){l.push(b.apply(d,n));if(v)break;do if(p++,(b=e[q[p] ])&&l.push(b.apply(d,n)),v)break;while(b)}else e[b.zIndex]=b;else if(l.push(b.apply(d,n)),v)break;v=f;a=r;return l};w._events=y;w.listeners=function(a){a=a.split(k);var d=y,f,n,u,p,b,q,e,l=[d],r=[];u=0;for(p=a.length;u<p;u++){e=[];b=0;for(q=l.length;b<q;b++)for(d=l[b].n,f=[d[a[u] ],d["*"] ],n=2;n--;)if(d=
f[n])e.push(d),r=r.concat(d.f||[]);l=e}return r};w.on=function(a,d){a=String(a);if("function"!=typeof d)return function(){};for(var f=a.split(L),n=0,u=f.length;n<u;n++)(function(a){a=a.split(k);for(var b=y,f,e=0,l=a.length;e<l;e++)b=b.n,b=b.hasOwnProperty(a[e])&&b[a[e] ]||(b[a[e] ]={n:{}});b.f=b.f||[];e=0;for(l=b.f.length;e<l;e++)if(b.f[e]==d){f=!0;break}!f&&b.f.push(d)})(f[n]);return function(a){+a==+a&&(d.zIndex=+a)}};w.f=function(a){var d=[].slice.call(arguments,1);return function(){w.apply(null,
[a,null].concat(d).concat([].slice.call(arguments,0)))}};w.stop=function(){v=1};w.nt=function(k){return k?(new RegExp("(?:\\.|\\/|^)"+k+"(?:\\.|\\/|$)")).test(a):a};w.nts=function(){return a.split(k)};w.off=w.unbind=function(a,d){if(a){var f=a.split(L);if(1<f.length)for(var n=0,u=f.length;n<u;n++)w.off(f[n],d);else{for(var f=a.split(k),p,b,q,e,l=[y],n=0,u=f.length;n<u;n++)for(e=0;e<l.length;e+=q.length-2){q=[e,1];p=l[e].n;if("*"!=f[n])p[f[n] ]&&q.push(p[f[n] ]);else for(b in p)p.hasOwnProperty(b)&&
q.push(p[b]);l.splice.apply(l,q)}n=0;for(u=l.length;n<u;n++)for(p=l[n];p.n;){if(d){if(p.f){e=0;for(f=p.f.length;e<f;e++)if(p.f[e]==d){p.f.splice(e,1);break}!p.f.length&&delete p.f}for(b in p.n)if(p.n.hasOwnProperty(b)&&p.n[b].f){q=p.n[b].f;e=0;for(f=q.length;e<f;e++)if(q[e]==d){q.splice(e,1);break}!q.length&&delete p.n[b].f}}else for(b in delete p.f,p.n)p.n.hasOwnProperty(b)&&p.n[b].f&&delete p.n[b].f;p=p.n}}}else w._events=y={n:{}}};w.once=function(a,d){var f=function(){w.unbind(a,f);return d.apply(this,
arguments)};return w.on(a,f)};w.version="0.4.2";w.toString=function(){return"You are running Eve 0.4.2"};"undefined"!=typeof module&&module.exports?module.exports=w:"function"===typeof define&&define.amd?define("eve",[],function(){return w}):N.eve=w})(this);
(function(N,k){"function"===typeof define&&define.amd?define("Snap.svg",["eve"],function(L){return k(N,L)}):k(N,N.eve)})(this,function(N,k){var L=function(a){var k={},y=N.requestAnimationFrame||N.webkitRequestAnimationFrame||N.mozRequestAnimationFrame||N.oRequestAnimationFrame||N.msRequestAnimationFrame||function(a){setTimeout(a,16)},M=Array.isArray||function(a){return a instanceof Array||"[object Array]"==Object.prototype.toString.call(a)},A=0,w="M"+(+new Date).toString(36),z=function(a){if(null==
a)return this.s;var b=this.s-a;this.b+=this.dur*b;this.B+=this.dur*b;this.s=a},d=function(a){if(null==a)return this.spd;this.spd=a},f=function(a){if(null==a)return this.dur;this.s=this.s*a/this.dur;this.dur=a},n=function(){delete k[this.id];this.update();a("mina.stop."+this.id,this)},u=function(){this.pdif||(delete k[this.id],this.update(),this.pdif=this.get()-this.b)},p=function(){this.pdif&&(this.b=this.get()-this.pdif,delete this.pdif,k[this.id]=this)},b=function(){var a;if(M(this.start)){a=[];
for(var b=0,e=this.start.length;b<e;b++)a[b]=+this.start[b]+(this.end[b]-this.start[b])*this.easing(this.s)}else a=+this.start+(this.end-this.start)*this.easing(this.s);this.set(a)},q=function(){var l=0,b;for(b in k)if(k.hasOwnProperty(b)){var e=k[b],f=e.get();l++;e.s=(f-e.b)/(e.dur/e.spd);1<=e.s&&(delete k[b],e.s=1,l--,function(b){setTimeout(function(){a("mina.finish."+b.id,b)})}(e));e.update()}l&&y(q)},e=function(a,r,s,x,G,h,J){a={id:w+(A++).toString(36),start:a,end:r,b:s,s:0,dur:x-s,spd:1,get:G,
set:h,easing:J||e.linear,status:z,speed:d,duration:f,stop:n,pause:u,resume:p,update:b};k[a.id]=a;r=0;for(var K in k)if(k.hasOwnProperty(K)&&(r++,2==r))break;1==r&&y(q);return a};e.time=Date.now||function(){return+new Date};e.getById=function(a){return k[a]||null};e.linear=function(a){return a};e.easeout=function(a){return Math.pow(a,1.7)};e.easein=function(a){return Math.pow(a,0.48)};e.easeinout=function(a){if(1==a)return 1;if(0==a)return 0;var b=0.48-a/1.04,e=Math.sqrt(0.1734+b*b);a=e-b;a=Math.pow(Math.abs(a),
1/3)*(0>a?-1:1);b=-e-b;b=Math.pow(Math.abs(b),1/3)*(0>b?-1:1);a=a+b+0.5;return 3*(1-a)*a*a+a*a*a};e.backin=function(a){return 1==a?1:a*a*(2.70158*a-1.70158)};e.backout=function(a){if(0==a)return 0;a-=1;return a*a*(2.70158*a+1.70158)+1};e.elastic=function(a){return a==!!a?a:Math.pow(2,-10*a)*Math.sin(2*(a-0.075)*Math.PI/0.3)+1};e.bounce=function(a){a<1/2.75?a*=7.5625*a:a<2/2.75?(a-=1.5/2.75,a=7.5625*a*a+0.75):a<2.5/2.75?(a-=2.25/2.75,a=7.5625*a*a+0.9375):(a-=2.625/2.75,a=7.5625*a*a+0.984375);return a};
return N.mina=e}("undefined"==typeof k?function(){}:k),C=function(){function a(c,t){if(c){if(c.tagName)return x(c);if(y(c,"array")&&a.set)return a.set.apply(a,c);if(c instanceof e)return c;if(null==t)return c=G.doc.querySelector(c),x(c)}return new s(null==c?"100%":c,null==t?"100%":t)}function v(c,a){if(a){"#text"==c&&(c=G.doc.createTextNode(a.text||""));"string"==typeof c&&(c=v(c));if("string"==typeof a)return"xlink:"==a.substring(0,6)?c.getAttributeNS(m,a.substring(6)):"xml:"==a.substring(0,4)?c.getAttributeNS(la,
a.substring(4)):c.getAttribute(a);for(var da in a)if(a[h](da)){var b=J(a[da]);b?"xlink:"==da.substring(0,6)?c.setAttributeNS(m,da.substring(6),b):"xml:"==da.substring(0,4)?c.setAttributeNS(la,da.substring(4),b):c.setAttribute(da,b):c.removeAttribute(da)}}else c=G.doc.createElementNS(la,c);return c}function y(c,a){a=J.prototype.toLowerCase.call(a);return"finite"==a?isFinite(c):"array"==a&&(c instanceof Array||Array.isArray&&Array.isArray(c))?!0:"null"==a&&null===c||a==typeof c&&null!==c||"object"==
a&&c===Object(c)||$.call(c).slice(8,-1).toLowerCase()==a}function M(c){if("function"==typeof c||Object(c)!==c)return c;var a=new c.constructor,b;for(b in c)c[h](b)&&(a[b]=M(c[b]));return a}function A(c,a,b){function m(){var e=Array.prototype.slice.call(arguments,0),f=e.join("\u2400"),d=m.cache=m.cache||{},l=m.count=m.count||[];if(d[h](f)){a:for(var e=l,l=f,B=0,H=e.length;B<H;B++)if(e[B]===l){e.push(e.splice(B,1)[0]);break a}return b?b(d[f]):d[f]}1E3<=l.length&&delete d[l.shift()];l.push(f);d[f]=c.apply(a,
e);return b?b(d[f]):d[f]}return m}function w(c,a,b,m,e,f){return null==e?(c-=b,a-=m,c||a?(180*I.atan2(-a,-c)/C+540)%360:0):w(c,a,e,f)-w(b,m,e,f)}function z(c){return c%360*C/180}function d(c){var a=[];c=c.replace(/(?:^|\s)(\w+)\(([^)]+)\)/g,function(c,b,m){m=m.split(/\s*,\s*|\s+/);"rotate"==b&&1==m.length&&m.push(0,0);"scale"==b&&(2<m.length?m=m.slice(0,2):2==m.length&&m.push(0,0),1==m.length&&m.push(m[0],0,0));"skewX"==b?a.push(["m",1,0,I.tan(z(m[0])),1,0,0]):"skewY"==b?a.push(["m",1,I.tan(z(m[0])),
0,1,0,0]):a.push([b.charAt(0)].concat(m));return c});return a}function f(c,t){var b=O(c),m=new a.Matrix;if(b)for(var e=0,f=b.length;e<f;e++){var h=b[e],d=h.length,B=J(h[0]).toLowerCase(),H=h[0]!=B,l=H?m.invert():0,E;"t"==B&&2==d?m.translate(h[1],0):"t"==B&&3==d?H?(d=l.x(0,0),B=l.y(0,0),H=l.x(h[1],h[2]),l=l.y(h[1],h[2]),m.translate(H-d,l-B)):m.translate(h[1],h[2]):"r"==B?2==d?(E=E||t,m.rotate(h[1],E.x+E.width/2,E.y+E.height/2)):4==d&&(H?(H=l.x(h[2],h[3]),l=l.y(h[2],h[3]),m.rotate(h[1],H,l)):m.rotate(h[1],
h[2],h[3])):"s"==B?2==d||3==d?(E=E||t,m.scale(h[1],h[d-1],E.x+E.width/2,E.y+E.height/2)):4==d?H?(H=l.x(h[2],h[3]),l=l.y(h[2],h[3]),m.scale(h[1],h[1],H,l)):m.scale(h[1],h[1],h[2],h[3]):5==d&&(H?(H=l.x(h[3],h[4]),l=l.y(h[3],h[4]),m.scale(h[1],h[2],H,l)):m.scale(h[1],h[2],h[3],h[4])):"m"==B&&7==d&&m.add(h[1],h[2],h[3],h[4],h[5],h[6])}return m}function n(c,t){if(null==t){var m=!0;t="linearGradient"==c.type||"radialGradient"==c.type?c.node.getAttribute("gradientTransform"):"pattern"==c.type?c.node.getAttribute("patternTransform"):
c.node.getAttribute("transform");if(!t)return new a.Matrix;t=d(t)}else t=a._.rgTransform.test(t)?J(t).replace(/\.{3}|\u2026/g,c._.transform||aa):d(t),y(t,"array")&&(t=a.path?a.path.toString.call(t):J(t)),c._.transform=t;var b=f(t,c.getBBox(1));if(m)return b;c.matrix=b}function u(c){c=c.node.ownerSVGElement&&x(c.node.ownerSVGElement)||c.node.parentNode&&x(c.node.parentNode)||a.select("svg")||a(0,0);var t=c.select("defs"),t=null==t?!1:t.node;t||(t=r("defs",c.node).node);return t}function p(c){return c.node.ownerSVGElement&&
x(c.node.ownerSVGElement)||a.select("svg")}function b(c,a,m){function b(c){if(null==c)return aa;if(c==+c)return c;v(B,{width:c});try{return B.getBBox().width}catch(a){return 0}}function h(c){if(null==c)return aa;if(c==+c)return c;v(B,{height:c});try{return B.getBBox().height}catch(a){return 0}}function e(b,B){null==a?d[b]=B(c.attr(b)||0):b==a&&(d=B(null==m?c.attr(b)||0:m))}var f=p(c).node,d={},B=f.querySelector(".svg---mgr");B||(B=v("rect"),v(B,{x:-9E9,y:-9E9,width:10,height:10,"class":"svg---mgr",
fill:"none"}),f.appendChild(B));switch(c.type){case "rect":e("rx",b),e("ry",h);case "image":e("width",b),e("height",h);case "text":e("x",b);e("y",h);break;case "circle":e("cx",b);e("cy",h);e("r",b);break;case "ellipse":e("cx",b);e("cy",h);e("rx",b);e("ry",h);break;case "line":e("x1",b);e("x2",b);e("y1",h);e("y2",h);break;case "marker":e("refX",b);e("markerWidth",b);e("refY",h);e("markerHeight",h);break;case "radialGradient":e("fx",b);e("fy",h);break;case "tspan":e("dx",b);e("dy",h);break;default:e(a,
b)}f.removeChild(B);return d}function q(c){y(c,"array")||(c=Array.prototype.slice.call(arguments,0));for(var a=0,b=0,m=this.node;this[a];)delete this[a++];for(a=0;a<c.length;a++)"set"==c[a].type?c[a].forEach(function(c){m.appendChild(c.node)}):m.appendChild(c[a].node);for(var h=m.childNodes,a=0;a<h.length;a++)this[b++]=x(h[a]);return this}function e(c){if(c.snap in E)return E[c.snap];var a=this.id=V(),b;try{b=c.ownerSVGElement}catch(m){}this.node=c;b&&(this.paper=new s(b));this.type=c.tagName;this.anims=
{};this._={transform:[]};c.snap=a;E[a]=this;"g"==this.type&&(this.add=q);if(this.type in{g:1,mask:1,pattern:1})for(var e in s.prototype)s.prototype[h](e)&&(this[e]=s.prototype[e])}function l(c){this.node=c}function r(c,a){var b=v(c);a.appendChild(b);return x(b)}function s(c,a){var b,m,f,d=s.prototype;if(c&&"svg"==c.tagName){if(c.snap in E)return E[c.snap];var l=c.ownerDocument;b=new e(c);m=c.getElementsByTagName("desc")[0];f=c.getElementsByTagName("defs")[0];m||(m=v("desc"),m.appendChild(l.createTextNode("Created with Snap")),
b.node.appendChild(m));f||(f=v("defs"),b.node.appendChild(f));b.defs=f;for(var ca in d)d[h](ca)&&(b[ca]=d[ca]);b.paper=b.root=b}else b=r("svg",G.doc.body),v(b.node,{height:a,version:1.1,width:c,xmlns:la});return b}function x(c){return!c||c instanceof e||c instanceof l?c:c.tagName&&"svg"==c.tagName.toLowerCase()?new s(c):c.tagName&&"object"==c.tagName.toLowerCase()&&"image/svg+xml"==c.type?new s(c.contentDocument.getElementsByTagName("svg")[0]):new e(c)}a.version="0.3.0";a.toString=function(){return"Snap v"+
this.version};a._={};var G={win:N,doc:N.document};a._.glob=G;var h="hasOwnProperty",J=String,K=parseFloat,U=parseInt,I=Math,P=I.max,Q=I.min,Y=I.abs,C=I.PI,aa="",$=Object.prototype.toString,F=/^\s*((#[a-f\d]{6})|(#[a-f\d]{3})|rgba?\(\s*([\d\.]+%?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+%?(?:\s*,\s*[\d\.]+%?)?)\s*\)|hsba?\(\s*([\d\.]+(?:deg|\xb0|%)?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+(?:%?\s*,\s*[\d\.]+)?%?)\s*\)|hsla?\(\s*([\d\.]+(?:deg|\xb0|%)?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+(?:%?\s*,\s*[\d\.]+)?%?)\s*\))\s*$/i;a._.separator=
RegExp("[,\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]+");var S=RegExp("[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*"),X={hs:1,rg:1},W=RegExp("([a-z])[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029,]*((-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*)+)",
"ig"),ma=RegExp("([rstm])[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029,]*((-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*)+)","ig"),Z=RegExp("(-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?)[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*",
"ig"),na=0,ba="S"+(+new Date).toString(36),V=function(){return ba+(na++).toString(36)},m="http://www.w3.org/1999/xlink",la="http://www.w3.org/2000/svg",E={},ca=a.url=function(c){return"url('#"+c+"')"};a._.$=v;a._.id=V;a.format=function(){var c=/\{([^\}]+)\}/g,a=/(?:(?:^|\.)(.+?)(?=\[|\.|$|\()|\[('|")(.+?)\2\])(\(\))?/g,b=function(c,b,m){var h=m;b.replace(a,function(c,a,b,m,t){a=a||m;h&&(a in h&&(h=h[a]),"function"==typeof h&&t&&(h=h()))});return h=(null==h||h==m?c:h)+""};return function(a,m){return J(a).replace(c,
function(c,a){return b(c,a,m)})}}();a._.clone=M;a._.cacher=A;a.rad=z;a.deg=function(c){return 180*c/C%360};a.angle=w;a.is=y;a.snapTo=function(c,a,b){b=y(b,"finite")?b:10;if(y(c,"array"))for(var m=c.length;m--;){if(Y(c[m]-a)<=b)return c[m]}else{c=+c;m=a%c;if(m<b)return a-m;if(m>c-b)return a-m+c}return a};a.getRGB=A(function(c){if(!c||(c=J(c)).indexOf("-")+1)return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka};if("none"==c)return{r:-1,g:-1,b:-1,hex:"none",toString:ka};!X[h](c.toLowerCase().substring(0,
2))&&"#"!=c.charAt()&&(c=T(c));if(!c)return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka};var b,m,e,f,d;if(c=c.match(F)){c[2]&&(e=U(c[2].substring(5),16),m=U(c[2].substring(3,5),16),b=U(c[2].substring(1,3),16));c[3]&&(e=U((d=c[3].charAt(3))+d,16),m=U((d=c[3].charAt(2))+d,16),b=U((d=c[3].charAt(1))+d,16));c[4]&&(d=c[4].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b*=2.55),m=K(d[1]),"%"==d[1].slice(-1)&&(m*=2.55),e=K(d[2]),"%"==d[2].slice(-1)&&(e*=2.55),"rgba"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),
d[3]&&"%"==d[3].slice(-1)&&(f/=100));if(c[5])return d=c[5].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b/=100),m=K(d[1]),"%"==d[1].slice(-1)&&(m/=100),e=K(d[2]),"%"==d[2].slice(-1)&&(e/=100),"deg"!=d[0].slice(-3)&&"\u00b0"!=d[0].slice(-1)||(b/=360),"hsba"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),d[3]&&"%"==d[3].slice(-1)&&(f/=100),a.hsb2rgb(b,m,e,f);if(c[6])return d=c[6].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b/=100),m=K(d[1]),"%"==d[1].slice(-1)&&(m/=100),e=K(d[2]),"%"==d[2].slice(-1)&&(e/=100),
"deg"!=d[0].slice(-3)&&"\u00b0"!=d[0].slice(-1)||(b/=360),"hsla"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),d[3]&&"%"==d[3].slice(-1)&&(f/=100),a.hsl2rgb(b,m,e,f);b=Q(I.round(b),255);m=Q(I.round(m),255);e=Q(I.round(e),255);f=Q(P(f,0),1);c={r:b,g:m,b:e,toString:ka};c.hex="#"+(16777216|e|m<<8|b<<16).toString(16).slice(1);c.opacity=y(f,"finite")?f:1;return c}return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka}},a);a.hsb=A(function(c,b,m){return a.hsb2rgb(c,b,m).hex});a.hsl=A(function(c,b,m){return a.hsl2rgb(c,
b,m).hex});a.rgb=A(function(c,a,b,m){if(y(m,"finite")){var e=I.round;return"rgba("+[e(c),e(a),e(b),+m.toFixed(2)]+")"}return"#"+(16777216|b|a<<8|c<<16).toString(16).slice(1)});var T=function(c){var a=G.doc.getElementsByTagName("head")[0]||G.doc.getElementsByTagName("svg")[0];T=A(function(c){if("red"==c.toLowerCase())return"rgb(255, 0, 0)";a.style.color="rgb(255, 0, 0)";a.style.color=c;c=G.doc.defaultView.getComputedStyle(a,aa).getPropertyValue("color");return"rgb(255, 0, 0)"==c?null:c});return T(c)},
qa=function(){return"hsb("+[this.h,this.s,this.b]+")"},ra=function(){return"hsl("+[this.h,this.s,this.l]+")"},ka=function(){return 1==this.opacity||null==this.opacity?this.hex:"rgba("+[this.r,this.g,this.b,this.opacity]+")"},D=function(c,b,m){null==b&&y(c,"object")&&"r"in c&&"g"in c&&"b"in c&&(m=c.b,b=c.g,c=c.r);null==b&&y(c,string)&&(m=a.getRGB(c),c=m.r,b=m.g,m=m.b);if(1<c||1<b||1<m)c/=255,b/=255,m/=255;return[c,b,m]},oa=function(c,b,m,e){c=I.round(255*c);b=I.round(255*b);m=I.round(255*m);c={r:c,
g:b,b:m,opacity:y(e,"finite")?e:1,hex:a.rgb(c,b,m),toString:ka};y(e,"finite")&&(c.opacity=e);return c};a.color=function(c){var b;y(c,"object")&&"h"in c&&"s"in c&&"b"in c?(b=a.hsb2rgb(c),c.r=b.r,c.g=b.g,c.b=b.b,c.opacity=1,c.hex=b.hex):y(c,"object")&&"h"in c&&"s"in c&&"l"in c?(b=a.hsl2rgb(c),c.r=b.r,c.g=b.g,c.b=b.b,c.opacity=1,c.hex=b.hex):(y(c,"string")&&(c=a.getRGB(c)),y(c,"object")&&"r"in c&&"g"in c&&"b"in c&&!("error"in c)?(b=a.rgb2hsl(c),c.h=b.h,c.s=b.s,c.l=b.l,b=a.rgb2hsb(c),c.v=b.b):(c={hex:"none"},
c.r=c.g=c.b=c.h=c.s=c.v=c.l=-1,c.error=1));c.toString=ka;return c};a.hsb2rgb=function(c,a,b,m){y(c,"object")&&"h"in c&&"s"in c&&"b"in c&&(b=c.b,a=c.s,c=c.h,m=c.o);var e,h,d;c=360*c%360/60;d=b*a;a=d*(1-Y(c%2-1));b=e=h=b-d;c=~~c;b+=[d,a,0,0,a,d][c];e+=[a,d,d,a,0,0][c];h+=[0,0,a,d,d,a][c];return oa(b,e,h,m)};a.hsl2rgb=function(c,a,b,m){y(c,"object")&&"h"in c&&"s"in c&&"l"in c&&(b=c.l,a=c.s,c=c.h);if(1<c||1<a||1<b)c/=360,a/=100,b/=100;var e,h,d;c=360*c%360/60;d=2*a*(0.5>b?b:1-b);a=d*(1-Y(c%2-1));b=e=
h=b-d/2;c=~~c;b+=[d,a,0,0,a,d][c];e+=[a,d,d,a,0,0][c];h+=[0,0,a,d,d,a][c];return oa(b,e,h,m)};a.rgb2hsb=function(c,a,b){b=D(c,a,b);c=b[0];a=b[1];b=b[2];var m,e;m=P(c,a,b);e=m-Q(c,a,b);c=((0==e?0:m==c?(a-b)/e:m==a?(b-c)/e+2:(c-a)/e+4)+360)%6*60/360;return{h:c,s:0==e?0:e/m,b:m,toString:qa}};a.rgb2hsl=function(c,a,b){b=D(c,a,b);c=b[0];a=b[1];b=b[2];var m,e,h;m=P(c,a,b);e=Q(c,a,b);h=m-e;c=((0==h?0:m==c?(a-b)/h:m==a?(b-c)/h+2:(c-a)/h+4)+360)%6*60/360;m=(m+e)/2;return{h:c,s:0==h?0:0.5>m?h/(2*m):h/(2-2*
m),l:m,toString:ra}};a.parsePathString=function(c){if(!c)return null;var b=a.path(c);if(b.arr)return a.path.clone(b.arr);var m={a:7,c:6,o:2,h:1,l:2,m:2,r:4,q:4,s:4,t:2,v:1,u:3,z:0},e=[];y(c,"array")&&y(c[0],"array")&&(e=a.path.clone(c));e.length||J(c).replace(W,function(c,a,b){var h=[];c=a.toLowerCase();b.replace(Z,function(c,a){a&&h.push(+a)});"m"==c&&2<h.length&&(e.push([a].concat(h.splice(0,2))),c="l",a="m"==a?"l":"L");"o"==c&&1==h.length&&e.push([a,h[0] ]);if("r"==c)e.push([a].concat(h));else for(;h.length>=
m[c]&&(e.push([a].concat(h.splice(0,m[c]))),m[c]););});e.toString=a.path.toString;b.arr=a.path.clone(e);return e};var O=a.parseTransformString=function(c){if(!c)return null;var b=[];y(c,"array")&&y(c[0],"array")&&(b=a.path.clone(c));b.length||J(c).replace(ma,function(c,a,m){var e=[];a.toLowerCase();m.replace(Z,function(c,a){a&&e.push(+a)});b.push([a].concat(e))});b.toString=a.path.toString;return b};a._.svgTransform2string=d;a._.rgTransform=RegExp("^[a-z][\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*-?\\.?\\d",
"i");a._.transform2matrix=f;a._unit2px=b;a._.getSomeDefs=u;a._.getSomeSVG=p;a.select=function(c){return x(G.doc.querySelector(c))};a.selectAll=function(c){c=G.doc.querySelectorAll(c);for(var b=(a.set||Array)(),m=0;m<c.length;m++)b.push(x(c[m]));return b};setInterval(function(){for(var c in E)if(E[h](c)){var a=E[c],b=a.node;("svg"!=a.type&&!b.ownerSVGElement||"svg"==a.type&&(!b.parentNode||"ownerSVGElement"in b.parentNode&&!b.ownerSVGElement))&&delete E[c]}},1E4);(function(c){function m(c){function a(c,
b){var m=v(c.node,b);(m=(m=m&&m.match(d))&&m[2])&&"#"==m.charAt()&&(m=m.substring(1))&&(f[m]=(f[m]||[]).concat(function(a){var m={};m[b]=ca(a);v(c.node,m)}))}function b(c){var a=v(c.node,"xlink:href");a&&"#"==a.charAt()&&(a=a.substring(1))&&(f[a]=(f[a]||[]).concat(function(a){c.attr("xlink:href","#"+a)}))}var e=c.selectAll("*"),h,d=/^\s*url\(("|'|)(.*)\1\)\s*$/;c=[];for(var f={},l=0,E=e.length;l<E;l++){h=e[l];a(h,"fill");a(h,"stroke");a(h,"filter");a(h,"mask");a(h,"clip-path");b(h);var t=v(h.node,
"id");t&&(v(h.node,{id:h.id}),c.push({old:t,id:h.id}))}l=0;for(E=c.length;l<E;l++)if(e=f[c[l].old])for(h=0,t=e.length;h<t;h++)e[h](c[l].id)}function e(c,a,b){return function(m){m=m.slice(c,a);1==m.length&&(m=m[0]);return b?b(m):m}}function d(c){return function(){var a=c?"<"+this.type:"",b=this.node.attributes,m=this.node.childNodes;if(c)for(var e=0,h=b.length;e<h;e++)a+=" "+b[e].name+'="'+b[e].value.replace(/"/g,'\\"')+'"';if(m.length){c&&(a+=">");e=0;for(h=m.length;e<h;e++)3==m[e].nodeType?a+=m[e].nodeValue:
1==m[e].nodeType&&(a+=x(m[e]).toString());c&&(a+="</"+this.type+">")}else c&&(a+="/>");return a}}c.attr=function(c,a){if(!c)return this;if(y(c,"string"))if(1<arguments.length){var b={};b[c]=a;c=b}else return k("snap.util.getattr."+c,this).firstDefined();for(var m in c)c[h](m)&&k("snap.util.attr."+m,this,c[m]);return this};c.getBBox=function(c){if(!a.Matrix||!a.path)return this.node.getBBox();var b=this,m=new a.Matrix;if(b.removed)return a._.box();for(;"use"==b.type;)if(c||(m=m.add(b.transform().localMatrix.translate(b.attr("x")||
0,b.attr("y")||0))),b.original)b=b.original;else var e=b.attr("xlink:href"),b=b.original=b.node.ownerDocument.getElementById(e.substring(e.indexOf("#")+1));var e=b._,h=a.path.get[b.type]||a.path.get.deflt;try{if(c)return e.bboxwt=h?a.path.getBBox(b.realPath=h(b)):a._.box(b.node.getBBox()),a._.box(e.bboxwt);b.realPath=h(b);b.matrix=b.transform().localMatrix;e.bbox=a.path.getBBox(a.path.map(b.realPath,m.add(b.matrix)));return a._.box(e.bbox)}catch(d){return a._.box()}};var f=function(){return this.string};
c.transform=function(c){var b=this._;if(null==c){var m=this;c=new a.Matrix(this.node.getCTM());for(var e=n(this),h=[e],d=new a.Matrix,l=e.toTransformString(),b=J(e)==J(this.matrix)?J(b.transform):l;"svg"!=m.type&&(m=m.parent());)h.push(n(m));for(m=h.length;m--;)d.add(h[m]);return{string:b,globalMatrix:c,totalMatrix:d,localMatrix:e,diffMatrix:c.clone().add(e.invert()),global:c.toTransformString(),total:d.toTransformString(),local:l,toString:f}}c instanceof a.Matrix?this.matrix=c:n(this,c);this.node&&
("linearGradient"==this.type||"radialGradient"==this.type?v(this.node,{gradientTransform:this.matrix}):"pattern"==this.type?v(this.node,{patternTransform:this.matrix}):v(this.node,{transform:this.matrix}));return this};c.parent=function(){return x(this.node.parentNode)};c.append=c.add=function(c){if(c){if("set"==c.type){var a=this;c.forEach(function(c){a.add(c)});return this}c=x(c);this.node.appendChild(c.node);c.paper=this.paper}return this};c.appendTo=function(c){c&&(c=x(c),c.append(this));return this};
c.prepend=function(c){if(c){if("set"==c.type){var a=this,b;c.forEach(function(c){b?b.after(c):a.prepend(c);b=c});return this}c=x(c);var m=c.parent();this.node.insertBefore(c.node,this.node.firstChild);this.add&&this.add();c.paper=this.paper;this.parent()&&this.parent().add();m&&m.add()}return this};c.prependTo=function(c){c=x(c);c.prepend(this);return this};c.before=function(c){if("set"==c.type){var a=this;c.forEach(function(c){var b=c.parent();a.node.parentNode.insertBefore(c.node,a.node);b&&b.add()});
this.parent().add();return this}c=x(c);var b=c.parent();this.node.parentNode.insertBefore(c.node,this.node);this.parent()&&this.parent().add();b&&b.add();c.paper=this.paper;return this};c.after=function(c){c=x(c);var a=c.parent();this.node.nextSibling?this.node.parentNode.insertBefore(c.node,this.node.nextSibling):this.node.parentNode.appendChild(c.node);this.parent()&&this.parent().add();a&&a.add();c.paper=this.paper;return this};c.insertBefore=function(c){c=x(c);var a=this.parent();c.node.parentNode.insertBefore(this.node,
c.node);this.paper=c.paper;a&&a.add();c.parent()&&c.parent().add();return this};c.insertAfter=function(c){c=x(c);var a=this.parent();c.node.parentNode.insertBefore(this.node,c.node.nextSibling);this.paper=c.paper;a&&a.add();c.parent()&&c.parent().add();return this};c.remove=function(){var c=this.parent();this.node.parentNode&&this.node.parentNode.removeChild(this.node);delete this.paper;this.removed=!0;c&&c.add();return this};c.select=function(c){return x(this.node.querySelector(c))};c.selectAll=
function(c){c=this.node.querySelectorAll(c);for(var b=(a.set||Array)(),m=0;m<c.length;m++)b.push(x(c[m]));return b};c.asPX=function(c,a){null==a&&(a=this.attr(c));return+b(this,c,a)};c.use=function(){var c,a=this.node.id;a||(a=this.id,v(this.node,{id:a}));c="linearGradient"==this.type||"radialGradient"==this.type||"pattern"==this.type?r(this.type,this.node.parentNode):r("use",this.node.parentNode);v(c.node,{"xlink:href":"#"+a});c.original=this;return c};var l=/\S+/g;c.addClass=function(c){var a=(c||
"").match(l)||[];c=this.node;var b=c.className.baseVal,m=b.match(l)||[],e,h,d;if(a.length){for(e=0;d=a[e++];)h=m.indexOf(d),~h||m.push(d);a=m.join(" ");b!=a&&(c.className.baseVal=a)}return this};c.removeClass=function(c){var a=(c||"").match(l)||[];c=this.node;var b=c.className.baseVal,m=b.match(l)||[],e,h;if(m.length){for(e=0;h=a[e++];)h=m.indexOf(h),~h&&m.splice(h,1);a=m.join(" ");b!=a&&(c.className.baseVal=a)}return this};c.hasClass=function(c){return!!~(this.node.className.baseVal.match(l)||[]).indexOf(c)};
c.toggleClass=function(c,a){if(null!=a)return a?this.addClass(c):this.removeClass(c);var b=(c||"").match(l)||[],m=this.node,e=m.className.baseVal,h=e.match(l)||[],d,f,E;for(d=0;E=b[d++];)f=h.indexOf(E),~f?h.splice(f,1):h.push(E);b=h.join(" ");e!=b&&(m.className.baseVal=b);return this};c.clone=function(){var c=x(this.node.cloneNode(!0));v(c.node,"id")&&v(c.node,{id:c.id});m(c);c.insertAfter(this);return c};c.toDefs=function(){u(this).appendChild(this.node);return this};c.pattern=c.toPattern=function(c,
a,b,m){var e=r("pattern",u(this));null==c&&(c=this.getBBox());y(c,"object")&&"x"in c&&(a=c.y,b=c.width,m=c.height,c=c.x);v(e.node,{x:c,y:a,width:b,height:m,patternUnits:"userSpaceOnUse",id:e.id,viewBox:[c,a,b,m].join(" ")});e.node.appendChild(this.node);return e};c.marker=function(c,a,b,m,e,h){var d=r("marker",u(this));null==c&&(c=this.getBBox());y(c,"object")&&"x"in c&&(a=c.y,b=c.width,m=c.height,e=c.refX||c.cx,h=c.refY||c.cy,c=c.x);v(d.node,{viewBox:[c,a,b,m].join(" "),markerWidth:b,markerHeight:m,
orient:"auto",refX:e||0,refY:h||0,id:d.id});d.node.appendChild(this.node);return d};var E=function(c,a,b,m){"function"!=typeof b||b.length||(m=b,b=L.linear);this.attr=c;this.dur=a;b&&(this.easing=b);m&&(this.callback=m)};a._.Animation=E;a.animation=function(c,a,b,m){return new E(c,a,b,m)};c.inAnim=function(){var c=[],a;for(a in this.anims)this.anims[h](a)&&function(a){c.push({anim:new E(a._attrs,a.dur,a.easing,a._callback),mina:a,curStatus:a.status(),status:function(c){return a.status(c)},stop:function(){a.stop()}})}(this.anims[a]);
return c};a.animate=function(c,a,b,m,e,h){"function"!=typeof e||e.length||(h=e,e=L.linear);var d=L.time();c=L(c,a,d,d+m,L.time,b,e);h&&k.once("mina.finish."+c.id,h);return c};c.stop=function(){for(var c=this.inAnim(),a=0,b=c.length;a<b;a++)c[a].stop();return this};c.animate=function(c,a,b,m){"function"!=typeof b||b.length||(m=b,b=L.linear);c instanceof E&&(m=c.callback,b=c.easing,a=b.dur,c=c.attr);var d=[],f=[],l={},t,ca,n,T=this,q;for(q in c)if(c[h](q)){T.equal?(n=T.equal(q,J(c[q])),t=n.from,ca=
n.to,n=n.f):(t=+T.attr(q),ca=+c[q]);var la=y(t,"array")?t.length:1;l[q]=e(d.length,d.length+la,n);d=d.concat(t);f=f.concat(ca)}t=L.time();var p=L(d,f,t,t+a,L.time,function(c){var a={},b;for(b in l)l[h](b)&&(a[b]=l[b](c));T.attr(a)},b);T.anims[p.id]=p;p._attrs=c;p._callback=m;k("snap.animcreated."+T.id,p);k.once("mina.finish."+p.id,function(){delete T.anims[p.id];m&&m.call(T)});k.once("mina.stop."+p.id,function(){delete T.anims[p.id]});return T};var T={};c.data=function(c,b){var m=T[this.id]=T[this.id]||
{};if(0==arguments.length)return k("snap.data.get."+this.id,this,m,null),m;if(1==arguments.length){if(a.is(c,"object")){for(var e in c)c[h](e)&&this.data(e,c[e]);return this}k("snap.data.get."+this.id,this,m[c],c);return m[c]}m[c]=b;k("snap.data.set."+this.id,this,b,c);return this};c.removeData=function(c){null==c?T[this.id]={}:T[this.id]&&delete T[this.id][c];return this};c.outerSVG=c.toString=d(1);c.innerSVG=d()})(e.prototype);a.parse=function(c){var a=G.doc.createDocumentFragment(),b=!0,m=G.doc.createElement("div");
c=J(c);c.match(/^\s*<\s*svg(?:\s|>)/)||(c="<svg>"+c+"</svg>",b=!1);m.innerHTML=c;if(c=m.getElementsByTagName("svg")[0])if(b)a=c;else for(;c.firstChild;)a.appendChild(c.firstChild);m.innerHTML=aa;return new l(a)};l.prototype.select=e.prototype.select;l.prototype.selectAll=e.prototype.selectAll;a.fragment=function(){for(var c=Array.prototype.slice.call(arguments,0),b=G.doc.createDocumentFragment(),m=0,e=c.length;m<e;m++){var h=c[m];h.node&&h.node.nodeType&&b.appendChild(h.node);h.nodeType&&b.appendChild(h);
"string"==typeof h&&b.appendChild(a.parse(h).node)}return new l(b)};a._.make=r;a._.wrap=x;s.prototype.el=function(c,a){var b=r(c,this.node);a&&b.attr(a);return b};k.on("snap.util.getattr",function(){var c=k.nt(),c=c.substring(c.lastIndexOf(".")+1),a=c.replace(/[A-Z]/g,function(c){return"-"+c.toLowerCase()});return pa[h](a)?this.node.ownerDocument.defaultView.getComputedStyle(this.node,null).getPropertyValue(a):v(this.node,c)});var pa={"alignment-baseline":0,"baseline-shift":0,clip:0,"clip-path":0,
"clip-rule":0,color:0,"color-interpolation":0,"color-interpolation-filters":0,"color-profile":0,"color-rendering":0,cursor:0,direction:0,display:0,"dominant-baseline":0,"enable-background":0,fill:0,"fill-opacity":0,"fill-rule":0,filter:0,"flood-color":0,"flood-opacity":0,font:0,"font-family":0,"font-size":0,"font-size-adjust":0,"font-stretch":0,"font-style":0,"font-variant":0,"font-weight":0,"glyph-orientation-horizontal":0,"glyph-orientation-vertical":0,"image-rendering":0,kerning:0,"letter-spacing":0,
"lighting-color":0,marker:0,"marker-end":0,"marker-mid":0,"marker-start":0,mask:0,opacity:0,overflow:0,"pointer-events":0,"shape-rendering":0,"stop-color":0,"stop-opacity":0,stroke:0,"stroke-dasharray":0,"stroke-dashoffset":0,"stroke-linecap":0,"stroke-linejoin":0,"stroke-miterlimit":0,"stroke-opacity":0,"stroke-width":0,"text-anchor":0,"text-decoration":0,"text-rendering":0,"unicode-bidi":0,visibility:0,"word-spacing":0,"writing-mode":0};k.on("snap.util.attr",function(c){var a=k.nt(),b={},a=a.substring(a.lastIndexOf(".")+
1);b[a]=c;var m=a.replace(/-(\w)/gi,function(c,a){return a.toUpperCase()}),a=a.replace(/[A-Z]/g,function(c){return"-"+c.toLowerCase()});pa[h](a)?this.node.style[m]=null==c?aa:c:v(this.node,b)});a.ajax=function(c,a,b,m){var e=new XMLHttpRequest,h=V();if(e){if(y(a,"function"))m=b,b=a,a=null;else if(y(a,"object")){var d=[],f;for(f in a)a.hasOwnProperty(f)&&d.push(encodeURIComponent(f)+"="+encodeURIComponent(a[f]));a=d.join("&")}e.open(a?"POST":"GET",c,!0);a&&(e.setRequestHeader("X-Requested-With","XMLHttpRequest"),
e.setRequestHeader("Content-type","application/x-www-form-urlencoded"));b&&(k.once("snap.ajax."+h+".0",b),k.once("snap.ajax."+h+".200",b),k.once("snap.ajax."+h+".304",b));e.onreadystatechange=function(){4==e.readyState&&k("snap.ajax."+h+"."+e.status,m,e)};if(4==e.readyState)return e;e.send(a);return e}};a.load=function(c,b,m){a.ajax(c,function(c){c=a.parse(c.responseText);m?b.call(m,c):b(c)})};a.getElementByPoint=function(c,a){var b,m,e=G.doc.elementFromPoint(c,a);if(G.win.opera&&"svg"==e.tagName){b=
e;m=b.getBoundingClientRect();b=b.ownerDocument;var h=b.body,d=b.documentElement;b=m.top+(g.win.pageYOffset||d.scrollTop||h.scrollTop)-(d.clientTop||h.clientTop||0);m=m.left+(g.win.pageXOffset||d.scrollLeft||h.scrollLeft)-(d.clientLeft||h.clientLeft||0);h=e.createSVGRect();h.x=c-m;h.y=a-b;h.width=h.height=1;b=e.getIntersectionList(h,null);b.length&&(e=b[b.length-1])}return e?x(e):null};a.plugin=function(c){c(a,e,s,G,l)};return G.win.Snap=a}();C.plugin(function(a,k,y,M,A){function w(a,d,f,b,q,e){null==
d&&"[object SVGMatrix]"==z.call(a)?(this.a=a.a,this.b=a.b,this.c=a.c,this.d=a.d,this.e=a.e,this.f=a.f):null!=a?(this.a=+a,this.b=+d,this.c=+f,this.d=+b,this.e=+q,this.f=+e):(this.a=1,this.c=this.b=0,this.d=1,this.f=this.e=0)}var z=Object.prototype.toString,d=String,f=Math;(function(n){function k(a){return a[0]*a[0]+a[1]*a[1]}function p(a){var d=f.sqrt(k(a));a[0]&&(a[0]/=d);a[1]&&(a[1]/=d)}n.add=function(a,d,e,f,n,p){var k=[[],[],[] ],u=[[this.a,this.c,this.e],[this.b,this.d,this.f],[0,0,1] ];d=[[a,
e,n],[d,f,p],[0,0,1] ];a&&a instanceof w&&(d=[[a.a,a.c,a.e],[a.b,a.d,a.f],[0,0,1] ]);for(a=0;3>a;a++)for(e=0;3>e;e++){for(f=n=0;3>f;f++)n+=u[a][f]*d[f][e];k[a][e]=n}this.a=k[0][0];this.b=k[1][0];this.c=k[0][1];this.d=k[1][1];this.e=k[0][2];this.f=k[1][2];return this};n.invert=function(){var a=this.a*this.d-this.b*this.c;return new w(this.d/a,-this.b/a,-this.c/a,this.a/a,(this.c*this.f-this.d*this.e)/a,(this.b*this.e-this.a*this.f)/a)};n.clone=function(){return new w(this.a,this.b,this.c,this.d,this.e,
this.f)};n.translate=function(a,d){return this.add(1,0,0,1,a,d)};n.scale=function(a,d,e,f){null==d&&(d=a);(e||f)&&this.add(1,0,0,1,e,f);this.add(a,0,0,d,0,0);(e||f)&&this.add(1,0,0,1,-e,-f);return this};n.rotate=function(b,d,e){b=a.rad(b);d=d||0;e=e||0;var l=+f.cos(b).toFixed(9);b=+f.sin(b).toFixed(9);this.add(l,b,-b,l,d,e);return this.add(1,0,0,1,-d,-e)};n.x=function(a,d){return a*this.a+d*this.c+this.e};n.y=function(a,d){return a*this.b+d*this.d+this.f};n.get=function(a){return+this[d.fromCharCode(97+
a)].toFixed(4)};n.toString=function(){return"matrix("+[this.get(0),this.get(1),this.get(2),this.get(3),this.get(4),this.get(5)].join()+")"};n.offset=function(){return[this.e.toFixed(4),this.f.toFixed(4)]};n.determinant=function(){return this.a*this.d-this.b*this.c};n.split=function(){var b={};b.dx=this.e;b.dy=this.f;var d=[[this.a,this.c],[this.b,this.d] ];b.scalex=f.sqrt(k(d[0]));p(d[0]);b.shear=d[0][0]*d[1][0]+d[0][1]*d[1][1];d[1]=[d[1][0]-d[0][0]*b.shear,d[1][1]-d[0][1]*b.shear];b.scaley=f.sqrt(k(d[1]));
p(d[1]);b.shear/=b.scaley;0>this.determinant()&&(b.scalex=-b.scalex);var e=-d[0][1],d=d[1][1];0>d?(b.rotate=a.deg(f.acos(d)),0>e&&(b.rotate=360-b.rotate)):b.rotate=a.deg(f.asin(e));b.isSimple=!+b.shear.toFixed(9)&&(b.scalex.toFixed(9)==b.scaley.toFixed(9)||!b.rotate);b.isSuperSimple=!+b.shear.toFixed(9)&&b.scalex.toFixed(9)==b.scaley.toFixed(9)&&!b.rotate;b.noRotation=!+b.shear.toFixed(9)&&!b.rotate;return b};n.toTransformString=function(a){a=a||this.split();if(+a.shear.toFixed(9))return"m"+[this.get(0),
this.get(1),this.get(2),this.get(3),this.get(4),this.get(5)];a.scalex=+a.scalex.toFixed(4);a.scaley=+a.scaley.toFixed(4);a.rotate=+a.rotate.toFixed(4);return(a.dx||a.dy?"t"+[+a.dx.toFixed(4),+a.dy.toFixed(4)]:"")+(1!=a.scalex||1!=a.scaley?"s"+[a.scalex,a.scaley,0,0]:"")+(a.rotate?"r"+[+a.rotate.toFixed(4),0,0]:"")}})(w.prototype);a.Matrix=w;a.matrix=function(a,d,f,b,k,e){return new w(a,d,f,b,k,e)}});C.plugin(function(a,v,y,M,A){function w(h){return function(d){k.stop();d instanceof A&&1==d.node.childNodes.length&&
("radialGradient"==d.node.firstChild.tagName||"linearGradient"==d.node.firstChild.tagName||"pattern"==d.node.firstChild.tagName)&&(d=d.node.firstChild,b(this).appendChild(d),d=u(d));if(d instanceof v)if("radialGradient"==d.type||"linearGradient"==d.type||"pattern"==d.type){d.node.id||e(d.node,{id:d.id});var f=l(d.node.id)}else f=d.attr(h);else f=a.color(d),f.error?(f=a(b(this).ownerSVGElement).gradient(d))?(f.node.id||e(f.node,{id:f.id}),f=l(f.node.id)):f=d:f=r(f);d={};d[h]=f;e(this.node,d);this.node.style[h]=
x}}function z(a){k.stop();a==+a&&(a+="px");this.node.style.fontSize=a}function d(a){var b=[];a=a.childNodes;for(var e=0,f=a.length;e<f;e++){var l=a[e];3==l.nodeType&&b.push(l.nodeValue);"tspan"==l.tagName&&(1==l.childNodes.length&&3==l.firstChild.nodeType?b.push(l.firstChild.nodeValue):b.push(d(l)))}return b}function f(){k.stop();return this.node.style.fontSize}var n=a._.make,u=a._.wrap,p=a.is,b=a._.getSomeDefs,q=/^url\(#?([^)]+)\)$/,e=a._.$,l=a.url,r=String,s=a._.separator,x="";k.on("snap.util.attr.mask",
function(a){if(a instanceof v||a instanceof A){k.stop();a instanceof A&&1==a.node.childNodes.length&&(a=a.node.firstChild,b(this).appendChild(a),a=u(a));if("mask"==a.type)var d=a;else d=n("mask",b(this)),d.node.appendChild(a.node);!d.node.id&&e(d.node,{id:d.id});e(this.node,{mask:l(d.id)})}});(function(a){k.on("snap.util.attr.clip",a);k.on("snap.util.attr.clip-path",a);k.on("snap.util.attr.clipPath",a)})(function(a){if(a instanceof v||a instanceof A){k.stop();if("clipPath"==a.type)var d=a;else d=
n("clipPath",b(this)),d.node.appendChild(a.node),!d.node.id&&e(d.node,{id:d.id});e(this.node,{"clip-path":l(d.id)})}});k.on("snap.util.attr.fill",w("fill"));k.on("snap.util.attr.stroke",w("stroke"));var G=/^([lr])(?:\(([^)]*)\))?(.*)$/i;k.on("snap.util.grad.parse",function(a){a=r(a);var b=a.match(G);if(!b)return null;a=b[1];var e=b[2],b=b[3],e=e.split(/\s*,\s*/).map(function(a){return+a==a?+a:a});1==e.length&&0==e[0]&&(e=[]);b=b.split("-");b=b.map(function(a){a=a.split(":");var b={color:a[0]};a[1]&&
(b.offset=parseFloat(a[1]));return b});return{type:a,params:e,stops:b}});k.on("snap.util.attr.d",function(b){k.stop();p(b,"array")&&p(b[0],"array")&&(b=a.path.toString.call(b));b=r(b);b.match(/[ruo]/i)&&(b=a.path.toAbsolute(b));e(this.node,{d:b})})(-1);k.on("snap.util.attr.#text",function(a){k.stop();a=r(a);for(a=M.doc.createTextNode(a);this.node.firstChild;)this.node.removeChild(this.node.firstChild);this.node.appendChild(a)})(-1);k.on("snap.util.attr.path",function(a){k.stop();this.attr({d:a})})(-1);
k.on("snap.util.attr.class",function(a){k.stop();this.node.className.baseVal=a})(-1);k.on("snap.util.attr.viewBox",function(a){a=p(a,"object")&&"x"in a?[a.x,a.y,a.width,a.height].join(" "):p(a,"array")?a.join(" "):a;e(this.node,{viewBox:a});k.stop()})(-1);k.on("snap.util.attr.transform",function(a){this.transform(a);k.stop()})(-1);k.on("snap.util.attr.r",function(a){"rect"==this.type&&(k.stop(),e(this.node,{rx:a,ry:a}))})(-1);k.on("snap.util.attr.textpath",function(a){k.stop();if("text"==this.type){var d,
f;if(!a&&this.textPath){for(a=this.textPath;a.node.firstChild;)this.node.appendChild(a.node.firstChild);a.remove();delete this.textPath}else if(p(a,"string")?(d=b(this),a=u(d.parentNode).path(a),d.appendChild(a.node),d=a.id,a.attr({id:d})):(a=u(a),a instanceof v&&(d=a.attr("id"),d||(d=a.id,a.attr({id:d})))),d)if(a=this.textPath,f=this.node,a)a.attr({"xlink:href":"#"+d});else{for(a=e("textPath",{"xlink:href":"#"+d});f.firstChild;)a.appendChild(f.firstChild);f.appendChild(a);this.textPath=u(a)}}})(-1);
k.on("snap.util.attr.text",function(a){if("text"==this.type){for(var b=this.node,d=function(a){var b=e("tspan");if(p(a,"array"))for(var f=0;f<a.length;f++)b.appendChild(d(a[f]));else b.appendChild(M.doc.createTextNode(a));b.normalize&&b.normalize();return b};b.firstChild;)b.removeChild(b.firstChild);for(a=d(a);a.firstChild;)b.appendChild(a.firstChild)}k.stop()})(-1);k.on("snap.util.attr.fontSize",z)(-1);k.on("snap.util.attr.font-size",z)(-1);k.on("snap.util.getattr.transform",function(){k.stop();
return this.transform()})(-1);k.on("snap.util.getattr.textpath",function(){k.stop();return this.textPath})(-1);(function(){function b(d){return function(){k.stop();var b=M.doc.defaultView.getComputedStyle(this.node,null).getPropertyValue("marker-"+d);return"none"==b?b:a(M.doc.getElementById(b.match(q)[1]))}}function d(a){return function(b){k.stop();var d="marker"+a.charAt(0).toUpperCase()+a.substring(1);if(""==b||!b)this.node.style[d]="none";else if("marker"==b.type){var f=b.node.id;f||e(b.node,{id:b.id});
this.node.style[d]=l(f)}}}k.on("snap.util.getattr.marker-end",b("end"))(-1);k.on("snap.util.getattr.markerEnd",b("end"))(-1);k.on("snap.util.getattr.marker-start",b("start"))(-1);k.on("snap.util.getattr.markerStart",b("start"))(-1);k.on("snap.util.getattr.marker-mid",b("mid"))(-1);k.on("snap.util.getattr.markerMid",b("mid"))(-1);k.on("snap.util.attr.marker-end",d("end"))(-1);k.on("snap.util.attr.markerEnd",d("end"))(-1);k.on("snap.util.attr.marker-start",d("start"))(-1);k.on("snap.util.attr.markerStart",
d("start"))(-1);k.on("snap.util.attr.marker-mid",d("mid"))(-1);k.on("snap.util.attr.markerMid",d("mid"))(-1)})();k.on("snap.util.getattr.r",function(){if("rect"==this.type&&e(this.node,"rx")==e(this.node,"ry"))return k.stop(),e(this.node,"rx")})(-1);k.on("snap.util.getattr.text",function(){if("text"==this.type||"tspan"==this.type){k.stop();var a=d(this.node);return 1==a.length?a[0]:a}})(-1);k.on("snap.util.getattr.#text",function(){return this.node.textContent})(-1);k.on("snap.util.getattr.viewBox",
function(){k.stop();var b=e(this.node,"viewBox");if(b)return b=b.split(s),a._.box(+b[0],+b[1],+b[2],+b[3])})(-1);k.on("snap.util.getattr.points",function(){var a=e(this.node,"points");k.stop();if(a)return a.split(s)})(-1);k.on("snap.util.getattr.path",function(){var a=e(this.node,"d");k.stop();return a})(-1);k.on("snap.util.getattr.class",function(){return this.node.className.baseVal})(-1);k.on("snap.util.getattr.fontSize",f)(-1);k.on("snap.util.getattr.font-size",f)(-1)});C.plugin(function(a,v,y,
M,A){function w(a){return a}function z(a){return function(b){return+b.toFixed(3)+a}}var d={"+":function(a,b){return a+b},"-":function(a,b){return a-b},"/":function(a,b){return a/b},"*":function(a,b){return a*b}},f=String,n=/[a-z]+$/i,u=/^\s*([+\-\/*])\s*=\s*([\d.eE+\-]+)\s*([^\d\s]+)?\s*$/;k.on("snap.util.attr",function(a){if(a=f(a).match(u)){var b=k.nt(),b=b.substring(b.lastIndexOf(".")+1),q=this.attr(b),e={};k.stop();var l=a[3]||"",r=q.match(n),s=d[a[1] ];r&&r==l?a=s(parseFloat(q),+a[2]):(q=this.asPX(b),
a=s(this.asPX(b),this.asPX(b,a[2]+l)));isNaN(q)||isNaN(a)||(e[b]=a,this.attr(e))}})(-10);k.on("snap.util.equal",function(a,b){var q=f(this.attr(a)||""),e=f(b).match(u);if(e){k.stop();var l=e[3]||"",r=q.match(n),s=d[e[1] ];if(r&&r==l)return{from:parseFloat(q),to:s(parseFloat(q),+e[2]),f:z(r)};q=this.asPX(a);return{from:q,to:s(q,this.asPX(a,e[2]+l)),f:w}}})(-10)});C.plugin(function(a,v,y,M,A){var w=y.prototype,z=a.is;w.rect=function(a,d,k,p,b,q){var e;null==q&&(q=b);z(a,"object")&&"[object Object]"==
a?e=a:null!=a&&(e={x:a,y:d,width:k,height:p},null!=b&&(e.rx=b,e.ry=q));return this.el("rect",e)};w.circle=function(a,d,k){var p;z(a,"object")&&"[object Object]"==a?p=a:null!=a&&(p={cx:a,cy:d,r:k});return this.el("circle",p)};var d=function(){function a(){this.parentNode.removeChild(this)}return function(d,k){var p=M.doc.createElement("img"),b=M.doc.body;p.style.cssText="position:absolute;left:-9999em;top:-9999em";p.onload=function(){k.call(p);p.onload=p.onerror=null;b.removeChild(p)};p.onerror=a;
b.appendChild(p);p.src=d}}();w.image=function(f,n,k,p,b){var q=this.el("image");if(z(f,"object")&&"src"in f)q.attr(f);else if(null!=f){var e={"xlink:href":f,preserveAspectRatio:"none"};null!=n&&null!=k&&(e.x=n,e.y=k);null!=p&&null!=b?(e.width=p,e.height=b):d(f,function(){a._.$(q.node,{width:this.offsetWidth,height:this.offsetHeight})});a._.$(q.node,e)}return q};w.ellipse=function(a,d,k,p){var b;z(a,"object")&&"[object Object]"==a?b=a:null!=a&&(b={cx:a,cy:d,rx:k,ry:p});return this.el("ellipse",b)};
w.path=function(a){var d;z(a,"object")&&!z(a,"array")?d=a:a&&(d={d:a});return this.el("path",d)};w.group=w.g=function(a){var d=this.el("g");1==arguments.length&&a&&!a.type?d.attr(a):arguments.length&&d.add(Array.prototype.slice.call(arguments,0));return d};w.svg=function(a,d,k,p,b,q,e,l){var r={};z(a,"object")&&null==d?r=a:(null!=a&&(r.x=a),null!=d&&(r.y=d),null!=k&&(r.width=k),null!=p&&(r.height=p),null!=b&&null!=q&&null!=e&&null!=l&&(r.viewBox=[b,q,e,l]));return this.el("svg",r)};w.mask=function(a){var d=
this.el("mask");1==arguments.length&&a&&!a.type?d.attr(a):arguments.length&&d.add(Array.prototype.slice.call(arguments,0));return d};w.ptrn=function(a,d,k,p,b,q,e,l){if(z(a,"object"))var r=a;else arguments.length?(r={},null!=a&&(r.x=a),null!=d&&(r.y=d),null!=k&&(r.width=k),null!=p&&(r.height=p),null!=b&&null!=q&&null!=e&&null!=l&&(r.viewBox=[b,q,e,l])):r={patternUnits:"userSpaceOnUse"};return this.el("pattern",r)};w.use=function(a){return null!=a?(make("use",this.node),a instanceof v&&(a.attr("id")||
a.attr({id:ID()}),a=a.attr("id")),this.el("use",{"xlink:href":a})):v.prototype.use.call(this)};w.text=function(a,d,k){var p={};z(a,"object")?p=a:null!=a&&(p={x:a,y:d,text:k||""});return this.el("text",p)};w.line=function(a,d,k,p){var b={};z(a,"object")?b=a:null!=a&&(b={x1:a,x2:k,y1:d,y2:p});return this.el("line",b)};w.polyline=function(a){1<arguments.length&&(a=Array.prototype.slice.call(arguments,0));var d={};z(a,"object")&&!z(a,"array")?d=a:null!=a&&(d={points:a});return this.el("polyline",d)};
w.polygon=function(a){1<arguments.length&&(a=Array.prototype.slice.call(arguments,0));var d={};z(a,"object")&&!z(a,"array")?d=a:null!=a&&(d={points:a});return this.el("polygon",d)};(function(){function d(){return this.selectAll("stop")}function n(b,d){var f=e("stop"),k={offset:+d+"%"};b=a.color(b);k["stop-color"]=b.hex;1>b.opacity&&(k["stop-opacity"]=b.opacity);e(f,k);this.node.appendChild(f);return this}function u(){if("linearGradient"==this.type){var b=e(this.node,"x1")||0,d=e(this.node,"x2")||
1,f=e(this.node,"y1")||0,k=e(this.node,"y2")||0;return a._.box(b,f,math.abs(d-b),math.abs(k-f))}b=this.node.r||0;return a._.box((this.node.cx||0.5)-b,(this.node.cy||0.5)-b,2*b,2*b)}function p(a,d){function f(a,b){for(var d=(b-u)/(a-w),e=w;e<a;e++)h[e].offset=+(+u+d*(e-w)).toFixed(2);w=a;u=b}var n=k("snap.util.grad.parse",null,d).firstDefined(),p;if(!n)return null;n.params.unshift(a);p="l"==n.type.toLowerCase()?b.apply(0,n.params):q.apply(0,n.params);n.type!=n.type.toLowerCase()&&e(p.node,{gradientUnits:"userSpaceOnUse"});
var h=n.stops,n=h.length,u=0,w=0;n--;for(var v=0;v<n;v++)"offset"in h[v]&&f(v,h[v].offset);h[n].offset=h[n].offset||100;f(n,h[n].offset);for(v=0;v<=n;v++){var y=h[v];p.addStop(y.color,y.offset)}return p}function b(b,k,p,q,w){b=a._.make("linearGradient",b);b.stops=d;b.addStop=n;b.getBBox=u;null!=k&&e(b.node,{x1:k,y1:p,x2:q,y2:w});return b}function q(b,k,p,q,w,h){b=a._.make("radialGradient",b);b.stops=d;b.addStop=n;b.getBBox=u;null!=k&&e(b.node,{cx:k,cy:p,r:q});null!=w&&null!=h&&e(b.node,{fx:w,fy:h});
return b}var e=a._.$;w.gradient=function(a){return p(this.defs,a)};w.gradientLinear=function(a,d,e,f){return b(this.defs,a,d,e,f)};w.gradientRadial=function(a,b,d,e,f){return q(this.defs,a,b,d,e,f)};w.toString=function(){var b=this.node.ownerDocument,d=b.createDocumentFragment(),b=b.createElement("div"),e=this.node.cloneNode(!0);d.appendChild(b);b.appendChild(e);a._.$(e,{xmlns:"http://www.w3.org/2000/svg"});b=b.innerHTML;d.removeChild(d.firstChild);return b};w.clear=function(){for(var a=this.node.firstChild,
b;a;)b=a.nextSibling,"defs"!=a.tagName?a.parentNode.removeChild(a):w.clear.call({node:a}),a=b}})()});C.plugin(function(a,k,y,M){function A(a){var b=A.ps=A.ps||{};b[a]?b[a].sleep=100:b[a]={sleep:100};setTimeout(function(){for(var d in b)b[L](d)&&d!=a&&(b[d].sleep--,!b[d].sleep&&delete b[d])});return b[a]}function w(a,b,d,e){null==a&&(a=b=d=e=0);null==b&&(b=a.y,d=a.width,e=a.height,a=a.x);return{x:a,y:b,width:d,w:d,height:e,h:e,x2:a+d,y2:b+e,cx:a+d/2,cy:b+e/2,r1:F.min(d,e)/2,r2:F.max(d,e)/2,r0:F.sqrt(d*
d+e*e)/2,path:s(a,b,d,e),vb:[a,b,d,e].join(" ")}}function z(){return this.join(",").replace(N,"$1")}function d(a){a=C(a);a.toString=z;return a}function f(a,b,d,h,f,k,l,n,p){if(null==p)return e(a,b,d,h,f,k,l,n);if(0>p||e(a,b,d,h,f,k,l,n)<p)p=void 0;else{var q=0.5,O=1-q,s;for(s=e(a,b,d,h,f,k,l,n,O);0.01<Z(s-p);)q/=2,O+=(s<p?1:-1)*q,s=e(a,b,d,h,f,k,l,n,O);p=O}return u(a,b,d,h,f,k,l,n,p)}function n(b,d){function e(a){return+(+a).toFixed(3)}return a._.cacher(function(a,h,l){a instanceof k&&(a=a.attr("d"));
a=I(a);for(var n,p,D,q,O="",s={},c=0,t=0,r=a.length;t<r;t++){D=a[t];if("M"==D[0])n=+D[1],p=+D[2];else{q=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6]);if(c+q>h){if(d&&!s.start){n=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6],h-c);O+=["C"+e(n.start.x),e(n.start.y),e(n.m.x),e(n.m.y),e(n.x),e(n.y)];if(l)return O;s.start=O;O=["M"+e(n.x),e(n.y)+"C"+e(n.n.x),e(n.n.y),e(n.end.x),e(n.end.y),e(D[5]),e(D[6])].join();c+=q;n=+D[5];p=+D[6];continue}if(!b&&!d)return n=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6],h-c)}c+=q;n=+D[5];p=+D[6]}O+=
D.shift()+D}s.end=O;return n=b?c:d?s:u(n,p,D[0],D[1],D[2],D[3],D[4],D[5],1)},null,a._.clone)}function u(a,b,d,e,h,f,k,l,n){var p=1-n,q=ma(p,3),s=ma(p,2),c=n*n,t=c*n,r=q*a+3*s*n*d+3*p*n*n*h+t*k,q=q*b+3*s*n*e+3*p*n*n*f+t*l,s=a+2*n*(d-a)+c*(h-2*d+a),t=b+2*n*(e-b)+c*(f-2*e+b),x=d+2*n*(h-d)+c*(k-2*h+d),c=e+2*n*(f-e)+c*(l-2*f+e);a=p*a+n*d;b=p*b+n*e;h=p*h+n*k;f=p*f+n*l;l=90-180*F.atan2(s-x,t-c)/S;return{x:r,y:q,m:{x:s,y:t},n:{x:x,y:c},start:{x:a,y:b},end:{x:h,y:f},alpha:l}}function p(b,d,e,h,f,n,k,l){a.is(b,
"array")||(b=[b,d,e,h,f,n,k,l]);b=U.apply(null,b);return w(b.min.x,b.min.y,b.max.x-b.min.x,b.max.y-b.min.y)}function b(a,b,d){return b>=a.x&&b<=a.x+a.width&&d>=a.y&&d<=a.y+a.height}function q(a,d){a=w(a);d=w(d);return b(d,a.x,a.y)||b(d,a.x2,a.y)||b(d,a.x,a.y2)||b(d,a.x2,a.y2)||b(a,d.x,d.y)||b(a,d.x2,d.y)||b(a,d.x,d.y2)||b(a,d.x2,d.y2)||(a.x<d.x2&&a.x>d.x||d.x<a.x2&&d.x>a.x)&&(a.y<d.y2&&a.y>d.y||d.y<a.y2&&d.y>a.y)}function e(a,b,d,e,h,f,n,k,l){null==l&&(l=1);l=(1<l?1:0>l?0:l)/2;for(var p=[-0.1252,
0.1252,-0.3678,0.3678,-0.5873,0.5873,-0.7699,0.7699,-0.9041,0.9041,-0.9816,0.9816],q=[0.2491,0.2491,0.2335,0.2335,0.2032,0.2032,0.1601,0.1601,0.1069,0.1069,0.0472,0.0472],s=0,c=0;12>c;c++)var t=l*p[c]+l,r=t*(t*(-3*a+9*d-9*h+3*n)+6*a-12*d+6*h)-3*a+3*d,t=t*(t*(-3*b+9*e-9*f+3*k)+6*b-12*e+6*f)-3*b+3*e,s=s+q[c]*F.sqrt(r*r+t*t);return l*s}function l(a,b,d){a=I(a);b=I(b);for(var h,f,l,n,k,s,r,O,x,c,t=d?0:[],w=0,v=a.length;w<v;w++)if(x=a[w],"M"==x[0])h=k=x[1],f=s=x[2];else{"C"==x[0]?(x=[h,f].concat(x.slice(1)),
h=x[6],f=x[7]):(x=[h,f,h,f,k,s,k,s],h=k,f=s);for(var G=0,y=b.length;G<y;G++)if(c=b[G],"M"==c[0])l=r=c[1],n=O=c[2];else{"C"==c[0]?(c=[l,n].concat(c.slice(1)),l=c[6],n=c[7]):(c=[l,n,l,n,r,O,r,O],l=r,n=O);var z;var K=x,B=c;z=d;var H=p(K),J=p(B);if(q(H,J)){for(var H=e.apply(0,K),J=e.apply(0,B),H=~~(H/8),J=~~(J/8),U=[],A=[],F={},M=z?0:[],P=0;P<H+1;P++){var C=u.apply(0,K.concat(P/H));U.push({x:C.x,y:C.y,t:P/H})}for(P=0;P<J+1;P++)C=u.apply(0,B.concat(P/J)),A.push({x:C.x,y:C.y,t:P/J});for(P=0;P<H;P++)for(K=
0;K<J;K++){var Q=U[P],L=U[P+1],B=A[K],C=A[K+1],N=0.001>Z(L.x-Q.x)?"y":"x",S=0.001>Z(C.x-B.x)?"y":"x",R;R=Q.x;var Y=Q.y,V=L.x,ea=L.y,fa=B.x,ga=B.y,ha=C.x,ia=C.y;if(W(R,V)<X(fa,ha)||X(R,V)>W(fa,ha)||W(Y,ea)<X(ga,ia)||X(Y,ea)>W(ga,ia))R=void 0;else{var $=(R*ea-Y*V)*(fa-ha)-(R-V)*(fa*ia-ga*ha),aa=(R*ea-Y*V)*(ga-ia)-(Y-ea)*(fa*ia-ga*ha),ja=(R-V)*(ga-ia)-(Y-ea)*(fa-ha);if(ja){var $=$/ja,aa=aa/ja,ja=+$.toFixed(2),ba=+aa.toFixed(2);R=ja<+X(R,V).toFixed(2)||ja>+W(R,V).toFixed(2)||ja<+X(fa,ha).toFixed(2)||
ja>+W(fa,ha).toFixed(2)||ba<+X(Y,ea).toFixed(2)||ba>+W(Y,ea).toFixed(2)||ba<+X(ga,ia).toFixed(2)||ba>+W(ga,ia).toFixed(2)?void 0:{x:$,y:aa}}else R=void 0}R&&F[R.x.toFixed(4)]!=R.y.toFixed(4)&&(F[R.x.toFixed(4)]=R.y.toFixed(4),Q=Q.t+Z((R[N]-Q[N])/(L[N]-Q[N]))*(L.t-Q.t),B=B.t+Z((R[S]-B[S])/(C[S]-B[S]))*(C.t-B.t),0<=Q&&1>=Q&&0<=B&&1>=B&&(z?M++:M.push({x:R.x,y:R.y,t1:Q,t2:B})))}z=M}else z=z?0:[];if(d)t+=z;else{H=0;for(J=z.length;H<J;H++)z[H].segment1=w,z[H].segment2=G,z[H].bez1=x,z[H].bez2=c;t=t.concat(z)}}}return t}
function r(a){var b=A(a);if(b.bbox)return C(b.bbox);if(!a)return w();a=I(a);for(var d=0,e=0,h=[],f=[],l,n=0,k=a.length;n<k;n++)l=a[n],"M"==l[0]?(d=l[1],e=l[2],h.push(d),f.push(e)):(d=U(d,e,l[1],l[2],l[3],l[4],l[5],l[6]),h=h.concat(d.min.x,d.max.x),f=f.concat(d.min.y,d.max.y),d=l[5],e=l[6]);a=X.apply(0,h);l=X.apply(0,f);h=W.apply(0,h);f=W.apply(0,f);f=w(a,l,h-a,f-l);b.bbox=C(f);return f}function s(a,b,d,e,h){if(h)return[["M",+a+ +h,b],["l",d-2*h,0],["a",h,h,0,0,1,h,h],["l",0,e-2*h],["a",h,h,0,0,1,
-h,h],["l",2*h-d,0],["a",h,h,0,0,1,-h,-h],["l",0,2*h-e],["a",h,h,0,0,1,h,-h],["z"] ];a=[["M",a,b],["l",d,0],["l",0,e],["l",-d,0],["z"] ];a.toString=z;return a}function x(a,b,d,e,h){null==h&&null==e&&(e=d);a=+a;b=+b;d=+d;e=+e;if(null!=h){var f=Math.PI/180,l=a+d*Math.cos(-e*f);a+=d*Math.cos(-h*f);var n=b+d*Math.sin(-e*f);b+=d*Math.sin(-h*f);d=[["M",l,n],["A",d,d,0,+(180<h-e),0,a,b] ]}else d=[["M",a,b],["m",0,-e],["a",d,e,0,1,1,0,2*e],["a",d,e,0,1,1,0,-2*e],["z"] ];d.toString=z;return d}function G(b){var e=
A(b);if(e.abs)return d(e.abs);Q(b,"array")&&Q(b&&b[0],"array")||(b=a.parsePathString(b));if(!b||!b.length)return[["M",0,0] ];var h=[],f=0,l=0,n=0,k=0,p=0;"M"==b[0][0]&&(f=+b[0][1],l=+b[0][2],n=f,k=l,p++,h[0]=["M",f,l]);for(var q=3==b.length&&"M"==b[0][0]&&"R"==b[1][0].toUpperCase()&&"Z"==b[2][0].toUpperCase(),s,r,w=p,c=b.length;w<c;w++){h.push(s=[]);r=b[w];p=r[0];if(p!=p.toUpperCase())switch(s[0]=p.toUpperCase(),s[0]){case "A":s[1]=r[1];s[2]=r[2];s[3]=r[3];s[4]=r[4];s[5]=r[5];s[6]=+r[6]+f;s[7]=+r[7]+
l;break;case "V":s[1]=+r[1]+l;break;case "H":s[1]=+r[1]+f;break;case "R":for(var t=[f,l].concat(r.slice(1)),u=2,v=t.length;u<v;u++)t[u]=+t[u]+f,t[++u]=+t[u]+l;h.pop();h=h.concat(P(t,q));break;case "O":h.pop();t=x(f,l,r[1],r[2]);t.push(t[0]);h=h.concat(t);break;case "U":h.pop();h=h.concat(x(f,l,r[1],r[2],r[3]));s=["U"].concat(h[h.length-1].slice(-2));break;case "M":n=+r[1]+f,k=+r[2]+l;default:for(u=1,v=r.length;u<v;u++)s[u]=+r[u]+(u%2?f:l)}else if("R"==p)t=[f,l].concat(r.slice(1)),h.pop(),h=h.concat(P(t,
q)),s=["R"].concat(r.slice(-2));else if("O"==p)h.pop(),t=x(f,l,r[1],r[2]),t.push(t[0]),h=h.concat(t);else if("U"==p)h.pop(),h=h.concat(x(f,l,r[1],r[2],r[3])),s=["U"].concat(h[h.length-1].slice(-2));else for(t=0,u=r.length;t<u;t++)s[t]=r[t];p=p.toUpperCase();if("O"!=p)switch(s[0]){case "Z":f=+n;l=+k;break;case "H":f=s[1];break;case "V":l=s[1];break;case "M":n=s[s.length-2],k=s[s.length-1];default:f=s[s.length-2],l=s[s.length-1]}}h.toString=z;e.abs=d(h);return h}function h(a,b,d,e){return[a,b,d,e,d,
e]}function J(a,b,d,e,h,f){var l=1/3,n=2/3;return[l*a+n*d,l*b+n*e,l*h+n*d,l*f+n*e,h,f]}function K(b,d,e,h,f,l,n,k,p,s){var r=120*S/180,q=S/180*(+f||0),c=[],t,x=a._.cacher(function(a,b,c){var d=a*F.cos(c)-b*F.sin(c);a=a*F.sin(c)+b*F.cos(c);return{x:d,y:a}});if(s)v=s[0],t=s[1],l=s[2],u=s[3];else{t=x(b,d,-q);b=t.x;d=t.y;t=x(k,p,-q);k=t.x;p=t.y;F.cos(S/180*f);F.sin(S/180*f);t=(b-k)/2;v=(d-p)/2;u=t*t/(e*e)+v*v/(h*h);1<u&&(u=F.sqrt(u),e*=u,h*=u);var u=e*e,w=h*h,u=(l==n?-1:1)*F.sqrt(Z((u*w-u*v*v-w*t*t)/
(u*v*v+w*t*t)));l=u*e*v/h+(b+k)/2;var u=u*-h*t/e+(d+p)/2,v=F.asin(((d-u)/h).toFixed(9));t=F.asin(((p-u)/h).toFixed(9));v=b<l?S-v:v;t=k<l?S-t:t;0>v&&(v=2*S+v);0>t&&(t=2*S+t);n&&v>t&&(v-=2*S);!n&&t>v&&(t-=2*S)}if(Z(t-v)>r){var c=t,w=k,G=p;t=v+r*(n&&t>v?1:-1);k=l+e*F.cos(t);p=u+h*F.sin(t);c=K(k,p,e,h,f,0,n,w,G,[t,c,l,u])}l=t-v;f=F.cos(v);r=F.sin(v);n=F.cos(t);t=F.sin(t);l=F.tan(l/4);e=4/3*e*l;l*=4/3*h;h=[b,d];b=[b+e*r,d-l*f];d=[k+e*t,p-l*n];k=[k,p];b[0]=2*h[0]-b[0];b[1]=2*h[1]-b[1];if(s)return[b,d,k].concat(c);
c=[b,d,k].concat(c).join().split(",");s=[];k=0;for(p=c.length;k<p;k++)s[k]=k%2?x(c[k-1],c[k],q).y:x(c[k],c[k+1],q).x;return s}function U(a,b,d,e,h,f,l,k){for(var n=[],p=[[],[] ],s,r,c,t,q=0;2>q;++q)0==q?(r=6*a-12*d+6*h,s=-3*a+9*d-9*h+3*l,c=3*d-3*a):(r=6*b-12*e+6*f,s=-3*b+9*e-9*f+3*k,c=3*e-3*b),1E-12>Z(s)?1E-12>Z(r)||(s=-c/r,0<s&&1>s&&n.push(s)):(t=r*r-4*c*s,c=F.sqrt(t),0>t||(t=(-r+c)/(2*s),0<t&&1>t&&n.push(t),s=(-r-c)/(2*s),0<s&&1>s&&n.push(s)));for(r=q=n.length;q--;)s=n[q],c=1-s,p[0][q]=c*c*c*a+3*
c*c*s*d+3*c*s*s*h+s*s*s*l,p[1][q]=c*c*c*b+3*c*c*s*e+3*c*s*s*f+s*s*s*k;p[0][r]=a;p[1][r]=b;p[0][r+1]=l;p[1][r+1]=k;p[0].length=p[1].length=r+2;return{min:{x:X.apply(0,p[0]),y:X.apply(0,p[1])},max:{x:W.apply(0,p[0]),y:W.apply(0,p[1])}}}function I(a,b){var e=!b&&A(a);if(!b&&e.curve)return d(e.curve);var f=G(a),l=b&&G(b),n={x:0,y:0,bx:0,by:0,X:0,Y:0,qx:null,qy:null},k={x:0,y:0,bx:0,by:0,X:0,Y:0,qx:null,qy:null},p=function(a,b,c){if(!a)return["C",b.x,b.y,b.x,b.y,b.x,b.y];a[0]in{T:1,Q:1}||(b.qx=b.qy=null);
switch(a[0]){case "M":b.X=a[1];b.Y=a[2];break;case "A":a=["C"].concat(K.apply(0,[b.x,b.y].concat(a.slice(1))));break;case "S":"C"==c||"S"==c?(c=2*b.x-b.bx,b=2*b.y-b.by):(c=b.x,b=b.y);a=["C",c,b].concat(a.slice(1));break;case "T":"Q"==c||"T"==c?(b.qx=2*b.x-b.qx,b.qy=2*b.y-b.qy):(b.qx=b.x,b.qy=b.y);a=["C"].concat(J(b.x,b.y,b.qx,b.qy,a[1],a[2]));break;case "Q":b.qx=a[1];b.qy=a[2];a=["C"].concat(J(b.x,b.y,a[1],a[2],a[3],a[4]));break;case "L":a=["C"].concat(h(b.x,b.y,a[1],a[2]));break;case "H":a=["C"].concat(h(b.x,
b.y,a[1],b.y));break;case "V":a=["C"].concat(h(b.x,b.y,b.x,a[1]));break;case "Z":a=["C"].concat(h(b.x,b.y,b.X,b.Y))}return a},s=function(a,b){if(7<a[b].length){a[b].shift();for(var c=a[b];c.length;)q[b]="A",l&&(u[b]="A"),a.splice(b++,0,["C"].concat(c.splice(0,6)));a.splice(b,1);v=W(f.length,l&&l.length||0)}},r=function(a,b,c,d,e){a&&b&&"M"==a[e][0]&&"M"!=b[e][0]&&(b.splice(e,0,["M",d.x,d.y]),c.bx=0,c.by=0,c.x=a[e][1],c.y=a[e][2],v=W(f.length,l&&l.length||0))},q=[],u=[],c="",t="",x=0,v=W(f.length,
l&&l.length||0);for(;x<v;x++){f[x]&&(c=f[x][0]);"C"!=c&&(q[x]=c,x&&(t=q[x-1]));f[x]=p(f[x],n,t);"A"!=q[x]&&"C"==c&&(q[x]="C");s(f,x);l&&(l[x]&&(c=l[x][0]),"C"!=c&&(u[x]=c,x&&(t=u[x-1])),l[x]=p(l[x],k,t),"A"!=u[x]&&"C"==c&&(u[x]="C"),s(l,x));r(f,l,n,k,x);r(l,f,k,n,x);var w=f[x],z=l&&l[x],y=w.length,U=l&&z.length;n.x=w[y-2];n.y=w[y-1];n.bx=$(w[y-4])||n.x;n.by=$(w[y-3])||n.y;k.bx=l&&($(z[U-4])||k.x);k.by=l&&($(z[U-3])||k.y);k.x=l&&z[U-2];k.y=l&&z[U-1]}l||(e.curve=d(f));return l?[f,l]:f}function P(a,
b){for(var d=[],e=0,h=a.length;h-2*!b>e;e+=2){var f=[{x:+a[e-2],y:+a[e-1]},{x:+a[e],y:+a[e+1]},{x:+a[e+2],y:+a[e+3]},{x:+a[e+4],y:+a[e+5]}];b?e?h-4==e?f[3]={x:+a[0],y:+a[1]}:h-2==e&&(f[2]={x:+a[0],y:+a[1]},f[3]={x:+a[2],y:+a[3]}):f[0]={x:+a[h-2],y:+a[h-1]}:h-4==e?f[3]=f[2]:e||(f[0]={x:+a[e],y:+a[e+1]});d.push(["C",(-f[0].x+6*f[1].x+f[2].x)/6,(-f[0].y+6*f[1].y+f[2].y)/6,(f[1].x+6*f[2].x-f[3].x)/6,(f[1].y+6*f[2].y-f[3].y)/6,f[2].x,f[2].y])}return d}y=k.prototype;var Q=a.is,C=a._.clone,L="hasOwnProperty",
N=/,?([a-z]),?/gi,$=parseFloat,F=Math,S=F.PI,X=F.min,W=F.max,ma=F.pow,Z=F.abs;M=n(1);var na=n(),ba=n(0,1),V=a._unit2px;a.path=A;a.path.getTotalLength=M;a.path.getPointAtLength=na;a.path.getSubpath=function(a,b,d){if(1E-6>this.getTotalLength(a)-d)return ba(a,b).end;a=ba(a,d,1);return b?ba(a,b).end:a};y.getTotalLength=function(){if(this.node.getTotalLength)return this.node.getTotalLength()};y.getPointAtLength=function(a){return na(this.attr("d"),a)};y.getSubpath=function(b,d){return a.path.getSubpath(this.attr("d"),
b,d)};a._.box=w;a.path.findDotsAtSegment=u;a.path.bezierBBox=p;a.path.isPointInsideBBox=b;a.path.isBBoxIntersect=q;a.path.intersection=function(a,b){return l(a,b)};a.path.intersectionNumber=function(a,b){return l(a,b,1)};a.path.isPointInside=function(a,d,e){var h=r(a);return b(h,d,e)&&1==l(a,[["M",d,e],["H",h.x2+10] ],1)%2};a.path.getBBox=r;a.path.get={path:function(a){return a.attr("path")},circle:function(a){a=V(a);return x(a.cx,a.cy,a.r)},ellipse:function(a){a=V(a);return x(a.cx||0,a.cy||0,a.rx,
a.ry)},rect:function(a){a=V(a);return s(a.x||0,a.y||0,a.width,a.height,a.rx,a.ry)},image:function(a){a=V(a);return s(a.x||0,a.y||0,a.width,a.height)},line:function(a){return"M"+[a.attr("x1")||0,a.attr("y1")||0,a.attr("x2"),a.attr("y2")]},polyline:function(a){return"M"+a.attr("points")},polygon:function(a){return"M"+a.attr("points")+"z"},deflt:function(a){a=a.node.getBBox();return s(a.x,a.y,a.width,a.height)}};a.path.toRelative=function(b){var e=A(b),h=String.prototype.toLowerCase;if(e.rel)return d(e.rel);
a.is(b,"array")&&a.is(b&&b[0],"array")||(b=a.parsePathString(b));var f=[],l=0,n=0,k=0,p=0,s=0;"M"==b[0][0]&&(l=b[0][1],n=b[0][2],k=l,p=n,s++,f.push(["M",l,n]));for(var r=b.length;s<r;s++){var q=f[s]=[],x=b[s];if(x[0]!=h.call(x[0]))switch(q[0]=h.call(x[0]),q[0]){case "a":q[1]=x[1];q[2]=x[2];q[3]=x[3];q[4]=x[4];q[5]=x[5];q[6]=+(x[6]-l).toFixed(3);q[7]=+(x[7]-n).toFixed(3);break;case "v":q[1]=+(x[1]-n).toFixed(3);break;case "m":k=x[1],p=x[2];default:for(var c=1,t=x.length;c<t;c++)q[c]=+(x[c]-(c%2?l:
n)).toFixed(3)}else for(f[s]=[],"m"==x[0]&&(k=x[1]+l,p=x[2]+n),q=0,c=x.length;q<c;q++)f[s][q]=x[q];x=f[s].length;switch(f[s][0]){case "z":l=k;n=p;break;case "h":l+=+f[s][x-1];break;case "v":n+=+f[s][x-1];break;default:l+=+f[s][x-2],n+=+f[s][x-1]}}f.toString=z;e.rel=d(f);return f};a.path.toAbsolute=G;a.path.toCubic=I;a.path.map=function(a,b){if(!b)return a;var d,e,h,f,l,n,k;a=I(a);h=0;for(l=a.length;h<l;h++)for(k=a[h],f=1,n=k.length;f<n;f+=2)d=b.x(k[f],k[f+1]),e=b.y(k[f],k[f+1]),k[f]=d,k[f+1]=e;return a};
a.path.toString=z;a.path.clone=d});C.plugin(function(a,v,y,C){var A=Math.max,w=Math.min,z=function(a){this.items=[];this.bindings={};this.length=0;this.type="set";if(a)for(var f=0,n=a.length;f<n;f++)a[f]&&(this[this.items.length]=this.items[this.items.length]=a[f],this.length++)};v=z.prototype;v.push=function(){for(var a,f,n=0,k=arguments.length;n<k;n++)if(a=arguments[n])f=this.items.length,this[f]=this.items[f]=a,this.length++;return this};v.pop=function(){this.length&&delete this[this.length--];
return this.items.pop()};v.forEach=function(a,f){for(var n=0,k=this.items.length;n<k&&!1!==a.call(f,this.items[n],n);n++);return this};v.animate=function(d,f,n,u){"function"!=typeof n||n.length||(u=n,n=L.linear);d instanceof a._.Animation&&(u=d.callback,n=d.easing,f=n.dur,d=d.attr);var p=arguments;if(a.is(d,"array")&&a.is(p[p.length-1],"array"))var b=!0;var q,e=function(){q?this.b=q:q=this.b},l=0,r=u&&function(){l++==this.length&&u.call(this)};return this.forEach(function(a,l){k.once("snap.animcreated."+
a.id,e);b?p[l]&&a.animate.apply(a,p[l]):a.animate(d,f,n,r)})};v.remove=function(){for(;this.length;)this.pop().remove();return this};v.bind=function(a,f,k){var u={};if("function"==typeof f)this.bindings[a]=f;else{var p=k||a;this.bindings[a]=function(a){u[p]=a;f.attr(u)}}return this};v.attr=function(a){var f={},k;for(k in a)if(this.bindings[k])this.bindings[k](a[k]);else f[k]=a[k];a=0;for(k=this.items.length;a<k;a++)this.items[a].attr(f);return this};v.clear=function(){for(;this.length;)this.pop()};
v.splice=function(a,f,k){a=0>a?A(this.length+a,0):a;f=A(0,w(this.length-a,f));var u=[],p=[],b=[],q;for(q=2;q<arguments.length;q++)b.push(arguments[q]);for(q=0;q<f;q++)p.push(this[a+q]);for(;q<this.length-a;q++)u.push(this[a+q]);var e=b.length;for(q=0;q<e+u.length;q++)this.items[a+q]=this[a+q]=q<e?b[q]:u[q-e];for(q=this.items.length=this.length-=f-e;this[q];)delete this[q++];return new z(p)};v.exclude=function(a){for(var f=0,k=this.length;f<k;f++)if(this[f]==a)return this.splice(f,1),!0;return!1};
v.insertAfter=function(a){for(var f=this.items.length;f--;)this.items[f].insertAfter(a);return this};v.getBBox=function(){for(var a=[],f=[],k=[],u=[],p=this.items.length;p--;)if(!this.items[p].removed){var b=this.items[p].getBBox();a.push(b.x);f.push(b.y);k.push(b.x+b.width);u.push(b.y+b.height)}a=w.apply(0,a);f=w.apply(0,f);k=A.apply(0,k);u=A.apply(0,u);return{x:a,y:f,x2:k,y2:u,width:k-a,height:u-f,cx:a+(k-a)/2,cy:f+(u-f)/2}};v.clone=function(a){a=new z;for(var f=0,k=this.items.length;f<k;f++)a.push(this.items[f].clone());
return a};v.toString=function(){return"Snap\u2018s set"};v.type="set";a.set=function(){var a=new z;arguments.length&&a.push.apply(a,Array.prototype.slice.call(arguments,0));return a}});C.plugin(function(a,v,y,C){function A(a){var b=a[0];switch(b.toLowerCase()){case "t":return[b,0,0];case "m":return[b,1,0,0,1,0,0];case "r":return 4==a.length?[b,0,a[2],a[3] ]:[b,0];case "s":return 5==a.length?[b,1,1,a[3],a[4] ]:3==a.length?[b,1,1]:[b,1]}}function w(b,d,f){d=q(d).replace(/\.{3}|\u2026/g,b);b=a.parseTransformString(b)||
[];d=a.parseTransformString(d)||[];for(var k=Math.max(b.length,d.length),p=[],v=[],h=0,w,z,y,I;h<k;h++){y=b[h]||A(d[h]);I=d[h]||A(y);if(y[0]!=I[0]||"r"==y[0].toLowerCase()&&(y[2]!=I[2]||y[3]!=I[3])||"s"==y[0].toLowerCase()&&(y[3]!=I[3]||y[4]!=I[4])){b=a._.transform2matrix(b,f());d=a._.transform2matrix(d,f());p=[["m",b.a,b.b,b.c,b.d,b.e,b.f] ];v=[["m",d.a,d.b,d.c,d.d,d.e,d.f] ];break}p[h]=[];v[h]=[];w=0;for(z=Math.max(y.length,I.length);w<z;w++)w in y&&(p[h][w]=y[w]),w in I&&(v[h][w]=I[w])}return{from:u(p),
to:u(v),f:n(p)}}function z(a){return a}function d(a){return function(b){return+b.toFixed(3)+a}}function f(b){return a.rgb(b[0],b[1],b[2])}function n(a){var b=0,d,f,k,n,h,p,q=[];d=0;for(f=a.length;d<f;d++){h="[";p=['"'+a[d][0]+'"'];k=1;for(n=a[d].length;k<n;k++)p[k]="val["+b++ +"]";h+=p+"]";q[d]=h}return Function("val","return Snap.path.toString.call(["+q+"])")}function u(a){for(var b=[],d=0,f=a.length;d<f;d++)for(var k=1,n=a[d].length;k<n;k++)b.push(a[d][k]);return b}var p={},b=/[a-z]+$/i,q=String;
p.stroke=p.fill="colour";v.prototype.equal=function(a,b){return k("snap.util.equal",this,a,b).firstDefined()};k.on("snap.util.equal",function(e,k){var r,s;r=q(this.attr(e)||"");var x=this;if(r==+r&&k==+k)return{from:+r,to:+k,f:z};if("colour"==p[e])return r=a.color(r),s=a.color(k),{from:[r.r,r.g,r.b,r.opacity],to:[s.r,s.g,s.b,s.opacity],f:f};if("transform"==e||"gradientTransform"==e||"patternTransform"==e)return k instanceof a.Matrix&&(k=k.toTransformString()),a._.rgTransform.test(k)||(k=a._.svgTransform2string(k)),
w(r,k,function(){return x.getBBox(1)});if("d"==e||"path"==e)return r=a.path.toCubic(r,k),{from:u(r[0]),to:u(r[1]),f:n(r[0])};if("points"==e)return r=q(r).split(a._.separator),s=q(k).split(a._.separator),{from:r,to:s,f:function(a){return a}};aUnit=r.match(b);s=q(k).match(b);return aUnit&&aUnit==s?{from:parseFloat(r),to:parseFloat(k),f:d(aUnit)}:{from:this.asPX(e),to:this.asPX(e,k),f:z}})});C.plugin(function(a,v,y,C){var A=v.prototype,w="createTouch"in C.doc;v="click dblclick mousedown mousemove mouseout mouseover mouseup touchstart touchmove touchend touchcancel".split(" ");
var z={mousedown:"touchstart",mousemove:"touchmove",mouseup:"touchend"},d=function(a,b){var d="y"==a?"scrollTop":"scrollLeft",e=b&&b.node?b.node.ownerDocument:C.doc;return e[d in e.documentElement?"documentElement":"body"][d]},f=function(){this.returnValue=!1},n=function(){return this.originalEvent.preventDefault()},u=function(){this.cancelBubble=!0},p=function(){return this.originalEvent.stopPropagation()},b=function(){if(C.doc.addEventListener)return function(a,b,e,f){var k=w&&z[b]?z[b]:b,l=function(k){var l=
d("y",f),q=d("x",f);if(w&&z.hasOwnProperty(b))for(var r=0,u=k.targetTouches&&k.targetTouches.length;r<u;r++)if(k.targetTouches[r].target==a||a.contains(k.targetTouches[r].target)){u=k;k=k.targetTouches[r];k.originalEvent=u;k.preventDefault=n;k.stopPropagation=p;break}return e.call(f,k,k.clientX+q,k.clientY+l)};b!==k&&a.addEventListener(b,l,!1);a.addEventListener(k,l,!1);return function(){b!==k&&a.removeEventListener(b,l,!1);a.removeEventListener(k,l,!1);return!0}};if(C.doc.attachEvent)return function(a,
b,e,h){var k=function(a){a=a||h.node.ownerDocument.window.event;var b=d("y",h),k=d("x",h),k=a.clientX+k,b=a.clientY+b;a.preventDefault=a.preventDefault||f;a.stopPropagation=a.stopPropagation||u;return e.call(h,a,k,b)};a.attachEvent("on"+b,k);return function(){a.detachEvent("on"+b,k);return!0}}}(),q=[],e=function(a){for(var b=a.clientX,e=a.clientY,f=d("y"),l=d("x"),n,p=q.length;p--;){n=q[p];if(w)for(var r=a.touches&&a.touches.length,u;r--;){if(u=a.touches[r],u.identifier==n.el._drag.id||n.el.node.contains(u.target)){b=
u.clientX;e=u.clientY;(a.originalEvent?a.originalEvent:a).preventDefault();break}}else a.preventDefault();b+=l;e+=f;k("snap.drag.move."+n.el.id,n.move_scope||n.el,b-n.el._drag.x,e-n.el._drag.y,b,e,a)}},l=function(b){a.unmousemove(e).unmouseup(l);for(var d=q.length,f;d--;)f=q[d],f.el._drag={},k("snap.drag.end."+f.el.id,f.end_scope||f.start_scope||f.move_scope||f.el,b);q=[]};for(y=v.length;y--;)(function(d){a[d]=A[d]=function(e,f){a.is(e,"function")&&(this.events=this.events||[],this.events.push({name:d,
f:e,unbind:b(this.node||document,d,e,f||this)}));return this};a["un"+d]=A["un"+d]=function(a){for(var b=this.events||[],e=b.length;e--;)if(b[e].name==d&&(b[e].f==a||!a)){b[e].unbind();b.splice(e,1);!b.length&&delete this.events;break}return this}})(v[y]);A.hover=function(a,b,d,e){return this.mouseover(a,d).mouseout(b,e||d)};A.unhover=function(a,b){return this.unmouseover(a).unmouseout(b)};var r=[];A.drag=function(b,d,f,h,n,p){function u(r,v,w){(r.originalEvent||r).preventDefault();this._drag.x=v;
this._drag.y=w;this._drag.id=r.identifier;!q.length&&a.mousemove(e).mouseup(l);q.push({el:this,move_scope:h,start_scope:n,end_scope:p});d&&k.on("snap.drag.start."+this.id,d);b&&k.on("snap.drag.move."+this.id,b);f&&k.on("snap.drag.end."+this.id,f);k("snap.drag.start."+this.id,n||h||this,v,w,r)}if(!arguments.length){var v;return this.drag(function(a,b){this.attr({transform:v+(v?"T":"t")+[a,b]})},function(){v=this.transform().local})}this._drag={};r.push({el:this,start:u});this.mousedown(u);return this};
A.undrag=function(){for(var b=r.length;b--;)r[b].el==this&&(this.unmousedown(r[b].start),r.splice(b,1),k.unbind("snap.drag.*."+this.id));!r.length&&a.unmousemove(e).unmouseup(l);return this}});C.plugin(function(a,v,y,C){y=y.prototype;var A=/^\s*url\((.+)\)/,w=String,z=a._.$;a.filter={};y.filter=function(d){var f=this;"svg"!=f.type&&(f=f.paper);d=a.parse(w(d));var k=a._.id(),u=z("filter");z(u,{id:k,filterUnits:"userSpaceOnUse"});u.appendChild(d.node);f.defs.appendChild(u);return new v(u)};k.on("snap.util.getattr.filter",
function(){k.stop();var d=z(this.node,"filter");if(d)return(d=w(d).match(A))&&a.select(d[1])});k.on("snap.util.attr.filter",function(d){if(d instanceof v&&"filter"==d.type){k.stop();var f=d.node.id;f||(z(d.node,{id:d.id}),f=d.id);z(this.node,{filter:a.url(f)})}d&&"none"!=d||(k.stop(),this.node.removeAttribute("filter"))});a.filter.blur=function(d,f){null==d&&(d=2);return a.format('<feGaussianBlur stdDeviation="{def}"/>',{def:null==f?d:[d,f]})};a.filter.blur.toString=function(){return this()};a.filter.shadow=
function(d,f,k,u,p){"string"==typeof k&&(p=u=k,k=4);"string"!=typeof u&&(p=u,u="#000");null==k&&(k=4);null==p&&(p=1);null==d&&(d=0,f=2);null==f&&(f=d);u=a.color(u||"#000");return a.format('<feGaussianBlur in="SourceAlpha" stdDeviation="{blur}"/><feOffset dx="{dx}" dy="{dy}" result="offsetblur"/><feFlood flood-color="{color}"/><feComposite in2="offsetblur" operator="in"/><feComponentTransfer><feFuncA type="linear" slope="{opacity}"/></feComponentTransfer><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>',
{color:u,dx:d,dy:f,blur:k,opacity:p})};a.filter.shadow.toString=function(){return this()};a.filter.grayscale=function(d){null==d&&(d=1);return a.format('<feColorMatrix type="matrix" values="{a} {b} {c} 0 0 {d} {e} {f} 0 0 {g} {b} {h} 0 0 0 0 0 1 0"/>',{a:0.2126+0.7874*(1-d),b:0.7152-0.7152*(1-d),c:0.0722-0.0722*(1-d),d:0.2126-0.2126*(1-d),e:0.7152+0.2848*(1-d),f:0.0722-0.0722*(1-d),g:0.2126-0.2126*(1-d),h:0.0722+0.9278*(1-d)})};a.filter.grayscale.toString=function(){return this()};a.filter.sepia=
function(d){null==d&&(d=1);return a.format('<feColorMatrix type="matrix" values="{a} {b} {c} 0 0 {d} {e} {f} 0 0 {g} {h} {i} 0 0 0 0 0 1 0"/>',{a:0.393+0.607*(1-d),b:0.769-0.769*(1-d),c:0.189-0.189*(1-d),d:0.349-0.349*(1-d),e:0.686+0.314*(1-d),f:0.168-0.168*(1-d),g:0.272-0.272*(1-d),h:0.534-0.534*(1-d),i:0.131+0.869*(1-d)})};a.filter.sepia.toString=function(){return this()};a.filter.saturate=function(d){null==d&&(d=1);return a.format('<feColorMatrix type="saturate" values="{amount}"/>',{amount:1-
d})};a.filter.saturate.toString=function(){return this()};a.filter.hueRotate=function(d){return a.format('<feColorMatrix type="hueRotate" values="{angle}"/>',{angle:d||0})};a.filter.hueRotate.toString=function(){return this()};a.filter.invert=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="table" tableValues="{amount} {amount2}"/><feFuncG type="table" tableValues="{amount} {amount2}"/><feFuncB type="table" tableValues="{amount} {amount2}"/></feComponentTransfer>',{amount:d,
amount2:1-d})};a.filter.invert.toString=function(){return this()};a.filter.brightness=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="linear" slope="{amount}"/><feFuncG type="linear" slope="{amount}"/><feFuncB type="linear" slope="{amount}"/></feComponentTransfer>',{amount:d})};a.filter.brightness.toString=function(){return this()};a.filter.contrast=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="linear" slope="{amount}" intercept="{amount2}"/><feFuncG type="linear" slope="{amount}" intercept="{amount2}"/><feFuncB type="linear" slope="{amount}" intercept="{amount2}"/></feComponentTransfer>',
{amount:d,amount2:0.5-d/2})};a.filter.contrast.toString=function(){return this()}});return C});

]]> </script>
<script> <![CDATA[

(function (glob, factory) {
    // AMD support
    if (typeof define === "function" && define.amd) {
        // Define as an anonymous module
        define("Gadfly", ["Snap.svg"], function (Snap) {
            return factory(Snap);
        });
    } else {
        // Browser globals (glob is window)
        // Snap adds itself to window
        glob.Gadfly = factory(glob.Snap);
    }
}(this, function (Snap) {

var Gadfly = {};

// Get an x/y coordinate value in pixels
var xPX = function(fig, x) {
    var client_box = fig.node.getBoundingClientRect();
    return x * fig.node.viewBox.baseVal.width / client_box.width;
};

var yPX = function(fig, y) {
    var client_box = fig.node.getBoundingClientRect();
    return y * fig.node.viewBox.baseVal.height / client_box.height;
};


Snap.plugin(function (Snap, Element, Paper, global) {
    // Traverse upwards from a snap element to find and return the first
    // note with the "plotroot" class.
    Element.prototype.plotroot = function () {
        var element = this;
        while (!element.hasClass("plotroot") && element.parent() != null) {
            element = element.parent();
        }
        return element;
    };

    Element.prototype.svgroot = function () {
        var element = this;
        while (element.node.nodeName != "svg" && element.parent() != null) {
            element = element.parent();
        }
        return element;
    };

    Element.prototype.plotbounds = function () {
        var root = this.plotroot()
        var bbox = root.select(".guide.background").node.getBBox();
        return {
            x0: bbox.x,
            x1: bbox.x + bbox.width,
            y0: bbox.y,
            y1: bbox.y + bbox.height
        };
    };

    Element.prototype.plotcenter = function () {
        var root = this.plotroot()
        var bbox = root.select(".guide.background").node.getBBox();
        return {
            x: bbox.x + bbox.width / 2,
            y: bbox.y + bbox.height / 2
        };
    };

    // Emulate IE style mouseenter/mouseleave events, since Microsoft always
    // does everything right.
    // See: http://www.dynamic-tools.net/toolbox/isMouseLeaveOrEnter/
    var events = ["mouseenter", "mouseleave"];

    for (i in events) {
        (function (event_name) {
            var event_name = events[i];
            Element.prototype[event_name] = function (fn, scope) {
                if (Snap.is(fn, "function")) {
                    var fn2 = function (event) {
                        if (event.type != "mouseover" && event.type != "mouseout") {
                            return;
                        }

                        var reltg = event.relatedTarget ? event.relatedTarget :
                            event.type == "mouseout" ? event.toElement : event.fromElement;
                        while (reltg && reltg != this.node) reltg = reltg.parentNode;

                        if (reltg != this.node) {
                            return fn.apply(this, event);
                        }
                    };

                    if (event_name == "mouseenter") {
                        this.mouseover(fn2, scope);
                    } else {
                        this.mouseout(fn2, scope);
                    }
                }
                return this;
            };
        })(events[i]);
    }


    Element.prototype.mousewheel = function (fn, scope) {
        if (Snap.is(fn, "function")) {
            var el = this;
            var fn2 = function (event) {
                fn.apply(el, [event]);
            };
        }

        this.node.addEventListener(
            /Firefox/i.test(navigator.userAgent) ? "DOMMouseScroll" : "mousewheel",
            fn2);

        return this;
    };


    // Snap's attr function can be too slow for things like panning/zooming.
    // This is a function to directly update element attributes without going
    // through eve.
    Element.prototype.attribute = function(key, val) {
        if (val === undefined) {
            return this.node.getAttribute(key);
        } else {
            this.node.setAttribute(key, val);
            return this;
        }
    };
});


// When the plot is moused over, emphasize the grid lines.
Gadfly.plot_mouseover = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);

    var xgridlines = root.select(".xgridlines"),
        ygridlines = root.select(".ygridlines");

    xgridlines.data("unfocused_strokedash",
                    xgridlines.attribute("stroke-dasharray").replace(/(\d)(,|$)/g, "$1mm$2"));
    ygridlines.data("unfocused_strokedash",
                    ygridlines.attribute("stroke-dasharray").replace(/(\d)(,|$)/g, "$1mm$2"));

    // emphasize grid lines
    var destcolor = root.data("focused_xgrid_color");
    xgridlines.attribute("stroke-dasharray", "none")
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    destcolor = root.data("focused_ygrid_color");
    ygridlines.attribute("stroke-dasharray", "none")
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    // reveal zoom slider
    root.select(".zoomslider")
        .animate({opacity: 1.0}, 250);
};


// Unemphasize grid lines on mouse out.
Gadfly.plot_mouseout = function(event) {
    var root = this.plotroot();
    var xgridlines = root.select(".xgridlines"),
        ygridlines = root.select(".ygridlines");

    var destcolor = root.data("unfocused_xgrid_color");

    xgridlines.attribute("stroke-dasharray", xgridlines.data("unfocused_strokedash"))
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    destcolor = root.data("unfocused_ygrid_color");
    ygridlines.attribute("stroke-dasharray", ygridlines.data("unfocused_strokedash"))
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    // hide zoom slider
    root.select(".zoomslider")
        .animate({opacity: 0.0}, 250);
};


var set_geometry_transform = function(root, tx, ty, scale) {
    var xscalable = root.hasClass("xscalable"),
        yscalable = root.hasClass("yscalable");

    var old_scale = root.data("scale");

    var xscale = xscalable ? scale : 1.0,
        yscale = yscalable ? scale : 1.0;

    tx = xscalable ? tx : 0.0;
    ty = yscalable ? ty : 0.0;

    var t = new Snap.Matrix().translate(tx, ty).scale(xscale, yscale);

    root.selectAll(".geometry, image")
        .forEach(function (element, i) {
            element.transform(t);
        });

    bounds = root.plotbounds();

    if (yscalable) {
        var xfixed_t = new Snap.Matrix().translate(0, ty).scale(1.0, yscale);
        root.selectAll(".xfixed")
            .forEach(function (element, i) {
                element.transform(xfixed_t);
            });

        root.select(".ylabels")
            .transform(xfixed_t)
            .selectAll("text")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var cx = element.asPX("x"),
                        cy = element.asPX("y");
                    var st = element.data("static_transform");
                    unscale_t = new Snap.Matrix();
                    unscale_t.scale(1, 1/scale, cx, cy).add(st);
                    element.transform(unscale_t);

                    var y = cy * scale + ty;
                    element.attr("visibility",
                        bounds.y0 <= y && y <= bounds.y1 ? "visible" : "hidden");
                }
            });
    }

    if (xscalable) {
        var yfixed_t = new Snap.Matrix().translate(tx, 0).scale(xscale, 1.0);
        var xtrans = new Snap.Matrix().translate(tx, 0);
        root.selectAll(".yfixed")
            .forEach(function (element, i) {
                element.transform(yfixed_t);
            });

        root.select(".xlabels")
            .transform(yfixed_t)
            .selectAll("text")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var cx = element.asPX("x"),
                        cy = element.asPX("y");
                    var st = element.data("static_transform");
                    unscale_t = new Snap.Matrix();
                    unscale_t.scale(1/scale, 1, cx, cy).add(st);

                    element.transform(unscale_t);

                    var x = cx * scale + tx;
                    element.attr("visibility",
                        bounds.x0 <= x && x <= bounds.x1 ? "visible" : "hidden");
                    }
            });
    }

    // we must unscale anything that is scale invariance: widths, raiduses, etc.
    var size_attribs = ["font-size"];
    var unscaled_selection = ".geometry, .geometry *";
    if (xscalable) {
        size_attribs.push("rx");
        unscaled_selection += ", .xgridlines";
    }
    if (yscalable) {
        size_attribs.push("ry");
        unscaled_selection += ", .ygridlines";
    }

    root.selectAll(unscaled_selection)
        .forEach(function (element, i) {
            // circle need special help
            if (element.node.nodeName == "circle") {
                var cx = element.attribute("cx"),
                    cy = element.attribute("cy");
                unscale_t = new Snap.Matrix().scale(1/xscale, 1/yscale,
                                                        cx, cy);
                element.transform(unscale_t);
                return;
            }

            for (i in size_attribs) {
                var key = size_attribs[i];
                var val = parseFloat(element.attribute(key));
                if (val !== undefined && val != 0 && !isNaN(val)) {
                    element.attribute(key, val * old_scale / scale);
                }
            }
        });
};


// Find the most appropriate tick scale and update label visibility.
var update_tickscale = function(root, scale, axis) {
    if (!root.hasClass(axis + "scalable")) return;

    var tickscales = root.data(axis + "tickscales");
    var best_tickscale = 1.0;
    var best_tickscale_dist = Infinity;
    for (tickscale in tickscales) {
        var dist = Math.abs(Math.log(tickscale) - Math.log(scale));
        if (dist < best_tickscale_dist) {
            best_tickscale_dist = dist;
            best_tickscale = tickscale;
        }
    }

    if (best_tickscale != root.data(axis + "tickscale")) {
        root.data(axis + "tickscale", best_tickscale);
        var mark_inscale_gridlines = function (element, i) {
            var inscale = element.attr("gadfly:scale") == best_tickscale;
            element.attribute("gadfly:inscale", inscale);
            element.attr("visibility", inscale ? "visible" : "hidden");
        };

        var mark_inscale_labels = function (element, i) {
            var inscale = element.attr("gadfly:scale") == best_tickscale;
            element.attribute("gadfly:inscale", inscale);
            element.attr("visibility", inscale ? "visible" : "hidden");
        };

        root.select("." + axis + "gridlines").selectAll("path").forEach(mark_inscale_gridlines);
        root.select("." + axis + "labels").selectAll("text").forEach(mark_inscale_labels);
    }
};


var set_plot_pan_zoom = function(root, tx, ty, scale) {
    var old_scale = root.data("scale");
    var bounds = root.plotbounds();

    var width = bounds.x1 - bounds.x0,
        height = bounds.y1 - bounds.y0;

    // compute the viewport derived from tx, ty, and scale
    var x_min = -width * scale - (scale * width - width),
        x_max = width * scale,
        y_min = -height * scale - (scale * height - height),
        y_max = height * scale;

    var x0 = bounds.x0 - scale * bounds.x0,
        y0 = bounds.y0 - scale * bounds.y0;

    var tx = Math.max(Math.min(tx - x0, x_max), x_min),
        ty = Math.max(Math.min(ty - y0, y_max), y_min);

    tx += x0;
    ty += y0;

    // when the scale change, we may need to alter which set of
    // ticks is being displayed
    if (scale != old_scale) {
        update_tickscale(root, scale, "x");
        update_tickscale(root, scale, "y");
    }

    set_geometry_transform(root, tx, ty, scale);

    root.data("scale", scale);
    root.data("tx", tx);
    root.data("ty", ty);
};


var scale_centered_translation = function(root, scale) {
    var bounds = root.plotbounds();

    var width = bounds.x1 - bounds.x0,
        height = bounds.y1 - bounds.y0;

    var tx0 = root.data("tx"),
        ty0 = root.data("ty");

    var scale0 = root.data("scale");

    // how off from center the current view is
    var xoff = tx0 - (bounds.x0 * (1 - scale0) + (width * (1 - scale0)) / 2),
        yoff = ty0 - (bounds.y0 * (1 - scale0) + (height * (1 - scale0)) / 2);

    // rescale offsets
    xoff = xoff * scale / scale0;
    yoff = yoff * scale / scale0;

    // adjust for the panel position being scaled
    var x_edge_adjust = bounds.x0 * (1 - scale),
        y_edge_adjust = bounds.y0 * (1 - scale);

    return {
        x: xoff + x_edge_adjust + (width - width * scale) / 2,
        y: yoff + y_edge_adjust + (height - height * scale) / 2
    };
};


// Initialize data for panning zooming if it isn't already.
var init_pan_zoom = function(root) {
    if (root.data("zoompan-ready")) {
        return;
    }

    // The non-scaling-stroke trick. Rather than try to correct for the
    // stroke-width when zooming, we force it to a fixed value.
    var px_per_mm = root.node.getCTM().a;

    // Drag events report deltas in pixels, which we'd like to convert to
    // millimeters.
    root.data("px_per_mm", px_per_mm);

    root.selectAll("path")
        .forEach(function (element, i) {
        sw = element.asPX("stroke-width") * px_per_mm;
        if (sw > 0) {
            element.attribute("stroke-width", sw);
            element.attribute("vector-effect", "non-scaling-stroke");
        }
    });

    // Store ticks labels original tranformation
    root.selectAll(".xlabels > text, .ylabels > text")
        .forEach(function (element, i) {
            var lm = element.transform().localMatrix;
            element.data("static_transform",
                new Snap.Matrix(lm.a, lm.b, lm.c, lm.d, lm.e, lm.f));
        });

    var xgridlines = root.select(".xgridlines");
    var ygridlines = root.select(".ygridlines");
    var xlabels = root.select(".xlabels");
    var ylabels = root.select(".ylabels");

    if (root.data("tx") === undefined) root.data("tx", 0);
    if (root.data("ty") === undefined) root.data("ty", 0);
    if (root.data("scale") === undefined) root.data("scale", 1.0);
    if (root.data("xtickscales") === undefined) {

        // index all the tick scales that are listed
        var xtickscales = {};
        var ytickscales = {};
        var add_x_tick_scales = function (element, i) {
            xtickscales[element.attribute("gadfly:scale")] = true;
        };
        var add_y_tick_scales = function (element, i) {
            ytickscales[element.attribute("gadfly:scale")] = true;
        };

        if (xgridlines) xgridlines.selectAll("path").forEach(add_x_tick_scales);
        if (ygridlines) ygridlines.selectAll("path").forEach(add_y_tick_scales);
        if (xlabels) xlabels.selectAll("text").forEach(add_x_tick_scales);
        if (ylabels) ylabels.selectAll("text").forEach(add_y_tick_scales);

        root.data("xtickscales", xtickscales);
        root.data("ytickscales", ytickscales);
        root.data("xtickscale", 1.0);
    }

    var min_scale = 1.0, max_scale = 1.0;
    for (scale in xtickscales) {
        min_scale = Math.min(min_scale, scale);
        max_scale = Math.max(max_scale, scale);
    }
    for (scale in ytickscales) {
        min_scale = Math.min(min_scale, scale);
        max_scale = Math.max(max_scale, scale);
    }
    root.data("min_scale", min_scale);
    root.data("max_scale", max_scale);

    // store the original positions of labels
    if (xlabels) {
        xlabels.selectAll("text")
               .forEach(function (element, i) {
                   element.data("x", element.asPX("x"));
               });
    }

    if (ylabels) {
        ylabels.selectAll("text")
               .forEach(function (element, i) {
                   element.data("y", element.asPX("y"));
               });
    }

    // mark grid lines and ticks as in or out of scale.
    var mark_inscale = function (element, i) {
        element.attribute("gadfly:inscale", element.attribute("gadfly:scale") == 1.0);
    };

    if (xgridlines) xgridlines.selectAll("path").forEach(mark_inscale);
    if (ygridlines) ygridlines.selectAll("path").forEach(mark_inscale);
    if (xlabels) xlabels.selectAll("text").forEach(mark_inscale);
    if (ylabels) ylabels.selectAll("text").forEach(mark_inscale);

    // figure out the upper ond lower bounds on panning using the maximum
    // and minum grid lines
    var bounds = root.plotbounds();
    var pan_bounds = {
        x0: 0.0,
        y0: 0.0,
        x1: 0.0,
        y1: 0.0
    };

    if (xgridlines) {
        xgridlines
            .selectAll("path")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var bbox = element.node.getBBox();
                    if (bounds.x1 - bbox.x < pan_bounds.x0) {
                        pan_bounds.x0 = bounds.x1 - bbox.x;
                    }
                    if (bounds.x0 - bbox.x > pan_bounds.x1) {
                        pan_bounds.x1 = bounds.x0 - bbox.x;
                    }
                    element.attr("visibility", "visible");
                }
            });
    }

    if (ygridlines) {
        ygridlines
            .selectAll("path")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var bbox = element.node.getBBox();
                    if (bounds.y1 - bbox.y < pan_bounds.y0) {
                        pan_bounds.y0 = bounds.y1 - bbox.y;
                    }
                    if (bounds.y0 - bbox.y > pan_bounds.y1) {
                        pan_bounds.y1 = bounds.y0 - bbox.y;
                    }
                    element.attr("visibility", "visible");
                }
            });
    }

    // nudge these values a little
    pan_bounds.x0 -= 5;
    pan_bounds.x1 += 5;
    pan_bounds.y0 -= 5;
    pan_bounds.y1 += 5;
    root.data("pan_bounds", pan_bounds);

    root.data("zoompan-ready", true)
};


// Panning
Gadfly.guide_background_drag_onmove = function(dx, dy, x, y, event) {
    var root = this.plotroot();
    var px_per_mm = root.data("px_per_mm");
    dx /= px_per_mm;
    dy /= px_per_mm;

    var tx0 = root.data("tx"),
        ty0 = root.data("ty");

    var dx0 = root.data("dx"),
        dy0 = root.data("dy");

    root.data("dx", dx);
    root.data("dy", dy);

    dx = dx - dx0;
    dy = dy - dy0;

    var tx = tx0 + dx,
        ty = ty0 + dy;

    set_plot_pan_zoom(root, tx, ty, root.data("scale"));
};


Gadfly.guide_background_drag_onstart = function(x, y, event) {
    var root = this.plotroot();
    root.data("dx", 0);
    root.data("dy", 0);
    init_pan_zoom(root);
};


Gadfly.guide_background_drag_onend = function(event) {
    var root = this.plotroot();
};


Gadfly.guide_background_scroll = function(event) {
    if (event.shiftKey) {
        var root = this.plotroot();
        init_pan_zoom(root);
        var new_scale = root.data("scale") * Math.pow(2, 0.002 * event.wheelDelta);
        new_scale = Math.max(
            root.data("min_scale"),
            Math.min(root.data("max_scale"), new_scale))
        update_plot_scale(root, new_scale);
        event.stopPropagation();
    }
};


Gadfly.zoomslider_button_mouseover = function(event) {
    this.select(".button_logo")
         .animate({fill: this.data("mouseover_color")}, 100);
};


Gadfly.zoomslider_button_mouseout = function(event) {
     this.select(".button_logo")
         .animate({fill: this.data("mouseout_color")}, 100);
};


Gadfly.zoomslider_zoomout_click = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);
    var min_scale = root.data("min_scale"),
        scale = root.data("scale");
    Snap.animate(
        scale,
        Math.max(min_scale, scale / 1.5),
        function (new_scale) {
            update_plot_scale(root, new_scale);
        },
        200);
};


Gadfly.zoomslider_zoomin_click = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);
    var max_scale = root.data("max_scale"),
        scale = root.data("scale");

    Snap.animate(
        scale,
        Math.min(max_scale, scale * 1.5),
        function (new_scale) {
            update_plot_scale(root, new_scale);
        },
        200);
};


Gadfly.zoomslider_track_click = function(event) {
    // TODO
};


Gadfly.zoomslider_thumb_mousedown = function(event) {
    this.animate({fill: this.data("mouseover_color")}, 100);
};


Gadfly.zoomslider_thumb_mouseup = function(event) {
    this.animate({fill: this.data("mouseout_color")}, 100);
};


// compute the position in [0, 1] of the zoom slider thumb from the current scale
var slider_position_from_scale = function(scale, min_scale, max_scale) {
    if (scale >= 1.0) {
        return 0.5 + 0.5 * (Math.log(scale) / Math.log(max_scale));
    }
    else {
        return 0.5 * (Math.log(scale) - Math.log(min_scale)) / (0 - Math.log(min_scale));
    }
}


var update_plot_scale = function(root, new_scale) {
    var trans = scale_centered_translation(root, new_scale);
    set_plot_pan_zoom(root, trans.x, trans.y, new_scale);

    root.selectAll(".zoomslider_thumb")
        .forEach(function (element, i) {
            var min_pos = element.data("min_pos"),
                max_pos = element.data("max_pos"),
                min_scale = root.data("min_scale"),
                max_scale = root.data("max_scale");
            var xmid = (min_pos + max_pos) / 2;
            var xpos = slider_position_from_scale(new_scale, min_scale, max_scale);
            element.transform(new Snap.Matrix().translate(
                Math.max(min_pos, Math.min(
                         max_pos, min_pos + (max_pos - min_pos) * xpos)) - xmid, 0));
    });
};


Gadfly.zoomslider_thumb_dragmove = function(dx, dy, x, y) {
    var root = this.plotroot();
    var min_pos = this.data("min_pos"),
        max_pos = this.data("max_pos"),
        min_scale = root.data("min_scale"),
        max_scale = root.data("max_scale"),
        old_scale = root.data("old_scale");

    var px_per_mm = root.data("px_per_mm");
    dx /= px_per_mm;
    dy /= px_per_mm;

    var xmid = (min_pos + max_pos) / 2;
    var xpos = slider_position_from_scale(old_scale, min_scale, max_scale) +
                   dx / (max_pos - min_pos);

    // compute the new scale
    var new_scale;
    if (xpos >= 0.5) {
        new_scale = Math.exp(2.0 * (xpos - 0.5) * Math.log(max_scale));
    }
    else {
        new_scale = Math.exp(2.0 * xpos * (0 - Math.log(min_scale)) +
                        Math.log(min_scale));
    }
    new_scale = Math.min(max_scale, Math.max(min_scale, new_scale));

    update_plot_scale(root, new_scale);
};


Gadfly.zoomslider_thumb_dragstart = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);

    // keep track of what the scale was when we started dragging
    root.data("old_scale", root.data("scale"));
};


Gadfly.zoomslider_thumb_dragend = function(event) {
};


var toggle_color_class = function(root, color_class, ison) {
    var guides = root.selectAll(".guide." + color_class + ",.guide ." + color_class);
    var geoms = root.selectAll(".geometry." + color_class + ",.geometry ." + color_class);
    if (ison) {
        guides.animate({opacity: 0.5}, 250);
        geoms.animate({opacity: 0.0}, 250);
    } else {
        guides.animate({opacity: 1.0}, 250);
        geoms.animate({opacity: 1.0}, 250);
    }
};


Gadfly.colorkey_swatch_click = function(event) {
    var root = this.plotroot();
    var color_class = this.data("color_class");

    if (event.shiftKey) {
        root.selectAll(".colorkey text")
            .forEach(function (element) {
                var other_color_class = element.data("color_class");
                if (other_color_class != color_class) {
                    toggle_color_class(root, other_color_class,
                                       element.attr("opacity") == 1.0);
                }
            });
    } else {
        toggle_color_class(root, color_class, this.attr("opacity") == 1.0);
    }
};


return Gadfly;

}));


//@ sourceURL=gadfly.js

(function (glob, factory) {
    // AMD support
      if (typeof require === "function" && typeof define === "function" && define.amd) {
        require(["Snap.svg", "Gadfly"], function (Snap, Gadfly) {
            factory(Snap, Gadfly);
        });
      } else {
          factory(glob.Snap, glob.Gadfly);
      }
})(window, function (Snap, Gadfly) {
    var fig = Snap("#fig-f3ea444778014fbf80c9634001e4f6c2");
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-4")
   .drag(function() {}, function() {}, function() {});
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-6")
   .data("color_class", "color_0")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-7")
   .data("color_class", "color_1")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-9")
   .data("color_class", "color_0")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-10")
   .data("color_class", "color_1")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-13")
   .mouseenter(Gadfly.plot_mouseover)
.mouseleave(Gadfly.plot_mouseout)
.mousewheel(Gadfly.guide_background_scroll)
.drag(Gadfly.guide_background_drag_onmove,
      Gadfly.guide_background_drag_onstart,
      Gadfly.guide_background_drag_onend)
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-17")
   .plotroot().data("unfocused_ygrid_color", "#D0D0E0")
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-17")
   .plotroot().data("focused_ygrid_color", "#A0A0A0")
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-132")
   .plotroot().data("unfocused_xgrid_color", "#D0D0E0")
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-132")
   .plotroot().data("focused_xgrid_color", "#A0A0A0")
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-296")
   .data("mouseover_color", "#cd5c5c")
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-296")
   .data("mouseout_color", "#6a6a6a")
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-296")
   .click(Gadfly.zoomslider_zoomin_click)
.mouseenter(Gadfly.zoomslider_button_mouseover)
.mouseleave(Gadfly.zoomslider_button_mouseout)
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-300")
   .data("max_pos", 106.11)
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-300")
   .data("min_pos", 89.11)
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-300")
   .click(Gadfly.zoomslider_track_click);
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-302")
   .data("max_pos", 106.11)
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-302")
   .data("min_pos", 89.11)
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-302")
   .data("mouseover_color", "#cd5c5c")
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-302")
   .data("mouseout_color", "#6a6a6a")
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-302")
   .drag(Gadfly.zoomslider_thumb_dragmove,
     Gadfly.zoomslider_thumb_dragstart,
     Gadfly.zoomslider_thumb_dragend)
.mousedown(Gadfly.zoomslider_thumb_mousedown)
.mouseup(Gadfly.zoomslider_thumb_mouseup)
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-304")
   .data("mouseover_color", "#cd5c5c")
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-304")
   .data("mouseout_color", "#6a6a6a")
;
fig.select("#fig-f3ea444778014fbf80c9634001e4f6c2-element-304")
   .click(Gadfly.zoomslider_zoomout_click)
.mouseenter(Gadfly.zoomslider_button_mouseover)
.mouseleave(Gadfly.zoomslider_button_mouseout)
;
    });
]]> </script>
</svg>




We see the number of Children survived is lower than the number of victims.

Let try to combine the gender and child variables to see how that impacts the survival rate. We go back to our friendly bug Gadfly to construct a stacked histogram with subplots.


    plot(train, xgroup="Child", x="Sex", y="Survived", color="Survived", Geom.subplot_grid(Geom.histogram(position=:stack)), Scale.color_discrete_manual("red","green"))




<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:gadfly="http://www.gadflyjl.org/ns"
     version="1.2"
     width="141.42mm" height="100mm" viewBox="0 0 141.42 100"
     stroke="none"
     fill="#000000"
     stroke-width="0.3"
     font-size="3.88"

     id="fig-e97312c08242438abf7e68ad37f1e090">
<g class="plotroot yscalable" id="fig-e97312c08242438abf7e68ad37f1e090-element-1">
  <g font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" fill="#564A55" stroke="#000000" stroke-opacity="0.000" id="fig-e97312c08242438abf7e68ad37f1e090-element-2">
    <text x="67.36" y="88.39" text-anchor="middle" dy="0.6em">Sex <tspan style="dominant-baseline:inherit" font-style="italic"><tspan style="dominant-baseline:inherit" font-weight="bold">by</tspan></tspan> Child</text>
  </g>
  <g class="guide colorkey" id="fig-e97312c08242438abf7e68ad37f1e090-element-3">
    <g fill="#4C404B" font-size="2.82" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" id="fig-e97312c08242438abf7e68ad37f1e090-element-4">
      <text x="125.93" y="45.19" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-5" class="color_0">0</text>
      <text x="125.93" y="48.82" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-6" class="color_1">1</text>
    </g>
    <g stroke="#000000" stroke-opacity="0.000" id="fig-e97312c08242438abf7e68ad37f1e090-element-7">
      <rect x="123.11" y="44.29" width="1.81" height="1.81" id="fig-e97312c08242438abf7e68ad37f1e090-element-8" class="color_0" fill="#FF0000"/>
      <rect x="123.11" y="47.91" width="1.81" height="1.81" id="fig-e97312c08242438abf7e68ad37f1e090-element-9" class="color_1" fill="#008000"/>
    </g>
    <g fill="#362A35" font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" stroke="#000000" stroke-opacity="0.000" id="fig-e97312c08242438abf7e68ad37f1e090-element-10">
      <text x="123.11" y="41.37" id="fig-e97312c08242438abf7e68ad37f1e090-element-11">Survived</text>
    </g>
  </g>
  <g clip-path="url(#fig-e97312c08242438abf7e68ad37f1e090-element-13)" id="fig-e97312c08242438abf7e68ad37f1e090-element-12">
    <g class="plotpanel" id="fig-e97312c08242438abf7e68ad37f1e090-element-14">
      <g font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" fill="#564A55" stroke="#000000" stroke-opacity="0.000" id="fig-e97312c08242438abf7e68ad37f1e090-element-15">
        <text x="96.49" y="80.27" text-anchor="end" dy="0.35em" transform="rotate(-90, 96.49, 80.27)" id="fig-e97312c08242438abf7e68ad37f1e090-element-16">1</text>
      </g>
      <g font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" fill="#564A55" stroke="#000000" stroke-opacity="0.000" id="fig-e97312c08242438abf7e68ad37f1e090-element-17">
        <text x="44.25" y="80.27" text-anchor="end" dy="0.35em" transform="rotate(-90, 44.25, 80.27)" id="fig-e97312c08242438abf7e68ad37f1e090-element-18">0</text>
      </g>
      <g class="guide xlabels" font-size="2.82" font-family="'PT Sans Caption','Helvetica Neue','Helvetica',sans-serif" fill="#6C606B" id="fig-e97312c08242438abf7e68ad37f1e090-element-19">
        <text x="84.68" y="76.27" text-anchor="middle" id="fig-e97312c08242438abf7e68ad37f1e090-element-20" visibility="visible" gadfly:scale="1.0">male</text>
        <text x="108.3" y="76.27" text-anchor="middle" id="fig-e97312c08242438abf7e68ad37f1e090-element-21" visibility="visible" gadfly:scale="1.0">female</text>
      </g>
      <g class="guide xlabels" font-size="2.82" font-family="'PT Sans Caption','Helvetica Neue','Helvetica',sans-serif" fill="#6C606B" id="fig-e97312c08242438abf7e68ad37f1e090-element-22">
        <text x="31.94" y="76.27" text-anchor="middle" id="fig-e97312c08242438abf7e68ad37f1e090-element-23" visibility="visible" gadfly:scale="1.0">male</text>
        <text x="56.56" y="76.27" text-anchor="middle" id="fig-e97312c08242438abf7e68ad37f1e090-element-24" visibility="visible" gadfly:scale="1.0">female</text>
      </g>
      <g clip-path="url(#fig-e97312c08242438abf7e68ad37f1e090-element-26)" id="fig-e97312c08242438abf7e68ad37f1e090-element-25">
        <g pointer-events="visible" opacity="1" fill="#000000" fill-opacity="0.000" stroke="#000000" stroke-opacity="0.000" class="guide background" id="fig-e97312c08242438abf7e68ad37f1e090-element-27">
          <rect x="72.87" y="7" width="47.24" height="65.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-28"/>
        </g>
        <g class="guide ygridlines xfixed" stroke-dasharray="0.5,0.5" stroke-width="0.2" stroke="#D0D0E0" id="fig-e97312c08242438abf7e68ad37f1e090-element-29">
          <path fill="none" d="M72.87,147.59 L 120.11 147.59" id="fig-e97312c08242438abf7e68ad37f1e090-element-30" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,132.19 L 120.11 132.19" id="fig-e97312c08242438abf7e68ad37f1e090-element-31" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,116.8 L 120.11 116.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-32" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,101.4 L 120.11 101.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-33" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,86 L 120.11 86" id="fig-e97312c08242438abf7e68ad37f1e090-element-34" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,70.6 L 120.11 70.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-35" visibility="visible" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,55.2 L 120.11 55.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-36" visibility="visible" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,39.8 L 120.11 39.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-37" visibility="visible" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,24.4 L 120.11 24.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-38" visibility="visible" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,9 L 120.11 9" id="fig-e97312c08242438abf7e68ad37f1e090-element-39" visibility="visible" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,-6.4 L 120.11 -6.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-40" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,-21.8 L 120.11 -21.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-41" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,-37.2 L 120.11 -37.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-42" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,-52.6 L 120.11 -52.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-43" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,-68 L 120.11 -68" id="fig-e97312c08242438abf7e68ad37f1e090-element-44" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M72.87,132.19 L 120.11 132.19" id="fig-e97312c08242438abf7e68ad37f1e090-element-45" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,129.12 L 120.11 129.12" id="fig-e97312c08242438abf7e68ad37f1e090-element-46" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,126.04 L 120.11 126.04" id="fig-e97312c08242438abf7e68ad37f1e090-element-47" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,122.96 L 120.11 122.96" id="fig-e97312c08242438abf7e68ad37f1e090-element-48" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,119.88 L 120.11 119.88" id="fig-e97312c08242438abf7e68ad37f1e090-element-49" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,116.8 L 120.11 116.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-50" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,113.72 L 120.11 113.72" id="fig-e97312c08242438abf7e68ad37f1e090-element-51" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,110.64 L 120.11 110.64" id="fig-e97312c08242438abf7e68ad37f1e090-element-52" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,107.56 L 120.11 107.56" id="fig-e97312c08242438abf7e68ad37f1e090-element-53" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,104.48 L 120.11 104.48" id="fig-e97312c08242438abf7e68ad37f1e090-element-54" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,101.4 L 120.11 101.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-55" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,98.32 L 120.11 98.32" id="fig-e97312c08242438abf7e68ad37f1e090-element-56" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,95.24 L 120.11 95.24" id="fig-e97312c08242438abf7e68ad37f1e090-element-57" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,92.16 L 120.11 92.16" id="fig-e97312c08242438abf7e68ad37f1e090-element-58" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,89.08 L 120.11 89.08" id="fig-e97312c08242438abf7e68ad37f1e090-element-59" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,86 L 120.11 86" id="fig-e97312c08242438abf7e68ad37f1e090-element-60" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,82.92 L 120.11 82.92" id="fig-e97312c08242438abf7e68ad37f1e090-element-61" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,79.84 L 120.11 79.84" id="fig-e97312c08242438abf7e68ad37f1e090-element-62" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,76.76 L 120.11 76.76" id="fig-e97312c08242438abf7e68ad37f1e090-element-63" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,73.68 L 120.11 73.68" id="fig-e97312c08242438abf7e68ad37f1e090-element-64" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,70.6 L 120.11 70.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-65" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,67.52 L 120.11 67.52" id="fig-e97312c08242438abf7e68ad37f1e090-element-66" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,64.44 L 120.11 64.44" id="fig-e97312c08242438abf7e68ad37f1e090-element-67" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,61.36 L 120.11 61.36" id="fig-e97312c08242438abf7e68ad37f1e090-element-68" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,58.28 L 120.11 58.28" id="fig-e97312c08242438abf7e68ad37f1e090-element-69" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,55.2 L 120.11 55.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-70" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,52.12 L 120.11 52.12" id="fig-e97312c08242438abf7e68ad37f1e090-element-71" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,49.04 L 120.11 49.04" id="fig-e97312c08242438abf7e68ad37f1e090-element-72" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,45.96 L 120.11 45.96" id="fig-e97312c08242438abf7e68ad37f1e090-element-73" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,42.88 L 120.11 42.88" id="fig-e97312c08242438abf7e68ad37f1e090-element-74" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,39.8 L 120.11 39.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-75" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,36.72 L 120.11 36.72" id="fig-e97312c08242438abf7e68ad37f1e090-element-76" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,33.64 L 120.11 33.64" id="fig-e97312c08242438abf7e68ad37f1e090-element-77" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,30.56 L 120.11 30.56" id="fig-e97312c08242438abf7e68ad37f1e090-element-78" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,27.48 L 120.11 27.48" id="fig-e97312c08242438abf7e68ad37f1e090-element-79" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,24.4 L 120.11 24.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-80" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,21.32 L 120.11 21.32" id="fig-e97312c08242438abf7e68ad37f1e090-element-81" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,18.24 L 120.11 18.24" id="fig-e97312c08242438abf7e68ad37f1e090-element-82" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,15.16 L 120.11 15.16" id="fig-e97312c08242438abf7e68ad37f1e090-element-83" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,12.08 L 120.11 12.08" id="fig-e97312c08242438abf7e68ad37f1e090-element-84" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,9 L 120.11 9" id="fig-e97312c08242438abf7e68ad37f1e090-element-85" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,5.92 L 120.11 5.92" id="fig-e97312c08242438abf7e68ad37f1e090-element-86" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,2.84 L 120.11 2.84" id="fig-e97312c08242438abf7e68ad37f1e090-element-87" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-0.24 L 120.11 -0.24" id="fig-e97312c08242438abf7e68ad37f1e090-element-88" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-3.32 L 120.11 -3.32" id="fig-e97312c08242438abf7e68ad37f1e090-element-89" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-6.4 L 120.11 -6.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-90" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-9.48 L 120.11 -9.48" id="fig-e97312c08242438abf7e68ad37f1e090-element-91" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-12.56 L 120.11 -12.56" id="fig-e97312c08242438abf7e68ad37f1e090-element-92" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-15.64 L 120.11 -15.64" id="fig-e97312c08242438abf7e68ad37f1e090-element-93" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-18.72 L 120.11 -18.72" id="fig-e97312c08242438abf7e68ad37f1e090-element-94" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-21.8 L 120.11 -21.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-95" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-24.88 L 120.11 -24.88" id="fig-e97312c08242438abf7e68ad37f1e090-element-96" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-27.96 L 120.11 -27.96" id="fig-e97312c08242438abf7e68ad37f1e090-element-97" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-31.04 L 120.11 -31.04" id="fig-e97312c08242438abf7e68ad37f1e090-element-98" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-34.12 L 120.11 -34.12" id="fig-e97312c08242438abf7e68ad37f1e090-element-99" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-37.2 L 120.11 -37.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-100" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-40.28 L 120.11 -40.28" id="fig-e97312c08242438abf7e68ad37f1e090-element-101" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-43.36 L 120.11 -43.36" id="fig-e97312c08242438abf7e68ad37f1e090-element-102" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-46.44 L 120.11 -46.44" id="fig-e97312c08242438abf7e68ad37f1e090-element-103" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-49.52 L 120.11 -49.52" id="fig-e97312c08242438abf7e68ad37f1e090-element-104" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,-52.6 L 120.11 -52.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-105" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M72.87,147.59 L 120.11 147.59" id="fig-e97312c08242438abf7e68ad37f1e090-element-106" visibility="hidden" gadfly:scale="0.5"/>
          <path fill="none" d="M72.87,70.6 L 120.11 70.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-107" visibility="hidden" gadfly:scale="0.5"/>
          <path fill="none" d="M72.87,-6.4 L 120.11 -6.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-108" visibility="hidden" gadfly:scale="0.5"/>
          <path fill="none" d="M72.87,-83.4 L 120.11 -83.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-109" visibility="hidden" gadfly:scale="0.5"/>
          <path fill="none" d="M72.87,132.19 L 120.11 132.19" id="fig-e97312c08242438abf7e68ad37f1e090-element-110" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,124.5 L 120.11 124.5" id="fig-e97312c08242438abf7e68ad37f1e090-element-111" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,116.8 L 120.11 116.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-112" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,109.1 L 120.11 109.1" id="fig-e97312c08242438abf7e68ad37f1e090-element-113" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,101.4 L 120.11 101.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-114" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,93.7 L 120.11 93.7" id="fig-e97312c08242438abf7e68ad37f1e090-element-115" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,86 L 120.11 86" id="fig-e97312c08242438abf7e68ad37f1e090-element-116" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,78.3 L 120.11 78.3" id="fig-e97312c08242438abf7e68ad37f1e090-element-117" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,70.6 L 120.11 70.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-118" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,62.9 L 120.11 62.9" id="fig-e97312c08242438abf7e68ad37f1e090-element-119" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,55.2 L 120.11 55.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-120" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,47.5 L 120.11 47.5" id="fig-e97312c08242438abf7e68ad37f1e090-element-121" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,39.8 L 120.11 39.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-122" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,32.1 L 120.11 32.1" id="fig-e97312c08242438abf7e68ad37f1e090-element-123" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,24.4 L 120.11 24.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-124" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,16.7 L 120.11 16.7" id="fig-e97312c08242438abf7e68ad37f1e090-element-125" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,9 L 120.11 9" id="fig-e97312c08242438abf7e68ad37f1e090-element-126" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,1.3 L 120.11 1.3" id="fig-e97312c08242438abf7e68ad37f1e090-element-127" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,-6.4 L 120.11 -6.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-128" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,-14.1 L 120.11 -14.1" id="fig-e97312c08242438abf7e68ad37f1e090-element-129" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,-21.8 L 120.11 -21.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-130" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,-29.5 L 120.11 -29.5" id="fig-e97312c08242438abf7e68ad37f1e090-element-131" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,-37.2 L 120.11 -37.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-132" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,-44.9 L 120.11 -44.9" id="fig-e97312c08242438abf7e68ad37f1e090-element-133" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M72.87,-52.6 L 120.11 -52.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-134" visibility="hidden" gadfly:scale="5.0"/>
        </g>
        <g class="guide xgridlines yfixed" stroke-dasharray="0.5,0.5" stroke-width="0.2" stroke="#D0D0E0" visibility="visible" id="fig-e97312c08242438abf7e68ad37f1e090-element-135">
          <path fill="none" d="M96.49,7 L 96.49 72.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-136" gadfly:scale="1.0"/>
        </g>
        <g class="plotpanel" id="fig-e97312c08242438abf7e68ad37f1e090-element-137">
          <g shape-rendering="crispEdges" stroke-width="0.3" id="fig-e97312c08242438abf7e68ad37f1e090-element-138">
            <g stroke="#000000" stroke-opacity="0.000" class="geometry" id="fig-e97312c08242438abf7e68ad37f1e090-element-139">
              <rect x="72.85" y="64.59" width="23.67" height="6.01" id="fig-e97312c08242438abf7e68ad37f1e090-element-140" fill="#008000"/>
              <rect x="96.47" y="59.2" width="23.67" height="11.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-141" fill="#008000"/>
              <rect x="72.85" y="42.57" width="23.67" height="22.02" id="fig-e97312c08242438abf7e68ad37f1e090-element-142" fill="#FF0000"/>
              <rect x="96.47" y="53.97" width="23.67" height="5.24" id="fig-e97312c08242438abf7e68ad37f1e090-element-143" fill="#FF0000"/>
            </g>
          </g>
        </g>
      </g>
      <g clip-path="url(#fig-e97312c08242438abf7e68ad37f1e090-element-145)" id="fig-e97312c08242438abf7e68ad37f1e090-element-144">
        <g pointer-events="visible" opacity="1" fill="#000000" fill-opacity="0.000" stroke="#000000" stroke-opacity="0.000" class="guide background" id="fig-e97312c08242438abf7e68ad37f1e090-element-146">
          <rect x="19.63" y="7" width="49.24" height="65.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-147"/>
        </g>
        <g class="guide ygridlines xfixed" stroke-dasharray="0.5,0.5" stroke-width="0.2" stroke="#D0D0E0" id="fig-e97312c08242438abf7e68ad37f1e090-element-148">
          <path fill="none" d="M19.63,147.59 L 68.87 147.59" id="fig-e97312c08242438abf7e68ad37f1e090-element-149" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,132.19 L 68.87 132.19" id="fig-e97312c08242438abf7e68ad37f1e090-element-150" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,116.8 L 68.87 116.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-151" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,101.4 L 68.87 101.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-152" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,86 L 68.87 86" id="fig-e97312c08242438abf7e68ad37f1e090-element-153" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,70.6 L 68.87 70.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-154" visibility="visible" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,55.2 L 68.87 55.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-155" visibility="visible" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,39.8 L 68.87 39.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-156" visibility="visible" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,24.4 L 68.87 24.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-157" visibility="visible" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,9 L 68.87 9" id="fig-e97312c08242438abf7e68ad37f1e090-element-158" visibility="visible" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,-6.4 L 68.87 -6.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-159" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,-21.8 L 68.87 -21.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-160" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,-37.2 L 68.87 -37.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-161" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,-52.6 L 68.87 -52.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-162" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,-68 L 68.87 -68" id="fig-e97312c08242438abf7e68ad37f1e090-element-163" visibility="hidden" gadfly:scale="1.0"/>
          <path fill="none" d="M19.63,132.19 L 68.87 132.19" id="fig-e97312c08242438abf7e68ad37f1e090-element-164" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,129.12 L 68.87 129.12" id="fig-e97312c08242438abf7e68ad37f1e090-element-165" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,126.04 L 68.87 126.04" id="fig-e97312c08242438abf7e68ad37f1e090-element-166" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,122.96 L 68.87 122.96" id="fig-e97312c08242438abf7e68ad37f1e090-element-167" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,119.88 L 68.87 119.88" id="fig-e97312c08242438abf7e68ad37f1e090-element-168" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,116.8 L 68.87 116.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-169" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,113.72 L 68.87 113.72" id="fig-e97312c08242438abf7e68ad37f1e090-element-170" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,110.64 L 68.87 110.64" id="fig-e97312c08242438abf7e68ad37f1e090-element-171" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,107.56 L 68.87 107.56" id="fig-e97312c08242438abf7e68ad37f1e090-element-172" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,104.48 L 68.87 104.48" id="fig-e97312c08242438abf7e68ad37f1e090-element-173" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,101.4 L 68.87 101.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-174" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,98.32 L 68.87 98.32" id="fig-e97312c08242438abf7e68ad37f1e090-element-175" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,95.24 L 68.87 95.24" id="fig-e97312c08242438abf7e68ad37f1e090-element-176" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,92.16 L 68.87 92.16" id="fig-e97312c08242438abf7e68ad37f1e090-element-177" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,89.08 L 68.87 89.08" id="fig-e97312c08242438abf7e68ad37f1e090-element-178" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,86 L 68.87 86" id="fig-e97312c08242438abf7e68ad37f1e090-element-179" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,82.92 L 68.87 82.92" id="fig-e97312c08242438abf7e68ad37f1e090-element-180" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,79.84 L 68.87 79.84" id="fig-e97312c08242438abf7e68ad37f1e090-element-181" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,76.76 L 68.87 76.76" id="fig-e97312c08242438abf7e68ad37f1e090-element-182" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,73.68 L 68.87 73.68" id="fig-e97312c08242438abf7e68ad37f1e090-element-183" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,70.6 L 68.87 70.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-184" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,67.52 L 68.87 67.52" id="fig-e97312c08242438abf7e68ad37f1e090-element-185" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,64.44 L 68.87 64.44" id="fig-e97312c08242438abf7e68ad37f1e090-element-186" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,61.36 L 68.87 61.36" id="fig-e97312c08242438abf7e68ad37f1e090-element-187" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,58.28 L 68.87 58.28" id="fig-e97312c08242438abf7e68ad37f1e090-element-188" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,55.2 L 68.87 55.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-189" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,52.12 L 68.87 52.12" id="fig-e97312c08242438abf7e68ad37f1e090-element-190" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,49.04 L 68.87 49.04" id="fig-e97312c08242438abf7e68ad37f1e090-element-191" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,45.96 L 68.87 45.96" id="fig-e97312c08242438abf7e68ad37f1e090-element-192" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,42.88 L 68.87 42.88" id="fig-e97312c08242438abf7e68ad37f1e090-element-193" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,39.8 L 68.87 39.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-194" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,36.72 L 68.87 36.72" id="fig-e97312c08242438abf7e68ad37f1e090-element-195" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,33.64 L 68.87 33.64" id="fig-e97312c08242438abf7e68ad37f1e090-element-196" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,30.56 L 68.87 30.56" id="fig-e97312c08242438abf7e68ad37f1e090-element-197" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,27.48 L 68.87 27.48" id="fig-e97312c08242438abf7e68ad37f1e090-element-198" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,24.4 L 68.87 24.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-199" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,21.32 L 68.87 21.32" id="fig-e97312c08242438abf7e68ad37f1e090-element-200" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,18.24 L 68.87 18.24" id="fig-e97312c08242438abf7e68ad37f1e090-element-201" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,15.16 L 68.87 15.16" id="fig-e97312c08242438abf7e68ad37f1e090-element-202" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,12.08 L 68.87 12.08" id="fig-e97312c08242438abf7e68ad37f1e090-element-203" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,9 L 68.87 9" id="fig-e97312c08242438abf7e68ad37f1e090-element-204" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,5.92 L 68.87 5.92" id="fig-e97312c08242438abf7e68ad37f1e090-element-205" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,2.84 L 68.87 2.84" id="fig-e97312c08242438abf7e68ad37f1e090-element-206" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-0.24 L 68.87 -0.24" id="fig-e97312c08242438abf7e68ad37f1e090-element-207" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-3.32 L 68.87 -3.32" id="fig-e97312c08242438abf7e68ad37f1e090-element-208" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-6.4 L 68.87 -6.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-209" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-9.48 L 68.87 -9.48" id="fig-e97312c08242438abf7e68ad37f1e090-element-210" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-12.56 L 68.87 -12.56" id="fig-e97312c08242438abf7e68ad37f1e090-element-211" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-15.64 L 68.87 -15.64" id="fig-e97312c08242438abf7e68ad37f1e090-element-212" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-18.72 L 68.87 -18.72" id="fig-e97312c08242438abf7e68ad37f1e090-element-213" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-21.8 L 68.87 -21.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-214" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-24.88 L 68.87 -24.88" id="fig-e97312c08242438abf7e68ad37f1e090-element-215" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-27.96 L 68.87 -27.96" id="fig-e97312c08242438abf7e68ad37f1e090-element-216" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-31.04 L 68.87 -31.04" id="fig-e97312c08242438abf7e68ad37f1e090-element-217" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-34.12 L 68.87 -34.12" id="fig-e97312c08242438abf7e68ad37f1e090-element-218" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-37.2 L 68.87 -37.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-219" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-40.28 L 68.87 -40.28" id="fig-e97312c08242438abf7e68ad37f1e090-element-220" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-43.36 L 68.87 -43.36" id="fig-e97312c08242438abf7e68ad37f1e090-element-221" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-46.44 L 68.87 -46.44" id="fig-e97312c08242438abf7e68ad37f1e090-element-222" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-49.52 L 68.87 -49.52" id="fig-e97312c08242438abf7e68ad37f1e090-element-223" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,-52.6 L 68.87 -52.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-224" visibility="hidden" gadfly:scale="10.0"/>
          <path fill="none" d="M19.63,147.59 L 68.87 147.59" id="fig-e97312c08242438abf7e68ad37f1e090-element-225" visibility="hidden" gadfly:scale="0.5"/>
          <path fill="none" d="M19.63,70.6 L 68.87 70.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-226" visibility="hidden" gadfly:scale="0.5"/>
          <path fill="none" d="M19.63,-6.4 L 68.87 -6.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-227" visibility="hidden" gadfly:scale="0.5"/>
          <path fill="none" d="M19.63,-83.4 L 68.87 -83.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-228" visibility="hidden" gadfly:scale="0.5"/>
          <path fill="none" d="M19.63,132.19 L 68.87 132.19" id="fig-e97312c08242438abf7e68ad37f1e090-element-229" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,124.5 L 68.87 124.5" id="fig-e97312c08242438abf7e68ad37f1e090-element-230" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,116.8 L 68.87 116.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-231" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,109.1 L 68.87 109.1" id="fig-e97312c08242438abf7e68ad37f1e090-element-232" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,101.4 L 68.87 101.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-233" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,93.7 L 68.87 93.7" id="fig-e97312c08242438abf7e68ad37f1e090-element-234" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,86 L 68.87 86" id="fig-e97312c08242438abf7e68ad37f1e090-element-235" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,78.3 L 68.87 78.3" id="fig-e97312c08242438abf7e68ad37f1e090-element-236" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,70.6 L 68.87 70.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-237" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,62.9 L 68.87 62.9" id="fig-e97312c08242438abf7e68ad37f1e090-element-238" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,55.2 L 68.87 55.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-239" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,47.5 L 68.87 47.5" id="fig-e97312c08242438abf7e68ad37f1e090-element-240" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,39.8 L 68.87 39.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-241" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,32.1 L 68.87 32.1" id="fig-e97312c08242438abf7e68ad37f1e090-element-242" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,24.4 L 68.87 24.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-243" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,16.7 L 68.87 16.7" id="fig-e97312c08242438abf7e68ad37f1e090-element-244" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,9 L 68.87 9" id="fig-e97312c08242438abf7e68ad37f1e090-element-245" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,1.3 L 68.87 1.3" id="fig-e97312c08242438abf7e68ad37f1e090-element-246" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,-6.4 L 68.87 -6.4" id="fig-e97312c08242438abf7e68ad37f1e090-element-247" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,-14.1 L 68.87 -14.1" id="fig-e97312c08242438abf7e68ad37f1e090-element-248" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,-21.8 L 68.87 -21.8" id="fig-e97312c08242438abf7e68ad37f1e090-element-249" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,-29.5 L 68.87 -29.5" id="fig-e97312c08242438abf7e68ad37f1e090-element-250" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,-37.2 L 68.87 -37.2" id="fig-e97312c08242438abf7e68ad37f1e090-element-251" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,-44.9 L 68.87 -44.9" id="fig-e97312c08242438abf7e68ad37f1e090-element-252" visibility="hidden" gadfly:scale="5.0"/>
          <path fill="none" d="M19.63,-52.6 L 68.87 -52.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-253" visibility="hidden" gadfly:scale="5.0"/>
        </g>
        <g class="guide xgridlines yfixed" stroke-dasharray="0.5,0.5" stroke-width="0.2" stroke="#D0D0E0" visibility="visible" id="fig-e97312c08242438abf7e68ad37f1e090-element-254">
          <path fill="none" d="M44.25,7 L 44.25 72.6" id="fig-e97312c08242438abf7e68ad37f1e090-element-255" gadfly:scale="1.0"/>
        </g>
        <g class="plotpanel" id="fig-e97312c08242438abf7e68ad37f1e090-element-256">
          <g shape-rendering="crispEdges" stroke-width="0.3" id="fig-e97312c08242438abf7e68ad37f1e090-element-257">
            <g stroke="#000000" stroke-opacity="0.000" class="geometry" id="fig-e97312c08242438abf7e68ad37f1e090-element-258">
              <rect x="19.61" y="59.82" width="24.67" height="10.78" id="fig-e97312c08242438abf7e68ad37f1e090-element-259" fill="#008000"/>
              <rect x="44.23" y="46.11" width="24.67" height="24.49" id="fig-e97312c08242438abf7e68ad37f1e090-element-260" fill="#008000"/>
              <rect x="19.61" y="9.77" width="24.67" height="50.05" id="fig-e97312c08242438abf7e68ad37f1e090-element-261" fill="#FF0000"/>
              <rect x="44.23" y="38.87" width="24.67" height="7.24" id="fig-e97312c08242438abf7e68ad37f1e090-element-262" fill="#FF0000"/>
            </g>
          </g>
        </g>
      </g>
      <g class="guide ylabels" font-size="2.82" font-family="'PT Sans Caption','Helvetica Neue','Helvetica',sans-serif" fill="#6C606B" id="fig-e97312c08242438abf7e68ad37f1e090-element-263">
        <text x="18.63" y="147.59" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-264" visibility="hidden" gadfly:scale="1.0">-500</text>
        <text x="18.63" y="132.19" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-265" visibility="hidden" gadfly:scale="1.0">-400</text>
        <text x="18.63" y="116.8" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-266" visibility="hidden" gadfly:scale="1.0">-300</text>
        <text x="18.63" y="101.4" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-267" visibility="hidden" gadfly:scale="1.0">-200</text>
        <text x="18.63" y="86" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-268" visibility="hidden" gadfly:scale="1.0">-100</text>
        <text x="18.63" y="70.6" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-269" visibility="visible" gadfly:scale="1.0">0</text>
        <text x="18.63" y="55.2" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-270" visibility="visible" gadfly:scale="1.0">100</text>
        <text x="18.63" y="39.8" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-271" visibility="visible" gadfly:scale="1.0">200</text>
        <text x="18.63" y="24.4" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-272" visibility="visible" gadfly:scale="1.0">300</text>
        <text x="18.63" y="9" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-273" visibility="visible" gadfly:scale="1.0">400</text>
        <text x="18.63" y="-6.4" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-274" visibility="hidden" gadfly:scale="1.0">500</text>
        <text x="18.63" y="-21.8" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-275" visibility="hidden" gadfly:scale="1.0">600</text>
        <text x="18.63" y="-37.2" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-276" visibility="hidden" gadfly:scale="1.0">700</text>
        <text x="18.63" y="-52.6" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-277" visibility="hidden" gadfly:scale="1.0">800</text>
        <text x="18.63" y="-68" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-278" visibility="hidden" gadfly:scale="1.0">900</text>
        <text x="18.63" y="132.19" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-279" visibility="hidden" gadfly:scale="10.0">-400</text>
        <text x="18.63" y="129.12" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-280" visibility="hidden" gadfly:scale="10.0">-380</text>
        <text x="18.63" y="126.04" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-281" visibility="hidden" gadfly:scale="10.0">-360</text>
        <text x="18.63" y="122.96" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-282" visibility="hidden" gadfly:scale="10.0">-340</text>
        <text x="18.63" y="119.88" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-283" visibility="hidden" gadfly:scale="10.0">-320</text>
        <text x="18.63" y="116.8" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-284" visibility="hidden" gadfly:scale="10.0">-300</text>
        <text x="18.63" y="113.72" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-285" visibility="hidden" gadfly:scale="10.0">-280</text>
        <text x="18.63" y="110.64" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-286" visibility="hidden" gadfly:scale="10.0">-260</text>
        <text x="18.63" y="107.56" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-287" visibility="hidden" gadfly:scale="10.0">-240</text>
        <text x="18.63" y="104.48" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-288" visibility="hidden" gadfly:scale="10.0">-220</text>
        <text x="18.63" y="101.4" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-289" visibility="hidden" gadfly:scale="10.0">-200</text>
        <text x="18.63" y="98.32" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-290" visibility="hidden" gadfly:scale="10.0">-180</text>
        <text x="18.63" y="95.24" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-291" visibility="hidden" gadfly:scale="10.0">-160</text>
        <text x="18.63" y="92.16" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-292" visibility="hidden" gadfly:scale="10.0">-140</text>
        <text x="18.63" y="89.08" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-293" visibility="hidden" gadfly:scale="10.0">-120</text>
        <text x="18.63" y="86" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-294" visibility="hidden" gadfly:scale="10.0">-100</text>
        <text x="18.63" y="82.92" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-295" visibility="hidden" gadfly:scale="10.0">-80</text>
        <text x="18.63" y="79.84" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-296" visibility="hidden" gadfly:scale="10.0">-60</text>
        <text x="18.63" y="76.76" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-297" visibility="hidden" gadfly:scale="10.0">-40</text>
        <text x="18.63" y="73.68" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-298" visibility="hidden" gadfly:scale="10.0">-20</text>
        <text x="18.63" y="70.6" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-299" visibility="hidden" gadfly:scale="10.0">0</text>
        <text x="18.63" y="67.52" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-300" visibility="hidden" gadfly:scale="10.0">20</text>
        <text x="18.63" y="64.44" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-301" visibility="hidden" gadfly:scale="10.0">40</text>
        <text x="18.63" y="61.36" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-302" visibility="hidden" gadfly:scale="10.0">60</text>
        <text x="18.63" y="58.28" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-303" visibility="hidden" gadfly:scale="10.0">80</text>
        <text x="18.63" y="55.2" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-304" visibility="hidden" gadfly:scale="10.0">100</text>
        <text x="18.63" y="52.12" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-305" visibility="hidden" gadfly:scale="10.0">120</text>
        <text x="18.63" y="49.04" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-306" visibility="hidden" gadfly:scale="10.0">140</text>
        <text x="18.63" y="45.96" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-307" visibility="hidden" gadfly:scale="10.0">160</text>
        <text x="18.63" y="42.88" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-308" visibility="hidden" gadfly:scale="10.0">180</text>
        <text x="18.63" y="39.8" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-309" visibility="hidden" gadfly:scale="10.0">200</text>
        <text x="18.63" y="36.72" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-310" visibility="hidden" gadfly:scale="10.0">220</text>
        <text x="18.63" y="33.64" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-311" visibility="hidden" gadfly:scale="10.0">240</text>
        <text x="18.63" y="30.56" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-312" visibility="hidden" gadfly:scale="10.0">260</text>
        <text x="18.63" y="27.48" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-313" visibility="hidden" gadfly:scale="10.0">280</text>
        <text x="18.63" y="24.4" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-314" visibility="hidden" gadfly:scale="10.0">300</text>
        <text x="18.63" y="21.32" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-315" visibility="hidden" gadfly:scale="10.0">320</text>
        <text x="18.63" y="18.24" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-316" visibility="hidden" gadfly:scale="10.0">340</text>
        <text x="18.63" y="15.16" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-317" visibility="hidden" gadfly:scale="10.0">360</text>
        <text x="18.63" y="12.08" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-318" visibility="hidden" gadfly:scale="10.0">380</text>
        <text x="18.63" y="9" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-319" visibility="hidden" gadfly:scale="10.0">400</text>
        <text x="18.63" y="5.92" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-320" visibility="hidden" gadfly:scale="10.0">420</text>
        <text x="18.63" y="2.84" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-321" visibility="hidden" gadfly:scale="10.0">440</text>
        <text x="18.63" y="-0.24" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-322" visibility="hidden" gadfly:scale="10.0">460</text>
        <text x="18.63" y="-3.32" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-323" visibility="hidden" gadfly:scale="10.0">480</text>
        <text x="18.63" y="-6.4" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-324" visibility="hidden" gadfly:scale="10.0">500</text>
        <text x="18.63" y="-9.48" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-325" visibility="hidden" gadfly:scale="10.0">520</text>
        <text x="18.63" y="-12.56" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-326" visibility="hidden" gadfly:scale="10.0">540</text>
        <text x="18.63" y="-15.64" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-327" visibility="hidden" gadfly:scale="10.0">560</text>
        <text x="18.63" y="-18.72" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-328" visibility="hidden" gadfly:scale="10.0">580</text>
        <text x="18.63" y="-21.8" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-329" visibility="hidden" gadfly:scale="10.0">600</text>
        <text x="18.63" y="-24.88" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-330" visibility="hidden" gadfly:scale="10.0">620</text>
        <text x="18.63" y="-27.96" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-331" visibility="hidden" gadfly:scale="10.0">640</text>
        <text x="18.63" y="-31.04" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-332" visibility="hidden" gadfly:scale="10.0">660</text>
        <text x="18.63" y="-34.12" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-333" visibility="hidden" gadfly:scale="10.0">680</text>
        <text x="18.63" y="-37.2" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-334" visibility="hidden" gadfly:scale="10.0">700</text>
        <text x="18.63" y="-40.28" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-335" visibility="hidden" gadfly:scale="10.0">720</text>
        <text x="18.63" y="-43.36" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-336" visibility="hidden" gadfly:scale="10.0">740</text>
        <text x="18.63" y="-46.44" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-337" visibility="hidden" gadfly:scale="10.0">760</text>
        <text x="18.63" y="-49.52" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-338" visibility="hidden" gadfly:scale="10.0">780</text>
        <text x="18.63" y="-52.6" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-339" visibility="hidden" gadfly:scale="10.0">800</text>
        <text x="18.63" y="147.59" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-340" visibility="hidden" gadfly:scale="0.5">-500</text>
        <text x="18.63" y="70.6" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-341" visibility="hidden" gadfly:scale="0.5">0</text>
        <text x="18.63" y="-6.4" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-342" visibility="hidden" gadfly:scale="0.5">500</text>
        <text x="18.63" y="-83.4" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-343" visibility="hidden" gadfly:scale="0.5">1000</text>
        <text x="18.63" y="132.19" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-344" visibility="hidden" gadfly:scale="5.0">-400</text>
        <text x="18.63" y="124.5" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-345" visibility="hidden" gadfly:scale="5.0">-350</text>
        <text x="18.63" y="116.8" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-346" visibility="hidden" gadfly:scale="5.0">-300</text>
        <text x="18.63" y="109.1" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-347" visibility="hidden" gadfly:scale="5.0">-250</text>
        <text x="18.63" y="101.4" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-348" visibility="hidden" gadfly:scale="5.0">-200</text>
        <text x="18.63" y="93.7" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-349" visibility="hidden" gadfly:scale="5.0">-150</text>
        <text x="18.63" y="86" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-350" visibility="hidden" gadfly:scale="5.0">-100</text>
        <text x="18.63" y="78.3" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-351" visibility="hidden" gadfly:scale="5.0">-50</text>
        <text x="18.63" y="70.6" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-352" visibility="hidden" gadfly:scale="5.0">0</text>
        <text x="18.63" y="62.9" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-353" visibility="hidden" gadfly:scale="5.0">50</text>
        <text x="18.63" y="55.2" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-354" visibility="hidden" gadfly:scale="5.0">100</text>
        <text x="18.63" y="47.5" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-355" visibility="hidden" gadfly:scale="5.0">150</text>
        <text x="18.63" y="39.8" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-356" visibility="hidden" gadfly:scale="5.0">200</text>
        <text x="18.63" y="32.1" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-357" visibility="hidden" gadfly:scale="5.0">250</text>
        <text x="18.63" y="24.4" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-358" visibility="hidden" gadfly:scale="5.0">300</text>
        <text x="18.63" y="16.7" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-359" visibility="hidden" gadfly:scale="5.0">350</text>
        <text x="18.63" y="9" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-360" visibility="hidden" gadfly:scale="5.0">400</text>
        <text x="18.63" y="1.3" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-361" visibility="hidden" gadfly:scale="5.0">450</text>
        <text x="18.63" y="-6.4" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-362" visibility="hidden" gadfly:scale="5.0">500</text>
        <text x="18.63" y="-14.1" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-363" visibility="hidden" gadfly:scale="5.0">550</text>
        <text x="18.63" y="-21.8" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-364" visibility="hidden" gadfly:scale="5.0">600</text>
        <text x="18.63" y="-29.5" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-365" visibility="hidden" gadfly:scale="5.0">650</text>
        <text x="18.63" y="-37.2" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-366" visibility="hidden" gadfly:scale="5.0">700</text>
        <text x="18.63" y="-44.9" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-367" visibility="hidden" gadfly:scale="5.0">750</text>
        <text x="18.63" y="-52.6" text-anchor="end" dy="0.35em" id="fig-e97312c08242438abf7e68ad37f1e090-element-368" visibility="hidden" gadfly:scale="5.0">800</text>
      </g>
    </g>
  </g>
  <g font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" fill="#564A55" stroke="#000000" stroke-opacity="0.000" id="fig-e97312c08242438abf7e68ad37f1e090-element-369">
    <text x="8.81" y="43.19" text-anchor="middle" dy="0.35em" transform="rotate(-90, 8.81, 45.19)" id="fig-e97312c08242438abf7e68ad37f1e090-element-370">Survived</text>
  </g>
</g>
<defs>
<clipPath id="fig-e97312c08242438abf7e68ad37f1e090-element-26">
  <path d="M72.87,7 L 120.11 7 120.11 72.6 72.87 72.6" />
</clipPath
><clipPath id="fig-e97312c08242438abf7e68ad37f1e090-element-145">
  <path d="M19.63,7 L 68.87 7 68.87 72.6 19.63 72.6" />
</clipPath
><clipPath id="fig-e97312c08242438abf7e68ad37f1e090-element-13">
  <path d="M12.61,5 L 122.11 5 122.11 85.39 12.61 85.39" />
</clipPath
></defs>
<script> <![CDATA[
(function(N){var k=/[\.\/]/,L=/\s*,\s*/,C=function(a,d){return a-d},a,v,y={n:{}},M=function(){for(var a=0,d=this.length;a<d;a++)if("undefined"!=typeof this[a])return this[a]},A=function(){for(var a=this.length;--a;)if("undefined"!=typeof this[a])return this[a]},w=function(k,d){k=String(k);var f=v,n=Array.prototype.slice.call(arguments,2),u=w.listeners(k),p=0,b,q=[],e={},l=[],r=a;l.firstDefined=M;l.lastDefined=A;a=k;for(var s=v=0,x=u.length;s<x;s++)"zIndex"in u[s]&&(q.push(u[s].zIndex),0>u[s].zIndex&&
(e[u[s].zIndex]=u[s]));for(q.sort(C);0>q[p];)if(b=e[q[p++] ],l.push(b.apply(d,n)),v)return v=f,l;for(s=0;s<x;s++)if(b=u[s],"zIndex"in b)if(b.zIndex==q[p]){l.push(b.apply(d,n));if(v)break;do if(p++,(b=e[q[p] ])&&l.push(b.apply(d,n)),v)break;while(b)}else e[b.zIndex]=b;else if(l.push(b.apply(d,n)),v)break;v=f;a=r;return l};w._events=y;w.listeners=function(a){a=a.split(k);var d=y,f,n,u,p,b,q,e,l=[d],r=[];u=0;for(p=a.length;u<p;u++){e=[];b=0;for(q=l.length;b<q;b++)for(d=l[b].n,f=[d[a[u] ],d["*"] ],n=2;n--;)if(d=
f[n])e.push(d),r=r.concat(d.f||[]);l=e}return r};w.on=function(a,d){a=String(a);if("function"!=typeof d)return function(){};for(var f=a.split(L),n=0,u=f.length;n<u;n++)(function(a){a=a.split(k);for(var b=y,f,e=0,l=a.length;e<l;e++)b=b.n,b=b.hasOwnProperty(a[e])&&b[a[e] ]||(b[a[e] ]={n:{}});b.f=b.f||[];e=0;for(l=b.f.length;e<l;e++)if(b.f[e]==d){f=!0;break}!f&&b.f.push(d)})(f[n]);return function(a){+a==+a&&(d.zIndex=+a)}};w.f=function(a){var d=[].slice.call(arguments,1);return function(){w.apply(null,
[a,null].concat(d).concat([].slice.call(arguments,0)))}};w.stop=function(){v=1};w.nt=function(k){return k?(new RegExp("(?:\\.|\\/|^)"+k+"(?:\\.|\\/|$)")).test(a):a};w.nts=function(){return a.split(k)};w.off=w.unbind=function(a,d){if(a){var f=a.split(L);if(1<f.length)for(var n=0,u=f.length;n<u;n++)w.off(f[n],d);else{for(var f=a.split(k),p,b,q,e,l=[y],n=0,u=f.length;n<u;n++)for(e=0;e<l.length;e+=q.length-2){q=[e,1];p=l[e].n;if("*"!=f[n])p[f[n] ]&&q.push(p[f[n] ]);else for(b in p)p.hasOwnProperty(b)&&
q.push(p[b]);l.splice.apply(l,q)}n=0;for(u=l.length;n<u;n++)for(p=l[n];p.n;){if(d){if(p.f){e=0;for(f=p.f.length;e<f;e++)if(p.f[e]==d){p.f.splice(e,1);break}!p.f.length&&delete p.f}for(b in p.n)if(p.n.hasOwnProperty(b)&&p.n[b].f){q=p.n[b].f;e=0;for(f=q.length;e<f;e++)if(q[e]==d){q.splice(e,1);break}!q.length&&delete p.n[b].f}}else for(b in delete p.f,p.n)p.n.hasOwnProperty(b)&&p.n[b].f&&delete p.n[b].f;p=p.n}}}else w._events=y={n:{}}};w.once=function(a,d){var f=function(){w.unbind(a,f);return d.apply(this,
arguments)};return w.on(a,f)};w.version="0.4.2";w.toString=function(){return"You are running Eve 0.4.2"};"undefined"!=typeof module&&module.exports?module.exports=w:"function"===typeof define&&define.amd?define("eve",[],function(){return w}):N.eve=w})(this);
(function(N,k){"function"===typeof define&&define.amd?define("Snap.svg",["eve"],function(L){return k(N,L)}):k(N,N.eve)})(this,function(N,k){var L=function(a){var k={},y=N.requestAnimationFrame||N.webkitRequestAnimationFrame||N.mozRequestAnimationFrame||N.oRequestAnimationFrame||N.msRequestAnimationFrame||function(a){setTimeout(a,16)},M=Array.isArray||function(a){return a instanceof Array||"[object Array]"==Object.prototype.toString.call(a)},A=0,w="M"+(+new Date).toString(36),z=function(a){if(null==
a)return this.s;var b=this.s-a;this.b+=this.dur*b;this.B+=this.dur*b;this.s=a},d=function(a){if(null==a)return this.spd;this.spd=a},f=function(a){if(null==a)return this.dur;this.s=this.s*a/this.dur;this.dur=a},n=function(){delete k[this.id];this.update();a("mina.stop."+this.id,this)},u=function(){this.pdif||(delete k[this.id],this.update(),this.pdif=this.get()-this.b)},p=function(){this.pdif&&(this.b=this.get()-this.pdif,delete this.pdif,k[this.id]=this)},b=function(){var a;if(M(this.start)){a=[];
for(var b=0,e=this.start.length;b<e;b++)a[b]=+this.start[b]+(this.end[b]-this.start[b])*this.easing(this.s)}else a=+this.start+(this.end-this.start)*this.easing(this.s);this.set(a)},q=function(){var l=0,b;for(b in k)if(k.hasOwnProperty(b)){var e=k[b],f=e.get();l++;e.s=(f-e.b)/(e.dur/e.spd);1<=e.s&&(delete k[b],e.s=1,l--,function(b){setTimeout(function(){a("mina.finish."+b.id,b)})}(e));e.update()}l&&y(q)},e=function(a,r,s,x,G,h,J){a={id:w+(A++).toString(36),start:a,end:r,b:s,s:0,dur:x-s,spd:1,get:G,
set:h,easing:J||e.linear,status:z,speed:d,duration:f,stop:n,pause:u,resume:p,update:b};k[a.id]=a;r=0;for(var K in k)if(k.hasOwnProperty(K)&&(r++,2==r))break;1==r&&y(q);return a};e.time=Date.now||function(){return+new Date};e.getById=function(a){return k[a]||null};e.linear=function(a){return a};e.easeout=function(a){return Math.pow(a,1.7)};e.easein=function(a){return Math.pow(a,0.48)};e.easeinout=function(a){if(1==a)return 1;if(0==a)return 0;var b=0.48-a/1.04,e=Math.sqrt(0.1734+b*b);a=e-b;a=Math.pow(Math.abs(a),
1/3)*(0>a?-1:1);b=-e-b;b=Math.pow(Math.abs(b),1/3)*(0>b?-1:1);a=a+b+0.5;return 3*(1-a)*a*a+a*a*a};e.backin=function(a){return 1==a?1:a*a*(2.70158*a-1.70158)};e.backout=function(a){if(0==a)return 0;a-=1;return a*a*(2.70158*a+1.70158)+1};e.elastic=function(a){return a==!!a?a:Math.pow(2,-10*a)*Math.sin(2*(a-0.075)*Math.PI/0.3)+1};e.bounce=function(a){a<1/2.75?a*=7.5625*a:a<2/2.75?(a-=1.5/2.75,a=7.5625*a*a+0.75):a<2.5/2.75?(a-=2.25/2.75,a=7.5625*a*a+0.9375):(a-=2.625/2.75,a=7.5625*a*a+0.984375);return a};
return N.mina=e}("undefined"==typeof k?function(){}:k),C=function(){function a(c,t){if(c){if(c.tagName)return x(c);if(y(c,"array")&&a.set)return a.set.apply(a,c);if(c instanceof e)return c;if(null==t)return c=G.doc.querySelector(c),x(c)}return new s(null==c?"100%":c,null==t?"100%":t)}function v(c,a){if(a){"#text"==c&&(c=G.doc.createTextNode(a.text||""));"string"==typeof c&&(c=v(c));if("string"==typeof a)return"xlink:"==a.substring(0,6)?c.getAttributeNS(m,a.substring(6)):"xml:"==a.substring(0,4)?c.getAttributeNS(la,
a.substring(4)):c.getAttribute(a);for(var da in a)if(a[h](da)){var b=J(a[da]);b?"xlink:"==da.substring(0,6)?c.setAttributeNS(m,da.substring(6),b):"xml:"==da.substring(0,4)?c.setAttributeNS(la,da.substring(4),b):c.setAttribute(da,b):c.removeAttribute(da)}}else c=G.doc.createElementNS(la,c);return c}function y(c,a){a=J.prototype.toLowerCase.call(a);return"finite"==a?isFinite(c):"array"==a&&(c instanceof Array||Array.isArray&&Array.isArray(c))?!0:"null"==a&&null===c||a==typeof c&&null!==c||"object"==
a&&c===Object(c)||$.call(c).slice(8,-1).toLowerCase()==a}function M(c){if("function"==typeof c||Object(c)!==c)return c;var a=new c.constructor,b;for(b in c)c[h](b)&&(a[b]=M(c[b]));return a}function A(c,a,b){function m(){var e=Array.prototype.slice.call(arguments,0),f=e.join("\u2400"),d=m.cache=m.cache||{},l=m.count=m.count||[];if(d[h](f)){a:for(var e=l,l=f,B=0,H=e.length;B<H;B++)if(e[B]===l){e.push(e.splice(B,1)[0]);break a}return b?b(d[f]):d[f]}1E3<=l.length&&delete d[l.shift()];l.push(f);d[f]=c.apply(a,
e);return b?b(d[f]):d[f]}return m}function w(c,a,b,m,e,f){return null==e?(c-=b,a-=m,c||a?(180*I.atan2(-a,-c)/C+540)%360:0):w(c,a,e,f)-w(b,m,e,f)}function z(c){return c%360*C/180}function d(c){var a=[];c=c.replace(/(?:^|\s)(\w+)\(([^)]+)\)/g,function(c,b,m){m=m.split(/\s*,\s*|\s+/);"rotate"==b&&1==m.length&&m.push(0,0);"scale"==b&&(2<m.length?m=m.slice(0,2):2==m.length&&m.push(0,0),1==m.length&&m.push(m[0],0,0));"skewX"==b?a.push(["m",1,0,I.tan(z(m[0])),1,0,0]):"skewY"==b?a.push(["m",1,I.tan(z(m[0])),
0,1,0,0]):a.push([b.charAt(0)].concat(m));return c});return a}function f(c,t){var b=O(c),m=new a.Matrix;if(b)for(var e=0,f=b.length;e<f;e++){var h=b[e],d=h.length,B=J(h[0]).toLowerCase(),H=h[0]!=B,l=H?m.invert():0,E;"t"==B&&2==d?m.translate(h[1],0):"t"==B&&3==d?H?(d=l.x(0,0),B=l.y(0,0),H=l.x(h[1],h[2]),l=l.y(h[1],h[2]),m.translate(H-d,l-B)):m.translate(h[1],h[2]):"r"==B?2==d?(E=E||t,m.rotate(h[1],E.x+E.width/2,E.y+E.height/2)):4==d&&(H?(H=l.x(h[2],h[3]),l=l.y(h[2],h[3]),m.rotate(h[1],H,l)):m.rotate(h[1],
h[2],h[3])):"s"==B?2==d||3==d?(E=E||t,m.scale(h[1],h[d-1],E.x+E.width/2,E.y+E.height/2)):4==d?H?(H=l.x(h[2],h[3]),l=l.y(h[2],h[3]),m.scale(h[1],h[1],H,l)):m.scale(h[1],h[1],h[2],h[3]):5==d&&(H?(H=l.x(h[3],h[4]),l=l.y(h[3],h[4]),m.scale(h[1],h[2],H,l)):m.scale(h[1],h[2],h[3],h[4])):"m"==B&&7==d&&m.add(h[1],h[2],h[3],h[4],h[5],h[6])}return m}function n(c,t){if(null==t){var m=!0;t="linearGradient"==c.type||"radialGradient"==c.type?c.node.getAttribute("gradientTransform"):"pattern"==c.type?c.node.getAttribute("patternTransform"):
c.node.getAttribute("transform");if(!t)return new a.Matrix;t=d(t)}else t=a._.rgTransform.test(t)?J(t).replace(/\.{3}|\u2026/g,c._.transform||aa):d(t),y(t,"array")&&(t=a.path?a.path.toString.call(t):J(t)),c._.transform=t;var b=f(t,c.getBBox(1));if(m)return b;c.matrix=b}function u(c){c=c.node.ownerSVGElement&&x(c.node.ownerSVGElement)||c.node.parentNode&&x(c.node.parentNode)||a.select("svg")||a(0,0);var t=c.select("defs"),t=null==t?!1:t.node;t||(t=r("defs",c.node).node);return t}function p(c){return c.node.ownerSVGElement&&
x(c.node.ownerSVGElement)||a.select("svg")}function b(c,a,m){function b(c){if(null==c)return aa;if(c==+c)return c;v(B,{width:c});try{return B.getBBox().width}catch(a){return 0}}function h(c){if(null==c)return aa;if(c==+c)return c;v(B,{height:c});try{return B.getBBox().height}catch(a){return 0}}function e(b,B){null==a?d[b]=B(c.attr(b)||0):b==a&&(d=B(null==m?c.attr(b)||0:m))}var f=p(c).node,d={},B=f.querySelector(".svg---mgr");B||(B=v("rect"),v(B,{x:-9E9,y:-9E9,width:10,height:10,"class":"svg---mgr",
fill:"none"}),f.appendChild(B));switch(c.type){case "rect":e("rx",b),e("ry",h);case "image":e("width",b),e("height",h);case "text":e("x",b);e("y",h);break;case "circle":e("cx",b);e("cy",h);e("r",b);break;case "ellipse":e("cx",b);e("cy",h);e("rx",b);e("ry",h);break;case "line":e("x1",b);e("x2",b);e("y1",h);e("y2",h);break;case "marker":e("refX",b);e("markerWidth",b);e("refY",h);e("markerHeight",h);break;case "radialGradient":e("fx",b);e("fy",h);break;case "tspan":e("dx",b);e("dy",h);break;default:e(a,
b)}f.removeChild(B);return d}function q(c){y(c,"array")||(c=Array.prototype.slice.call(arguments,0));for(var a=0,b=0,m=this.node;this[a];)delete this[a++];for(a=0;a<c.length;a++)"set"==c[a].type?c[a].forEach(function(c){m.appendChild(c.node)}):m.appendChild(c[a].node);for(var h=m.childNodes,a=0;a<h.length;a++)this[b++]=x(h[a]);return this}function e(c){if(c.snap in E)return E[c.snap];var a=this.id=V(),b;try{b=c.ownerSVGElement}catch(m){}this.node=c;b&&(this.paper=new s(b));this.type=c.tagName;this.anims=
{};this._={transform:[]};c.snap=a;E[a]=this;"g"==this.type&&(this.add=q);if(this.type in{g:1,mask:1,pattern:1})for(var e in s.prototype)s.prototype[h](e)&&(this[e]=s.prototype[e])}function l(c){this.node=c}function r(c,a){var b=v(c);a.appendChild(b);return x(b)}function s(c,a){var b,m,f,d=s.prototype;if(c&&"svg"==c.tagName){if(c.snap in E)return E[c.snap];var l=c.ownerDocument;b=new e(c);m=c.getElementsByTagName("desc")[0];f=c.getElementsByTagName("defs")[0];m||(m=v("desc"),m.appendChild(l.createTextNode("Created with Snap")),
b.node.appendChild(m));f||(f=v("defs"),b.node.appendChild(f));b.defs=f;for(var ca in d)d[h](ca)&&(b[ca]=d[ca]);b.paper=b.root=b}else b=r("svg",G.doc.body),v(b.node,{height:a,version:1.1,width:c,xmlns:la});return b}function x(c){return!c||c instanceof e||c instanceof l?c:c.tagName&&"svg"==c.tagName.toLowerCase()?new s(c):c.tagName&&"object"==c.tagName.toLowerCase()&&"image/svg+xml"==c.type?new s(c.contentDocument.getElementsByTagName("svg")[0]):new e(c)}a.version="0.3.0";a.toString=function(){return"Snap v"+
this.version};a._={};var G={win:N,doc:N.document};a._.glob=G;var h="hasOwnProperty",J=String,K=parseFloat,U=parseInt,I=Math,P=I.max,Q=I.min,Y=I.abs,C=I.PI,aa="",$=Object.prototype.toString,F=/^\s*((#[a-f\d]{6})|(#[a-f\d]{3})|rgba?\(\s*([\d\.]+%?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+%?(?:\s*,\s*[\d\.]+%?)?)\s*\)|hsba?\(\s*([\d\.]+(?:deg|\xb0|%)?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+(?:%?\s*,\s*[\d\.]+)?%?)\s*\)|hsla?\(\s*([\d\.]+(?:deg|\xb0|%)?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+(?:%?\s*,\s*[\d\.]+)?%?)\s*\))\s*$/i;a._.separator=
RegExp("[,\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]+");var S=RegExp("[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*"),X={hs:1,rg:1},W=RegExp("([a-z])[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029,]*((-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*)+)",
"ig"),ma=RegExp("([rstm])[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029,]*((-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*)+)","ig"),Z=RegExp("(-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?)[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*",
"ig"),na=0,ba="S"+(+new Date).toString(36),V=function(){return ba+(na++).toString(36)},m="http://www.w3.org/1999/xlink",la="http://www.w3.org/2000/svg",E={},ca=a.url=function(c){return"url('#"+c+"')"};a._.$=v;a._.id=V;a.format=function(){var c=/\{([^\}]+)\}/g,a=/(?:(?:^|\.)(.+?)(?=\[|\.|$|\()|\[('|")(.+?)\2\])(\(\))?/g,b=function(c,b,m){var h=m;b.replace(a,function(c,a,b,m,t){a=a||m;h&&(a in h&&(h=h[a]),"function"==typeof h&&t&&(h=h()))});return h=(null==h||h==m?c:h)+""};return function(a,m){return J(a).replace(c,
function(c,a){return b(c,a,m)})}}();a._.clone=M;a._.cacher=A;a.rad=z;a.deg=function(c){return 180*c/C%360};a.angle=w;a.is=y;a.snapTo=function(c,a,b){b=y(b,"finite")?b:10;if(y(c,"array"))for(var m=c.length;m--;){if(Y(c[m]-a)<=b)return c[m]}else{c=+c;m=a%c;if(m<b)return a-m;if(m>c-b)return a-m+c}return a};a.getRGB=A(function(c){if(!c||(c=J(c)).indexOf("-")+1)return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka};if("none"==c)return{r:-1,g:-1,b:-1,hex:"none",toString:ka};!X[h](c.toLowerCase().substring(0,
2))&&"#"!=c.charAt()&&(c=T(c));if(!c)return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka};var b,m,e,f,d;if(c=c.match(F)){c[2]&&(e=U(c[2].substring(5),16),m=U(c[2].substring(3,5),16),b=U(c[2].substring(1,3),16));c[3]&&(e=U((d=c[3].charAt(3))+d,16),m=U((d=c[3].charAt(2))+d,16),b=U((d=c[3].charAt(1))+d,16));c[4]&&(d=c[4].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b*=2.55),m=K(d[1]),"%"==d[1].slice(-1)&&(m*=2.55),e=K(d[2]),"%"==d[2].slice(-1)&&(e*=2.55),"rgba"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),
d[3]&&"%"==d[3].slice(-1)&&(f/=100));if(c[5])return d=c[5].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b/=100),m=K(d[1]),"%"==d[1].slice(-1)&&(m/=100),e=K(d[2]),"%"==d[2].slice(-1)&&(e/=100),"deg"!=d[0].slice(-3)&&"\u00b0"!=d[0].slice(-1)||(b/=360),"hsba"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),d[3]&&"%"==d[3].slice(-1)&&(f/=100),a.hsb2rgb(b,m,e,f);if(c[6])return d=c[6].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b/=100),m=K(d[1]),"%"==d[1].slice(-1)&&(m/=100),e=K(d[2]),"%"==d[2].slice(-1)&&(e/=100),
"deg"!=d[0].slice(-3)&&"\u00b0"!=d[0].slice(-1)||(b/=360),"hsla"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),d[3]&&"%"==d[3].slice(-1)&&(f/=100),a.hsl2rgb(b,m,e,f);b=Q(I.round(b),255);m=Q(I.round(m),255);e=Q(I.round(e),255);f=Q(P(f,0),1);c={r:b,g:m,b:e,toString:ka};c.hex="#"+(16777216|e|m<<8|b<<16).toString(16).slice(1);c.opacity=y(f,"finite")?f:1;return c}return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka}},a);a.hsb=A(function(c,b,m){return a.hsb2rgb(c,b,m).hex});a.hsl=A(function(c,b,m){return a.hsl2rgb(c,
b,m).hex});a.rgb=A(function(c,a,b,m){if(y(m,"finite")){var e=I.round;return"rgba("+[e(c),e(a),e(b),+m.toFixed(2)]+")"}return"#"+(16777216|b|a<<8|c<<16).toString(16).slice(1)});var T=function(c){var a=G.doc.getElementsByTagName("head")[0]||G.doc.getElementsByTagName("svg")[0];T=A(function(c){if("red"==c.toLowerCase())return"rgb(255, 0, 0)";a.style.color="rgb(255, 0, 0)";a.style.color=c;c=G.doc.defaultView.getComputedStyle(a,aa).getPropertyValue("color");return"rgb(255, 0, 0)"==c?null:c});return T(c)},
qa=function(){return"hsb("+[this.h,this.s,this.b]+")"},ra=function(){return"hsl("+[this.h,this.s,this.l]+")"},ka=function(){return 1==this.opacity||null==this.opacity?this.hex:"rgba("+[this.r,this.g,this.b,this.opacity]+")"},D=function(c,b,m){null==b&&y(c,"object")&&"r"in c&&"g"in c&&"b"in c&&(m=c.b,b=c.g,c=c.r);null==b&&y(c,string)&&(m=a.getRGB(c),c=m.r,b=m.g,m=m.b);if(1<c||1<b||1<m)c/=255,b/=255,m/=255;return[c,b,m]},oa=function(c,b,m,e){c=I.round(255*c);b=I.round(255*b);m=I.round(255*m);c={r:c,
g:b,b:m,opacity:y(e,"finite")?e:1,hex:a.rgb(c,b,m),toString:ka};y(e,"finite")&&(c.opacity=e);return c};a.color=function(c){var b;y(c,"object")&&"h"in c&&"s"in c&&"b"in c?(b=a.hsb2rgb(c),c.r=b.r,c.g=b.g,c.b=b.b,c.opacity=1,c.hex=b.hex):y(c,"object")&&"h"in c&&"s"in c&&"l"in c?(b=a.hsl2rgb(c),c.r=b.r,c.g=b.g,c.b=b.b,c.opacity=1,c.hex=b.hex):(y(c,"string")&&(c=a.getRGB(c)),y(c,"object")&&"r"in c&&"g"in c&&"b"in c&&!("error"in c)?(b=a.rgb2hsl(c),c.h=b.h,c.s=b.s,c.l=b.l,b=a.rgb2hsb(c),c.v=b.b):(c={hex:"none"},
c.r=c.g=c.b=c.h=c.s=c.v=c.l=-1,c.error=1));c.toString=ka;return c};a.hsb2rgb=function(c,a,b,m){y(c,"object")&&"h"in c&&"s"in c&&"b"in c&&(b=c.b,a=c.s,c=c.h,m=c.o);var e,h,d;c=360*c%360/60;d=b*a;a=d*(1-Y(c%2-1));b=e=h=b-d;c=~~c;b+=[d,a,0,0,a,d][c];e+=[a,d,d,a,0,0][c];h+=[0,0,a,d,d,a][c];return oa(b,e,h,m)};a.hsl2rgb=function(c,a,b,m){y(c,"object")&&"h"in c&&"s"in c&&"l"in c&&(b=c.l,a=c.s,c=c.h);if(1<c||1<a||1<b)c/=360,a/=100,b/=100;var e,h,d;c=360*c%360/60;d=2*a*(0.5>b?b:1-b);a=d*(1-Y(c%2-1));b=e=
h=b-d/2;c=~~c;b+=[d,a,0,0,a,d][c];e+=[a,d,d,a,0,0][c];h+=[0,0,a,d,d,a][c];return oa(b,e,h,m)};a.rgb2hsb=function(c,a,b){b=D(c,a,b);c=b[0];a=b[1];b=b[2];var m,e;m=P(c,a,b);e=m-Q(c,a,b);c=((0==e?0:m==c?(a-b)/e:m==a?(b-c)/e+2:(c-a)/e+4)+360)%6*60/360;return{h:c,s:0==e?0:e/m,b:m,toString:qa}};a.rgb2hsl=function(c,a,b){b=D(c,a,b);c=b[0];a=b[1];b=b[2];var m,e,h;m=P(c,a,b);e=Q(c,a,b);h=m-e;c=((0==h?0:m==c?(a-b)/h:m==a?(b-c)/h+2:(c-a)/h+4)+360)%6*60/360;m=(m+e)/2;return{h:c,s:0==h?0:0.5>m?h/(2*m):h/(2-2*
m),l:m,toString:ra}};a.parsePathString=function(c){if(!c)return null;var b=a.path(c);if(b.arr)return a.path.clone(b.arr);var m={a:7,c:6,o:2,h:1,l:2,m:2,r:4,q:4,s:4,t:2,v:1,u:3,z:0},e=[];y(c,"array")&&y(c[0],"array")&&(e=a.path.clone(c));e.length||J(c).replace(W,function(c,a,b){var h=[];c=a.toLowerCase();b.replace(Z,function(c,a){a&&h.push(+a)});"m"==c&&2<h.length&&(e.push([a].concat(h.splice(0,2))),c="l",a="m"==a?"l":"L");"o"==c&&1==h.length&&e.push([a,h[0] ]);if("r"==c)e.push([a].concat(h));else for(;h.length>=
m[c]&&(e.push([a].concat(h.splice(0,m[c]))),m[c]););});e.toString=a.path.toString;b.arr=a.path.clone(e);return e};var O=a.parseTransformString=function(c){if(!c)return null;var b=[];y(c,"array")&&y(c[0],"array")&&(b=a.path.clone(c));b.length||J(c).replace(ma,function(c,a,m){var e=[];a.toLowerCase();m.replace(Z,function(c,a){a&&e.push(+a)});b.push([a].concat(e))});b.toString=a.path.toString;return b};a._.svgTransform2string=d;a._.rgTransform=RegExp("^[a-z][\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*-?\\.?\\d",
"i");a._.transform2matrix=f;a._unit2px=b;a._.getSomeDefs=u;a._.getSomeSVG=p;a.select=function(c){return x(G.doc.querySelector(c))};a.selectAll=function(c){c=G.doc.querySelectorAll(c);for(var b=(a.set||Array)(),m=0;m<c.length;m++)b.push(x(c[m]));return b};setInterval(function(){for(var c in E)if(E[h](c)){var a=E[c],b=a.node;("svg"!=a.type&&!b.ownerSVGElement||"svg"==a.type&&(!b.parentNode||"ownerSVGElement"in b.parentNode&&!b.ownerSVGElement))&&delete E[c]}},1E4);(function(c){function m(c){function a(c,
b){var m=v(c.node,b);(m=(m=m&&m.match(d))&&m[2])&&"#"==m.charAt()&&(m=m.substring(1))&&(f[m]=(f[m]||[]).concat(function(a){var m={};m[b]=ca(a);v(c.node,m)}))}function b(c){var a=v(c.node,"xlink:href");a&&"#"==a.charAt()&&(a=a.substring(1))&&(f[a]=(f[a]||[]).concat(function(a){c.attr("xlink:href","#"+a)}))}var e=c.selectAll("*"),h,d=/^\s*url\(("|'|)(.*)\1\)\s*$/;c=[];for(var f={},l=0,E=e.length;l<E;l++){h=e[l];a(h,"fill");a(h,"stroke");a(h,"filter");a(h,"mask");a(h,"clip-path");b(h);var t=v(h.node,
"id");t&&(v(h.node,{id:h.id}),c.push({old:t,id:h.id}))}l=0;for(E=c.length;l<E;l++)if(e=f[c[l].old])for(h=0,t=e.length;h<t;h++)e[h](c[l].id)}function e(c,a,b){return function(m){m=m.slice(c,a);1==m.length&&(m=m[0]);return b?b(m):m}}function d(c){return function(){var a=c?"<"+this.type:"",b=this.node.attributes,m=this.node.childNodes;if(c)for(var e=0,h=b.length;e<h;e++)a+=" "+b[e].name+'="'+b[e].value.replace(/"/g,'\\"')+'"';if(m.length){c&&(a+=">");e=0;for(h=m.length;e<h;e++)3==m[e].nodeType?a+=m[e].nodeValue:
1==m[e].nodeType&&(a+=x(m[e]).toString());c&&(a+="</"+this.type+">")}else c&&(a+="/>");return a}}c.attr=function(c,a){if(!c)return this;if(y(c,"string"))if(1<arguments.length){var b={};b[c]=a;c=b}else return k("snap.util.getattr."+c,this).firstDefined();for(var m in c)c[h](m)&&k("snap.util.attr."+m,this,c[m]);return this};c.getBBox=function(c){if(!a.Matrix||!a.path)return this.node.getBBox();var b=this,m=new a.Matrix;if(b.removed)return a._.box();for(;"use"==b.type;)if(c||(m=m.add(b.transform().localMatrix.translate(b.attr("x")||
0,b.attr("y")||0))),b.original)b=b.original;else var e=b.attr("xlink:href"),b=b.original=b.node.ownerDocument.getElementById(e.substring(e.indexOf("#")+1));var e=b._,h=a.path.get[b.type]||a.path.get.deflt;try{if(c)return e.bboxwt=h?a.path.getBBox(b.realPath=h(b)):a._.box(b.node.getBBox()),a._.box(e.bboxwt);b.realPath=h(b);b.matrix=b.transform().localMatrix;e.bbox=a.path.getBBox(a.path.map(b.realPath,m.add(b.matrix)));return a._.box(e.bbox)}catch(d){return a._.box()}};var f=function(){return this.string};
c.transform=function(c){var b=this._;if(null==c){var m=this;c=new a.Matrix(this.node.getCTM());for(var e=n(this),h=[e],d=new a.Matrix,l=e.toTransformString(),b=J(e)==J(this.matrix)?J(b.transform):l;"svg"!=m.type&&(m=m.parent());)h.push(n(m));for(m=h.length;m--;)d.add(h[m]);return{string:b,globalMatrix:c,totalMatrix:d,localMatrix:e,diffMatrix:c.clone().add(e.invert()),global:c.toTransformString(),total:d.toTransformString(),local:l,toString:f}}c instanceof a.Matrix?this.matrix=c:n(this,c);this.node&&
("linearGradient"==this.type||"radialGradient"==this.type?v(this.node,{gradientTransform:this.matrix}):"pattern"==this.type?v(this.node,{patternTransform:this.matrix}):v(this.node,{transform:this.matrix}));return this};c.parent=function(){return x(this.node.parentNode)};c.append=c.add=function(c){if(c){if("set"==c.type){var a=this;c.forEach(function(c){a.add(c)});return this}c=x(c);this.node.appendChild(c.node);c.paper=this.paper}return this};c.appendTo=function(c){c&&(c=x(c),c.append(this));return this};
c.prepend=function(c){if(c){if("set"==c.type){var a=this,b;c.forEach(function(c){b?b.after(c):a.prepend(c);b=c});return this}c=x(c);var m=c.parent();this.node.insertBefore(c.node,this.node.firstChild);this.add&&this.add();c.paper=this.paper;this.parent()&&this.parent().add();m&&m.add()}return this};c.prependTo=function(c){c=x(c);c.prepend(this);return this};c.before=function(c){if("set"==c.type){var a=this;c.forEach(function(c){var b=c.parent();a.node.parentNode.insertBefore(c.node,a.node);b&&b.add()});
this.parent().add();return this}c=x(c);var b=c.parent();this.node.parentNode.insertBefore(c.node,this.node);this.parent()&&this.parent().add();b&&b.add();c.paper=this.paper;return this};c.after=function(c){c=x(c);var a=c.parent();this.node.nextSibling?this.node.parentNode.insertBefore(c.node,this.node.nextSibling):this.node.parentNode.appendChild(c.node);this.parent()&&this.parent().add();a&&a.add();c.paper=this.paper;return this};c.insertBefore=function(c){c=x(c);var a=this.parent();c.node.parentNode.insertBefore(this.node,
c.node);this.paper=c.paper;a&&a.add();c.parent()&&c.parent().add();return this};c.insertAfter=function(c){c=x(c);var a=this.parent();c.node.parentNode.insertBefore(this.node,c.node.nextSibling);this.paper=c.paper;a&&a.add();c.parent()&&c.parent().add();return this};c.remove=function(){var c=this.parent();this.node.parentNode&&this.node.parentNode.removeChild(this.node);delete this.paper;this.removed=!0;c&&c.add();return this};c.select=function(c){return x(this.node.querySelector(c))};c.selectAll=
function(c){c=this.node.querySelectorAll(c);for(var b=(a.set||Array)(),m=0;m<c.length;m++)b.push(x(c[m]));return b};c.asPX=function(c,a){null==a&&(a=this.attr(c));return+b(this,c,a)};c.use=function(){var c,a=this.node.id;a||(a=this.id,v(this.node,{id:a}));c="linearGradient"==this.type||"radialGradient"==this.type||"pattern"==this.type?r(this.type,this.node.parentNode):r("use",this.node.parentNode);v(c.node,{"xlink:href":"#"+a});c.original=this;return c};var l=/\S+/g;c.addClass=function(c){var a=(c||
"").match(l)||[];c=this.node;var b=c.className.baseVal,m=b.match(l)||[],e,h,d;if(a.length){for(e=0;d=a[e++];)h=m.indexOf(d),~h||m.push(d);a=m.join(" ");b!=a&&(c.className.baseVal=a)}return this};c.removeClass=function(c){var a=(c||"").match(l)||[];c=this.node;var b=c.className.baseVal,m=b.match(l)||[],e,h;if(m.length){for(e=0;h=a[e++];)h=m.indexOf(h),~h&&m.splice(h,1);a=m.join(" ");b!=a&&(c.className.baseVal=a)}return this};c.hasClass=function(c){return!!~(this.node.className.baseVal.match(l)||[]).indexOf(c)};
c.toggleClass=function(c,a){if(null!=a)return a?this.addClass(c):this.removeClass(c);var b=(c||"").match(l)||[],m=this.node,e=m.className.baseVal,h=e.match(l)||[],d,f,E;for(d=0;E=b[d++];)f=h.indexOf(E),~f?h.splice(f,1):h.push(E);b=h.join(" ");e!=b&&(m.className.baseVal=b);return this};c.clone=function(){var c=x(this.node.cloneNode(!0));v(c.node,"id")&&v(c.node,{id:c.id});m(c);c.insertAfter(this);return c};c.toDefs=function(){u(this).appendChild(this.node);return this};c.pattern=c.toPattern=function(c,
a,b,m){var e=r("pattern",u(this));null==c&&(c=this.getBBox());y(c,"object")&&"x"in c&&(a=c.y,b=c.width,m=c.height,c=c.x);v(e.node,{x:c,y:a,width:b,height:m,patternUnits:"userSpaceOnUse",id:e.id,viewBox:[c,a,b,m].join(" ")});e.node.appendChild(this.node);return e};c.marker=function(c,a,b,m,e,h){var d=r("marker",u(this));null==c&&(c=this.getBBox());y(c,"object")&&"x"in c&&(a=c.y,b=c.width,m=c.height,e=c.refX||c.cx,h=c.refY||c.cy,c=c.x);v(d.node,{viewBox:[c,a,b,m].join(" "),markerWidth:b,markerHeight:m,
orient:"auto",refX:e||0,refY:h||0,id:d.id});d.node.appendChild(this.node);return d};var E=function(c,a,b,m){"function"!=typeof b||b.length||(m=b,b=L.linear);this.attr=c;this.dur=a;b&&(this.easing=b);m&&(this.callback=m)};a._.Animation=E;a.animation=function(c,a,b,m){return new E(c,a,b,m)};c.inAnim=function(){var c=[],a;for(a in this.anims)this.anims[h](a)&&function(a){c.push({anim:new E(a._attrs,a.dur,a.easing,a._callback),mina:a,curStatus:a.status(),status:function(c){return a.status(c)},stop:function(){a.stop()}})}(this.anims[a]);
return c};a.animate=function(c,a,b,m,e,h){"function"!=typeof e||e.length||(h=e,e=L.linear);var d=L.time();c=L(c,a,d,d+m,L.time,b,e);h&&k.once("mina.finish."+c.id,h);return c};c.stop=function(){for(var c=this.inAnim(),a=0,b=c.length;a<b;a++)c[a].stop();return this};c.animate=function(c,a,b,m){"function"!=typeof b||b.length||(m=b,b=L.linear);c instanceof E&&(m=c.callback,b=c.easing,a=b.dur,c=c.attr);var d=[],f=[],l={},t,ca,n,T=this,q;for(q in c)if(c[h](q)){T.equal?(n=T.equal(q,J(c[q])),t=n.from,ca=
n.to,n=n.f):(t=+T.attr(q),ca=+c[q]);var la=y(t,"array")?t.length:1;l[q]=e(d.length,d.length+la,n);d=d.concat(t);f=f.concat(ca)}t=L.time();var p=L(d,f,t,t+a,L.time,function(c){var a={},b;for(b in l)l[h](b)&&(a[b]=l[b](c));T.attr(a)},b);T.anims[p.id]=p;p._attrs=c;p._callback=m;k("snap.animcreated."+T.id,p);k.once("mina.finish."+p.id,function(){delete T.anims[p.id];m&&m.call(T)});k.once("mina.stop."+p.id,function(){delete T.anims[p.id]});return T};var T={};c.data=function(c,b){var m=T[this.id]=T[this.id]||
{};if(0==arguments.length)return k("snap.data.get."+this.id,this,m,null),m;if(1==arguments.length){if(a.is(c,"object")){for(var e in c)c[h](e)&&this.data(e,c[e]);return this}k("snap.data.get."+this.id,this,m[c],c);return m[c]}m[c]=b;k("snap.data.set."+this.id,this,b,c);return this};c.removeData=function(c){null==c?T[this.id]={}:T[this.id]&&delete T[this.id][c];return this};c.outerSVG=c.toString=d(1);c.innerSVG=d()})(e.prototype);a.parse=function(c){var a=G.doc.createDocumentFragment(),b=!0,m=G.doc.createElement("div");
c=J(c);c.match(/^\s*<\s*svg(?:\s|>)/)||(c="<svg>"+c+"</svg>",b=!1);m.innerHTML=c;if(c=m.getElementsByTagName("svg")[0])if(b)a=c;else for(;c.firstChild;)a.appendChild(c.firstChild);m.innerHTML=aa;return new l(a)};l.prototype.select=e.prototype.select;l.prototype.selectAll=e.prototype.selectAll;a.fragment=function(){for(var c=Array.prototype.slice.call(arguments,0),b=G.doc.createDocumentFragment(),m=0,e=c.length;m<e;m++){var h=c[m];h.node&&h.node.nodeType&&b.appendChild(h.node);h.nodeType&&b.appendChild(h);
"string"==typeof h&&b.appendChild(a.parse(h).node)}return new l(b)};a._.make=r;a._.wrap=x;s.prototype.el=function(c,a){var b=r(c,this.node);a&&b.attr(a);return b};k.on("snap.util.getattr",function(){var c=k.nt(),c=c.substring(c.lastIndexOf(".")+1),a=c.replace(/[A-Z]/g,function(c){return"-"+c.toLowerCase()});return pa[h](a)?this.node.ownerDocument.defaultView.getComputedStyle(this.node,null).getPropertyValue(a):v(this.node,c)});var pa={"alignment-baseline":0,"baseline-shift":0,clip:0,"clip-path":0,
"clip-rule":0,color:0,"color-interpolation":0,"color-interpolation-filters":0,"color-profile":0,"color-rendering":0,cursor:0,direction:0,display:0,"dominant-baseline":0,"enable-background":0,fill:0,"fill-opacity":0,"fill-rule":0,filter:0,"flood-color":0,"flood-opacity":0,font:0,"font-family":0,"font-size":0,"font-size-adjust":0,"font-stretch":0,"font-style":0,"font-variant":0,"font-weight":0,"glyph-orientation-horizontal":0,"glyph-orientation-vertical":0,"image-rendering":0,kerning:0,"letter-spacing":0,
"lighting-color":0,marker:0,"marker-end":0,"marker-mid":0,"marker-start":0,mask:0,opacity:0,overflow:0,"pointer-events":0,"shape-rendering":0,"stop-color":0,"stop-opacity":0,stroke:0,"stroke-dasharray":0,"stroke-dashoffset":0,"stroke-linecap":0,"stroke-linejoin":0,"stroke-miterlimit":0,"stroke-opacity":0,"stroke-width":0,"text-anchor":0,"text-decoration":0,"text-rendering":0,"unicode-bidi":0,visibility:0,"word-spacing":0,"writing-mode":0};k.on("snap.util.attr",function(c){var a=k.nt(),b={},a=a.substring(a.lastIndexOf(".")+
1);b[a]=c;var m=a.replace(/-(\w)/gi,function(c,a){return a.toUpperCase()}),a=a.replace(/[A-Z]/g,function(c){return"-"+c.toLowerCase()});pa[h](a)?this.node.style[m]=null==c?aa:c:v(this.node,b)});a.ajax=function(c,a,b,m){var e=new XMLHttpRequest,h=V();if(e){if(y(a,"function"))m=b,b=a,a=null;else if(y(a,"object")){var d=[],f;for(f in a)a.hasOwnProperty(f)&&d.push(encodeURIComponent(f)+"="+encodeURIComponent(a[f]));a=d.join("&")}e.open(a?"POST":"GET",c,!0);a&&(e.setRequestHeader("X-Requested-With","XMLHttpRequest"),
e.setRequestHeader("Content-type","application/x-www-form-urlencoded"));b&&(k.once("snap.ajax."+h+".0",b),k.once("snap.ajax."+h+".200",b),k.once("snap.ajax."+h+".304",b));e.onreadystatechange=function(){4==e.readyState&&k("snap.ajax."+h+"."+e.status,m,e)};if(4==e.readyState)return e;e.send(a);return e}};a.load=function(c,b,m){a.ajax(c,function(c){c=a.parse(c.responseText);m?b.call(m,c):b(c)})};a.getElementByPoint=function(c,a){var b,m,e=G.doc.elementFromPoint(c,a);if(G.win.opera&&"svg"==e.tagName){b=
e;m=b.getBoundingClientRect();b=b.ownerDocument;var h=b.body,d=b.documentElement;b=m.top+(g.win.pageYOffset||d.scrollTop||h.scrollTop)-(d.clientTop||h.clientTop||0);m=m.left+(g.win.pageXOffset||d.scrollLeft||h.scrollLeft)-(d.clientLeft||h.clientLeft||0);h=e.createSVGRect();h.x=c-m;h.y=a-b;h.width=h.height=1;b=e.getIntersectionList(h,null);b.length&&(e=b[b.length-1])}return e?x(e):null};a.plugin=function(c){c(a,e,s,G,l)};return G.win.Snap=a}();C.plugin(function(a,k,y,M,A){function w(a,d,f,b,q,e){null==
d&&"[object SVGMatrix]"==z.call(a)?(this.a=a.a,this.b=a.b,this.c=a.c,this.d=a.d,this.e=a.e,this.f=a.f):null!=a?(this.a=+a,this.b=+d,this.c=+f,this.d=+b,this.e=+q,this.f=+e):(this.a=1,this.c=this.b=0,this.d=1,this.f=this.e=0)}var z=Object.prototype.toString,d=String,f=Math;(function(n){function k(a){return a[0]*a[0]+a[1]*a[1]}function p(a){var d=f.sqrt(k(a));a[0]&&(a[0]/=d);a[1]&&(a[1]/=d)}n.add=function(a,d,e,f,n,p){var k=[[],[],[] ],u=[[this.a,this.c,this.e],[this.b,this.d,this.f],[0,0,1] ];d=[[a,
e,n],[d,f,p],[0,0,1] ];a&&a instanceof w&&(d=[[a.a,a.c,a.e],[a.b,a.d,a.f],[0,0,1] ]);for(a=0;3>a;a++)for(e=0;3>e;e++){for(f=n=0;3>f;f++)n+=u[a][f]*d[f][e];k[a][e]=n}this.a=k[0][0];this.b=k[1][0];this.c=k[0][1];this.d=k[1][1];this.e=k[0][2];this.f=k[1][2];return this};n.invert=function(){var a=this.a*this.d-this.b*this.c;return new w(this.d/a,-this.b/a,-this.c/a,this.a/a,(this.c*this.f-this.d*this.e)/a,(this.b*this.e-this.a*this.f)/a)};n.clone=function(){return new w(this.a,this.b,this.c,this.d,this.e,
this.f)};n.translate=function(a,d){return this.add(1,0,0,1,a,d)};n.scale=function(a,d,e,f){null==d&&(d=a);(e||f)&&this.add(1,0,0,1,e,f);this.add(a,0,0,d,0,0);(e||f)&&this.add(1,0,0,1,-e,-f);return this};n.rotate=function(b,d,e){b=a.rad(b);d=d||0;e=e||0;var l=+f.cos(b).toFixed(9);b=+f.sin(b).toFixed(9);this.add(l,b,-b,l,d,e);return this.add(1,0,0,1,-d,-e)};n.x=function(a,d){return a*this.a+d*this.c+this.e};n.y=function(a,d){return a*this.b+d*this.d+this.f};n.get=function(a){return+this[d.fromCharCode(97+
a)].toFixed(4)};n.toString=function(){return"matrix("+[this.get(0),this.get(1),this.get(2),this.get(3),this.get(4),this.get(5)].join()+")"};n.offset=function(){return[this.e.toFixed(4),this.f.toFixed(4)]};n.determinant=function(){return this.a*this.d-this.b*this.c};n.split=function(){var b={};b.dx=this.e;b.dy=this.f;var d=[[this.a,this.c],[this.b,this.d] ];b.scalex=f.sqrt(k(d[0]));p(d[0]);b.shear=d[0][0]*d[1][0]+d[0][1]*d[1][1];d[1]=[d[1][0]-d[0][0]*b.shear,d[1][1]-d[0][1]*b.shear];b.scaley=f.sqrt(k(d[1]));
p(d[1]);b.shear/=b.scaley;0>this.determinant()&&(b.scalex=-b.scalex);var e=-d[0][1],d=d[1][1];0>d?(b.rotate=a.deg(f.acos(d)),0>e&&(b.rotate=360-b.rotate)):b.rotate=a.deg(f.asin(e));b.isSimple=!+b.shear.toFixed(9)&&(b.scalex.toFixed(9)==b.scaley.toFixed(9)||!b.rotate);b.isSuperSimple=!+b.shear.toFixed(9)&&b.scalex.toFixed(9)==b.scaley.toFixed(9)&&!b.rotate;b.noRotation=!+b.shear.toFixed(9)&&!b.rotate;return b};n.toTransformString=function(a){a=a||this.split();if(+a.shear.toFixed(9))return"m"+[this.get(0),
this.get(1),this.get(2),this.get(3),this.get(4),this.get(5)];a.scalex=+a.scalex.toFixed(4);a.scaley=+a.scaley.toFixed(4);a.rotate=+a.rotate.toFixed(4);return(a.dx||a.dy?"t"+[+a.dx.toFixed(4),+a.dy.toFixed(4)]:"")+(1!=a.scalex||1!=a.scaley?"s"+[a.scalex,a.scaley,0,0]:"")+(a.rotate?"r"+[+a.rotate.toFixed(4),0,0]:"")}})(w.prototype);a.Matrix=w;a.matrix=function(a,d,f,b,k,e){return new w(a,d,f,b,k,e)}});C.plugin(function(a,v,y,M,A){function w(h){return function(d){k.stop();d instanceof A&&1==d.node.childNodes.length&&
("radialGradient"==d.node.firstChild.tagName||"linearGradient"==d.node.firstChild.tagName||"pattern"==d.node.firstChild.tagName)&&(d=d.node.firstChild,b(this).appendChild(d),d=u(d));if(d instanceof v)if("radialGradient"==d.type||"linearGradient"==d.type||"pattern"==d.type){d.node.id||e(d.node,{id:d.id});var f=l(d.node.id)}else f=d.attr(h);else f=a.color(d),f.error?(f=a(b(this).ownerSVGElement).gradient(d))?(f.node.id||e(f.node,{id:f.id}),f=l(f.node.id)):f=d:f=r(f);d={};d[h]=f;e(this.node,d);this.node.style[h]=
x}}function z(a){k.stop();a==+a&&(a+="px");this.node.style.fontSize=a}function d(a){var b=[];a=a.childNodes;for(var e=0,f=a.length;e<f;e++){var l=a[e];3==l.nodeType&&b.push(l.nodeValue);"tspan"==l.tagName&&(1==l.childNodes.length&&3==l.firstChild.nodeType?b.push(l.firstChild.nodeValue):b.push(d(l)))}return b}function f(){k.stop();return this.node.style.fontSize}var n=a._.make,u=a._.wrap,p=a.is,b=a._.getSomeDefs,q=/^url\(#?([^)]+)\)$/,e=a._.$,l=a.url,r=String,s=a._.separator,x="";k.on("snap.util.attr.mask",
function(a){if(a instanceof v||a instanceof A){k.stop();a instanceof A&&1==a.node.childNodes.length&&(a=a.node.firstChild,b(this).appendChild(a),a=u(a));if("mask"==a.type)var d=a;else d=n("mask",b(this)),d.node.appendChild(a.node);!d.node.id&&e(d.node,{id:d.id});e(this.node,{mask:l(d.id)})}});(function(a){k.on("snap.util.attr.clip",a);k.on("snap.util.attr.clip-path",a);k.on("snap.util.attr.clipPath",a)})(function(a){if(a instanceof v||a instanceof A){k.stop();if("clipPath"==a.type)var d=a;else d=
n("clipPath",b(this)),d.node.appendChild(a.node),!d.node.id&&e(d.node,{id:d.id});e(this.node,{"clip-path":l(d.id)})}});k.on("snap.util.attr.fill",w("fill"));k.on("snap.util.attr.stroke",w("stroke"));var G=/^([lr])(?:\(([^)]*)\))?(.*)$/i;k.on("snap.util.grad.parse",function(a){a=r(a);var b=a.match(G);if(!b)return null;a=b[1];var e=b[2],b=b[3],e=e.split(/\s*,\s*/).map(function(a){return+a==a?+a:a});1==e.length&&0==e[0]&&(e=[]);b=b.split("-");b=b.map(function(a){a=a.split(":");var b={color:a[0]};a[1]&&
(b.offset=parseFloat(a[1]));return b});return{type:a,params:e,stops:b}});k.on("snap.util.attr.d",function(b){k.stop();p(b,"array")&&p(b[0],"array")&&(b=a.path.toString.call(b));b=r(b);b.match(/[ruo]/i)&&(b=a.path.toAbsolute(b));e(this.node,{d:b})})(-1);k.on("snap.util.attr.#text",function(a){k.stop();a=r(a);for(a=M.doc.createTextNode(a);this.node.firstChild;)this.node.removeChild(this.node.firstChild);this.node.appendChild(a)})(-1);k.on("snap.util.attr.path",function(a){k.stop();this.attr({d:a})})(-1);
k.on("snap.util.attr.class",function(a){k.stop();this.node.className.baseVal=a})(-1);k.on("snap.util.attr.viewBox",function(a){a=p(a,"object")&&"x"in a?[a.x,a.y,a.width,a.height].join(" "):p(a,"array")?a.join(" "):a;e(this.node,{viewBox:a});k.stop()})(-1);k.on("snap.util.attr.transform",function(a){this.transform(a);k.stop()})(-1);k.on("snap.util.attr.r",function(a){"rect"==this.type&&(k.stop(),e(this.node,{rx:a,ry:a}))})(-1);k.on("snap.util.attr.textpath",function(a){k.stop();if("text"==this.type){var d,
f;if(!a&&this.textPath){for(a=this.textPath;a.node.firstChild;)this.node.appendChild(a.node.firstChild);a.remove();delete this.textPath}else if(p(a,"string")?(d=b(this),a=u(d.parentNode).path(a),d.appendChild(a.node),d=a.id,a.attr({id:d})):(a=u(a),a instanceof v&&(d=a.attr("id"),d||(d=a.id,a.attr({id:d})))),d)if(a=this.textPath,f=this.node,a)a.attr({"xlink:href":"#"+d});else{for(a=e("textPath",{"xlink:href":"#"+d});f.firstChild;)a.appendChild(f.firstChild);f.appendChild(a);this.textPath=u(a)}}})(-1);
k.on("snap.util.attr.text",function(a){if("text"==this.type){for(var b=this.node,d=function(a){var b=e("tspan");if(p(a,"array"))for(var f=0;f<a.length;f++)b.appendChild(d(a[f]));else b.appendChild(M.doc.createTextNode(a));b.normalize&&b.normalize();return b};b.firstChild;)b.removeChild(b.firstChild);for(a=d(a);a.firstChild;)b.appendChild(a.firstChild)}k.stop()})(-1);k.on("snap.util.attr.fontSize",z)(-1);k.on("snap.util.attr.font-size",z)(-1);k.on("snap.util.getattr.transform",function(){k.stop();
return this.transform()})(-1);k.on("snap.util.getattr.textpath",function(){k.stop();return this.textPath})(-1);(function(){function b(d){return function(){k.stop();var b=M.doc.defaultView.getComputedStyle(this.node,null).getPropertyValue("marker-"+d);return"none"==b?b:a(M.doc.getElementById(b.match(q)[1]))}}function d(a){return function(b){k.stop();var d="marker"+a.charAt(0).toUpperCase()+a.substring(1);if(""==b||!b)this.node.style[d]="none";else if("marker"==b.type){var f=b.node.id;f||e(b.node,{id:b.id});
this.node.style[d]=l(f)}}}k.on("snap.util.getattr.marker-end",b("end"))(-1);k.on("snap.util.getattr.markerEnd",b("end"))(-1);k.on("snap.util.getattr.marker-start",b("start"))(-1);k.on("snap.util.getattr.markerStart",b("start"))(-1);k.on("snap.util.getattr.marker-mid",b("mid"))(-1);k.on("snap.util.getattr.markerMid",b("mid"))(-1);k.on("snap.util.attr.marker-end",d("end"))(-1);k.on("snap.util.attr.markerEnd",d("end"))(-1);k.on("snap.util.attr.marker-start",d("start"))(-1);k.on("snap.util.attr.markerStart",
d("start"))(-1);k.on("snap.util.attr.marker-mid",d("mid"))(-1);k.on("snap.util.attr.markerMid",d("mid"))(-1)})();k.on("snap.util.getattr.r",function(){if("rect"==this.type&&e(this.node,"rx")==e(this.node,"ry"))return k.stop(),e(this.node,"rx")})(-1);k.on("snap.util.getattr.text",function(){if("text"==this.type||"tspan"==this.type){k.stop();var a=d(this.node);return 1==a.length?a[0]:a}})(-1);k.on("snap.util.getattr.#text",function(){return this.node.textContent})(-1);k.on("snap.util.getattr.viewBox",
function(){k.stop();var b=e(this.node,"viewBox");if(b)return b=b.split(s),a._.box(+b[0],+b[1],+b[2],+b[3])})(-1);k.on("snap.util.getattr.points",function(){var a=e(this.node,"points");k.stop();if(a)return a.split(s)})(-1);k.on("snap.util.getattr.path",function(){var a=e(this.node,"d");k.stop();return a})(-1);k.on("snap.util.getattr.class",function(){return this.node.className.baseVal})(-1);k.on("snap.util.getattr.fontSize",f)(-1);k.on("snap.util.getattr.font-size",f)(-1)});C.plugin(function(a,v,y,
M,A){function w(a){return a}function z(a){return function(b){return+b.toFixed(3)+a}}var d={"+":function(a,b){return a+b},"-":function(a,b){return a-b},"/":function(a,b){return a/b},"*":function(a,b){return a*b}},f=String,n=/[a-z]+$/i,u=/^\s*([+\-\/*])\s*=\s*([\d.eE+\-]+)\s*([^\d\s]+)?\s*$/;k.on("snap.util.attr",function(a){if(a=f(a).match(u)){var b=k.nt(),b=b.substring(b.lastIndexOf(".")+1),q=this.attr(b),e={};k.stop();var l=a[3]||"",r=q.match(n),s=d[a[1] ];r&&r==l?a=s(parseFloat(q),+a[2]):(q=this.asPX(b),
a=s(this.asPX(b),this.asPX(b,a[2]+l)));isNaN(q)||isNaN(a)||(e[b]=a,this.attr(e))}})(-10);k.on("snap.util.equal",function(a,b){var q=f(this.attr(a)||""),e=f(b).match(u);if(e){k.stop();var l=e[3]||"",r=q.match(n),s=d[e[1] ];if(r&&r==l)return{from:parseFloat(q),to:s(parseFloat(q),+e[2]),f:z(r)};q=this.asPX(a);return{from:q,to:s(q,this.asPX(a,e[2]+l)),f:w}}})(-10)});C.plugin(function(a,v,y,M,A){var w=y.prototype,z=a.is;w.rect=function(a,d,k,p,b,q){var e;null==q&&(q=b);z(a,"object")&&"[object Object]"==
a?e=a:null!=a&&(e={x:a,y:d,width:k,height:p},null!=b&&(e.rx=b,e.ry=q));return this.el("rect",e)};w.circle=function(a,d,k){var p;z(a,"object")&&"[object Object]"==a?p=a:null!=a&&(p={cx:a,cy:d,r:k});return this.el("circle",p)};var d=function(){function a(){this.parentNode.removeChild(this)}return function(d,k){var p=M.doc.createElement("img"),b=M.doc.body;p.style.cssText="position:absolute;left:-9999em;top:-9999em";p.onload=function(){k.call(p);p.onload=p.onerror=null;b.removeChild(p)};p.onerror=a;
b.appendChild(p);p.src=d}}();w.image=function(f,n,k,p,b){var q=this.el("image");if(z(f,"object")&&"src"in f)q.attr(f);else if(null!=f){var e={"xlink:href":f,preserveAspectRatio:"none"};null!=n&&null!=k&&(e.x=n,e.y=k);null!=p&&null!=b?(e.width=p,e.height=b):d(f,function(){a._.$(q.node,{width:this.offsetWidth,height:this.offsetHeight})});a._.$(q.node,e)}return q};w.ellipse=function(a,d,k,p){var b;z(a,"object")&&"[object Object]"==a?b=a:null!=a&&(b={cx:a,cy:d,rx:k,ry:p});return this.el("ellipse",b)};
w.path=function(a){var d;z(a,"object")&&!z(a,"array")?d=a:a&&(d={d:a});return this.el("path",d)};w.group=w.g=function(a){var d=this.el("g");1==arguments.length&&a&&!a.type?d.attr(a):arguments.length&&d.add(Array.prototype.slice.call(arguments,0));return d};w.svg=function(a,d,k,p,b,q,e,l){var r={};z(a,"object")&&null==d?r=a:(null!=a&&(r.x=a),null!=d&&(r.y=d),null!=k&&(r.width=k),null!=p&&(r.height=p),null!=b&&null!=q&&null!=e&&null!=l&&(r.viewBox=[b,q,e,l]));return this.el("svg",r)};w.mask=function(a){var d=
this.el("mask");1==arguments.length&&a&&!a.type?d.attr(a):arguments.length&&d.add(Array.prototype.slice.call(arguments,0));return d};w.ptrn=function(a,d,k,p,b,q,e,l){if(z(a,"object"))var r=a;else arguments.length?(r={},null!=a&&(r.x=a),null!=d&&(r.y=d),null!=k&&(r.width=k),null!=p&&(r.height=p),null!=b&&null!=q&&null!=e&&null!=l&&(r.viewBox=[b,q,e,l])):r={patternUnits:"userSpaceOnUse"};return this.el("pattern",r)};w.use=function(a){return null!=a?(make("use",this.node),a instanceof v&&(a.attr("id")||
a.attr({id:ID()}),a=a.attr("id")),this.el("use",{"xlink:href":a})):v.prototype.use.call(this)};w.text=function(a,d,k){var p={};z(a,"object")?p=a:null!=a&&(p={x:a,y:d,text:k||""});return this.el("text",p)};w.line=function(a,d,k,p){var b={};z(a,"object")?b=a:null!=a&&(b={x1:a,x2:k,y1:d,y2:p});return this.el("line",b)};w.polyline=function(a){1<arguments.length&&(a=Array.prototype.slice.call(arguments,0));var d={};z(a,"object")&&!z(a,"array")?d=a:null!=a&&(d={points:a});return this.el("polyline",d)};
w.polygon=function(a){1<arguments.length&&(a=Array.prototype.slice.call(arguments,0));var d={};z(a,"object")&&!z(a,"array")?d=a:null!=a&&(d={points:a});return this.el("polygon",d)};(function(){function d(){return this.selectAll("stop")}function n(b,d){var f=e("stop"),k={offset:+d+"%"};b=a.color(b);k["stop-color"]=b.hex;1>b.opacity&&(k["stop-opacity"]=b.opacity);e(f,k);this.node.appendChild(f);return this}function u(){if("linearGradient"==this.type){var b=e(this.node,"x1")||0,d=e(this.node,"x2")||
1,f=e(this.node,"y1")||0,k=e(this.node,"y2")||0;return a._.box(b,f,math.abs(d-b),math.abs(k-f))}b=this.node.r||0;return a._.box((this.node.cx||0.5)-b,(this.node.cy||0.5)-b,2*b,2*b)}function p(a,d){function f(a,b){for(var d=(b-u)/(a-w),e=w;e<a;e++)h[e].offset=+(+u+d*(e-w)).toFixed(2);w=a;u=b}var n=k("snap.util.grad.parse",null,d).firstDefined(),p;if(!n)return null;n.params.unshift(a);p="l"==n.type.toLowerCase()?b.apply(0,n.params):q.apply(0,n.params);n.type!=n.type.toLowerCase()&&e(p.node,{gradientUnits:"userSpaceOnUse"});
var h=n.stops,n=h.length,u=0,w=0;n--;for(var v=0;v<n;v++)"offset"in h[v]&&f(v,h[v].offset);h[n].offset=h[n].offset||100;f(n,h[n].offset);for(v=0;v<=n;v++){var y=h[v];p.addStop(y.color,y.offset)}return p}function b(b,k,p,q,w){b=a._.make("linearGradient",b);b.stops=d;b.addStop=n;b.getBBox=u;null!=k&&e(b.node,{x1:k,y1:p,x2:q,y2:w});return b}function q(b,k,p,q,w,h){b=a._.make("radialGradient",b);b.stops=d;b.addStop=n;b.getBBox=u;null!=k&&e(b.node,{cx:k,cy:p,r:q});null!=w&&null!=h&&e(b.node,{fx:w,fy:h});
return b}var e=a._.$;w.gradient=function(a){return p(this.defs,a)};w.gradientLinear=function(a,d,e,f){return b(this.defs,a,d,e,f)};w.gradientRadial=function(a,b,d,e,f){return q(this.defs,a,b,d,e,f)};w.toString=function(){var b=this.node.ownerDocument,d=b.createDocumentFragment(),b=b.createElement("div"),e=this.node.cloneNode(!0);d.appendChild(b);b.appendChild(e);a._.$(e,{xmlns:"http://www.w3.org/2000/svg"});b=b.innerHTML;d.removeChild(d.firstChild);return b};w.clear=function(){for(var a=this.node.firstChild,
b;a;)b=a.nextSibling,"defs"!=a.tagName?a.parentNode.removeChild(a):w.clear.call({node:a}),a=b}})()});C.plugin(function(a,k,y,M){function A(a){var b=A.ps=A.ps||{};b[a]?b[a].sleep=100:b[a]={sleep:100};setTimeout(function(){for(var d in b)b[L](d)&&d!=a&&(b[d].sleep--,!b[d].sleep&&delete b[d])});return b[a]}function w(a,b,d,e){null==a&&(a=b=d=e=0);null==b&&(b=a.y,d=a.width,e=a.height,a=a.x);return{x:a,y:b,width:d,w:d,height:e,h:e,x2:a+d,y2:b+e,cx:a+d/2,cy:b+e/2,r1:F.min(d,e)/2,r2:F.max(d,e)/2,r0:F.sqrt(d*
d+e*e)/2,path:s(a,b,d,e),vb:[a,b,d,e].join(" ")}}function z(){return this.join(",").replace(N,"$1")}function d(a){a=C(a);a.toString=z;return a}function f(a,b,d,h,f,k,l,n,p){if(null==p)return e(a,b,d,h,f,k,l,n);if(0>p||e(a,b,d,h,f,k,l,n)<p)p=void 0;else{var q=0.5,O=1-q,s;for(s=e(a,b,d,h,f,k,l,n,O);0.01<Z(s-p);)q/=2,O+=(s<p?1:-1)*q,s=e(a,b,d,h,f,k,l,n,O);p=O}return u(a,b,d,h,f,k,l,n,p)}function n(b,d){function e(a){return+(+a).toFixed(3)}return a._.cacher(function(a,h,l){a instanceof k&&(a=a.attr("d"));
a=I(a);for(var n,p,D,q,O="",s={},c=0,t=0,r=a.length;t<r;t++){D=a[t];if("M"==D[0])n=+D[1],p=+D[2];else{q=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6]);if(c+q>h){if(d&&!s.start){n=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6],h-c);O+=["C"+e(n.start.x),e(n.start.y),e(n.m.x),e(n.m.y),e(n.x),e(n.y)];if(l)return O;s.start=O;O=["M"+e(n.x),e(n.y)+"C"+e(n.n.x),e(n.n.y),e(n.end.x),e(n.end.y),e(D[5]),e(D[6])].join();c+=q;n=+D[5];p=+D[6];continue}if(!b&&!d)return n=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6],h-c)}c+=q;n=+D[5];p=+D[6]}O+=
D.shift()+D}s.end=O;return n=b?c:d?s:u(n,p,D[0],D[1],D[2],D[3],D[4],D[5],1)},null,a._.clone)}function u(a,b,d,e,h,f,k,l,n){var p=1-n,q=ma(p,3),s=ma(p,2),c=n*n,t=c*n,r=q*a+3*s*n*d+3*p*n*n*h+t*k,q=q*b+3*s*n*e+3*p*n*n*f+t*l,s=a+2*n*(d-a)+c*(h-2*d+a),t=b+2*n*(e-b)+c*(f-2*e+b),x=d+2*n*(h-d)+c*(k-2*h+d),c=e+2*n*(f-e)+c*(l-2*f+e);a=p*a+n*d;b=p*b+n*e;h=p*h+n*k;f=p*f+n*l;l=90-180*F.atan2(s-x,t-c)/S;return{x:r,y:q,m:{x:s,y:t},n:{x:x,y:c},start:{x:a,y:b},end:{x:h,y:f},alpha:l}}function p(b,d,e,h,f,n,k,l){a.is(b,
"array")||(b=[b,d,e,h,f,n,k,l]);b=U.apply(null,b);return w(b.min.x,b.min.y,b.max.x-b.min.x,b.max.y-b.min.y)}function b(a,b,d){return b>=a.x&&b<=a.x+a.width&&d>=a.y&&d<=a.y+a.height}function q(a,d){a=w(a);d=w(d);return b(d,a.x,a.y)||b(d,a.x2,a.y)||b(d,a.x,a.y2)||b(d,a.x2,a.y2)||b(a,d.x,d.y)||b(a,d.x2,d.y)||b(a,d.x,d.y2)||b(a,d.x2,d.y2)||(a.x<d.x2&&a.x>d.x||d.x<a.x2&&d.x>a.x)&&(a.y<d.y2&&a.y>d.y||d.y<a.y2&&d.y>a.y)}function e(a,b,d,e,h,f,n,k,l){null==l&&(l=1);l=(1<l?1:0>l?0:l)/2;for(var p=[-0.1252,
0.1252,-0.3678,0.3678,-0.5873,0.5873,-0.7699,0.7699,-0.9041,0.9041,-0.9816,0.9816],q=[0.2491,0.2491,0.2335,0.2335,0.2032,0.2032,0.1601,0.1601,0.1069,0.1069,0.0472,0.0472],s=0,c=0;12>c;c++)var t=l*p[c]+l,r=t*(t*(-3*a+9*d-9*h+3*n)+6*a-12*d+6*h)-3*a+3*d,t=t*(t*(-3*b+9*e-9*f+3*k)+6*b-12*e+6*f)-3*b+3*e,s=s+q[c]*F.sqrt(r*r+t*t);return l*s}function l(a,b,d){a=I(a);b=I(b);for(var h,f,l,n,k,s,r,O,x,c,t=d?0:[],w=0,v=a.length;w<v;w++)if(x=a[w],"M"==x[0])h=k=x[1],f=s=x[2];else{"C"==x[0]?(x=[h,f].concat(x.slice(1)),
h=x[6],f=x[7]):(x=[h,f,h,f,k,s,k,s],h=k,f=s);for(var G=0,y=b.length;G<y;G++)if(c=b[G],"M"==c[0])l=r=c[1],n=O=c[2];else{"C"==c[0]?(c=[l,n].concat(c.slice(1)),l=c[6],n=c[7]):(c=[l,n,l,n,r,O,r,O],l=r,n=O);var z;var K=x,B=c;z=d;var H=p(K),J=p(B);if(q(H,J)){for(var H=e.apply(0,K),J=e.apply(0,B),H=~~(H/8),J=~~(J/8),U=[],A=[],F={},M=z?0:[],P=0;P<H+1;P++){var C=u.apply(0,K.concat(P/H));U.push({x:C.x,y:C.y,t:P/H})}for(P=0;P<J+1;P++)C=u.apply(0,B.concat(P/J)),A.push({x:C.x,y:C.y,t:P/J});for(P=0;P<H;P++)for(K=
0;K<J;K++){var Q=U[P],L=U[P+1],B=A[K],C=A[K+1],N=0.001>Z(L.x-Q.x)?"y":"x",S=0.001>Z(C.x-B.x)?"y":"x",R;R=Q.x;var Y=Q.y,V=L.x,ea=L.y,fa=B.x,ga=B.y,ha=C.x,ia=C.y;if(W(R,V)<X(fa,ha)||X(R,V)>W(fa,ha)||W(Y,ea)<X(ga,ia)||X(Y,ea)>W(ga,ia))R=void 0;else{var $=(R*ea-Y*V)*(fa-ha)-(R-V)*(fa*ia-ga*ha),aa=(R*ea-Y*V)*(ga-ia)-(Y-ea)*(fa*ia-ga*ha),ja=(R-V)*(ga-ia)-(Y-ea)*(fa-ha);if(ja){var $=$/ja,aa=aa/ja,ja=+$.toFixed(2),ba=+aa.toFixed(2);R=ja<+X(R,V).toFixed(2)||ja>+W(R,V).toFixed(2)||ja<+X(fa,ha).toFixed(2)||
ja>+W(fa,ha).toFixed(2)||ba<+X(Y,ea).toFixed(2)||ba>+W(Y,ea).toFixed(2)||ba<+X(ga,ia).toFixed(2)||ba>+W(ga,ia).toFixed(2)?void 0:{x:$,y:aa}}else R=void 0}R&&F[R.x.toFixed(4)]!=R.y.toFixed(4)&&(F[R.x.toFixed(4)]=R.y.toFixed(4),Q=Q.t+Z((R[N]-Q[N])/(L[N]-Q[N]))*(L.t-Q.t),B=B.t+Z((R[S]-B[S])/(C[S]-B[S]))*(C.t-B.t),0<=Q&&1>=Q&&0<=B&&1>=B&&(z?M++:M.push({x:R.x,y:R.y,t1:Q,t2:B})))}z=M}else z=z?0:[];if(d)t+=z;else{H=0;for(J=z.length;H<J;H++)z[H].segment1=w,z[H].segment2=G,z[H].bez1=x,z[H].bez2=c;t=t.concat(z)}}}return t}
function r(a){var b=A(a);if(b.bbox)return C(b.bbox);if(!a)return w();a=I(a);for(var d=0,e=0,h=[],f=[],l,n=0,k=a.length;n<k;n++)l=a[n],"M"==l[0]?(d=l[1],e=l[2],h.push(d),f.push(e)):(d=U(d,e,l[1],l[2],l[3],l[4],l[5],l[6]),h=h.concat(d.min.x,d.max.x),f=f.concat(d.min.y,d.max.y),d=l[5],e=l[6]);a=X.apply(0,h);l=X.apply(0,f);h=W.apply(0,h);f=W.apply(0,f);f=w(a,l,h-a,f-l);b.bbox=C(f);return f}function s(a,b,d,e,h){if(h)return[["M",+a+ +h,b],["l",d-2*h,0],["a",h,h,0,0,1,h,h],["l",0,e-2*h],["a",h,h,0,0,1,
-h,h],["l",2*h-d,0],["a",h,h,0,0,1,-h,-h],["l",0,2*h-e],["a",h,h,0,0,1,h,-h],["z"] ];a=[["M",a,b],["l",d,0],["l",0,e],["l",-d,0],["z"] ];a.toString=z;return a}function x(a,b,d,e,h){null==h&&null==e&&(e=d);a=+a;b=+b;d=+d;e=+e;if(null!=h){var f=Math.PI/180,l=a+d*Math.cos(-e*f);a+=d*Math.cos(-h*f);var n=b+d*Math.sin(-e*f);b+=d*Math.sin(-h*f);d=[["M",l,n],["A",d,d,0,+(180<h-e),0,a,b] ]}else d=[["M",a,b],["m",0,-e],["a",d,e,0,1,1,0,2*e],["a",d,e,0,1,1,0,-2*e],["z"] ];d.toString=z;return d}function G(b){var e=
A(b);if(e.abs)return d(e.abs);Q(b,"array")&&Q(b&&b[0],"array")||(b=a.parsePathString(b));if(!b||!b.length)return[["M",0,0] ];var h=[],f=0,l=0,n=0,k=0,p=0;"M"==b[0][0]&&(f=+b[0][1],l=+b[0][2],n=f,k=l,p++,h[0]=["M",f,l]);for(var q=3==b.length&&"M"==b[0][0]&&"R"==b[1][0].toUpperCase()&&"Z"==b[2][0].toUpperCase(),s,r,w=p,c=b.length;w<c;w++){h.push(s=[]);r=b[w];p=r[0];if(p!=p.toUpperCase())switch(s[0]=p.toUpperCase(),s[0]){case "A":s[1]=r[1];s[2]=r[2];s[3]=r[3];s[4]=r[4];s[5]=r[5];s[6]=+r[6]+f;s[7]=+r[7]+
l;break;case "V":s[1]=+r[1]+l;break;case "H":s[1]=+r[1]+f;break;case "R":for(var t=[f,l].concat(r.slice(1)),u=2,v=t.length;u<v;u++)t[u]=+t[u]+f,t[++u]=+t[u]+l;h.pop();h=h.concat(P(t,q));break;case "O":h.pop();t=x(f,l,r[1],r[2]);t.push(t[0]);h=h.concat(t);break;case "U":h.pop();h=h.concat(x(f,l,r[1],r[2],r[3]));s=["U"].concat(h[h.length-1].slice(-2));break;case "M":n=+r[1]+f,k=+r[2]+l;default:for(u=1,v=r.length;u<v;u++)s[u]=+r[u]+(u%2?f:l)}else if("R"==p)t=[f,l].concat(r.slice(1)),h.pop(),h=h.concat(P(t,
q)),s=["R"].concat(r.slice(-2));else if("O"==p)h.pop(),t=x(f,l,r[1],r[2]),t.push(t[0]),h=h.concat(t);else if("U"==p)h.pop(),h=h.concat(x(f,l,r[1],r[2],r[3])),s=["U"].concat(h[h.length-1].slice(-2));else for(t=0,u=r.length;t<u;t++)s[t]=r[t];p=p.toUpperCase();if("O"!=p)switch(s[0]){case "Z":f=+n;l=+k;break;case "H":f=s[1];break;case "V":l=s[1];break;case "M":n=s[s.length-2],k=s[s.length-1];default:f=s[s.length-2],l=s[s.length-1]}}h.toString=z;e.abs=d(h);return h}function h(a,b,d,e){return[a,b,d,e,d,
e]}function J(a,b,d,e,h,f){var l=1/3,n=2/3;return[l*a+n*d,l*b+n*e,l*h+n*d,l*f+n*e,h,f]}function K(b,d,e,h,f,l,n,k,p,s){var r=120*S/180,q=S/180*(+f||0),c=[],t,x=a._.cacher(function(a,b,c){var d=a*F.cos(c)-b*F.sin(c);a=a*F.sin(c)+b*F.cos(c);return{x:d,y:a}});if(s)v=s[0],t=s[1],l=s[2],u=s[3];else{t=x(b,d,-q);b=t.x;d=t.y;t=x(k,p,-q);k=t.x;p=t.y;F.cos(S/180*f);F.sin(S/180*f);t=(b-k)/2;v=(d-p)/2;u=t*t/(e*e)+v*v/(h*h);1<u&&(u=F.sqrt(u),e*=u,h*=u);var u=e*e,w=h*h,u=(l==n?-1:1)*F.sqrt(Z((u*w-u*v*v-w*t*t)/
(u*v*v+w*t*t)));l=u*e*v/h+(b+k)/2;var u=u*-h*t/e+(d+p)/2,v=F.asin(((d-u)/h).toFixed(9));t=F.asin(((p-u)/h).toFixed(9));v=b<l?S-v:v;t=k<l?S-t:t;0>v&&(v=2*S+v);0>t&&(t=2*S+t);n&&v>t&&(v-=2*S);!n&&t>v&&(t-=2*S)}if(Z(t-v)>r){var c=t,w=k,G=p;t=v+r*(n&&t>v?1:-1);k=l+e*F.cos(t);p=u+h*F.sin(t);c=K(k,p,e,h,f,0,n,w,G,[t,c,l,u])}l=t-v;f=F.cos(v);r=F.sin(v);n=F.cos(t);t=F.sin(t);l=F.tan(l/4);e=4/3*e*l;l*=4/3*h;h=[b,d];b=[b+e*r,d-l*f];d=[k+e*t,p-l*n];k=[k,p];b[0]=2*h[0]-b[0];b[1]=2*h[1]-b[1];if(s)return[b,d,k].concat(c);
c=[b,d,k].concat(c).join().split(",");s=[];k=0;for(p=c.length;k<p;k++)s[k]=k%2?x(c[k-1],c[k],q).y:x(c[k],c[k+1],q).x;return s}function U(a,b,d,e,h,f,l,k){for(var n=[],p=[[],[] ],s,r,c,t,q=0;2>q;++q)0==q?(r=6*a-12*d+6*h,s=-3*a+9*d-9*h+3*l,c=3*d-3*a):(r=6*b-12*e+6*f,s=-3*b+9*e-9*f+3*k,c=3*e-3*b),1E-12>Z(s)?1E-12>Z(r)||(s=-c/r,0<s&&1>s&&n.push(s)):(t=r*r-4*c*s,c=F.sqrt(t),0>t||(t=(-r+c)/(2*s),0<t&&1>t&&n.push(t),s=(-r-c)/(2*s),0<s&&1>s&&n.push(s)));for(r=q=n.length;q--;)s=n[q],c=1-s,p[0][q]=c*c*c*a+3*
c*c*s*d+3*c*s*s*h+s*s*s*l,p[1][q]=c*c*c*b+3*c*c*s*e+3*c*s*s*f+s*s*s*k;p[0][r]=a;p[1][r]=b;p[0][r+1]=l;p[1][r+1]=k;p[0].length=p[1].length=r+2;return{min:{x:X.apply(0,p[0]),y:X.apply(0,p[1])},max:{x:W.apply(0,p[0]),y:W.apply(0,p[1])}}}function I(a,b){var e=!b&&A(a);if(!b&&e.curve)return d(e.curve);var f=G(a),l=b&&G(b),n={x:0,y:0,bx:0,by:0,X:0,Y:0,qx:null,qy:null},k={x:0,y:0,bx:0,by:0,X:0,Y:0,qx:null,qy:null},p=function(a,b,c){if(!a)return["C",b.x,b.y,b.x,b.y,b.x,b.y];a[0]in{T:1,Q:1}||(b.qx=b.qy=null);
switch(a[0]){case "M":b.X=a[1];b.Y=a[2];break;case "A":a=["C"].concat(K.apply(0,[b.x,b.y].concat(a.slice(1))));break;case "S":"C"==c||"S"==c?(c=2*b.x-b.bx,b=2*b.y-b.by):(c=b.x,b=b.y);a=["C",c,b].concat(a.slice(1));break;case "T":"Q"==c||"T"==c?(b.qx=2*b.x-b.qx,b.qy=2*b.y-b.qy):(b.qx=b.x,b.qy=b.y);a=["C"].concat(J(b.x,b.y,b.qx,b.qy,a[1],a[2]));break;case "Q":b.qx=a[1];b.qy=a[2];a=["C"].concat(J(b.x,b.y,a[1],a[2],a[3],a[4]));break;case "L":a=["C"].concat(h(b.x,b.y,a[1],a[2]));break;case "H":a=["C"].concat(h(b.x,
b.y,a[1],b.y));break;case "V":a=["C"].concat(h(b.x,b.y,b.x,a[1]));break;case "Z":a=["C"].concat(h(b.x,b.y,b.X,b.Y))}return a},s=function(a,b){if(7<a[b].length){a[b].shift();for(var c=a[b];c.length;)q[b]="A",l&&(u[b]="A"),a.splice(b++,0,["C"].concat(c.splice(0,6)));a.splice(b,1);v=W(f.length,l&&l.length||0)}},r=function(a,b,c,d,e){a&&b&&"M"==a[e][0]&&"M"!=b[e][0]&&(b.splice(e,0,["M",d.x,d.y]),c.bx=0,c.by=0,c.x=a[e][1],c.y=a[e][2],v=W(f.length,l&&l.length||0))},q=[],u=[],c="",t="",x=0,v=W(f.length,
l&&l.length||0);for(;x<v;x++){f[x]&&(c=f[x][0]);"C"!=c&&(q[x]=c,x&&(t=q[x-1]));f[x]=p(f[x],n,t);"A"!=q[x]&&"C"==c&&(q[x]="C");s(f,x);l&&(l[x]&&(c=l[x][0]),"C"!=c&&(u[x]=c,x&&(t=u[x-1])),l[x]=p(l[x],k,t),"A"!=u[x]&&"C"==c&&(u[x]="C"),s(l,x));r(f,l,n,k,x);r(l,f,k,n,x);var w=f[x],z=l&&l[x],y=w.length,U=l&&z.length;n.x=w[y-2];n.y=w[y-1];n.bx=$(w[y-4])||n.x;n.by=$(w[y-3])||n.y;k.bx=l&&($(z[U-4])||k.x);k.by=l&&($(z[U-3])||k.y);k.x=l&&z[U-2];k.y=l&&z[U-1]}l||(e.curve=d(f));return l?[f,l]:f}function P(a,
b){for(var d=[],e=0,h=a.length;h-2*!b>e;e+=2){var f=[{x:+a[e-2],y:+a[e-1]},{x:+a[e],y:+a[e+1]},{x:+a[e+2],y:+a[e+3]},{x:+a[e+4],y:+a[e+5]}];b?e?h-4==e?f[3]={x:+a[0],y:+a[1]}:h-2==e&&(f[2]={x:+a[0],y:+a[1]},f[3]={x:+a[2],y:+a[3]}):f[0]={x:+a[h-2],y:+a[h-1]}:h-4==e?f[3]=f[2]:e||(f[0]={x:+a[e],y:+a[e+1]});d.push(["C",(-f[0].x+6*f[1].x+f[2].x)/6,(-f[0].y+6*f[1].y+f[2].y)/6,(f[1].x+6*f[2].x-f[3].x)/6,(f[1].y+6*f[2].y-f[3].y)/6,f[2].x,f[2].y])}return d}y=k.prototype;var Q=a.is,C=a._.clone,L="hasOwnProperty",
N=/,?([a-z]),?/gi,$=parseFloat,F=Math,S=F.PI,X=F.min,W=F.max,ma=F.pow,Z=F.abs;M=n(1);var na=n(),ba=n(0,1),V=a._unit2px;a.path=A;a.path.getTotalLength=M;a.path.getPointAtLength=na;a.path.getSubpath=function(a,b,d){if(1E-6>this.getTotalLength(a)-d)return ba(a,b).end;a=ba(a,d,1);return b?ba(a,b).end:a};y.getTotalLength=function(){if(this.node.getTotalLength)return this.node.getTotalLength()};y.getPointAtLength=function(a){return na(this.attr("d"),a)};y.getSubpath=function(b,d){return a.path.getSubpath(this.attr("d"),
b,d)};a._.box=w;a.path.findDotsAtSegment=u;a.path.bezierBBox=p;a.path.isPointInsideBBox=b;a.path.isBBoxIntersect=q;a.path.intersection=function(a,b){return l(a,b)};a.path.intersectionNumber=function(a,b){return l(a,b,1)};a.path.isPointInside=function(a,d,e){var h=r(a);return b(h,d,e)&&1==l(a,[["M",d,e],["H",h.x2+10] ],1)%2};a.path.getBBox=r;a.path.get={path:function(a){return a.attr("path")},circle:function(a){a=V(a);return x(a.cx,a.cy,a.r)},ellipse:function(a){a=V(a);return x(a.cx||0,a.cy||0,a.rx,
a.ry)},rect:function(a){a=V(a);return s(a.x||0,a.y||0,a.width,a.height,a.rx,a.ry)},image:function(a){a=V(a);return s(a.x||0,a.y||0,a.width,a.height)},line:function(a){return"M"+[a.attr("x1")||0,a.attr("y1")||0,a.attr("x2"),a.attr("y2")]},polyline:function(a){return"M"+a.attr("points")},polygon:function(a){return"M"+a.attr("points")+"z"},deflt:function(a){a=a.node.getBBox();return s(a.x,a.y,a.width,a.height)}};a.path.toRelative=function(b){var e=A(b),h=String.prototype.toLowerCase;if(e.rel)return d(e.rel);
a.is(b,"array")&&a.is(b&&b[0],"array")||(b=a.parsePathString(b));var f=[],l=0,n=0,k=0,p=0,s=0;"M"==b[0][0]&&(l=b[0][1],n=b[0][2],k=l,p=n,s++,f.push(["M",l,n]));for(var r=b.length;s<r;s++){var q=f[s]=[],x=b[s];if(x[0]!=h.call(x[0]))switch(q[0]=h.call(x[0]),q[0]){case "a":q[1]=x[1];q[2]=x[2];q[3]=x[3];q[4]=x[4];q[5]=x[5];q[6]=+(x[6]-l).toFixed(3);q[7]=+(x[7]-n).toFixed(3);break;case "v":q[1]=+(x[1]-n).toFixed(3);break;case "m":k=x[1],p=x[2];default:for(var c=1,t=x.length;c<t;c++)q[c]=+(x[c]-(c%2?l:
n)).toFixed(3)}else for(f[s]=[],"m"==x[0]&&(k=x[1]+l,p=x[2]+n),q=0,c=x.length;q<c;q++)f[s][q]=x[q];x=f[s].length;switch(f[s][0]){case "z":l=k;n=p;break;case "h":l+=+f[s][x-1];break;case "v":n+=+f[s][x-1];break;default:l+=+f[s][x-2],n+=+f[s][x-1]}}f.toString=z;e.rel=d(f);return f};a.path.toAbsolute=G;a.path.toCubic=I;a.path.map=function(a,b){if(!b)return a;var d,e,h,f,l,n,k;a=I(a);h=0;for(l=a.length;h<l;h++)for(k=a[h],f=1,n=k.length;f<n;f+=2)d=b.x(k[f],k[f+1]),e=b.y(k[f],k[f+1]),k[f]=d,k[f+1]=e;return a};
a.path.toString=z;a.path.clone=d});C.plugin(function(a,v,y,C){var A=Math.max,w=Math.min,z=function(a){this.items=[];this.bindings={};this.length=0;this.type="set";if(a)for(var f=0,n=a.length;f<n;f++)a[f]&&(this[this.items.length]=this.items[this.items.length]=a[f],this.length++)};v=z.prototype;v.push=function(){for(var a,f,n=0,k=arguments.length;n<k;n++)if(a=arguments[n])f=this.items.length,this[f]=this.items[f]=a,this.length++;return this};v.pop=function(){this.length&&delete this[this.length--];
return this.items.pop()};v.forEach=function(a,f){for(var n=0,k=this.items.length;n<k&&!1!==a.call(f,this.items[n],n);n++);return this};v.animate=function(d,f,n,u){"function"!=typeof n||n.length||(u=n,n=L.linear);d instanceof a._.Animation&&(u=d.callback,n=d.easing,f=n.dur,d=d.attr);var p=arguments;if(a.is(d,"array")&&a.is(p[p.length-1],"array"))var b=!0;var q,e=function(){q?this.b=q:q=this.b},l=0,r=u&&function(){l++==this.length&&u.call(this)};return this.forEach(function(a,l){k.once("snap.animcreated."+
a.id,e);b?p[l]&&a.animate.apply(a,p[l]):a.animate(d,f,n,r)})};v.remove=function(){for(;this.length;)this.pop().remove();return this};v.bind=function(a,f,k){var u={};if("function"==typeof f)this.bindings[a]=f;else{var p=k||a;this.bindings[a]=function(a){u[p]=a;f.attr(u)}}return this};v.attr=function(a){var f={},k;for(k in a)if(this.bindings[k])this.bindings[k](a[k]);else f[k]=a[k];a=0;for(k=this.items.length;a<k;a++)this.items[a].attr(f);return this};v.clear=function(){for(;this.length;)this.pop()};
v.splice=function(a,f,k){a=0>a?A(this.length+a,0):a;f=A(0,w(this.length-a,f));var u=[],p=[],b=[],q;for(q=2;q<arguments.length;q++)b.push(arguments[q]);for(q=0;q<f;q++)p.push(this[a+q]);for(;q<this.length-a;q++)u.push(this[a+q]);var e=b.length;for(q=0;q<e+u.length;q++)this.items[a+q]=this[a+q]=q<e?b[q]:u[q-e];for(q=this.items.length=this.length-=f-e;this[q];)delete this[q++];return new z(p)};v.exclude=function(a){for(var f=0,k=this.length;f<k;f++)if(this[f]==a)return this.splice(f,1),!0;return!1};
v.insertAfter=function(a){for(var f=this.items.length;f--;)this.items[f].insertAfter(a);return this};v.getBBox=function(){for(var a=[],f=[],k=[],u=[],p=this.items.length;p--;)if(!this.items[p].removed){var b=this.items[p].getBBox();a.push(b.x);f.push(b.y);k.push(b.x+b.width);u.push(b.y+b.height)}a=w.apply(0,a);f=w.apply(0,f);k=A.apply(0,k);u=A.apply(0,u);return{x:a,y:f,x2:k,y2:u,width:k-a,height:u-f,cx:a+(k-a)/2,cy:f+(u-f)/2}};v.clone=function(a){a=new z;for(var f=0,k=this.items.length;f<k;f++)a.push(this.items[f].clone());
return a};v.toString=function(){return"Snap\u2018s set"};v.type="set";a.set=function(){var a=new z;arguments.length&&a.push.apply(a,Array.prototype.slice.call(arguments,0));return a}});C.plugin(function(a,v,y,C){function A(a){var b=a[0];switch(b.toLowerCase()){case "t":return[b,0,0];case "m":return[b,1,0,0,1,0,0];case "r":return 4==a.length?[b,0,a[2],a[3] ]:[b,0];case "s":return 5==a.length?[b,1,1,a[3],a[4] ]:3==a.length?[b,1,1]:[b,1]}}function w(b,d,f){d=q(d).replace(/\.{3}|\u2026/g,b);b=a.parseTransformString(b)||
[];d=a.parseTransformString(d)||[];for(var k=Math.max(b.length,d.length),p=[],v=[],h=0,w,z,y,I;h<k;h++){y=b[h]||A(d[h]);I=d[h]||A(y);if(y[0]!=I[0]||"r"==y[0].toLowerCase()&&(y[2]!=I[2]||y[3]!=I[3])||"s"==y[0].toLowerCase()&&(y[3]!=I[3]||y[4]!=I[4])){b=a._.transform2matrix(b,f());d=a._.transform2matrix(d,f());p=[["m",b.a,b.b,b.c,b.d,b.e,b.f] ];v=[["m",d.a,d.b,d.c,d.d,d.e,d.f] ];break}p[h]=[];v[h]=[];w=0;for(z=Math.max(y.length,I.length);w<z;w++)w in y&&(p[h][w]=y[w]),w in I&&(v[h][w]=I[w])}return{from:u(p),
to:u(v),f:n(p)}}function z(a){return a}function d(a){return function(b){return+b.toFixed(3)+a}}function f(b){return a.rgb(b[0],b[1],b[2])}function n(a){var b=0,d,f,k,n,h,p,q=[];d=0;for(f=a.length;d<f;d++){h="[";p=['"'+a[d][0]+'"'];k=1;for(n=a[d].length;k<n;k++)p[k]="val["+b++ +"]";h+=p+"]";q[d]=h}return Function("val","return Snap.path.toString.call(["+q+"])")}function u(a){for(var b=[],d=0,f=a.length;d<f;d++)for(var k=1,n=a[d].length;k<n;k++)b.push(a[d][k]);return b}var p={},b=/[a-z]+$/i,q=String;
p.stroke=p.fill="colour";v.prototype.equal=function(a,b){return k("snap.util.equal",this,a,b).firstDefined()};k.on("snap.util.equal",function(e,k){var r,s;r=q(this.attr(e)||"");var x=this;if(r==+r&&k==+k)return{from:+r,to:+k,f:z};if("colour"==p[e])return r=a.color(r),s=a.color(k),{from:[r.r,r.g,r.b,r.opacity],to:[s.r,s.g,s.b,s.opacity],f:f};if("transform"==e||"gradientTransform"==e||"patternTransform"==e)return k instanceof a.Matrix&&(k=k.toTransformString()),a._.rgTransform.test(k)||(k=a._.svgTransform2string(k)),
w(r,k,function(){return x.getBBox(1)});if("d"==e||"path"==e)return r=a.path.toCubic(r,k),{from:u(r[0]),to:u(r[1]),f:n(r[0])};if("points"==e)return r=q(r).split(a._.separator),s=q(k).split(a._.separator),{from:r,to:s,f:function(a){return a}};aUnit=r.match(b);s=q(k).match(b);return aUnit&&aUnit==s?{from:parseFloat(r),to:parseFloat(k),f:d(aUnit)}:{from:this.asPX(e),to:this.asPX(e,k),f:z}})});C.plugin(function(a,v,y,C){var A=v.prototype,w="createTouch"in C.doc;v="click dblclick mousedown mousemove mouseout mouseover mouseup touchstart touchmove touchend touchcancel".split(" ");
var z={mousedown:"touchstart",mousemove:"touchmove",mouseup:"touchend"},d=function(a,b){var d="y"==a?"scrollTop":"scrollLeft",e=b&&b.node?b.node.ownerDocument:C.doc;return e[d in e.documentElement?"documentElement":"body"][d]},f=function(){this.returnValue=!1},n=function(){return this.originalEvent.preventDefault()},u=function(){this.cancelBubble=!0},p=function(){return this.originalEvent.stopPropagation()},b=function(){if(C.doc.addEventListener)return function(a,b,e,f){var k=w&&z[b]?z[b]:b,l=function(k){var l=
d("y",f),q=d("x",f);if(w&&z.hasOwnProperty(b))for(var r=0,u=k.targetTouches&&k.targetTouches.length;r<u;r++)if(k.targetTouches[r].target==a||a.contains(k.targetTouches[r].target)){u=k;k=k.targetTouches[r];k.originalEvent=u;k.preventDefault=n;k.stopPropagation=p;break}return e.call(f,k,k.clientX+q,k.clientY+l)};b!==k&&a.addEventListener(b,l,!1);a.addEventListener(k,l,!1);return function(){b!==k&&a.removeEventListener(b,l,!1);a.removeEventListener(k,l,!1);return!0}};if(C.doc.attachEvent)return function(a,
b,e,h){var k=function(a){a=a||h.node.ownerDocument.window.event;var b=d("y",h),k=d("x",h),k=a.clientX+k,b=a.clientY+b;a.preventDefault=a.preventDefault||f;a.stopPropagation=a.stopPropagation||u;return e.call(h,a,k,b)};a.attachEvent("on"+b,k);return function(){a.detachEvent("on"+b,k);return!0}}}(),q=[],e=function(a){for(var b=a.clientX,e=a.clientY,f=d("y"),l=d("x"),n,p=q.length;p--;){n=q[p];if(w)for(var r=a.touches&&a.touches.length,u;r--;){if(u=a.touches[r],u.identifier==n.el._drag.id||n.el.node.contains(u.target)){b=
u.clientX;e=u.clientY;(a.originalEvent?a.originalEvent:a).preventDefault();break}}else a.preventDefault();b+=l;e+=f;k("snap.drag.move."+n.el.id,n.move_scope||n.el,b-n.el._drag.x,e-n.el._drag.y,b,e,a)}},l=function(b){a.unmousemove(e).unmouseup(l);for(var d=q.length,f;d--;)f=q[d],f.el._drag={},k("snap.drag.end."+f.el.id,f.end_scope||f.start_scope||f.move_scope||f.el,b);q=[]};for(y=v.length;y--;)(function(d){a[d]=A[d]=function(e,f){a.is(e,"function")&&(this.events=this.events||[],this.events.push({name:d,
f:e,unbind:b(this.node||document,d,e,f||this)}));return this};a["un"+d]=A["un"+d]=function(a){for(var b=this.events||[],e=b.length;e--;)if(b[e].name==d&&(b[e].f==a||!a)){b[e].unbind();b.splice(e,1);!b.length&&delete this.events;break}return this}})(v[y]);A.hover=function(a,b,d,e){return this.mouseover(a,d).mouseout(b,e||d)};A.unhover=function(a,b){return this.unmouseover(a).unmouseout(b)};var r=[];A.drag=function(b,d,f,h,n,p){function u(r,v,w){(r.originalEvent||r).preventDefault();this._drag.x=v;
this._drag.y=w;this._drag.id=r.identifier;!q.length&&a.mousemove(e).mouseup(l);q.push({el:this,move_scope:h,start_scope:n,end_scope:p});d&&k.on("snap.drag.start."+this.id,d);b&&k.on("snap.drag.move."+this.id,b);f&&k.on("snap.drag.end."+this.id,f);k("snap.drag.start."+this.id,n||h||this,v,w,r)}if(!arguments.length){var v;return this.drag(function(a,b){this.attr({transform:v+(v?"T":"t")+[a,b]})},function(){v=this.transform().local})}this._drag={};r.push({el:this,start:u});this.mousedown(u);return this};
A.undrag=function(){for(var b=r.length;b--;)r[b].el==this&&(this.unmousedown(r[b].start),r.splice(b,1),k.unbind("snap.drag.*."+this.id));!r.length&&a.unmousemove(e).unmouseup(l);return this}});C.plugin(function(a,v,y,C){y=y.prototype;var A=/^\s*url\((.+)\)/,w=String,z=a._.$;a.filter={};y.filter=function(d){var f=this;"svg"!=f.type&&(f=f.paper);d=a.parse(w(d));var k=a._.id(),u=z("filter");z(u,{id:k,filterUnits:"userSpaceOnUse"});u.appendChild(d.node);f.defs.appendChild(u);return new v(u)};k.on("snap.util.getattr.filter",
function(){k.stop();var d=z(this.node,"filter");if(d)return(d=w(d).match(A))&&a.select(d[1])});k.on("snap.util.attr.filter",function(d){if(d instanceof v&&"filter"==d.type){k.stop();var f=d.node.id;f||(z(d.node,{id:d.id}),f=d.id);z(this.node,{filter:a.url(f)})}d&&"none"!=d||(k.stop(),this.node.removeAttribute("filter"))});a.filter.blur=function(d,f){null==d&&(d=2);return a.format('<feGaussianBlur stdDeviation="{def}"/>',{def:null==f?d:[d,f]})};a.filter.blur.toString=function(){return this()};a.filter.shadow=
function(d,f,k,u,p){"string"==typeof k&&(p=u=k,k=4);"string"!=typeof u&&(p=u,u="#000");null==k&&(k=4);null==p&&(p=1);null==d&&(d=0,f=2);null==f&&(f=d);u=a.color(u||"#000");return a.format('<feGaussianBlur in="SourceAlpha" stdDeviation="{blur}"/><feOffset dx="{dx}" dy="{dy}" result="offsetblur"/><feFlood flood-color="{color}"/><feComposite in2="offsetblur" operator="in"/><feComponentTransfer><feFuncA type="linear" slope="{opacity}"/></feComponentTransfer><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>',
{color:u,dx:d,dy:f,blur:k,opacity:p})};a.filter.shadow.toString=function(){return this()};a.filter.grayscale=function(d){null==d&&(d=1);return a.format('<feColorMatrix type="matrix" values="{a} {b} {c} 0 0 {d} {e} {f} 0 0 {g} {b} {h} 0 0 0 0 0 1 0"/>',{a:0.2126+0.7874*(1-d),b:0.7152-0.7152*(1-d),c:0.0722-0.0722*(1-d),d:0.2126-0.2126*(1-d),e:0.7152+0.2848*(1-d),f:0.0722-0.0722*(1-d),g:0.2126-0.2126*(1-d),h:0.0722+0.9278*(1-d)})};a.filter.grayscale.toString=function(){return this()};a.filter.sepia=
function(d){null==d&&(d=1);return a.format('<feColorMatrix type="matrix" values="{a} {b} {c} 0 0 {d} {e} {f} 0 0 {g} {h} {i} 0 0 0 0 0 1 0"/>',{a:0.393+0.607*(1-d),b:0.769-0.769*(1-d),c:0.189-0.189*(1-d),d:0.349-0.349*(1-d),e:0.686+0.314*(1-d),f:0.168-0.168*(1-d),g:0.272-0.272*(1-d),h:0.534-0.534*(1-d),i:0.131+0.869*(1-d)})};a.filter.sepia.toString=function(){return this()};a.filter.saturate=function(d){null==d&&(d=1);return a.format('<feColorMatrix type="saturate" values="{amount}"/>',{amount:1-
d})};a.filter.saturate.toString=function(){return this()};a.filter.hueRotate=function(d){return a.format('<feColorMatrix type="hueRotate" values="{angle}"/>',{angle:d||0})};a.filter.hueRotate.toString=function(){return this()};a.filter.invert=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="table" tableValues="{amount} {amount2}"/><feFuncG type="table" tableValues="{amount} {amount2}"/><feFuncB type="table" tableValues="{amount} {amount2}"/></feComponentTransfer>',{amount:d,
amount2:1-d})};a.filter.invert.toString=function(){return this()};a.filter.brightness=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="linear" slope="{amount}"/><feFuncG type="linear" slope="{amount}"/><feFuncB type="linear" slope="{amount}"/></feComponentTransfer>',{amount:d})};a.filter.brightness.toString=function(){return this()};a.filter.contrast=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="linear" slope="{amount}" intercept="{amount2}"/><feFuncG type="linear" slope="{amount}" intercept="{amount2}"/><feFuncB type="linear" slope="{amount}" intercept="{amount2}"/></feComponentTransfer>',
{amount:d,amount2:0.5-d/2})};a.filter.contrast.toString=function(){return this()}});return C});

]]> </script>
<script> <![CDATA[

(function (glob, factory) {
    // AMD support
    if (typeof define === "function" && define.amd) {
        // Define as an anonymous module
        define("Gadfly", ["Snap.svg"], function (Snap) {
            return factory(Snap);
        });
    } else {
        // Browser globals (glob is window)
        // Snap adds itself to window
        glob.Gadfly = factory(glob.Snap);
    }
}(this, function (Snap) {

var Gadfly = {};

// Get an x/y coordinate value in pixels
var xPX = function(fig, x) {
    var client_box = fig.node.getBoundingClientRect();
    return x * fig.node.viewBox.baseVal.width / client_box.width;
};

var yPX = function(fig, y) {
    var client_box = fig.node.getBoundingClientRect();
    return y * fig.node.viewBox.baseVal.height / client_box.height;
};


Snap.plugin(function (Snap, Element, Paper, global) {
    // Traverse upwards from a snap element to find and return the first
    // note with the "plotroot" class.
    Element.prototype.plotroot = function () {
        var element = this;
        while (!element.hasClass("plotroot") && element.parent() != null) {
            element = element.parent();
        }
        return element;
    };

    Element.prototype.svgroot = function () {
        var element = this;
        while (element.node.nodeName != "svg" && element.parent() != null) {
            element = element.parent();
        }
        return element;
    };

    Element.prototype.plotbounds = function () {
        var root = this.plotroot()
        var bbox = root.select(".guide.background").node.getBBox();
        return {
            x0: bbox.x,
            x1: bbox.x + bbox.width,
            y0: bbox.y,
            y1: bbox.y + bbox.height
        };
    };

    Element.prototype.plotcenter = function () {
        var root = this.plotroot()
        var bbox = root.select(".guide.background").node.getBBox();
        return {
            x: bbox.x + bbox.width / 2,
            y: bbox.y + bbox.height / 2
        };
    };

    // Emulate IE style mouseenter/mouseleave events, since Microsoft always
    // does everything right.
    // See: http://www.dynamic-tools.net/toolbox/isMouseLeaveOrEnter/
    var events = ["mouseenter", "mouseleave"];

    for (i in events) {
        (function (event_name) {
            var event_name = events[i];
            Element.prototype[event_name] = function (fn, scope) {
                if (Snap.is(fn, "function")) {
                    var fn2 = function (event) {
                        if (event.type != "mouseover" && event.type != "mouseout") {
                            return;
                        }

                        var reltg = event.relatedTarget ? event.relatedTarget :
                            event.type == "mouseout" ? event.toElement : event.fromElement;
                        while (reltg && reltg != this.node) reltg = reltg.parentNode;

                        if (reltg != this.node) {
                            return fn.apply(this, event);
                        }
                    };

                    if (event_name == "mouseenter") {
                        this.mouseover(fn2, scope);
                    } else {
                        this.mouseout(fn2, scope);
                    }
                }
                return this;
            };
        })(events[i]);
    }


    Element.prototype.mousewheel = function (fn, scope) {
        if (Snap.is(fn, "function")) {
            var el = this;
            var fn2 = function (event) {
                fn.apply(el, [event]);
            };
        }

        this.node.addEventListener(
            /Firefox/i.test(navigator.userAgent) ? "DOMMouseScroll" : "mousewheel",
            fn2);

        return this;
    };


    // Snap's attr function can be too slow for things like panning/zooming.
    // This is a function to directly update element attributes without going
    // through eve.
    Element.prototype.attribute = function(key, val) {
        if (val === undefined) {
            return this.node.getAttribute(key);
        } else {
            this.node.setAttribute(key, val);
            return this;
        }
    };
});


// When the plot is moused over, emphasize the grid lines.
Gadfly.plot_mouseover = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);

    var xgridlines = root.select(".xgridlines"),
        ygridlines = root.select(".ygridlines");

    xgridlines.data("unfocused_strokedash",
                    xgridlines.attribute("stroke-dasharray").replace(/(\d)(,|$)/g, "$1mm$2"));
    ygridlines.data("unfocused_strokedash",
                    ygridlines.attribute("stroke-dasharray").replace(/(\d)(,|$)/g, "$1mm$2"));

    // emphasize grid lines
    var destcolor = root.data("focused_xgrid_color");
    xgridlines.attribute("stroke-dasharray", "none")
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    destcolor = root.data("focused_ygrid_color");
    ygridlines.attribute("stroke-dasharray", "none")
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    // reveal zoom slider
    root.select(".zoomslider")
        .animate({opacity: 1.0}, 250);
};


// Unemphasize grid lines on mouse out.
Gadfly.plot_mouseout = function(event) {
    var root = this.plotroot();
    var xgridlines = root.select(".xgridlines"),
        ygridlines = root.select(".ygridlines");

    var destcolor = root.data("unfocused_xgrid_color");

    xgridlines.attribute("stroke-dasharray", xgridlines.data("unfocused_strokedash"))
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    destcolor = root.data("unfocused_ygrid_color");
    ygridlines.attribute("stroke-dasharray", ygridlines.data("unfocused_strokedash"))
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    // hide zoom slider
    root.select(".zoomslider")
        .animate({opacity: 0.0}, 250);
};


var set_geometry_transform = function(root, tx, ty, scale) {
    var xscalable = root.hasClass("xscalable"),
        yscalable = root.hasClass("yscalable");

    var old_scale = root.data("scale");

    var xscale = xscalable ? scale : 1.0,
        yscale = yscalable ? scale : 1.0;

    tx = xscalable ? tx : 0.0;
    ty = yscalable ? ty : 0.0;

    var t = new Snap.Matrix().translate(tx, ty).scale(xscale, yscale);

    root.selectAll(".geometry, image")
        .forEach(function (element, i) {
            element.transform(t);
        });

    bounds = root.plotbounds();

    if (yscalable) {
        var xfixed_t = new Snap.Matrix().translate(0, ty).scale(1.0, yscale);
        root.selectAll(".xfixed")
            .forEach(function (element, i) {
                element.transform(xfixed_t);
            });

        root.select(".ylabels")
            .transform(xfixed_t)
            .selectAll("text")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var cx = element.asPX("x"),
                        cy = element.asPX("y");
                    var st = element.data("static_transform");
                    unscale_t = new Snap.Matrix();
                    unscale_t.scale(1, 1/scale, cx, cy).add(st);
                    element.transform(unscale_t);

                    var y = cy * scale + ty;
                    element.attr("visibility",
                        bounds.y0 <= y && y <= bounds.y1 ? "visible" : "hidden");
                }
            });
    }

    if (xscalable) {
        var yfixed_t = new Snap.Matrix().translate(tx, 0).scale(xscale, 1.0);
        var xtrans = new Snap.Matrix().translate(tx, 0);
        root.selectAll(".yfixed")
            .forEach(function (element, i) {
                element.transform(yfixed_t);
            });

        root.select(".xlabels")
            .transform(yfixed_t)
            .selectAll("text")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var cx = element.asPX("x"),
                        cy = element.asPX("y");
                    var st = element.data("static_transform");
                    unscale_t = new Snap.Matrix();
                    unscale_t.scale(1/scale, 1, cx, cy).add(st);

                    element.transform(unscale_t);

                    var x = cx * scale + tx;
                    element.attr("visibility",
                        bounds.x0 <= x && x <= bounds.x1 ? "visible" : "hidden");
                    }
            });
    }

    // we must unscale anything that is scale invariance: widths, raiduses, etc.
    var size_attribs = ["font-size"];
    var unscaled_selection = ".geometry, .geometry *";
    if (xscalable) {
        size_attribs.push("rx");
        unscaled_selection += ", .xgridlines";
    }
    if (yscalable) {
        size_attribs.push("ry");
        unscaled_selection += ", .ygridlines";
    }

    root.selectAll(unscaled_selection)
        .forEach(function (element, i) {
            // circle need special help
            if (element.node.nodeName == "circle") {
                var cx = element.attribute("cx"),
                    cy = element.attribute("cy");
                unscale_t = new Snap.Matrix().scale(1/xscale, 1/yscale,
                                                        cx, cy);
                element.transform(unscale_t);
                return;
            }

            for (i in size_attribs) {
                var key = size_attribs[i];
                var val = parseFloat(element.attribute(key));
                if (val !== undefined && val != 0 && !isNaN(val)) {
                    element.attribute(key, val * old_scale / scale);
                }
            }
        });
};


// Find the most appropriate tick scale and update label visibility.
var update_tickscale = function(root, scale, axis) {
    if (!root.hasClass(axis + "scalable")) return;

    var tickscales = root.data(axis + "tickscales");
    var best_tickscale = 1.0;
    var best_tickscale_dist = Infinity;
    for (tickscale in tickscales) {
        var dist = Math.abs(Math.log(tickscale) - Math.log(scale));
        if (dist < best_tickscale_dist) {
            best_tickscale_dist = dist;
            best_tickscale = tickscale;
        }
    }

    if (best_tickscale != root.data(axis + "tickscale")) {
        root.data(axis + "tickscale", best_tickscale);
        var mark_inscale_gridlines = function (element, i) {
            var inscale = element.attr("gadfly:scale") == best_tickscale;
            element.attribute("gadfly:inscale", inscale);
            element.attr("visibility", inscale ? "visible" : "hidden");
        };

        var mark_inscale_labels = function (element, i) {
            var inscale = element.attr("gadfly:scale") == best_tickscale;
            element.attribute("gadfly:inscale", inscale);
            element.attr("visibility", inscale ? "visible" : "hidden");
        };

        root.select("." + axis + "gridlines").selectAll("path").forEach(mark_inscale_gridlines);
        root.select("." + axis + "labels").selectAll("text").forEach(mark_inscale_labels);
    }
};


var set_plot_pan_zoom = function(root, tx, ty, scale) {
    var old_scale = root.data("scale");
    var bounds = root.plotbounds();

    var width = bounds.x1 - bounds.x0,
        height = bounds.y1 - bounds.y0;

    // compute the viewport derived from tx, ty, and scale
    var x_min = -width * scale - (scale * width - width),
        x_max = width * scale,
        y_min = -height * scale - (scale * height - height),
        y_max = height * scale;

    var x0 = bounds.x0 - scale * bounds.x0,
        y0 = bounds.y0 - scale * bounds.y0;

    var tx = Math.max(Math.min(tx - x0, x_max), x_min),
        ty = Math.max(Math.min(ty - y0, y_max), y_min);

    tx += x0;
    ty += y0;

    // when the scale change, we may need to alter which set of
    // ticks is being displayed
    if (scale != old_scale) {
        update_tickscale(root, scale, "x");
        update_tickscale(root, scale, "y");
    }

    set_geometry_transform(root, tx, ty, scale);

    root.data("scale", scale);
    root.data("tx", tx);
    root.data("ty", ty);
};


var scale_centered_translation = function(root, scale) {
    var bounds = root.plotbounds();

    var width = bounds.x1 - bounds.x0,
        height = bounds.y1 - bounds.y0;

    var tx0 = root.data("tx"),
        ty0 = root.data("ty");

    var scale0 = root.data("scale");

    // how off from center the current view is
    var xoff = tx0 - (bounds.x0 * (1 - scale0) + (width * (1 - scale0)) / 2),
        yoff = ty0 - (bounds.y0 * (1 - scale0) + (height * (1 - scale0)) / 2);

    // rescale offsets
    xoff = xoff * scale / scale0;
    yoff = yoff * scale / scale0;

    // adjust for the panel position being scaled
    var x_edge_adjust = bounds.x0 * (1 - scale),
        y_edge_adjust = bounds.y0 * (1 - scale);

    return {
        x: xoff + x_edge_adjust + (width - width * scale) / 2,
        y: yoff + y_edge_adjust + (height - height * scale) / 2
    };
};


// Initialize data for panning zooming if it isn't already.
var init_pan_zoom = function(root) {
    if (root.data("zoompan-ready")) {
        return;
    }

    // The non-scaling-stroke trick. Rather than try to correct for the
    // stroke-width when zooming, we force it to a fixed value.
    var px_per_mm = root.node.getCTM().a;

    // Drag events report deltas in pixels, which we'd like to convert to
    // millimeters.
    root.data("px_per_mm", px_per_mm);

    root.selectAll("path")
        .forEach(function (element, i) {
        sw = element.asPX("stroke-width") * px_per_mm;
        if (sw > 0) {
            element.attribute("stroke-width", sw);
            element.attribute("vector-effect", "non-scaling-stroke");
        }
    });

    // Store ticks labels original tranformation
    root.selectAll(".xlabels > text, .ylabels > text")
        .forEach(function (element, i) {
            var lm = element.transform().localMatrix;
            element.data("static_transform",
                new Snap.Matrix(lm.a, lm.b, lm.c, lm.d, lm.e, lm.f));
        });

    var xgridlines = root.select(".xgridlines");
    var ygridlines = root.select(".ygridlines");
    var xlabels = root.select(".xlabels");
    var ylabels = root.select(".ylabels");

    if (root.data("tx") === undefined) root.data("tx", 0);
    if (root.data("ty") === undefined) root.data("ty", 0);
    if (root.data("scale") === undefined) root.data("scale", 1.0);
    if (root.data("xtickscales") === undefined) {

        // index all the tick scales that are listed
        var xtickscales = {};
        var ytickscales = {};
        var add_x_tick_scales = function (element, i) {
            xtickscales[element.attribute("gadfly:scale")] = true;
        };
        var add_y_tick_scales = function (element, i) {
            ytickscales[element.attribute("gadfly:scale")] = true;
        };

        if (xgridlines) xgridlines.selectAll("path").forEach(add_x_tick_scales);
        if (ygridlines) ygridlines.selectAll("path").forEach(add_y_tick_scales);
        if (xlabels) xlabels.selectAll("text").forEach(add_x_tick_scales);
        if (ylabels) ylabels.selectAll("text").forEach(add_y_tick_scales);

        root.data("xtickscales", xtickscales);
        root.data("ytickscales", ytickscales);
        root.data("xtickscale", 1.0);
    }

    var min_scale = 1.0, max_scale = 1.0;
    for (scale in xtickscales) {
        min_scale = Math.min(min_scale, scale);
        max_scale = Math.max(max_scale, scale);
    }
    for (scale in ytickscales) {
        min_scale = Math.min(min_scale, scale);
        max_scale = Math.max(max_scale, scale);
    }
    root.data("min_scale", min_scale);
    root.data("max_scale", max_scale);

    // store the original positions of labels
    if (xlabels) {
        xlabels.selectAll("text")
               .forEach(function (element, i) {
                   element.data("x", element.asPX("x"));
               });
    }

    if (ylabels) {
        ylabels.selectAll("text")
               .forEach(function (element, i) {
                   element.data("y", element.asPX("y"));
               });
    }

    // mark grid lines and ticks as in or out of scale.
    var mark_inscale = function (element, i) {
        element.attribute("gadfly:inscale", element.attribute("gadfly:scale") == 1.0);
    };

    if (xgridlines) xgridlines.selectAll("path").forEach(mark_inscale);
    if (ygridlines) ygridlines.selectAll("path").forEach(mark_inscale);
    if (xlabels) xlabels.selectAll("text").forEach(mark_inscale);
    if (ylabels) ylabels.selectAll("text").forEach(mark_inscale);

    // figure out the upper ond lower bounds on panning using the maximum
    // and minum grid lines
    var bounds = root.plotbounds();
    var pan_bounds = {
        x0: 0.0,
        y0: 0.0,
        x1: 0.0,
        y1: 0.0
    };

    if (xgridlines) {
        xgridlines
            .selectAll("path")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var bbox = element.node.getBBox();
                    if (bounds.x1 - bbox.x < pan_bounds.x0) {
                        pan_bounds.x0 = bounds.x1 - bbox.x;
                    }
                    if (bounds.x0 - bbox.x > pan_bounds.x1) {
                        pan_bounds.x1 = bounds.x0 - bbox.x;
                    }
                    element.attr("visibility", "visible");
                }
            });
    }

    if (ygridlines) {
        ygridlines
            .selectAll("path")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var bbox = element.node.getBBox();
                    if (bounds.y1 - bbox.y < pan_bounds.y0) {
                        pan_bounds.y0 = bounds.y1 - bbox.y;
                    }
                    if (bounds.y0 - bbox.y > pan_bounds.y1) {
                        pan_bounds.y1 = bounds.y0 - bbox.y;
                    }
                    element.attr("visibility", "visible");
                }
            });
    }

    // nudge these values a little
    pan_bounds.x0 -= 5;
    pan_bounds.x1 += 5;
    pan_bounds.y0 -= 5;
    pan_bounds.y1 += 5;
    root.data("pan_bounds", pan_bounds);

    root.data("zoompan-ready", true)
};


// Panning
Gadfly.guide_background_drag_onmove = function(dx, dy, x, y, event) {
    var root = this.plotroot();
    var px_per_mm = root.data("px_per_mm");
    dx /= px_per_mm;
    dy /= px_per_mm;

    var tx0 = root.data("tx"),
        ty0 = root.data("ty");

    var dx0 = root.data("dx"),
        dy0 = root.data("dy");

    root.data("dx", dx);
    root.data("dy", dy);

    dx = dx - dx0;
    dy = dy - dy0;

    var tx = tx0 + dx,
        ty = ty0 + dy;

    set_plot_pan_zoom(root, tx, ty, root.data("scale"));
};


Gadfly.guide_background_drag_onstart = function(x, y, event) {
    var root = this.plotroot();
    root.data("dx", 0);
    root.data("dy", 0);
    init_pan_zoom(root);
};


Gadfly.guide_background_drag_onend = function(event) {
    var root = this.plotroot();
};


Gadfly.guide_background_scroll = function(event) {
    if (event.shiftKey) {
        var root = this.plotroot();
        init_pan_zoom(root);
        var new_scale = root.data("scale") * Math.pow(2, 0.002 * event.wheelDelta);
        new_scale = Math.max(
            root.data("min_scale"),
            Math.min(root.data("max_scale"), new_scale))
        update_plot_scale(root, new_scale);
        event.stopPropagation();
    }
};


Gadfly.zoomslider_button_mouseover = function(event) {
    this.select(".button_logo")
         .animate({fill: this.data("mouseover_color")}, 100);
};


Gadfly.zoomslider_button_mouseout = function(event) {
     this.select(".button_logo")
         .animate({fill: this.data("mouseout_color")}, 100);
};


Gadfly.zoomslider_zoomout_click = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);
    var min_scale = root.data("min_scale"),
        scale = root.data("scale");
    Snap.animate(
        scale,
        Math.max(min_scale, scale / 1.5),
        function (new_scale) {
            update_plot_scale(root, new_scale);
        },
        200);
};


Gadfly.zoomslider_zoomin_click = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);
    var max_scale = root.data("max_scale"),
        scale = root.data("scale");

    Snap.animate(
        scale,
        Math.min(max_scale, scale * 1.5),
        function (new_scale) {
            update_plot_scale(root, new_scale);
        },
        200);
};


Gadfly.zoomslider_track_click = function(event) {
    // TODO
};


Gadfly.zoomslider_thumb_mousedown = function(event) {
    this.animate({fill: this.data("mouseover_color")}, 100);
};


Gadfly.zoomslider_thumb_mouseup = function(event) {
    this.animate({fill: this.data("mouseout_color")}, 100);
};


// compute the position in [0, 1] of the zoom slider thumb from the current scale
var slider_position_from_scale = function(scale, min_scale, max_scale) {
    if (scale >= 1.0) {
        return 0.5 + 0.5 * (Math.log(scale) / Math.log(max_scale));
    }
    else {
        return 0.5 * (Math.log(scale) - Math.log(min_scale)) / (0 - Math.log(min_scale));
    }
}


var update_plot_scale = function(root, new_scale) {
    var trans = scale_centered_translation(root, new_scale);
    set_plot_pan_zoom(root, trans.x, trans.y, new_scale);

    root.selectAll(".zoomslider_thumb")
        .forEach(function (element, i) {
            var min_pos = element.data("min_pos"),
                max_pos = element.data("max_pos"),
                min_scale = root.data("min_scale"),
                max_scale = root.data("max_scale");
            var xmid = (min_pos + max_pos) / 2;
            var xpos = slider_position_from_scale(new_scale, min_scale, max_scale);
            element.transform(new Snap.Matrix().translate(
                Math.max(min_pos, Math.min(
                         max_pos, min_pos + (max_pos - min_pos) * xpos)) - xmid, 0));
    });
};


Gadfly.zoomslider_thumb_dragmove = function(dx, dy, x, y) {
    var root = this.plotroot();
    var min_pos = this.data("min_pos"),
        max_pos = this.data("max_pos"),
        min_scale = root.data("min_scale"),
        max_scale = root.data("max_scale"),
        old_scale = root.data("old_scale");

    var px_per_mm = root.data("px_per_mm");
    dx /= px_per_mm;
    dy /= px_per_mm;

    var xmid = (min_pos + max_pos) / 2;
    var xpos = slider_position_from_scale(old_scale, min_scale, max_scale) +
                   dx / (max_pos - min_pos);

    // compute the new scale
    var new_scale;
    if (xpos >= 0.5) {
        new_scale = Math.exp(2.0 * (xpos - 0.5) * Math.log(max_scale));
    }
    else {
        new_scale = Math.exp(2.0 * xpos * (0 - Math.log(min_scale)) +
                        Math.log(min_scale));
    }
    new_scale = Math.min(max_scale, Math.max(min_scale, new_scale));

    update_plot_scale(root, new_scale);
};


Gadfly.zoomslider_thumb_dragstart = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);

    // keep track of what the scale was when we started dragging
    root.data("old_scale", root.data("scale"));
};


Gadfly.zoomslider_thumb_dragend = function(event) {
};


var toggle_color_class = function(root, color_class, ison) {
    var guides = root.selectAll(".guide." + color_class + ",.guide ." + color_class);
    var geoms = root.selectAll(".geometry." + color_class + ",.geometry ." + color_class);
    if (ison) {
        guides.animate({opacity: 0.5}, 250);
        geoms.animate({opacity: 0.0}, 250);
    } else {
        guides.animate({opacity: 1.0}, 250);
        geoms.animate({opacity: 1.0}, 250);
    }
};


Gadfly.colorkey_swatch_click = function(event) {
    var root = this.plotroot();
    var color_class = this.data("color_class");

    if (event.shiftKey) {
        root.selectAll(".colorkey text")
            .forEach(function (element) {
                var other_color_class = element.data("color_class");
                if (other_color_class != color_class) {
                    toggle_color_class(root, other_color_class,
                                       element.attr("opacity") == 1.0);
                }
            });
    } else {
        toggle_color_class(root, color_class, this.attr("opacity") == 1.0);
    }
};


return Gadfly;

}));


//@ sourceURL=gadfly.js

(function (glob, factory) {
    // AMD support
      if (typeof require === "function" && typeof define === "function" && define.amd) {
        require(["Snap.svg", "Gadfly"], function (Snap, Gadfly) {
            factory(Snap, Gadfly);
        });
      } else {
          factory(glob.Snap, glob.Gadfly);
      }
})(window, function (Snap, Gadfly) {
    var fig = Snap("#fig-e97312c08242438abf7e68ad37f1e090");
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-3")
   .drag(function() {}, function() {}, function() {});
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-5")
   .data("color_class", "color_0")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-6")
   .data("color_class", "color_1")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-8")
   .data("color_class", "color_0")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-9")
   .data("color_class", "color_1")
.click(Gadfly.colorkey_swatch_click)
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-12")
   .mouseenter(Gadfly.plot_mouseover)
.mouseleave(Gadfly.plot_mouseout)
.mousewheel(Gadfly.guide_background_scroll)
.drag(Gadfly.guide_background_drag_onmove,
      Gadfly.guide_background_drag_onstart,
      Gadfly.guide_background_drag_onend)
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-25")
   .mouseenter(Gadfly.plot_mouseover)
.mouseleave(Gadfly.plot_mouseout)
.mousewheel(Gadfly.guide_background_scroll)
.drag(Gadfly.guide_background_drag_onmove,
      Gadfly.guide_background_drag_onstart,
      Gadfly.guide_background_drag_onend)
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-29")
   .plotroot().data("unfocused_ygrid_color", "#D0D0E0")
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-29")
   .plotroot().data("focused_ygrid_color", "#A0A0A0")
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-135")
   .plotroot().data("unfocused_xgrid_color", "#D0D0E0")
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-135")
   .plotroot().data("focused_xgrid_color", "#A0A0A0")
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-144")
   .mouseenter(Gadfly.plot_mouseover)
.mouseleave(Gadfly.plot_mouseout)
.mousewheel(Gadfly.guide_background_scroll)
.drag(Gadfly.guide_background_drag_onmove,
      Gadfly.guide_background_drag_onstart,
      Gadfly.guide_background_drag_onend)
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-148")
   .plotroot().data("unfocused_ygrid_color", "#D0D0E0")
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-148")
   .plotroot().data("focused_ygrid_color", "#A0A0A0")
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-254")
   .plotroot().data("unfocused_xgrid_color", "#D0D0E0")
;
fig.select("#fig-e97312c08242438abf7e68ad37f1e090-element-254")
   .plotroot().data("focused_xgrid_color", "#A0A0A0")
;
    });
]]> </script>
</svg>




These kind of visual analysis is a powerful tool to perform the initial data mining before diving into any deeper machine learning algorithms for predictions.

We clearly see that Age and Gender influence the survival. But we cannot generate all combinations to create a ifelse model for prediction. Enter DecisionTrees. Decision trees do exactly what we did with Gender but at a larger scale where all dimensions (or features) are invovled in forming the decision tree.  
In our case the decision we need to make is Survived or Not.

Before doing that we need to handle the missing data in our feature variables. There are multiple ways to handle missing data and we wont go deeper into them. We will use a simple technique of filling the missing value with 

1. Mean value if the variable is numeric.
2. Max frequency categorical variable value for categorical variables.

From describe() above we have seen that Embarked and Age have some missing values (NA)

    countmap(train[:Embarked])
    
    Dict{Union(NAtype,UTF8String),Int64} with 4 entries:
      "Q" => 77
      "S" => 644
      "C" => 168
      NA  => 2
    
    train[isna(train[:Embarked]), :Embarked] = "S"
    
    "S"
    
    meanAge = mean(train[!isna(train[:Age]), :Age])
    
    29.69911764705882
    
    train[isna(train[:Age]), :Age] = meanAge
    
    29.69911764705882


Now that we knocked out the NAs out of our way, lets roll up our sleeves and grow a decision tree!
We will be using the `DecisionTree` julia package.  
Features form the predictors and labels are the response variables for the decision tree. We will start with building the Arrays for features and labels.


    using DecisionTree

    train[:Age] = float64(train[:Age])
    train[:Fare] = float64(train[:Fare])
    features = array(train[:,[:Pclass, :Sex, :Age, :SibSp, :Parch, :Fare, :Embarked, :Child]])
    labels=array(train[:Survived])


    891-element Array{Int64,1}:
     0
     1
     1
     1
     0
     0
     0
     0
     1
     1
     1
     1
     0
     ⋮
     1
     1
     0
     0
     0
     0
     0
     0
     1
     0
     1
     0


    features

    891x8 Array{Any,2}:
     3  "male"    22.0     1  0   7.25    "S"  0
     1  "female"  38.0     1  0  71.2833  "C"  0
     3  "female"  26.0     0  0   7.925   "S"  0
     1  "female"  35.0     1  0  53.1     "S"  0
     3  "male"    35.0     0  0   8.05    "S"  0
     3  "male"    29.6991  0  0   8.4583  "Q"  1
     1  "male"    54.0     0  0  51.8625  "S"  0
     3  "male"     2.0     3  1  21.075   "S"  1
     3  "female"  27.0     0  2  11.1333  "S"  0
     2  "female"  14.0     1  0  30.0708  "C"  1
     3  "female"   4.0     1  1  16.7     "S"  1
     1  "female"  58.0     0  0  26.55    "S"  0
     3  "male"    20.0     0  0   8.05    "S"  0
     ⋮                            ⋮             
     1  "female"  56.0     0  1  83.1583  "C"  0
     2  "female"  25.0     0  1  26.0     "S"  0
     3  "male"    33.0     0  0   7.8958  "S"  0
     3  "female"  22.0     0  0  10.5167  "S"  0
     2  "male"    28.0     0  0  10.5     "S"  0
     3  "male"    25.0     0  0   7.05    "S"  0
     3  "female"  39.0     0  5  29.125   "Q"  0
     2  "male"    27.0     0  0  13.0     "S"  0
     1  "female"  19.0     0  0  30.0     "S"  0
     3  "female"  29.6991  1  2  23.45    "S"  1
     1  "male"    26.0     0  0  30.0     "C"  0
     3  "male"    32.0     0  0   7.75    "Q"  0



## Stumps
A stump is the simplest decision tree, with only one split and two classification nodes (leaves). It tries to generate a single split with most predictive power. This is performed by splitting the dataset into 2 subset and returning the split with highest information gain. We will use the `build_stumps` api from DecisionTree and feed it with the above generated labels and features.


    stump = build_stump(labels, features)
    print_tree(stump)

    Feature 2, Threshold male
    L-> 1 : 233/314
    R-> 0 : 468/577
    
The stump picks feature 2, Sex to make the split with threshold being "male". 
The interpretation being - all males data points to the right leaf and all female datapoints go to the left leaf.
Hence the 314 on L and 577 on R - number of female and male datapoints (respectively) in train set.  
`L -> 1` implies Survived=1 is on left branch    
`R -> 0` implies Survived=0 is on right branch  
This creates a basic 1-rules tree with deciding factor being gender of the person.

Lets verify the split by predicting on the training data.

    predictions = apply_tree(stump, features)
    confusion_matrix(labels, predictions)
    
    2x2 Array{Int64,2}:
     468   81
     109  233
    
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7867564534231201
    Kappa:    0.5421129503407983

Since we predict male who survived and females who didnt survive wrongly we get a miclassification count of 81 + 109.
This gives use an accuracy of 1-190/891 = 0.7867


## Decision Trees
Now that we have our 1-level decision stump, lets bring it all together and build our first multi-split decision tree.  
More details coming soon ...


    model = build_tree(labels, features)
    # prune tree: merge leaves having >= 90% combined purity (default: 100%)
    model = prune_tree(model, 0.9)
    # pretty print of the tree, to a depth of 5 nodes (optional)
    print_tree(model, 5)
    # run n-fold cross validation for pruned tree,
    # using 90% purity threshold purning, and 3 CV folds
    accuracy = nfoldCV_tree(labels, features, 0.9, 3)

    Feature 2, Threshold male


    2x2 Array{Int64,2}:
     149  37
      35  76


    
    L-> Feature 1, Threshold 3
        L-> Feature 6, Threshold 29.0
            L-> Feature 6, Threshold 28.7125
                L-> Feature 3, Threshold 24.0
                    L-> 1 : 15/15
                    R-> 
                R-> 0 : 1/1
            R-> Feature 3, Threshold 3.0
                L-> 0 : 1/1
                R-> Feature 5, Threshold 2
                    L-> 1 : 84/84
                    R-> 
        R-> Feature 6, Threshold 23.45
            L-> Feature 8, Threshold 1
                L-> Feature 3, Threshold 37.0
                    L-> 
                    R-> 
                R-> Feature 6, Threshold 15.5
                    L-> 
                    R-> 
            R-> Feature 5, Threshold 1
                L-> 1 : 1/1
                R-> Feature 6, Threshold 31.3875
                    L-> 0 : 15/15
                    R-> 
    R-> Feature 6, Threshold 26.2875
        L-> Feature 3, Threshold 15.0
            L-> Feature 4, Threshold 3
                L-> Feature 3, Threshold 11.0
                    L-> 1 : 12/12
                    R-> 
                R-> 0 : 1/1
            R-> Feature 7, Threshold Q
                L-> Feature 6, Threshold 15.2458
                    L-> 
                    R-> 
                R-> Feature 6, Threshold 13.5
                    L-> 
                    R-> 0 : 62/64
        R-> Feature 4, Threshold 3
            L-> Feature 3, Threshold 16.0
                L-> 1 : 7/7
                R-> Feature 6, Threshold 26.55
                    L-> 1 : 4/4
                    R-> 
            R-> Feature 3, Threshold 4.0
                L-> Feature 3, Threshold 3.0
                    L-> 0 : 4/4
                    R-> 1 : 1/1
                R-> 0 : 18/18
    
    Fold 1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7575757575757576
    Kappa:    0.4840017373678876
    
    Fold 


    2x2 Array{Int64,2}:
     138  39
      38  82



    2x2 Array{Int64,2}:
     140  46
      34  77


    2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7407407407407407
    Kappa:    0.4623739332816136
    
    Fold 3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7306397306397306
    Kappa:    0.43686006825938567
    
    Mean Accuracy: 0.7429854096520763





    3-element Array{Float64,1}:
     0.757576
     0.740741
     0.73064 




    model = build_forest(labels, features, 2, 10, 0.5)
    # run n-fold cross validation for forests
    # using 2 random features, 10 trees, 3 folds and 0.5 of samples per tree (optional)
    accuracy = nfoldCV_forest(labels, features, 2, 10, 3, 0.5)

    
    Fold 1


    2x2 Array{Int64,2}:
     160  18
      52  67



    2x2 Array{Int64,2}:
     173  13
      32  79


    
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7643097643097643
    Kappa:    0.4848604985380841
    
    Fold 2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8484848484848485
    Kappa:    0.6647603280909022
    
    Fold 


    2x2 Array{Int64,2}:
     167  18
      43  69


    3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7946127946127947
    Kappa:    0.542673229837183
    
    Mean Accuracy: 0.8024691358024691





    3-element Array{Float64,1}:
     0.76431 
     0.848485
     0.794613




    # train adaptive-boosted stumps, using 7 iterations
    model, coeffs = build_adaboost_stumps(labels, features, 7);
    # run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
    accuracy = nfoldCV_stumps(labels, features, 7, 3)

    
    Fold 


    2x2 Array{Int64,2}:
     150  29
      59  59


    1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7037037037037037
    Kappa:    0.3532934131736527
    
    Fold 


    2x2 Array{Int64,2}:
     168  29
      39  61


    2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7710437710437711
    Kappa:    0.47447306791569094
    
    Fold 


    2x2 Array{Int64,2}:
     135  38
      43  81


    3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7272727272727273
    Kappa:    0.4360627300218008
    
    Mean Accuracy: 0.7340067340067341





    3-element Array{Float64,1}:
     0.703704
     0.771044
     0.727273




    model = build_tree(labels, features)
    # prune tree: merge leaves having >= 90% combined purity (default: 100%)
    model = prune_tree(model, 0.9)
    # pretty print of the tree, to a depth of 5 nodes (optional)
    print_tree(model, 5)
    # run n-fold cross validation for pruned tree,
    # using 90% purity threshold purning, and 3 CV folds
    accuracy = nfoldCV_tree(labels, features, 0.9, 3)
    purities = linspace(0.1, 1.0, 10)
    accuracies = zeros(length(purities));
    
    for i in 1:length(purities)
        accuracies[i] = mean(nfoldCV_tree(labels, features, purities[i], 10));
    end
    


    Feature 2, Threshold male


    2x2 Array{Int64,2}:
     142  31
      36  88


    
    L-> Feature 1, Threshold 3
        L-> Feature 6, Threshold 29.0
            L-> Feature 6, Threshold 28.7125
                L-> Feature 3, Threshold 24.0
                    L-> 1 : 15/15
                    R-> 
                R-> 0 : 1/1
            R-> Feature 3, Threshold 3.0
                L-> 0 : 1/1
                R-> Feature 5, Threshold 2
                    L-> 1 : 84/84
                    R-> 
        R-> Feature 6, Threshold 23.45
            L-> Feature 8, Threshold 1
                L-> Feature 3, Threshold 37.0
                    L-> 
                    R-> 
                R-> Feature 6, Threshold 15.5
                    L-> 
                    R-> 
            R-> Feature 5, Threshold 1
                L-> 1 : 1/1
                R-> Feature 6, Threshold 31.3875
                    L-> 0 : 15/15
                    R-> 
    R-> Feature 6, Threshold 26.2875
        L-> Feature 3, Threshold 15.0
            L-> Feature 4, Threshold 3
                L-> Feature 3, Threshold 11.0
                    L-> 1 : 12/12
                    R-> 
                R-> 0 : 1/1
            R-> Feature 7, Threshold Q
                L-> Feature 6, Threshold 15.2458
                    L-> 
                    R-> 
                R-> Feature 6, Threshold 13.5
                    L-> 
                    R-> 0 : 62/64
        R-> Feature 4, Threshold 3
            L-> Feature 3, Threshold 16.0
                L-> 1 : 7/7
                R-> Feature 6, Threshold 26.55
                    L-> 1 : 4/4
                    R-> 
            R-> Feature 3, Threshold 4.0
                L-> Feature 3, Threshold 3.0
                    L-> 0 : 4/4
                    R-> 1 : 1/1
                R-> 0 : 18/18
    
    Fold 1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7744107744107744
    Kappa:    0.533533369277292
    
    Fold 


    2x2 Array{Int64,2}:
     153  32
      35  77



    2x2 Array{Int64,2}:
     140  51
      25  81


    2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7744107744107744
    Kappa:    0.5172606195871037
    
    Fold 3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7441077441077442
    Kappa:    0.4714064914992273
    
    Mean Accuracy: 0.7643097643097644
    
    Fold 


    2x2 Array{Int64,2}:
     55  0
     34  0



    2x2 Array{Int64,2}:
     59  0
     30  0


    1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6179775280898876
    Kappa:    0.0
    
    Fold 2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6629213483146067
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     54  0
     35  0


    3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6067415730337079
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     58  0
     31  0



    2x2 Array{Int64,2}:
     52  0
     37  0


    4
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.651685393258427
    Kappa:    0.0
    
    Fold 5
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5842696629213483
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     59  0
     30  0


    6
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6629213483146067
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     63  0
     26  0


    7
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7078651685393258
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     50  0
     39  0



    2x2 Array{Int64,2}:
     51  0
     38  0


    8
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5617977528089888
    Kappa:    0.0
    
    Fold 9
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5730337078651685
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     47  0
     42  0



    2x2 Array{Int64,2}:
     52  0
     37  0


    10
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5280898876404494
    Kappa:    0.0
    
    Mean Accuracy: 0.6157303370786518
    
    Fold 1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5842696629213483
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     61  0
     28  0



    2x2 Array{Int64,2}:
     57  0
     32  0


    2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6853932584269663
    Kappa:    0.0
    
    Fold 3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6404494382022472
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     50  0
     39  0



    2x2 Array{Int64,2}:
     52  0
     37  0


    4
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5617977528089888
    Kappa:    0.0
    
    Fold 5
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5842696629213483
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     60  0
     29  0



    2x2 Array{Int64,2}:
     65  0
     24  0


    6
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6741573033707865
    Kappa:    0.0
    
    Fold 7
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7303370786516854
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     50  0
     39  0



    2x2 Array{Int64,2}:
     48  0
     41  0


    8
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5617977528089888
    Kappa:    0.0
    
    Fold 9
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5393258426966292
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     54  0
     35  0


    10
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6067415730337079
    Kappa:    0.0
    
    Mean Accuracy: 0.6168539325842697
    
    Fold 


    2x2 Array{Int64,2}:
     55  0
     34  0



    2x2 Array{Int64,2}:
     54  0
     35  0


    1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6179775280898876
    Kappa:    0.0
    
    Fold 2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6067415730337079
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     51  0
     38  0



    2x2 Array{Int64,2}:
     53  0
     36  0


    3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5730337078651685
    Kappa:    0.0
    
    Fold 4
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5955056179775281
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     54  0
     35  0


    5
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6067415730337079
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     48  0
     41  0



    2x2 Array{Int64,2}:
     55  0
     34  0


    6
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5393258426966292
    Kappa:    0.0
    
    Fold 7
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6179775280898876
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     60  0
     29  0


    8
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6741573033707865
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     59  0
     30  0



    2x2 Array{Int64,2}:
     59  0
     30  0


    9
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6629213483146067
    Kappa:    0.0
    
    Fold 10
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6629213483146067
    Kappa:    0.0
    
    Mean Accuracy: 0.6157303370786515
    
    Fold 


    2x2 Array{Int64,2}:
     54  0
     35  0



    2x2 Array{Int64,2}:
     54  0
     35  0


    1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6067415730337079
    Kappa:    0.0
    
    Fold 2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6067415730337079
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     54  0
     35  0



    2x2 Array{Int64,2}:
     53  0
     36  0


    3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6067415730337079
    Kappa:    0.0
    
    Fold 4
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5955056179775281
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     62  0
     27  0



    2x2 Array{Int64,2}:
     53  0
     36  0


    5
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6966292134831461
    Kappa:    0.0
    
    Fold 6
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5955056179775281
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     56  0
     33  0



    2x2 Array{Int64,2}:
     49  0
     40  0


    7
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6292134831460674
    Kappa:    0.0
    
    Fold 8
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.550561797752809
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     60  0
     29  0



    2x2 Array{Int64,2}:
     54  0
     35  0


    9
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6741573033707865
    Kappa:    0.0
    
    Fold 10
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6067415730337079
    Kappa:    0.0
    
    Mean Accuracy: 0.6168539325842696
    
    Fold 


    2x2 Array{Int64,2}:
     49  0
     40  0



    2x2 Array{Int64,2}:
     56  0
     33  0


    1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.550561797752809
    Kappa:    0.0
    
    Fold 2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6292134831460674
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     54  0
     35  0



    2x2 Array{Int64,2}:
     51  0
     38  0


    3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6067415730337079
    Kappa:    0.0
    
    Fold 4
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5730337078651685
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     57  0
     32  0



    2x2 Array{Int64,2}:
     60  0
     29  0


    5
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6404494382022472
    Kappa:    0.0
    
    Fold 6
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6741573033707865
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     56  0
     33  0



    2x2 Array{Int64,2}:
     47  0
     42  0


    7
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6292134831460674
    Kappa:    0.0
    
    Fold 8
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.5280898876404494
    Kappa:    0.0
    
    Fold 


    2x2 Array{Int64,2}:
     55  0
     34  0



    2x2 Array{Int64,2}:
     63  0
     26  0


    9
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.6179775280898876
    Kappa:    0.0
    
    Fold 10
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7078651685393258
    Kappa:    0.0
    
    Mean Accuracy: 0.6157303370786515
    
    Fold 


    2x2 Array{Int64,2}:
     54   3
      6  26



    2x2 Array{Int64,2}:
     52   4
      6  27


    1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.898876404494382
    Kappa:    0.7758186397984886
    
    Fold 2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8876404494382022
    Kappa:    0.7561643835616437
    
    Fold 


    2x2 Array{Int64,2}:
     46  12
     10  21



    2x2 Array{Int64,2}:
     43   6
     14  26


    3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7528089887640449
    Kappa:    0.4635616438356163
    
    Fold 4
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7752808988764045
    Kappa:    0.5374220374220374
    
    Fold 


    2x2 Array{Int64,2}:
     48  13
      7  21



    2x2 Array{Int64,2}:
     48  11
      7  23


    5
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7752808988764045
    Kappa:    0.5074709463198672
    
    Fold 6
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.797752808988764
    Kappa:    0.5618161925601749
    
    Fold 


    2x2 Array{Int64,2}:
     42   7
     11  29



    2x2 Array{Int64,2}:
     42   7
     11  29


    7
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.797752808988764
    Kappa:    0.5875386199794026
    
    Fold 8
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.797752808988764
    Kappa:    0.5875386199794026
    
    Fold 


    2x2 Array{Int64,2}:
     39  12
      8  30



    2x2 Array{Int64,2}:
     47  13
     11  18


    9
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7752808988764045
    Kappa:    0.5468431771894093
    
    Fold 10
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7303370786516854
    Kappa:    0.39695087521174477
    
    Mean Accuracy: 0.7988764044943821
    
    Fold 


    2x2 Array{Int64,2}:
     47   7
      8  27


    1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8314606741573034
    Kappa:    0.6450412124434991
    
    Fold 


    2x2 Array{Int64,2}:
     42  10
      9  28



    2x2 Array{Int64,2}:
     42   6
      8  33


    2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7865168539325843
    Kappa:    0.562257312969195
    
    Fold 3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8426966292134831
    Kappa:    0.6823049464558898
    
    Fold 


    2x2 Array{Int64,2}:
     47  15
      8  19



    2x2 Array{Int64,2}:
     52   7
     10  20


    4
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7415730337078652
    Kappa:    0.43028110214305604
    
    Fold 5
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8089887640449438
    Kappa:    0.561830292499276
    
    Fold 


    2x2 Array{Int64,2}:
     53   7
      6  23



    2x2 Array{Int64,2}:
     38  11
     13  27


    6
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8539325842696629
    Kappa:    0.6704642551979493
    
    Fold 7
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7303370786516854
    Kappa:    0.4525884161968222
    
    Fold 


    2x2 Array{Int64,2}:
     44   7
      8  30



    2x2 Array{Int64,2}:
     47   9
     12  21


    8
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8314606741573034
    Kappa:    0.654413668133575
    
    Fold 9
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7640449438202247
    Kappa:    0.4846980976013234
    
    Fold 


    2x2 Array{Int64,2}:
     44  13
     11  21



    2x2 Array{Int64,2}:
     40  12
      6  31


    10
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7303370786516854
    Kappa:    0.4223904813412656
    
    Mean Accuracy: 0.7921348314606742
    
    Fold 1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.797752808988764
    Kappa:    0.5931945149822243
    
    Fold 


    2x2 Array{Int64,2}:
     35  13
     12  29



    2x2 Array{Int64,2}:
     52   6
     11  20


    2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7191011235955056
    Kappa:    0.43570885112858226
    
    Fold 3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8089887640449438
    Kappa:    0.5628431089280554
    
    Fold 


    2x2 Array{Int64,2}:
     41  15
     11  22



    2x2 Array{Int64,2}:
     43   5
     11  30


    4
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7078651685393258
    Kappa:    0.38912354804646243
    
    Fold 5
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8202247191011236
    Kappa:    0.6343091936312275
    
    Fold 


    2x2 Array{Int64,2}:
     52   4
      9  24



    2x2 Array{Int64,2}:
     45  10
     11  23


    6
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8539325842696629
    Kappa:    0.6769058922088802
    
    Fold 7
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7640449438202247
    Kappa:    0.49744554987899964
    
    Fold 


    2x2 Array{Int64,2}:
     50   4
      7  28



    2x2 Array{Int64,2}:
     49  12
     10  18


    8
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8764044943820225
    Kappa:    0.7370400214880474
    
    Fold 9
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7528089887640449
    Kappa:    0.4376794945433658
    
    Fold 


    2x2 Array{Int64,2}:
     53   7
      7  22



    2x2 Array{Int64,2}:
     50  10
     11  18


    10
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8426966292134831
    Kappa:    0.6419540229885056
    
    Mean Accuracy: 0.79438202247191
    
    Fold 1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7640449438202247
    Kappa:    0.4581037982023774
    
    Fold 


    2x2 Array{Int64,2}:
     57   6
     10  16


    2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8202247191011236
    Kappa:    0.5447570332480819
    
    Fold 


    2x2 Array{Int64,2}:
     42   8
      9  30


    3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8089887640449438
    Kappa:    0.6109539727436358
    
    Fold 


    2x2 Array{Int64,2}:
     47   9
     11  22


    4
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7752808988764045
    Kappa:    0.5123287671232877
    
    Fold 


    2x2 Array{Int64,2}:
     44   9
     11  25



    2x2 Array{Int64,2}:
     36   9
     14  30


    5
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7752808988764045
    Kappa:    0.5293495505023798
    
    Fold 6
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7415730337078652
    Kappa:    0.48242730720606836
    
    Fold 


    2x2 Array{Int64,2}:
     51   7
     10  21



    2x2 Array{Int64,2}:
     46  10
      7  26


    7
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8089887640449438
    Kappa:    0.5695590327169274
    
    Fold 8
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8089887640449438
    Kappa:    0.5981407702523239
    
    Fold 


    2x2 Array{Int64,2}:
     45   8
      7  29



    2x2 Array{Int64,2}:
     47   8
     13  21


    9
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8314606741573034
    Kappa:    0.6517088442473259
    
    Fold 10
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7640449438202247
    Kappa:    0.48583218707015124
    
    Mean Accuracy: 0.7898876404494383
    
    Fold 


    2x2 Array{Int64,2}:
     44  12
      7  26



    2x2 Array{Int64,2}:
     46   9
     11  23


    1
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7865168539325843
    Kappa:    0.5562844397795854
    
    Fold 2
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7752808988764045
    Kappa:    0.5186587344510547
    
    Fold 


    2x2 Array{Int64,2}:
     50   8
     10  21



    2x2 Array{Int64,2}:
     45   9
     15  20


    3
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.797752808988764
    Kappa:    0.5477131564088085
    
    Fold 4
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7303370786516854
    Kappa:    0.4173486088379705
    
    Fold 


    2x2 Array{Int64,2}:
     44  10
     13  22



    2x2 Array{Int64,2}:
     43  12
      7  27


    5
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7415730337078652
    Kappa:    0.45017459038409896
    
    Fold 6
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7865168539325843
    Kappa:    0.5602080624187257
    
    Fold 


    2x2 Array{Int64,2}:
     42  17
      9  21



    2x2 Array{Int64,2}:
     41  10
     11  27


    7
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7078651685393258
    Kappa:    0.3865323435843054
    
    Fold 8
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7640449438202247
    Kappa:    0.5161791353870049
    
    Fold 


    2x2 Array{Int64,2}:
     46   9
     13  21


    9
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.7528089887640449
    Kappa:    0.4644420131291027
    
    Fold 


    2x2 Array{Int64,2}:
     41  10
      5  33


    10
    Classes:  {0,1}
    Matrix:   
    Accuracy: 0.8314606741573034
    Kappa:    0.6612534889621924
    
    Mean Accuracy: 0.7674157303370787



    plot(x=purities, y=accuracies, Geom.point, Geom.line)




<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:gadfly="http://www.gadflyjl.org/ns"
     version="1.2"
     width="141.42mm" height="100mm" viewBox="0 0 141.42 100"
     stroke="none"
     fill="#000000"
     stroke-width="0.3"
     font-size="3.88"

     id="fig-80d265d38d6f4718b346a3ae93949088">
<g class="plotroot xscalable yscalable" id="fig-80d265d38d6f4718b346a3ae93949088-element-1">
  <g font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" fill="#564A55" stroke="#000000" stroke-opacity="0.000" id="fig-80d265d38d6f4718b346a3ae93949088-element-2">
    <text x="77.46" y="88.39" text-anchor="middle" dy="0.6em">x</text>
  </g>
  <g class="guide xlabels" font-size="2.82" font-family="'PT Sans Caption','Helvetica Neue','Helvetica',sans-serif" fill="#6C606B" id="fig-80d265d38d6f4718b346a3ae93949088-element-3">
    <text x="-121.9" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">-1.25</text>
    <text x="-93.42" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">-1.00</text>
    <text x="-64.94" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">-0.75</text>
    <text x="-36.46" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">-0.50</text>
    <text x="-7.98" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">-0.25</text>
    <text x="20.5" y="84.39" text-anchor="middle" visibility="visible" gadfly:scale="1.0">0.00</text>
    <text x="48.98" y="84.39" text-anchor="middle" visibility="visible" gadfly:scale="1.0">0.25</text>
    <text x="77.46" y="84.39" text-anchor="middle" visibility="visible" gadfly:scale="1.0">0.50</text>
    <text x="105.94" y="84.39" text-anchor="middle" visibility="visible" gadfly:scale="1.0">0.75</text>
    <text x="134.42" y="84.39" text-anchor="middle" visibility="visible" gadfly:scale="1.0">1.00</text>
    <text x="162.9" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">1.25</text>
    <text x="191.38" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">1.50</text>
    <text x="219.86" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">1.75</text>
    <text x="248.34" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">2.00</text>
    <text x="276.82" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="1.0">2.25</text>
    <text x="-93.42" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-1.00</text>
    <text x="-87.73" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.95</text>
    <text x="-82.03" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.90</text>
    <text x="-76.33" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.85</text>
    <text x="-70.64" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.80</text>
    <text x="-64.94" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.75</text>
    <text x="-59.25" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.70</text>
    <text x="-53.55" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.65</text>
    <text x="-47.85" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.60</text>
    <text x="-42.16" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.55</text>
    <text x="-36.46" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.50</text>
    <text x="-30.77" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.45</text>
    <text x="-25.07" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.40</text>
    <text x="-19.37" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.35</text>
    <text x="-13.68" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.30</text>
    <text x="-7.98" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.25</text>
    <text x="-2.29" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.20</text>
    <text x="3.41" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.15</text>
    <text x="9.11" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.10</text>
    <text x="14.8" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">-0.05</text>
    <text x="20.5" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.00</text>
    <text x="26.2" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.05</text>
    <text x="31.89" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.10</text>
    <text x="37.59" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.15</text>
    <text x="43.28" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.20</text>
    <text x="48.98" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.25</text>
    <text x="54.68" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.30</text>
    <text x="60.37" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.35</text>
    <text x="66.07" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.40</text>
    <text x="71.76" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.45</text>
    <text x="77.46" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.50</text>
    <text x="83.16" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.55</text>
    <text x="88.85" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.60</text>
    <text x="94.55" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.65</text>
    <text x="100.24" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.70</text>
    <text x="105.94" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.75</text>
    <text x="111.64" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.80</text>
    <text x="117.33" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.85</text>
    <text x="123.03" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.90</text>
    <text x="128.73" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">0.95</text>
    <text x="134.42" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.00</text>
    <text x="140.12" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.05</text>
    <text x="145.81" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.10</text>
    <text x="151.51" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.15</text>
    <text x="157.21" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.20</text>
    <text x="162.9" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.25</text>
    <text x="168.6" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.30</text>
    <text x="174.29" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.35</text>
    <text x="179.99" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.40</text>
    <text x="185.69" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.45</text>
    <text x="191.38" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.50</text>
    <text x="197.08" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.55</text>
    <text x="202.77" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.60</text>
    <text x="208.47" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.65</text>
    <text x="214.17" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.70</text>
    <text x="219.86" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.75</text>
    <text x="225.56" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.80</text>
    <text x="231.26" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.85</text>
    <text x="236.95" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.90</text>
    <text x="242.65" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">1.95</text>
    <text x="248.34" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="10.0">2.00</text>
    <text x="-93.42" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="0.5">-1</text>
    <text x="20.5" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="0.5">0</text>
    <text x="134.42" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="0.5">1</text>
    <text x="248.34" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="0.5">2</text>
    <text x="-93.42" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-1.0</text>
    <text x="-82.03" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.9</text>
    <text x="-70.64" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.8</text>
    <text x="-59.25" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.7</text>
    <text x="-47.85" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.6</text>
    <text x="-36.46" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.5</text>
    <text x="-25.07" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.4</text>
    <text x="-13.68" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.3</text>
    <text x="-2.29" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.2</text>
    <text x="9.11" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">-0.1</text>
    <text x="20.5" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.0</text>
    <text x="31.89" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.1</text>
    <text x="43.28" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.2</text>
    <text x="54.68" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.3</text>
    <text x="66.07" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.4</text>
    <text x="77.46" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.5</text>
    <text x="88.85" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.6</text>
    <text x="100.24" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.7</text>
    <text x="111.64" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.8</text>
    <text x="123.03" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">0.9</text>
    <text x="134.42" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.0</text>
    <text x="145.81" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.1</text>
    <text x="157.21" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.2</text>
    <text x="168.6" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.3</text>
    <text x="179.99" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.4</text>
    <text x="191.38" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.5</text>
    <text x="202.77" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.6</text>
    <text x="214.17" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.7</text>
    <text x="225.56" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.8</text>
    <text x="236.95" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">1.9</text>
    <text x="248.34" y="84.39" text-anchor="middle" visibility="hidden" gadfly:scale="5.0">2.0</text>
  </g>
  <g clip-path="url(#fig-80d265d38d6f4718b346a3ae93949088-element-5)" id="fig-80d265d38d6f4718b346a3ae93949088-element-4">
    <g pointer-events="visible" opacity="1" fill="#000000" fill-opacity="0.000" stroke="#000000" stroke-opacity="0.000" class="guide background" id="fig-80d265d38d6f4718b346a3ae93949088-element-6">
      <rect x="18.5" y="5" width="117.92" height="75.72"/>
    </g>
    <g class="guide ygridlines xfixed" stroke-dasharray="0.5,0.5" stroke-width="0.2" stroke="#D0D0E0" id="fig-80d265d38d6f4718b346a3ae93949088-element-7">
      <path fill="none" d="M18.5,168.36 L 136.42 168.36" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,150.43 L 136.42 150.43" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,132.5 L 136.42 132.5" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,114.57 L 136.42 114.57" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,96.64 L 136.42 96.64" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,78.72 L 136.42 78.72" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,60.79 L 136.42 60.79" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,42.86 L 136.42 42.86" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,24.93 L 136.42 24.93" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,7 L 136.42 7" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,-10.93 L 136.42 -10.93" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,-28.86 L 136.42 -28.86" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,-46.79 L 136.42 -46.79" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,-64.71 L 136.42 -64.71" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,-82.64 L 136.42 -82.64" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M18.5,150.43 L 136.42 150.43" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,146.84 L 136.42 146.84" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,143.26 L 136.42 143.26" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,139.67 L 136.42 139.67" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,136.09 L 136.42 136.09" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,132.5 L 136.42 132.5" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,128.92 L 136.42 128.92" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,125.33 L 136.42 125.33" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,121.74 L 136.42 121.74" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,118.16 L 136.42 118.16" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,114.57 L 136.42 114.57" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,110.99 L 136.42 110.99" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,107.4 L 136.42 107.4" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,103.82 L 136.42 103.82" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,100.23 L 136.42 100.23" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,96.64 L 136.42 96.64" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,93.06 L 136.42 93.06" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,89.47 L 136.42 89.47" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,85.89 L 136.42 85.89" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,82.3 L 136.42 82.3" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,78.72 L 136.42 78.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,75.13 L 136.42 75.13" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,71.54 L 136.42 71.54" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,67.96 L 136.42 67.96" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,64.37 L 136.42 64.37" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,60.79 L 136.42 60.79" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,57.2 L 136.42 57.2" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,53.61 L 136.42 53.61" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,50.03 L 136.42 50.03" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,46.44 L 136.42 46.44" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,42.86 L 136.42 42.86" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,39.27 L 136.42 39.27" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,35.69 L 136.42 35.69" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,32.1 L 136.42 32.1" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,28.51 L 136.42 28.51" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,24.93 L 136.42 24.93" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,21.34 L 136.42 21.34" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,17.76 L 136.42 17.76" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,14.17 L 136.42 14.17" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,10.59 L 136.42 10.59" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,7 L 136.42 7" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,3.41 L 136.42 3.41" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-0.17 L 136.42 -0.17" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-3.76 L 136.42 -3.76" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-7.34 L 136.42 -7.34" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-10.93 L 136.42 -10.93" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-14.51 L 136.42 -14.51" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-18.1 L 136.42 -18.1" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-21.69 L 136.42 -21.69" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-25.27 L 136.42 -25.27" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-28.86 L 136.42 -28.86" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-32.44 L 136.42 -32.44" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-36.03 L 136.42 -36.03" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-39.61 L 136.42 -39.61" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-43.2 L 136.42 -43.2" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-46.79 L 136.42 -46.79" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-50.37 L 136.42 -50.37" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-53.96 L 136.42 -53.96" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-57.54 L 136.42 -57.54" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-61.13 L 136.42 -61.13" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,-64.71 L 136.42 -64.71" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M18.5,150.43 L 136.42 150.43" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M18.5,78.72 L 136.42 78.72" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M18.5,7 L 136.42 7" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M18.5,-64.71 L 136.42 -64.71" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M18.5,150.43 L 136.42 150.43" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,143.26 L 136.42 143.26" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,136.09 L 136.42 136.09" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,128.92 L 136.42 128.92" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,121.74 L 136.42 121.74" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,114.57 L 136.42 114.57" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,107.4 L 136.42 107.4" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,100.23 L 136.42 100.23" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,93.06 L 136.42 93.06" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,85.89 L 136.42 85.89" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,78.72 L 136.42 78.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,71.54 L 136.42 71.54" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,64.37 L 136.42 64.37" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,57.2 L 136.42 57.2" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,50.03 L 136.42 50.03" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,42.86 L 136.42 42.86" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,35.69 L 136.42 35.69" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,28.51 L 136.42 28.51" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,21.34 L 136.42 21.34" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,14.17 L 136.42 14.17" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,7 L 136.42 7" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,-0.17 L 136.42 -0.17" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,-7.34 L 136.42 -7.34" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,-14.51 L 136.42 -14.51" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,-21.69 L 136.42 -21.69" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,-28.86 L 136.42 -28.86" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,-36.03 L 136.42 -36.03" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,-43.2 L 136.42 -43.2" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,-50.37 L 136.42 -50.37" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,-57.54 L 136.42 -57.54" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M18.5,-64.71 L 136.42 -64.71" visibility="hidden" gadfly:scale="5.0"/>
    </g>
    <g class="guide xgridlines yfixed" stroke-dasharray="0.5,0.5" stroke-width="0.2" stroke="#D0D0E0" id="fig-80d265d38d6f4718b346a3ae93949088-element-8">
      <path fill="none" d="M-121.9,5 L -121.9 80.72" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M-93.42,5 L -93.42 80.72" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M-64.94,5 L -64.94 80.72" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M-36.46,5 L -36.46 80.72" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M-7.98,5 L -7.98 80.72" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M20.5,5 L 20.5 80.72" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M48.98,5 L 48.98 80.72" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M77.46,5 L 77.46 80.72" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M105.94,5 L 105.94 80.72" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M134.42,5 L 134.42 80.72" visibility="visible" gadfly:scale="1.0"/>
      <path fill="none" d="M162.9,5 L 162.9 80.72" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M191.38,5 L 191.38 80.72" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M219.86,5 L 219.86 80.72" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M248.34,5 L 248.34 80.72" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M276.82,5 L 276.82 80.72" visibility="hidden" gadfly:scale="1.0"/>
      <path fill="none" d="M-93.42,5 L -93.42 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-87.73,5 L -87.73 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-82.03,5 L -82.03 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-76.33,5 L -76.33 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-70.64,5 L -70.64 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-64.94,5 L -64.94 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-59.25,5 L -59.25 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-53.55,5 L -53.55 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-47.85,5 L -47.85 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-42.16,5 L -42.16 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-36.46,5 L -36.46 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-30.77,5 L -30.77 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-25.07,5 L -25.07 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-19.37,5 L -19.37 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-13.68,5 L -13.68 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-7.98,5 L -7.98 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-2.29,5 L -2.29 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M3.41,5 L 3.41 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M9.11,5 L 9.11 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M14.8,5 L 14.8 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M20.5,5 L 20.5 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M26.2,5 L 26.2 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M31.89,5 L 31.89 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M37.59,5 L 37.59 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M43.28,5 L 43.28 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M48.98,5 L 48.98 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M54.68,5 L 54.68 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M60.37,5 L 60.37 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M66.07,5 L 66.07 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M71.76,5 L 71.76 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M77.46,5 L 77.46 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M83.16,5 L 83.16 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M88.85,5 L 88.85 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M94.55,5 L 94.55 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M100.24,5 L 100.24 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M105.94,5 L 105.94 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M111.64,5 L 111.64 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M117.33,5 L 117.33 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M123.03,5 L 123.03 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M128.73,5 L 128.73 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M134.42,5 L 134.42 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M140.12,5 L 140.12 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M145.81,5 L 145.81 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M151.51,5 L 151.51 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M157.21,5 L 157.21 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M162.9,5 L 162.9 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M168.6,5 L 168.6 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M174.29,5 L 174.29 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M179.99,5 L 179.99 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M185.69,5 L 185.69 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M191.38,5 L 191.38 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M197.08,5 L 197.08 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M202.77,5 L 202.77 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M208.47,5 L 208.47 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M214.17,5 L 214.17 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M219.86,5 L 219.86 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M225.56,5 L 225.56 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M231.26,5 L 231.26 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M236.95,5 L 236.95 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M242.65,5 L 242.65 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M248.34,5 L 248.34 80.72" visibility="hidden" gadfly:scale="10.0"/>
      <path fill="none" d="M-93.42,5 L -93.42 80.72" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M20.5,5 L 20.5 80.72" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M134.42,5 L 134.42 80.72" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M248.34,5 L 248.34 80.72" visibility="hidden" gadfly:scale="0.5"/>
      <path fill="none" d="M-93.42,5 L -93.42 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-82.03,5 L -82.03 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-70.64,5 L -70.64 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-59.25,5 L -59.25 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-47.85,5 L -47.85 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-36.46,5 L -36.46 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-25.07,5 L -25.07 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-13.68,5 L -13.68 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M-2.29,5 L -2.29 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M9.11,5 L 9.11 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M20.5,5 L 20.5 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M31.89,5 L 31.89 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M43.28,5 L 43.28 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M54.68,5 L 54.68 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M66.07,5 L 66.07 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M77.46,5 L 77.46 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M88.85,5 L 88.85 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M100.24,5 L 100.24 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M111.64,5 L 111.64 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M123.03,5 L 123.03 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M134.42,5 L 134.42 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M145.81,5 L 145.81 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M157.21,5 L 157.21 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M168.6,5 L 168.6 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M179.99,5 L 179.99 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M191.38,5 L 191.38 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M202.77,5 L 202.77 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M214.17,5 L 214.17 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M225.56,5 L 225.56 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M236.95,5 L 236.95 80.72" visibility="hidden" gadfly:scale="5.0"/>
      <path fill="none" d="M248.34,5 L 248.34 80.72" visibility="hidden" gadfly:scale="5.0"/>
    </g>
    <g class="plotpanel" id="fig-80d265d38d6f4718b346a3ae93949088-element-9">
      <g stroke-width="0.3" fill="#000000" fill-opacity="0.000" class="geometry" stroke="#00BFFF" id="fig-80d265d38d6f4718b346a3ae93949088-element-10">
        <path fill="none" d="M31.89,73.07 L 43.28 72.67 54.68 73.07 66.07 72.67 77.46 73.07 88.85 7.4 100.24 9.82 111.64 9.01 123.03 10.63 134.42 18.68"/>
      </g>
      <g class="geometry" id="fig-80d265d38d6f4718b346a3ae93949088-element-11">
        <g class="color_AlphaColorValue{RGB{Float32},Float32}(RGB{Float32}(0.0f0,0.74736935f0,1.0f0),1.0f0)" stroke="#FFFFFF" stroke-width="0.3" fill="#00BFFF" id="fig-80d265d38d6f4718b346a3ae93949088-element-12">
          <circle cx="31.89" cy="73.07" r="0.9"/>
          <circle cx="43.28" cy="72.67" r="0.9"/>
          <circle cx="54.68" cy="73.07" r="0.9"/>
          <circle cx="66.07" cy="72.67" r="0.9"/>
          <circle cx="77.46" cy="73.07" r="0.9"/>
          <circle cx="88.85" cy="7.4" r="0.9"/>
          <circle cx="100.24" cy="9.82" r="0.9"/>
          <circle cx="111.64" cy="9.01" r="0.9"/>
          <circle cx="123.03" cy="10.63" r="0.9"/>
          <circle cx="134.42" cy="18.68" r="0.9"/>
        </g>
      </g>
    </g>
    <g opacity="0" class="guide zoomslider" stroke="#000000" stroke-opacity="0.000" id="fig-80d265d38d6f4718b346a3ae93949088-element-13">
      <g fill="#EAEAEA" stroke-width="0.3" stroke-opacity="0" stroke="#6A6A6A" id="fig-80d265d38d6f4718b346a3ae93949088-element-14">
        <rect x="129.42" y="8" width="4" height="4"/>
        <g class="button_logo" fill="#6A6A6A" id="fig-80d265d38d6f4718b346a3ae93949088-element-15">
          <path d="M130.22,9.6 L 131.02 9.6 131.02 8.8 131.82 8.8 131.82 9.6 132.62 9.6 132.62 10.4 131.82 10.4 131.82 11.2 131.02 11.2 131.02 10.4 130.22 10.4 z"/>
        </g>
      </g>
      <g fill="#EAEAEA" id="fig-80d265d38d6f4718b346a3ae93949088-element-16">
        <rect x="109.92" y="8" width="19" height="4"/>
      </g>
      <g class="zoomslider_thumb" fill="#6A6A6A" id="fig-80d265d38d6f4718b346a3ae93949088-element-17">
        <rect x="118.42" y="8" width="2" height="4"/>
      </g>
      <g fill="#EAEAEA" stroke-width="0.3" stroke-opacity="0" stroke="#6A6A6A" id="fig-80d265d38d6f4718b346a3ae93949088-element-18">
        <rect x="105.42" y="8" width="4" height="4"/>
        <g class="button_logo" fill="#6A6A6A" id="fig-80d265d38d6f4718b346a3ae93949088-element-19">
          <path d="M106.22,9.6 L 108.62 9.6 108.62 10.4 106.22 10.4 z"/>
        </g>
      </g>
    </g>
  </g>
  <g class="guide ylabels" font-size="2.82" font-family="'PT Sans Caption','Helvetica Neue','Helvetica',sans-serif" fill="#6C606B" id="fig-80d265d38d6f4718b346a3ae93949088-element-20">
    <text x="17.5" y="168.36" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="1.0">0.35</text>
    <text x="17.5" y="150.43" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="1.0">0.40</text>
    <text x="17.5" y="132.5" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="1.0">0.45</text>
    <text x="17.5" y="114.57" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="1.0">0.50</text>
    <text x="17.5" y="96.64" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="1.0">0.55</text>
    <text x="17.5" y="78.72" text-anchor="end" dy="0.35em" visibility="visible" gadfly:scale="1.0">0.60</text>
    <text x="17.5" y="60.79" text-anchor="end" dy="0.35em" visibility="visible" gadfly:scale="1.0">0.65</text>
    <text x="17.5" y="42.86" text-anchor="end" dy="0.35em" visibility="visible" gadfly:scale="1.0">0.70</text>
    <text x="17.5" y="24.93" text-anchor="end" dy="0.35em" visibility="visible" gadfly:scale="1.0">0.75</text>
    <text x="17.5" y="7" text-anchor="end" dy="0.35em" visibility="visible" gadfly:scale="1.0">0.80</text>
    <text x="17.5" y="-10.93" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="1.0">0.85</text>
    <text x="17.5" y="-28.86" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="1.0">0.90</text>
    <text x="17.5" y="-46.79" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="1.0">0.95</text>
    <text x="17.5" y="-64.71" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="1.0">1.00</text>
    <text x="17.5" y="-82.64" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="1.0">1.05</text>
    <text x="17.5" y="150.43" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.40</text>
    <text x="17.5" y="146.84" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.41</text>
    <text x="17.5" y="143.26" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.42</text>
    <text x="17.5" y="139.67" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.43</text>
    <text x="17.5" y="136.09" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.44</text>
    <text x="17.5" y="132.5" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.45</text>
    <text x="17.5" y="128.92" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.46</text>
    <text x="17.5" y="125.33" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.47</text>
    <text x="17.5" y="121.74" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.48</text>
    <text x="17.5" y="118.16" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.49</text>
    <text x="17.5" y="114.57" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.50</text>
    <text x="17.5" y="110.99" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.51</text>
    <text x="17.5" y="107.4" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.52</text>
    <text x="17.5" y="103.82" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.53</text>
    <text x="17.5" y="100.23" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.54</text>
    <text x="17.5" y="96.64" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.55</text>
    <text x="17.5" y="93.06" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.56</text>
    <text x="17.5" y="89.47" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.57</text>
    <text x="17.5" y="85.89" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.58</text>
    <text x="17.5" y="82.3" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.59</text>
    <text x="17.5" y="78.72" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.60</text>
    <text x="17.5" y="75.13" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.61</text>
    <text x="17.5" y="71.54" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.62</text>
    <text x="17.5" y="67.96" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.63</text>
    <text x="17.5" y="64.37" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.64</text>
    <text x="17.5" y="60.79" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.65</text>
    <text x="17.5" y="57.2" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.66</text>
    <text x="17.5" y="53.61" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.67</text>
    <text x="17.5" y="50.03" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.68</text>
    <text x="17.5" y="46.44" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.69</text>
    <text x="17.5" y="42.86" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.70</text>
    <text x="17.5" y="39.27" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.71</text>
    <text x="17.5" y="35.69" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.72</text>
    <text x="17.5" y="32.1" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.73</text>
    <text x="17.5" y="28.51" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.74</text>
    <text x="17.5" y="24.93" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.75</text>
    <text x="17.5" y="21.34" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.76</text>
    <text x="17.5" y="17.76" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.77</text>
    <text x="17.5" y="14.17" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.78</text>
    <text x="17.5" y="10.59" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.79</text>
    <text x="17.5" y="7" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.80</text>
    <text x="17.5" y="3.41" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.81</text>
    <text x="17.5" y="-0.17" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.82</text>
    <text x="17.5" y="-3.76" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.83</text>
    <text x="17.5" y="-7.34" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.84</text>
    <text x="17.5" y="-10.93" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.85</text>
    <text x="17.5" y="-14.51" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.86</text>
    <text x="17.5" y="-18.1" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.87</text>
    <text x="17.5" y="-21.69" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.88</text>
    <text x="17.5" y="-25.27" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.89</text>
    <text x="17.5" y="-28.86" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.90</text>
    <text x="17.5" y="-32.44" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.91</text>
    <text x="17.5" y="-36.03" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.92</text>
    <text x="17.5" y="-39.61" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.93</text>
    <text x="17.5" y="-43.2" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.94</text>
    <text x="17.5" y="-46.79" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.95</text>
    <text x="17.5" y="-50.37" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.96</text>
    <text x="17.5" y="-53.96" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.97</text>
    <text x="17.5" y="-57.54" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.98</text>
    <text x="17.5" y="-61.13" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">0.99</text>
    <text x="17.5" y="-64.71" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="10.0">1.00</text>
    <text x="17.5" y="150.43" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="0.5">0.4</text>
    <text x="17.5" y="78.72" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="0.5">0.6</text>
    <text x="17.5" y="7" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="0.5">0.8</text>
    <text x="17.5" y="-64.71" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="0.5">1.0</text>
    <text x="17.5" y="150.43" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.40</text>
    <text x="17.5" y="143.26" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.42</text>
    <text x="17.5" y="136.09" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.44</text>
    <text x="17.5" y="128.92" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.46</text>
    <text x="17.5" y="121.74" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.48</text>
    <text x="17.5" y="114.57" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.50</text>
    <text x="17.5" y="107.4" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.52</text>
    <text x="17.5" y="100.23" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.54</text>
    <text x="17.5" y="93.06" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.56</text>
    <text x="17.5" y="85.89" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.58</text>
    <text x="17.5" y="78.72" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.60</text>
    <text x="17.5" y="71.54" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.62</text>
    <text x="17.5" y="64.37" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.64</text>
    <text x="17.5" y="57.2" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.66</text>
    <text x="17.5" y="50.03" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.68</text>
    <text x="17.5" y="42.86" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.70</text>
    <text x="17.5" y="35.69" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.72</text>
    <text x="17.5" y="28.51" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.74</text>
    <text x="17.5" y="21.34" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.76</text>
    <text x="17.5" y="14.17" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.78</text>
    <text x="17.5" y="7" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.80</text>
    <text x="17.5" y="-0.17" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.82</text>
    <text x="17.5" y="-7.34" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.84</text>
    <text x="17.5" y="-14.51" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.86</text>
    <text x="17.5" y="-21.69" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.88</text>
    <text x="17.5" y="-28.86" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.90</text>
    <text x="17.5" y="-36.03" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.92</text>
    <text x="17.5" y="-43.2" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.94</text>
    <text x="17.5" y="-50.37" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.96</text>
    <text x="17.5" y="-57.54" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">0.98</text>
    <text x="17.5" y="-64.71" text-anchor="end" dy="0.35em" visibility="hidden" gadfly:scale="5.0">1.00</text>
  </g>
  <g font-size="3.88" font-family="'PT Sans','Helvetica Neue','Helvetica',sans-serif" fill="#564A55" stroke="#000000" stroke-opacity="0.000" id="fig-80d265d38d6f4718b346a3ae93949088-element-21">
    <text x="8.81" y="42.86" text-anchor="end" dy="0.35em">y</text>
  </g>
</g>
<defs>
<clipPath id="fig-80d265d38d6f4718b346a3ae93949088-element-5">
  <path d="M18.5,5 L 136.42 5 136.42 80.72 18.5 80.72" />
</clipPath
></defs>
<script> <![CDATA[
(function(N){var k=/[\.\/]/,L=/\s*,\s*/,C=function(a,d){return a-d},a,v,y={n:{}},M=function(){for(var a=0,d=this.length;a<d;a++)if("undefined"!=typeof this[a])return this[a]},A=function(){for(var a=this.length;--a;)if("undefined"!=typeof this[a])return this[a]},w=function(k,d){k=String(k);var f=v,n=Array.prototype.slice.call(arguments,2),u=w.listeners(k),p=0,b,q=[],e={},l=[],r=a;l.firstDefined=M;l.lastDefined=A;a=k;for(var s=v=0,x=u.length;s<x;s++)"zIndex"in u[s]&&(q.push(u[s].zIndex),0>u[s].zIndex&&
(e[u[s].zIndex]=u[s]));for(q.sort(C);0>q[p];)if(b=e[q[p++] ],l.push(b.apply(d,n)),v)return v=f,l;for(s=0;s<x;s++)if(b=u[s],"zIndex"in b)if(b.zIndex==q[p]){l.push(b.apply(d,n));if(v)break;do if(p++,(b=e[q[p] ])&&l.push(b.apply(d,n)),v)break;while(b)}else e[b.zIndex]=b;else if(l.push(b.apply(d,n)),v)break;v=f;a=r;return l};w._events=y;w.listeners=function(a){a=a.split(k);var d=y,f,n,u,p,b,q,e,l=[d],r=[];u=0;for(p=a.length;u<p;u++){e=[];b=0;for(q=l.length;b<q;b++)for(d=l[b].n,f=[d[a[u] ],d["*"] ],n=2;n--;)if(d=
f[n])e.push(d),r=r.concat(d.f||[]);l=e}return r};w.on=function(a,d){a=String(a);if("function"!=typeof d)return function(){};for(var f=a.split(L),n=0,u=f.length;n<u;n++)(function(a){a=a.split(k);for(var b=y,f,e=0,l=a.length;e<l;e++)b=b.n,b=b.hasOwnProperty(a[e])&&b[a[e] ]||(b[a[e] ]={n:{}});b.f=b.f||[];e=0;for(l=b.f.length;e<l;e++)if(b.f[e]==d){f=!0;break}!f&&b.f.push(d)})(f[n]);return function(a){+a==+a&&(d.zIndex=+a)}};w.f=function(a){var d=[].slice.call(arguments,1);return function(){w.apply(null,
[a,null].concat(d).concat([].slice.call(arguments,0)))}};w.stop=function(){v=1};w.nt=function(k){return k?(new RegExp("(?:\\.|\\/|^)"+k+"(?:\\.|\\/|$)")).test(a):a};w.nts=function(){return a.split(k)};w.off=w.unbind=function(a,d){if(a){var f=a.split(L);if(1<f.length)for(var n=0,u=f.length;n<u;n++)w.off(f[n],d);else{for(var f=a.split(k),p,b,q,e,l=[y],n=0,u=f.length;n<u;n++)for(e=0;e<l.length;e+=q.length-2){q=[e,1];p=l[e].n;if("*"!=f[n])p[f[n] ]&&q.push(p[f[n] ]);else for(b in p)p.hasOwnProperty(b)&&
q.push(p[b]);l.splice.apply(l,q)}n=0;for(u=l.length;n<u;n++)for(p=l[n];p.n;){if(d){if(p.f){e=0;for(f=p.f.length;e<f;e++)if(p.f[e]==d){p.f.splice(e,1);break}!p.f.length&&delete p.f}for(b in p.n)if(p.n.hasOwnProperty(b)&&p.n[b].f){q=p.n[b].f;e=0;for(f=q.length;e<f;e++)if(q[e]==d){q.splice(e,1);break}!q.length&&delete p.n[b].f}}else for(b in delete p.f,p.n)p.n.hasOwnProperty(b)&&p.n[b].f&&delete p.n[b].f;p=p.n}}}else w._events=y={n:{}}};w.once=function(a,d){var f=function(){w.unbind(a,f);return d.apply(this,
arguments)};return w.on(a,f)};w.version="0.4.2";w.toString=function(){return"You are running Eve 0.4.2"};"undefined"!=typeof module&&module.exports?module.exports=w:"function"===typeof define&&define.amd?define("eve",[],function(){return w}):N.eve=w})(this);
(function(N,k){"function"===typeof define&&define.amd?define("Snap.svg",["eve"],function(L){return k(N,L)}):k(N,N.eve)})(this,function(N,k){var L=function(a){var k={},y=N.requestAnimationFrame||N.webkitRequestAnimationFrame||N.mozRequestAnimationFrame||N.oRequestAnimationFrame||N.msRequestAnimationFrame||function(a){setTimeout(a,16)},M=Array.isArray||function(a){return a instanceof Array||"[object Array]"==Object.prototype.toString.call(a)},A=0,w="M"+(+new Date).toString(36),z=function(a){if(null==
a)return this.s;var b=this.s-a;this.b+=this.dur*b;this.B+=this.dur*b;this.s=a},d=function(a){if(null==a)return this.spd;this.spd=a},f=function(a){if(null==a)return this.dur;this.s=this.s*a/this.dur;this.dur=a},n=function(){delete k[this.id];this.update();a("mina.stop."+this.id,this)},u=function(){this.pdif||(delete k[this.id],this.update(),this.pdif=this.get()-this.b)},p=function(){this.pdif&&(this.b=this.get()-this.pdif,delete this.pdif,k[this.id]=this)},b=function(){var a;if(M(this.start)){a=[];
for(var b=0,e=this.start.length;b<e;b++)a[b]=+this.start[b]+(this.end[b]-this.start[b])*this.easing(this.s)}else a=+this.start+(this.end-this.start)*this.easing(this.s);this.set(a)},q=function(){var l=0,b;for(b in k)if(k.hasOwnProperty(b)){var e=k[b],f=e.get();l++;e.s=(f-e.b)/(e.dur/e.spd);1<=e.s&&(delete k[b],e.s=1,l--,function(b){setTimeout(function(){a("mina.finish."+b.id,b)})}(e));e.update()}l&&y(q)},e=function(a,r,s,x,G,h,J){a={id:w+(A++).toString(36),start:a,end:r,b:s,s:0,dur:x-s,spd:1,get:G,
set:h,easing:J||e.linear,status:z,speed:d,duration:f,stop:n,pause:u,resume:p,update:b};k[a.id]=a;r=0;for(var K in k)if(k.hasOwnProperty(K)&&(r++,2==r))break;1==r&&y(q);return a};e.time=Date.now||function(){return+new Date};e.getById=function(a){return k[a]||null};e.linear=function(a){return a};e.easeout=function(a){return Math.pow(a,1.7)};e.easein=function(a){return Math.pow(a,0.48)};e.easeinout=function(a){if(1==a)return 1;if(0==a)return 0;var b=0.48-a/1.04,e=Math.sqrt(0.1734+b*b);a=e-b;a=Math.pow(Math.abs(a),
1/3)*(0>a?-1:1);b=-e-b;b=Math.pow(Math.abs(b),1/3)*(0>b?-1:1);a=a+b+0.5;return 3*(1-a)*a*a+a*a*a};e.backin=function(a){return 1==a?1:a*a*(2.70158*a-1.70158)};e.backout=function(a){if(0==a)return 0;a-=1;return a*a*(2.70158*a+1.70158)+1};e.elastic=function(a){return a==!!a?a:Math.pow(2,-10*a)*Math.sin(2*(a-0.075)*Math.PI/0.3)+1};e.bounce=function(a){a<1/2.75?a*=7.5625*a:a<2/2.75?(a-=1.5/2.75,a=7.5625*a*a+0.75):a<2.5/2.75?(a-=2.25/2.75,a=7.5625*a*a+0.9375):(a-=2.625/2.75,a=7.5625*a*a+0.984375);return a};
return N.mina=e}("undefined"==typeof k?function(){}:k),C=function(){function a(c,t){if(c){if(c.tagName)return x(c);if(y(c,"array")&&a.set)return a.set.apply(a,c);if(c instanceof e)return c;if(null==t)return c=G.doc.querySelector(c),x(c)}return new s(null==c?"100%":c,null==t?"100%":t)}function v(c,a){if(a){"#text"==c&&(c=G.doc.createTextNode(a.text||""));"string"==typeof c&&(c=v(c));if("string"==typeof a)return"xlink:"==a.substring(0,6)?c.getAttributeNS(m,a.substring(6)):"xml:"==a.substring(0,4)?c.getAttributeNS(la,
a.substring(4)):c.getAttribute(a);for(var da in a)if(a[h](da)){var b=J(a[da]);b?"xlink:"==da.substring(0,6)?c.setAttributeNS(m,da.substring(6),b):"xml:"==da.substring(0,4)?c.setAttributeNS(la,da.substring(4),b):c.setAttribute(da,b):c.removeAttribute(da)}}else c=G.doc.createElementNS(la,c);return c}function y(c,a){a=J.prototype.toLowerCase.call(a);return"finite"==a?isFinite(c):"array"==a&&(c instanceof Array||Array.isArray&&Array.isArray(c))?!0:"null"==a&&null===c||a==typeof c&&null!==c||"object"==
a&&c===Object(c)||$.call(c).slice(8,-1).toLowerCase()==a}function M(c){if("function"==typeof c||Object(c)!==c)return c;var a=new c.constructor,b;for(b in c)c[h](b)&&(a[b]=M(c[b]));return a}function A(c,a,b){function m(){var e=Array.prototype.slice.call(arguments,0),f=e.join("\u2400"),d=m.cache=m.cache||{},l=m.count=m.count||[];if(d[h](f)){a:for(var e=l,l=f,B=0,H=e.length;B<H;B++)if(e[B]===l){e.push(e.splice(B,1)[0]);break a}return b?b(d[f]):d[f]}1E3<=l.length&&delete d[l.shift()];l.push(f);d[f]=c.apply(a,
e);return b?b(d[f]):d[f]}return m}function w(c,a,b,m,e,f){return null==e?(c-=b,a-=m,c||a?(180*I.atan2(-a,-c)/C+540)%360:0):w(c,a,e,f)-w(b,m,e,f)}function z(c){return c%360*C/180}function d(c){var a=[];c=c.replace(/(?:^|\s)(\w+)\(([^)]+)\)/g,function(c,b,m){m=m.split(/\s*,\s*|\s+/);"rotate"==b&&1==m.length&&m.push(0,0);"scale"==b&&(2<m.length?m=m.slice(0,2):2==m.length&&m.push(0,0),1==m.length&&m.push(m[0],0,0));"skewX"==b?a.push(["m",1,0,I.tan(z(m[0])),1,0,0]):"skewY"==b?a.push(["m",1,I.tan(z(m[0])),
0,1,0,0]):a.push([b.charAt(0)].concat(m));return c});return a}function f(c,t){var b=O(c),m=new a.Matrix;if(b)for(var e=0,f=b.length;e<f;e++){var h=b[e],d=h.length,B=J(h[0]).toLowerCase(),H=h[0]!=B,l=H?m.invert():0,E;"t"==B&&2==d?m.translate(h[1],0):"t"==B&&3==d?H?(d=l.x(0,0),B=l.y(0,0),H=l.x(h[1],h[2]),l=l.y(h[1],h[2]),m.translate(H-d,l-B)):m.translate(h[1],h[2]):"r"==B?2==d?(E=E||t,m.rotate(h[1],E.x+E.width/2,E.y+E.height/2)):4==d&&(H?(H=l.x(h[2],h[3]),l=l.y(h[2],h[3]),m.rotate(h[1],H,l)):m.rotate(h[1],
h[2],h[3])):"s"==B?2==d||3==d?(E=E||t,m.scale(h[1],h[d-1],E.x+E.width/2,E.y+E.height/2)):4==d?H?(H=l.x(h[2],h[3]),l=l.y(h[2],h[3]),m.scale(h[1],h[1],H,l)):m.scale(h[1],h[1],h[2],h[3]):5==d&&(H?(H=l.x(h[3],h[4]),l=l.y(h[3],h[4]),m.scale(h[1],h[2],H,l)):m.scale(h[1],h[2],h[3],h[4])):"m"==B&&7==d&&m.add(h[1],h[2],h[3],h[4],h[5],h[6])}return m}function n(c,t){if(null==t){var m=!0;t="linearGradient"==c.type||"radialGradient"==c.type?c.node.getAttribute("gradientTransform"):"pattern"==c.type?c.node.getAttribute("patternTransform"):
c.node.getAttribute("transform");if(!t)return new a.Matrix;t=d(t)}else t=a._.rgTransform.test(t)?J(t).replace(/\.{3}|\u2026/g,c._.transform||aa):d(t),y(t,"array")&&(t=a.path?a.path.toString.call(t):J(t)),c._.transform=t;var b=f(t,c.getBBox(1));if(m)return b;c.matrix=b}function u(c){c=c.node.ownerSVGElement&&x(c.node.ownerSVGElement)||c.node.parentNode&&x(c.node.parentNode)||a.select("svg")||a(0,0);var t=c.select("defs"),t=null==t?!1:t.node;t||(t=r("defs",c.node).node);return t}function p(c){return c.node.ownerSVGElement&&
x(c.node.ownerSVGElement)||a.select("svg")}function b(c,a,m){function b(c){if(null==c)return aa;if(c==+c)return c;v(B,{width:c});try{return B.getBBox().width}catch(a){return 0}}function h(c){if(null==c)return aa;if(c==+c)return c;v(B,{height:c});try{return B.getBBox().height}catch(a){return 0}}function e(b,B){null==a?d[b]=B(c.attr(b)||0):b==a&&(d=B(null==m?c.attr(b)||0:m))}var f=p(c).node,d={},B=f.querySelector(".svg---mgr");B||(B=v("rect"),v(B,{x:-9E9,y:-9E9,width:10,height:10,"class":"svg---mgr",
fill:"none"}),f.appendChild(B));switch(c.type){case "rect":e("rx",b),e("ry",h);case "image":e("width",b),e("height",h);case "text":e("x",b);e("y",h);break;case "circle":e("cx",b);e("cy",h);e("r",b);break;case "ellipse":e("cx",b);e("cy",h);e("rx",b);e("ry",h);break;case "line":e("x1",b);e("x2",b);e("y1",h);e("y2",h);break;case "marker":e("refX",b);e("markerWidth",b);e("refY",h);e("markerHeight",h);break;case "radialGradient":e("fx",b);e("fy",h);break;case "tspan":e("dx",b);e("dy",h);break;default:e(a,
b)}f.removeChild(B);return d}function q(c){y(c,"array")||(c=Array.prototype.slice.call(arguments,0));for(var a=0,b=0,m=this.node;this[a];)delete this[a++];for(a=0;a<c.length;a++)"set"==c[a].type?c[a].forEach(function(c){m.appendChild(c.node)}):m.appendChild(c[a].node);for(var h=m.childNodes,a=0;a<h.length;a++)this[b++]=x(h[a]);return this}function e(c){if(c.snap in E)return E[c.snap];var a=this.id=V(),b;try{b=c.ownerSVGElement}catch(m){}this.node=c;b&&(this.paper=new s(b));this.type=c.tagName;this.anims=
{};this._={transform:[]};c.snap=a;E[a]=this;"g"==this.type&&(this.add=q);if(this.type in{g:1,mask:1,pattern:1})for(var e in s.prototype)s.prototype[h](e)&&(this[e]=s.prototype[e])}function l(c){this.node=c}function r(c,a){var b=v(c);a.appendChild(b);return x(b)}function s(c,a){var b,m,f,d=s.prototype;if(c&&"svg"==c.tagName){if(c.snap in E)return E[c.snap];var l=c.ownerDocument;b=new e(c);m=c.getElementsByTagName("desc")[0];f=c.getElementsByTagName("defs")[0];m||(m=v("desc"),m.appendChild(l.createTextNode("Created with Snap")),
b.node.appendChild(m));f||(f=v("defs"),b.node.appendChild(f));b.defs=f;for(var ca in d)d[h](ca)&&(b[ca]=d[ca]);b.paper=b.root=b}else b=r("svg",G.doc.body),v(b.node,{height:a,version:1.1,width:c,xmlns:la});return b}function x(c){return!c||c instanceof e||c instanceof l?c:c.tagName&&"svg"==c.tagName.toLowerCase()?new s(c):c.tagName&&"object"==c.tagName.toLowerCase()&&"image/svg+xml"==c.type?new s(c.contentDocument.getElementsByTagName("svg")[0]):new e(c)}a.version="0.3.0";a.toString=function(){return"Snap v"+
this.version};a._={};var G={win:N,doc:N.document};a._.glob=G;var h="hasOwnProperty",J=String,K=parseFloat,U=parseInt,I=Math,P=I.max,Q=I.min,Y=I.abs,C=I.PI,aa="",$=Object.prototype.toString,F=/^\s*((#[a-f\d]{6})|(#[a-f\d]{3})|rgba?\(\s*([\d\.]+%?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+%?(?:\s*,\s*[\d\.]+%?)?)\s*\)|hsba?\(\s*([\d\.]+(?:deg|\xb0|%)?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+(?:%?\s*,\s*[\d\.]+)?%?)\s*\)|hsla?\(\s*([\d\.]+(?:deg|\xb0|%)?\s*,\s*[\d\.]+%?\s*,\s*[\d\.]+(?:%?\s*,\s*[\d\.]+)?%?)\s*\))\s*$/i;a._.separator=
RegExp("[,\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]+");var S=RegExp("[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*"),X={hs:1,rg:1},W=RegExp("([a-z])[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029,]*((-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*)+)",
"ig"),ma=RegExp("([rstm])[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029,]*((-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*)+)","ig"),Z=RegExp("(-?\\d*\\.?\\d*(?:e[\\-+]?\\d+)?)[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*,?[\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*",
"ig"),na=0,ba="S"+(+new Date).toString(36),V=function(){return ba+(na++).toString(36)},m="http://www.w3.org/1999/xlink",la="http://www.w3.org/2000/svg",E={},ca=a.url=function(c){return"url('#"+c+"')"};a._.$=v;a._.id=V;a.format=function(){var c=/\{([^\}]+)\}/g,a=/(?:(?:^|\.)(.+?)(?=\[|\.|$|\()|\[('|")(.+?)\2\])(\(\))?/g,b=function(c,b,m){var h=m;b.replace(a,function(c,a,b,m,t){a=a||m;h&&(a in h&&(h=h[a]),"function"==typeof h&&t&&(h=h()))});return h=(null==h||h==m?c:h)+""};return function(a,m){return J(a).replace(c,
function(c,a){return b(c,a,m)})}}();a._.clone=M;a._.cacher=A;a.rad=z;a.deg=function(c){return 180*c/C%360};a.angle=w;a.is=y;a.snapTo=function(c,a,b){b=y(b,"finite")?b:10;if(y(c,"array"))for(var m=c.length;m--;){if(Y(c[m]-a)<=b)return c[m]}else{c=+c;m=a%c;if(m<b)return a-m;if(m>c-b)return a-m+c}return a};a.getRGB=A(function(c){if(!c||(c=J(c)).indexOf("-")+1)return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka};if("none"==c)return{r:-1,g:-1,b:-1,hex:"none",toString:ka};!X[h](c.toLowerCase().substring(0,
2))&&"#"!=c.charAt()&&(c=T(c));if(!c)return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka};var b,m,e,f,d;if(c=c.match(F)){c[2]&&(e=U(c[2].substring(5),16),m=U(c[2].substring(3,5),16),b=U(c[2].substring(1,3),16));c[3]&&(e=U((d=c[3].charAt(3))+d,16),m=U((d=c[3].charAt(2))+d,16),b=U((d=c[3].charAt(1))+d,16));c[4]&&(d=c[4].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b*=2.55),m=K(d[1]),"%"==d[1].slice(-1)&&(m*=2.55),e=K(d[2]),"%"==d[2].slice(-1)&&(e*=2.55),"rgba"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),
d[3]&&"%"==d[3].slice(-1)&&(f/=100));if(c[5])return d=c[5].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b/=100),m=K(d[1]),"%"==d[1].slice(-1)&&(m/=100),e=K(d[2]),"%"==d[2].slice(-1)&&(e/=100),"deg"!=d[0].slice(-3)&&"\u00b0"!=d[0].slice(-1)||(b/=360),"hsba"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),d[3]&&"%"==d[3].slice(-1)&&(f/=100),a.hsb2rgb(b,m,e,f);if(c[6])return d=c[6].split(S),b=K(d[0]),"%"==d[0].slice(-1)&&(b/=100),m=K(d[1]),"%"==d[1].slice(-1)&&(m/=100),e=K(d[2]),"%"==d[2].slice(-1)&&(e/=100),
"deg"!=d[0].slice(-3)&&"\u00b0"!=d[0].slice(-1)||(b/=360),"hsla"==c[1].toLowerCase().slice(0,4)&&(f=K(d[3])),d[3]&&"%"==d[3].slice(-1)&&(f/=100),a.hsl2rgb(b,m,e,f);b=Q(I.round(b),255);m=Q(I.round(m),255);e=Q(I.round(e),255);f=Q(P(f,0),1);c={r:b,g:m,b:e,toString:ka};c.hex="#"+(16777216|e|m<<8|b<<16).toString(16).slice(1);c.opacity=y(f,"finite")?f:1;return c}return{r:-1,g:-1,b:-1,hex:"none",error:1,toString:ka}},a);a.hsb=A(function(c,b,m){return a.hsb2rgb(c,b,m).hex});a.hsl=A(function(c,b,m){return a.hsl2rgb(c,
b,m).hex});a.rgb=A(function(c,a,b,m){if(y(m,"finite")){var e=I.round;return"rgba("+[e(c),e(a),e(b),+m.toFixed(2)]+")"}return"#"+(16777216|b|a<<8|c<<16).toString(16).slice(1)});var T=function(c){var a=G.doc.getElementsByTagName("head")[0]||G.doc.getElementsByTagName("svg")[0];T=A(function(c){if("red"==c.toLowerCase())return"rgb(255, 0, 0)";a.style.color="rgb(255, 0, 0)";a.style.color=c;c=G.doc.defaultView.getComputedStyle(a,aa).getPropertyValue("color");return"rgb(255, 0, 0)"==c?null:c});return T(c)},
qa=function(){return"hsb("+[this.h,this.s,this.b]+")"},ra=function(){return"hsl("+[this.h,this.s,this.l]+")"},ka=function(){return 1==this.opacity||null==this.opacity?this.hex:"rgba("+[this.r,this.g,this.b,this.opacity]+")"},D=function(c,b,m){null==b&&y(c,"object")&&"r"in c&&"g"in c&&"b"in c&&(m=c.b,b=c.g,c=c.r);null==b&&y(c,string)&&(m=a.getRGB(c),c=m.r,b=m.g,m=m.b);if(1<c||1<b||1<m)c/=255,b/=255,m/=255;return[c,b,m]},oa=function(c,b,m,e){c=I.round(255*c);b=I.round(255*b);m=I.round(255*m);c={r:c,
g:b,b:m,opacity:y(e,"finite")?e:1,hex:a.rgb(c,b,m),toString:ka};y(e,"finite")&&(c.opacity=e);return c};a.color=function(c){var b;y(c,"object")&&"h"in c&&"s"in c&&"b"in c?(b=a.hsb2rgb(c),c.r=b.r,c.g=b.g,c.b=b.b,c.opacity=1,c.hex=b.hex):y(c,"object")&&"h"in c&&"s"in c&&"l"in c?(b=a.hsl2rgb(c),c.r=b.r,c.g=b.g,c.b=b.b,c.opacity=1,c.hex=b.hex):(y(c,"string")&&(c=a.getRGB(c)),y(c,"object")&&"r"in c&&"g"in c&&"b"in c&&!("error"in c)?(b=a.rgb2hsl(c),c.h=b.h,c.s=b.s,c.l=b.l,b=a.rgb2hsb(c),c.v=b.b):(c={hex:"none"},
c.r=c.g=c.b=c.h=c.s=c.v=c.l=-1,c.error=1));c.toString=ka;return c};a.hsb2rgb=function(c,a,b,m){y(c,"object")&&"h"in c&&"s"in c&&"b"in c&&(b=c.b,a=c.s,c=c.h,m=c.o);var e,h,d;c=360*c%360/60;d=b*a;a=d*(1-Y(c%2-1));b=e=h=b-d;c=~~c;b+=[d,a,0,0,a,d][c];e+=[a,d,d,a,0,0][c];h+=[0,0,a,d,d,a][c];return oa(b,e,h,m)};a.hsl2rgb=function(c,a,b,m){y(c,"object")&&"h"in c&&"s"in c&&"l"in c&&(b=c.l,a=c.s,c=c.h);if(1<c||1<a||1<b)c/=360,a/=100,b/=100;var e,h,d;c=360*c%360/60;d=2*a*(0.5>b?b:1-b);a=d*(1-Y(c%2-1));b=e=
h=b-d/2;c=~~c;b+=[d,a,0,0,a,d][c];e+=[a,d,d,a,0,0][c];h+=[0,0,a,d,d,a][c];return oa(b,e,h,m)};a.rgb2hsb=function(c,a,b){b=D(c,a,b);c=b[0];a=b[1];b=b[2];var m,e;m=P(c,a,b);e=m-Q(c,a,b);c=((0==e?0:m==c?(a-b)/e:m==a?(b-c)/e+2:(c-a)/e+4)+360)%6*60/360;return{h:c,s:0==e?0:e/m,b:m,toString:qa}};a.rgb2hsl=function(c,a,b){b=D(c,a,b);c=b[0];a=b[1];b=b[2];var m,e,h;m=P(c,a,b);e=Q(c,a,b);h=m-e;c=((0==h?0:m==c?(a-b)/h:m==a?(b-c)/h+2:(c-a)/h+4)+360)%6*60/360;m=(m+e)/2;return{h:c,s:0==h?0:0.5>m?h/(2*m):h/(2-2*
m),l:m,toString:ra}};a.parsePathString=function(c){if(!c)return null;var b=a.path(c);if(b.arr)return a.path.clone(b.arr);var m={a:7,c:6,o:2,h:1,l:2,m:2,r:4,q:4,s:4,t:2,v:1,u:3,z:0},e=[];y(c,"array")&&y(c[0],"array")&&(e=a.path.clone(c));e.length||J(c).replace(W,function(c,a,b){var h=[];c=a.toLowerCase();b.replace(Z,function(c,a){a&&h.push(+a)});"m"==c&&2<h.length&&(e.push([a].concat(h.splice(0,2))),c="l",a="m"==a?"l":"L");"o"==c&&1==h.length&&e.push([a,h[0] ]);if("r"==c)e.push([a].concat(h));else for(;h.length>=
m[c]&&(e.push([a].concat(h.splice(0,m[c]))),m[c]););});e.toString=a.path.toString;b.arr=a.path.clone(e);return e};var O=a.parseTransformString=function(c){if(!c)return null;var b=[];y(c,"array")&&y(c[0],"array")&&(b=a.path.clone(c));b.length||J(c).replace(ma,function(c,a,m){var e=[];a.toLowerCase();m.replace(Z,function(c,a){a&&e.push(+a)});b.push([a].concat(e))});b.toString=a.path.toString;return b};a._.svgTransform2string=d;a._.rgTransform=RegExp("^[a-z][\t\n\x0B\f\r \u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\u2028\u2029]*-?\\.?\\d",
"i");a._.transform2matrix=f;a._unit2px=b;a._.getSomeDefs=u;a._.getSomeSVG=p;a.select=function(c){return x(G.doc.querySelector(c))};a.selectAll=function(c){c=G.doc.querySelectorAll(c);for(var b=(a.set||Array)(),m=0;m<c.length;m++)b.push(x(c[m]));return b};setInterval(function(){for(var c in E)if(E[h](c)){var a=E[c],b=a.node;("svg"!=a.type&&!b.ownerSVGElement||"svg"==a.type&&(!b.parentNode||"ownerSVGElement"in b.parentNode&&!b.ownerSVGElement))&&delete E[c]}},1E4);(function(c){function m(c){function a(c,
b){var m=v(c.node,b);(m=(m=m&&m.match(d))&&m[2])&&"#"==m.charAt()&&(m=m.substring(1))&&(f[m]=(f[m]||[]).concat(function(a){var m={};m[b]=ca(a);v(c.node,m)}))}function b(c){var a=v(c.node,"xlink:href");a&&"#"==a.charAt()&&(a=a.substring(1))&&(f[a]=(f[a]||[]).concat(function(a){c.attr("xlink:href","#"+a)}))}var e=c.selectAll("*"),h,d=/^\s*url\(("|'|)(.*)\1\)\s*$/;c=[];for(var f={},l=0,E=e.length;l<E;l++){h=e[l];a(h,"fill");a(h,"stroke");a(h,"filter");a(h,"mask");a(h,"clip-path");b(h);var t=v(h.node,
"id");t&&(v(h.node,{id:h.id}),c.push({old:t,id:h.id}))}l=0;for(E=c.length;l<E;l++)if(e=f[c[l].old])for(h=0,t=e.length;h<t;h++)e[h](c[l].id)}function e(c,a,b){return function(m){m=m.slice(c,a);1==m.length&&(m=m[0]);return b?b(m):m}}function d(c){return function(){var a=c?"<"+this.type:"",b=this.node.attributes,m=this.node.childNodes;if(c)for(var e=0,h=b.length;e<h;e++)a+=" "+b[e].name+'="'+b[e].value.replace(/"/g,'\\"')+'"';if(m.length){c&&(a+=">");e=0;for(h=m.length;e<h;e++)3==m[e].nodeType?a+=m[e].nodeValue:
1==m[e].nodeType&&(a+=x(m[e]).toString());c&&(a+="</"+this.type+">")}else c&&(a+="/>");return a}}c.attr=function(c,a){if(!c)return this;if(y(c,"string"))if(1<arguments.length){var b={};b[c]=a;c=b}else return k("snap.util.getattr."+c,this).firstDefined();for(var m in c)c[h](m)&&k("snap.util.attr."+m,this,c[m]);return this};c.getBBox=function(c){if(!a.Matrix||!a.path)return this.node.getBBox();var b=this,m=new a.Matrix;if(b.removed)return a._.box();for(;"use"==b.type;)if(c||(m=m.add(b.transform().localMatrix.translate(b.attr("x")||
0,b.attr("y")||0))),b.original)b=b.original;else var e=b.attr("xlink:href"),b=b.original=b.node.ownerDocument.getElementById(e.substring(e.indexOf("#")+1));var e=b._,h=a.path.get[b.type]||a.path.get.deflt;try{if(c)return e.bboxwt=h?a.path.getBBox(b.realPath=h(b)):a._.box(b.node.getBBox()),a._.box(e.bboxwt);b.realPath=h(b);b.matrix=b.transform().localMatrix;e.bbox=a.path.getBBox(a.path.map(b.realPath,m.add(b.matrix)));return a._.box(e.bbox)}catch(d){return a._.box()}};var f=function(){return this.string};
c.transform=function(c){var b=this._;if(null==c){var m=this;c=new a.Matrix(this.node.getCTM());for(var e=n(this),h=[e],d=new a.Matrix,l=e.toTransformString(),b=J(e)==J(this.matrix)?J(b.transform):l;"svg"!=m.type&&(m=m.parent());)h.push(n(m));for(m=h.length;m--;)d.add(h[m]);return{string:b,globalMatrix:c,totalMatrix:d,localMatrix:e,diffMatrix:c.clone().add(e.invert()),global:c.toTransformString(),total:d.toTransformString(),local:l,toString:f}}c instanceof a.Matrix?this.matrix=c:n(this,c);this.node&&
("linearGradient"==this.type||"radialGradient"==this.type?v(this.node,{gradientTransform:this.matrix}):"pattern"==this.type?v(this.node,{patternTransform:this.matrix}):v(this.node,{transform:this.matrix}));return this};c.parent=function(){return x(this.node.parentNode)};c.append=c.add=function(c){if(c){if("set"==c.type){var a=this;c.forEach(function(c){a.add(c)});return this}c=x(c);this.node.appendChild(c.node);c.paper=this.paper}return this};c.appendTo=function(c){c&&(c=x(c),c.append(this));return this};
c.prepend=function(c){if(c){if("set"==c.type){var a=this,b;c.forEach(function(c){b?b.after(c):a.prepend(c);b=c});return this}c=x(c);var m=c.parent();this.node.insertBefore(c.node,this.node.firstChild);this.add&&this.add();c.paper=this.paper;this.parent()&&this.parent().add();m&&m.add()}return this};c.prependTo=function(c){c=x(c);c.prepend(this);return this};c.before=function(c){if("set"==c.type){var a=this;c.forEach(function(c){var b=c.parent();a.node.parentNode.insertBefore(c.node,a.node);b&&b.add()});
this.parent().add();return this}c=x(c);var b=c.parent();this.node.parentNode.insertBefore(c.node,this.node);this.parent()&&this.parent().add();b&&b.add();c.paper=this.paper;return this};c.after=function(c){c=x(c);var a=c.parent();this.node.nextSibling?this.node.parentNode.insertBefore(c.node,this.node.nextSibling):this.node.parentNode.appendChild(c.node);this.parent()&&this.parent().add();a&&a.add();c.paper=this.paper;return this};c.insertBefore=function(c){c=x(c);var a=this.parent();c.node.parentNode.insertBefore(this.node,
c.node);this.paper=c.paper;a&&a.add();c.parent()&&c.parent().add();return this};c.insertAfter=function(c){c=x(c);var a=this.parent();c.node.parentNode.insertBefore(this.node,c.node.nextSibling);this.paper=c.paper;a&&a.add();c.parent()&&c.parent().add();return this};c.remove=function(){var c=this.parent();this.node.parentNode&&this.node.parentNode.removeChild(this.node);delete this.paper;this.removed=!0;c&&c.add();return this};c.select=function(c){return x(this.node.querySelector(c))};c.selectAll=
function(c){c=this.node.querySelectorAll(c);for(var b=(a.set||Array)(),m=0;m<c.length;m++)b.push(x(c[m]));return b};c.asPX=function(c,a){null==a&&(a=this.attr(c));return+b(this,c,a)};c.use=function(){var c,a=this.node.id;a||(a=this.id,v(this.node,{id:a}));c="linearGradient"==this.type||"radialGradient"==this.type||"pattern"==this.type?r(this.type,this.node.parentNode):r("use",this.node.parentNode);v(c.node,{"xlink:href":"#"+a});c.original=this;return c};var l=/\S+/g;c.addClass=function(c){var a=(c||
"").match(l)||[];c=this.node;var b=c.className.baseVal,m=b.match(l)||[],e,h,d;if(a.length){for(e=0;d=a[e++];)h=m.indexOf(d),~h||m.push(d);a=m.join(" ");b!=a&&(c.className.baseVal=a)}return this};c.removeClass=function(c){var a=(c||"").match(l)||[];c=this.node;var b=c.className.baseVal,m=b.match(l)||[],e,h;if(m.length){for(e=0;h=a[e++];)h=m.indexOf(h),~h&&m.splice(h,1);a=m.join(" ");b!=a&&(c.className.baseVal=a)}return this};c.hasClass=function(c){return!!~(this.node.className.baseVal.match(l)||[]).indexOf(c)};
c.toggleClass=function(c,a){if(null!=a)return a?this.addClass(c):this.removeClass(c);var b=(c||"").match(l)||[],m=this.node,e=m.className.baseVal,h=e.match(l)||[],d,f,E;for(d=0;E=b[d++];)f=h.indexOf(E),~f?h.splice(f,1):h.push(E);b=h.join(" ");e!=b&&(m.className.baseVal=b);return this};c.clone=function(){var c=x(this.node.cloneNode(!0));v(c.node,"id")&&v(c.node,{id:c.id});m(c);c.insertAfter(this);return c};c.toDefs=function(){u(this).appendChild(this.node);return this};c.pattern=c.toPattern=function(c,
a,b,m){var e=r("pattern",u(this));null==c&&(c=this.getBBox());y(c,"object")&&"x"in c&&(a=c.y,b=c.width,m=c.height,c=c.x);v(e.node,{x:c,y:a,width:b,height:m,patternUnits:"userSpaceOnUse",id:e.id,viewBox:[c,a,b,m].join(" ")});e.node.appendChild(this.node);return e};c.marker=function(c,a,b,m,e,h){var d=r("marker",u(this));null==c&&(c=this.getBBox());y(c,"object")&&"x"in c&&(a=c.y,b=c.width,m=c.height,e=c.refX||c.cx,h=c.refY||c.cy,c=c.x);v(d.node,{viewBox:[c,a,b,m].join(" "),markerWidth:b,markerHeight:m,
orient:"auto",refX:e||0,refY:h||0,id:d.id});d.node.appendChild(this.node);return d};var E=function(c,a,b,m){"function"!=typeof b||b.length||(m=b,b=L.linear);this.attr=c;this.dur=a;b&&(this.easing=b);m&&(this.callback=m)};a._.Animation=E;a.animation=function(c,a,b,m){return new E(c,a,b,m)};c.inAnim=function(){var c=[],a;for(a in this.anims)this.anims[h](a)&&function(a){c.push({anim:new E(a._attrs,a.dur,a.easing,a._callback),mina:a,curStatus:a.status(),status:function(c){return a.status(c)},stop:function(){a.stop()}})}(this.anims[a]);
return c};a.animate=function(c,a,b,m,e,h){"function"!=typeof e||e.length||(h=e,e=L.linear);var d=L.time();c=L(c,a,d,d+m,L.time,b,e);h&&k.once("mina.finish."+c.id,h);return c};c.stop=function(){for(var c=this.inAnim(),a=0,b=c.length;a<b;a++)c[a].stop();return this};c.animate=function(c,a,b,m){"function"!=typeof b||b.length||(m=b,b=L.linear);c instanceof E&&(m=c.callback,b=c.easing,a=b.dur,c=c.attr);var d=[],f=[],l={},t,ca,n,T=this,q;for(q in c)if(c[h](q)){T.equal?(n=T.equal(q,J(c[q])),t=n.from,ca=
n.to,n=n.f):(t=+T.attr(q),ca=+c[q]);var la=y(t,"array")?t.length:1;l[q]=e(d.length,d.length+la,n);d=d.concat(t);f=f.concat(ca)}t=L.time();var p=L(d,f,t,t+a,L.time,function(c){var a={},b;for(b in l)l[h](b)&&(a[b]=l[b](c));T.attr(a)},b);T.anims[p.id]=p;p._attrs=c;p._callback=m;k("snap.animcreated."+T.id,p);k.once("mina.finish."+p.id,function(){delete T.anims[p.id];m&&m.call(T)});k.once("mina.stop."+p.id,function(){delete T.anims[p.id]});return T};var T={};c.data=function(c,b){var m=T[this.id]=T[this.id]||
{};if(0==arguments.length)return k("snap.data.get."+this.id,this,m,null),m;if(1==arguments.length){if(a.is(c,"object")){for(var e in c)c[h](e)&&this.data(e,c[e]);return this}k("snap.data.get."+this.id,this,m[c],c);return m[c]}m[c]=b;k("snap.data.set."+this.id,this,b,c);return this};c.removeData=function(c){null==c?T[this.id]={}:T[this.id]&&delete T[this.id][c];return this};c.outerSVG=c.toString=d(1);c.innerSVG=d()})(e.prototype);a.parse=function(c){var a=G.doc.createDocumentFragment(),b=!0,m=G.doc.createElement("div");
c=J(c);c.match(/^\s*<\s*svg(?:\s|>)/)||(c="<svg>"+c+"</svg>",b=!1);m.innerHTML=c;if(c=m.getElementsByTagName("svg")[0])if(b)a=c;else for(;c.firstChild;)a.appendChild(c.firstChild);m.innerHTML=aa;return new l(a)};l.prototype.select=e.prototype.select;l.prototype.selectAll=e.prototype.selectAll;a.fragment=function(){for(var c=Array.prototype.slice.call(arguments,0),b=G.doc.createDocumentFragment(),m=0,e=c.length;m<e;m++){var h=c[m];h.node&&h.node.nodeType&&b.appendChild(h.node);h.nodeType&&b.appendChild(h);
"string"==typeof h&&b.appendChild(a.parse(h).node)}return new l(b)};a._.make=r;a._.wrap=x;s.prototype.el=function(c,a){var b=r(c,this.node);a&&b.attr(a);return b};k.on("snap.util.getattr",function(){var c=k.nt(),c=c.substring(c.lastIndexOf(".")+1),a=c.replace(/[A-Z]/g,function(c){return"-"+c.toLowerCase()});return pa[h](a)?this.node.ownerDocument.defaultView.getComputedStyle(this.node,null).getPropertyValue(a):v(this.node,c)});var pa={"alignment-baseline":0,"baseline-shift":0,clip:0,"clip-path":0,
"clip-rule":0,color:0,"color-interpolation":0,"color-interpolation-filters":0,"color-profile":0,"color-rendering":0,cursor:0,direction:0,display:0,"dominant-baseline":0,"enable-background":0,fill:0,"fill-opacity":0,"fill-rule":0,filter:0,"flood-color":0,"flood-opacity":0,font:0,"font-family":0,"font-size":0,"font-size-adjust":0,"font-stretch":0,"font-style":0,"font-variant":0,"font-weight":0,"glyph-orientation-horizontal":0,"glyph-orientation-vertical":0,"image-rendering":0,kerning:0,"letter-spacing":0,
"lighting-color":0,marker:0,"marker-end":0,"marker-mid":0,"marker-start":0,mask:0,opacity:0,overflow:0,"pointer-events":0,"shape-rendering":0,"stop-color":0,"stop-opacity":0,stroke:0,"stroke-dasharray":0,"stroke-dashoffset":0,"stroke-linecap":0,"stroke-linejoin":0,"stroke-miterlimit":0,"stroke-opacity":0,"stroke-width":0,"text-anchor":0,"text-decoration":0,"text-rendering":0,"unicode-bidi":0,visibility:0,"word-spacing":0,"writing-mode":0};k.on("snap.util.attr",function(c){var a=k.nt(),b={},a=a.substring(a.lastIndexOf(".")+
1);b[a]=c;var m=a.replace(/-(\w)/gi,function(c,a){return a.toUpperCase()}),a=a.replace(/[A-Z]/g,function(c){return"-"+c.toLowerCase()});pa[h](a)?this.node.style[m]=null==c?aa:c:v(this.node,b)});a.ajax=function(c,a,b,m){var e=new XMLHttpRequest,h=V();if(e){if(y(a,"function"))m=b,b=a,a=null;else if(y(a,"object")){var d=[],f;for(f in a)a.hasOwnProperty(f)&&d.push(encodeURIComponent(f)+"="+encodeURIComponent(a[f]));a=d.join("&")}e.open(a?"POST":"GET",c,!0);a&&(e.setRequestHeader("X-Requested-With","XMLHttpRequest"),
e.setRequestHeader("Content-type","application/x-www-form-urlencoded"));b&&(k.once("snap.ajax."+h+".0",b),k.once("snap.ajax."+h+".200",b),k.once("snap.ajax."+h+".304",b));e.onreadystatechange=function(){4==e.readyState&&k("snap.ajax."+h+"."+e.status,m,e)};if(4==e.readyState)return e;e.send(a);return e}};a.load=function(c,b,m){a.ajax(c,function(c){c=a.parse(c.responseText);m?b.call(m,c):b(c)})};a.getElementByPoint=function(c,a){var b,m,e=G.doc.elementFromPoint(c,a);if(G.win.opera&&"svg"==e.tagName){b=
e;m=b.getBoundingClientRect();b=b.ownerDocument;var h=b.body,d=b.documentElement;b=m.top+(g.win.pageYOffset||d.scrollTop||h.scrollTop)-(d.clientTop||h.clientTop||0);m=m.left+(g.win.pageXOffset||d.scrollLeft||h.scrollLeft)-(d.clientLeft||h.clientLeft||0);h=e.createSVGRect();h.x=c-m;h.y=a-b;h.width=h.height=1;b=e.getIntersectionList(h,null);b.length&&(e=b[b.length-1])}return e?x(e):null};a.plugin=function(c){c(a,e,s,G,l)};return G.win.Snap=a}();C.plugin(function(a,k,y,M,A){function w(a,d,f,b,q,e){null==
d&&"[object SVGMatrix]"==z.call(a)?(this.a=a.a,this.b=a.b,this.c=a.c,this.d=a.d,this.e=a.e,this.f=a.f):null!=a?(this.a=+a,this.b=+d,this.c=+f,this.d=+b,this.e=+q,this.f=+e):(this.a=1,this.c=this.b=0,this.d=1,this.f=this.e=0)}var z=Object.prototype.toString,d=String,f=Math;(function(n){function k(a){return a[0]*a[0]+a[1]*a[1]}function p(a){var d=f.sqrt(k(a));a[0]&&(a[0]/=d);a[1]&&(a[1]/=d)}n.add=function(a,d,e,f,n,p){var k=[[],[],[] ],u=[[this.a,this.c,this.e],[this.b,this.d,this.f],[0,0,1] ];d=[[a,
e,n],[d,f,p],[0,0,1] ];a&&a instanceof w&&(d=[[a.a,a.c,a.e],[a.b,a.d,a.f],[0,0,1] ]);for(a=0;3>a;a++)for(e=0;3>e;e++){for(f=n=0;3>f;f++)n+=u[a][f]*d[f][e];k[a][e]=n}this.a=k[0][0];this.b=k[1][0];this.c=k[0][1];this.d=k[1][1];this.e=k[0][2];this.f=k[1][2];return this};n.invert=function(){var a=this.a*this.d-this.b*this.c;return new w(this.d/a,-this.b/a,-this.c/a,this.a/a,(this.c*this.f-this.d*this.e)/a,(this.b*this.e-this.a*this.f)/a)};n.clone=function(){return new w(this.a,this.b,this.c,this.d,this.e,
this.f)};n.translate=function(a,d){return this.add(1,0,0,1,a,d)};n.scale=function(a,d,e,f){null==d&&(d=a);(e||f)&&this.add(1,0,0,1,e,f);this.add(a,0,0,d,0,0);(e||f)&&this.add(1,0,0,1,-e,-f);return this};n.rotate=function(b,d,e){b=a.rad(b);d=d||0;e=e||0;var l=+f.cos(b).toFixed(9);b=+f.sin(b).toFixed(9);this.add(l,b,-b,l,d,e);return this.add(1,0,0,1,-d,-e)};n.x=function(a,d){return a*this.a+d*this.c+this.e};n.y=function(a,d){return a*this.b+d*this.d+this.f};n.get=function(a){return+this[d.fromCharCode(97+
a)].toFixed(4)};n.toString=function(){return"matrix("+[this.get(0),this.get(1),this.get(2),this.get(3),this.get(4),this.get(5)].join()+")"};n.offset=function(){return[this.e.toFixed(4),this.f.toFixed(4)]};n.determinant=function(){return this.a*this.d-this.b*this.c};n.split=function(){var b={};b.dx=this.e;b.dy=this.f;var d=[[this.a,this.c],[this.b,this.d] ];b.scalex=f.sqrt(k(d[0]));p(d[0]);b.shear=d[0][0]*d[1][0]+d[0][1]*d[1][1];d[1]=[d[1][0]-d[0][0]*b.shear,d[1][1]-d[0][1]*b.shear];b.scaley=f.sqrt(k(d[1]));
p(d[1]);b.shear/=b.scaley;0>this.determinant()&&(b.scalex=-b.scalex);var e=-d[0][1],d=d[1][1];0>d?(b.rotate=a.deg(f.acos(d)),0>e&&(b.rotate=360-b.rotate)):b.rotate=a.deg(f.asin(e));b.isSimple=!+b.shear.toFixed(9)&&(b.scalex.toFixed(9)==b.scaley.toFixed(9)||!b.rotate);b.isSuperSimple=!+b.shear.toFixed(9)&&b.scalex.toFixed(9)==b.scaley.toFixed(9)&&!b.rotate;b.noRotation=!+b.shear.toFixed(9)&&!b.rotate;return b};n.toTransformString=function(a){a=a||this.split();if(+a.shear.toFixed(9))return"m"+[this.get(0),
this.get(1),this.get(2),this.get(3),this.get(4),this.get(5)];a.scalex=+a.scalex.toFixed(4);a.scaley=+a.scaley.toFixed(4);a.rotate=+a.rotate.toFixed(4);return(a.dx||a.dy?"t"+[+a.dx.toFixed(4),+a.dy.toFixed(4)]:"")+(1!=a.scalex||1!=a.scaley?"s"+[a.scalex,a.scaley,0,0]:"")+(a.rotate?"r"+[+a.rotate.toFixed(4),0,0]:"")}})(w.prototype);a.Matrix=w;a.matrix=function(a,d,f,b,k,e){return new w(a,d,f,b,k,e)}});C.plugin(function(a,v,y,M,A){function w(h){return function(d){k.stop();d instanceof A&&1==d.node.childNodes.length&&
("radialGradient"==d.node.firstChild.tagName||"linearGradient"==d.node.firstChild.tagName||"pattern"==d.node.firstChild.tagName)&&(d=d.node.firstChild,b(this).appendChild(d),d=u(d));if(d instanceof v)if("radialGradient"==d.type||"linearGradient"==d.type||"pattern"==d.type){d.node.id||e(d.node,{id:d.id});var f=l(d.node.id)}else f=d.attr(h);else f=a.color(d),f.error?(f=a(b(this).ownerSVGElement).gradient(d))?(f.node.id||e(f.node,{id:f.id}),f=l(f.node.id)):f=d:f=r(f);d={};d[h]=f;e(this.node,d);this.node.style[h]=
x}}function z(a){k.stop();a==+a&&(a+="px");this.node.style.fontSize=a}function d(a){var b=[];a=a.childNodes;for(var e=0,f=a.length;e<f;e++){var l=a[e];3==l.nodeType&&b.push(l.nodeValue);"tspan"==l.tagName&&(1==l.childNodes.length&&3==l.firstChild.nodeType?b.push(l.firstChild.nodeValue):b.push(d(l)))}return b}function f(){k.stop();return this.node.style.fontSize}var n=a._.make,u=a._.wrap,p=a.is,b=a._.getSomeDefs,q=/^url\(#?([^)]+)\)$/,e=a._.$,l=a.url,r=String,s=a._.separator,x="";k.on("snap.util.attr.mask",
function(a){if(a instanceof v||a instanceof A){k.stop();a instanceof A&&1==a.node.childNodes.length&&(a=a.node.firstChild,b(this).appendChild(a),a=u(a));if("mask"==a.type)var d=a;else d=n("mask",b(this)),d.node.appendChild(a.node);!d.node.id&&e(d.node,{id:d.id});e(this.node,{mask:l(d.id)})}});(function(a){k.on("snap.util.attr.clip",a);k.on("snap.util.attr.clip-path",a);k.on("snap.util.attr.clipPath",a)})(function(a){if(a instanceof v||a instanceof A){k.stop();if("clipPath"==a.type)var d=a;else d=
n("clipPath",b(this)),d.node.appendChild(a.node),!d.node.id&&e(d.node,{id:d.id});e(this.node,{"clip-path":l(d.id)})}});k.on("snap.util.attr.fill",w("fill"));k.on("snap.util.attr.stroke",w("stroke"));var G=/^([lr])(?:\(([^)]*)\))?(.*)$/i;k.on("snap.util.grad.parse",function(a){a=r(a);var b=a.match(G);if(!b)return null;a=b[1];var e=b[2],b=b[3],e=e.split(/\s*,\s*/).map(function(a){return+a==a?+a:a});1==e.length&&0==e[0]&&(e=[]);b=b.split("-");b=b.map(function(a){a=a.split(":");var b={color:a[0]};a[1]&&
(b.offset=parseFloat(a[1]));return b});return{type:a,params:e,stops:b}});k.on("snap.util.attr.d",function(b){k.stop();p(b,"array")&&p(b[0],"array")&&(b=a.path.toString.call(b));b=r(b);b.match(/[ruo]/i)&&(b=a.path.toAbsolute(b));e(this.node,{d:b})})(-1);k.on("snap.util.attr.#text",function(a){k.stop();a=r(a);for(a=M.doc.createTextNode(a);this.node.firstChild;)this.node.removeChild(this.node.firstChild);this.node.appendChild(a)})(-1);k.on("snap.util.attr.path",function(a){k.stop();this.attr({d:a})})(-1);
k.on("snap.util.attr.class",function(a){k.stop();this.node.className.baseVal=a})(-1);k.on("snap.util.attr.viewBox",function(a){a=p(a,"object")&&"x"in a?[a.x,a.y,a.width,a.height].join(" "):p(a,"array")?a.join(" "):a;e(this.node,{viewBox:a});k.stop()})(-1);k.on("snap.util.attr.transform",function(a){this.transform(a);k.stop()})(-1);k.on("snap.util.attr.r",function(a){"rect"==this.type&&(k.stop(),e(this.node,{rx:a,ry:a}))})(-1);k.on("snap.util.attr.textpath",function(a){k.stop();if("text"==this.type){var d,
f;if(!a&&this.textPath){for(a=this.textPath;a.node.firstChild;)this.node.appendChild(a.node.firstChild);a.remove();delete this.textPath}else if(p(a,"string")?(d=b(this),a=u(d.parentNode).path(a),d.appendChild(a.node),d=a.id,a.attr({id:d})):(a=u(a),a instanceof v&&(d=a.attr("id"),d||(d=a.id,a.attr({id:d})))),d)if(a=this.textPath,f=this.node,a)a.attr({"xlink:href":"#"+d});else{for(a=e("textPath",{"xlink:href":"#"+d});f.firstChild;)a.appendChild(f.firstChild);f.appendChild(a);this.textPath=u(a)}}})(-1);
k.on("snap.util.attr.text",function(a){if("text"==this.type){for(var b=this.node,d=function(a){var b=e("tspan");if(p(a,"array"))for(var f=0;f<a.length;f++)b.appendChild(d(a[f]));else b.appendChild(M.doc.createTextNode(a));b.normalize&&b.normalize();return b};b.firstChild;)b.removeChild(b.firstChild);for(a=d(a);a.firstChild;)b.appendChild(a.firstChild)}k.stop()})(-1);k.on("snap.util.attr.fontSize",z)(-1);k.on("snap.util.attr.font-size",z)(-1);k.on("snap.util.getattr.transform",function(){k.stop();
return this.transform()})(-1);k.on("snap.util.getattr.textpath",function(){k.stop();return this.textPath})(-1);(function(){function b(d){return function(){k.stop();var b=M.doc.defaultView.getComputedStyle(this.node,null).getPropertyValue("marker-"+d);return"none"==b?b:a(M.doc.getElementById(b.match(q)[1]))}}function d(a){return function(b){k.stop();var d="marker"+a.charAt(0).toUpperCase()+a.substring(1);if(""==b||!b)this.node.style[d]="none";else if("marker"==b.type){var f=b.node.id;f||e(b.node,{id:b.id});
this.node.style[d]=l(f)}}}k.on("snap.util.getattr.marker-end",b("end"))(-1);k.on("snap.util.getattr.markerEnd",b("end"))(-1);k.on("snap.util.getattr.marker-start",b("start"))(-1);k.on("snap.util.getattr.markerStart",b("start"))(-1);k.on("snap.util.getattr.marker-mid",b("mid"))(-1);k.on("snap.util.getattr.markerMid",b("mid"))(-1);k.on("snap.util.attr.marker-end",d("end"))(-1);k.on("snap.util.attr.markerEnd",d("end"))(-1);k.on("snap.util.attr.marker-start",d("start"))(-1);k.on("snap.util.attr.markerStart",
d("start"))(-1);k.on("snap.util.attr.marker-mid",d("mid"))(-1);k.on("snap.util.attr.markerMid",d("mid"))(-1)})();k.on("snap.util.getattr.r",function(){if("rect"==this.type&&e(this.node,"rx")==e(this.node,"ry"))return k.stop(),e(this.node,"rx")})(-1);k.on("snap.util.getattr.text",function(){if("text"==this.type||"tspan"==this.type){k.stop();var a=d(this.node);return 1==a.length?a[0]:a}})(-1);k.on("snap.util.getattr.#text",function(){return this.node.textContent})(-1);k.on("snap.util.getattr.viewBox",
function(){k.stop();var b=e(this.node,"viewBox");if(b)return b=b.split(s),a._.box(+b[0],+b[1],+b[2],+b[3])})(-1);k.on("snap.util.getattr.points",function(){var a=e(this.node,"points");k.stop();if(a)return a.split(s)})(-1);k.on("snap.util.getattr.path",function(){var a=e(this.node,"d");k.stop();return a})(-1);k.on("snap.util.getattr.class",function(){return this.node.className.baseVal})(-1);k.on("snap.util.getattr.fontSize",f)(-1);k.on("snap.util.getattr.font-size",f)(-1)});C.plugin(function(a,v,y,
M,A){function w(a){return a}function z(a){return function(b){return+b.toFixed(3)+a}}var d={"+":function(a,b){return a+b},"-":function(a,b){return a-b},"/":function(a,b){return a/b},"*":function(a,b){return a*b}},f=String,n=/[a-z]+$/i,u=/^\s*([+\-\/*])\s*=\s*([\d.eE+\-]+)\s*([^\d\s]+)?\s*$/;k.on("snap.util.attr",function(a){if(a=f(a).match(u)){var b=k.nt(),b=b.substring(b.lastIndexOf(".")+1),q=this.attr(b),e={};k.stop();var l=a[3]||"",r=q.match(n),s=d[a[1] ];r&&r==l?a=s(parseFloat(q),+a[2]):(q=this.asPX(b),
a=s(this.asPX(b),this.asPX(b,a[2]+l)));isNaN(q)||isNaN(a)||(e[b]=a,this.attr(e))}})(-10);k.on("snap.util.equal",function(a,b){var q=f(this.attr(a)||""),e=f(b).match(u);if(e){k.stop();var l=e[3]||"",r=q.match(n),s=d[e[1] ];if(r&&r==l)return{from:parseFloat(q),to:s(parseFloat(q),+e[2]),f:z(r)};q=this.asPX(a);return{from:q,to:s(q,this.asPX(a,e[2]+l)),f:w}}})(-10)});C.plugin(function(a,v,y,M,A){var w=y.prototype,z=a.is;w.rect=function(a,d,k,p,b,q){var e;null==q&&(q=b);z(a,"object")&&"[object Object]"==
a?e=a:null!=a&&(e={x:a,y:d,width:k,height:p},null!=b&&(e.rx=b,e.ry=q));return this.el("rect",e)};w.circle=function(a,d,k){var p;z(a,"object")&&"[object Object]"==a?p=a:null!=a&&(p={cx:a,cy:d,r:k});return this.el("circle",p)};var d=function(){function a(){this.parentNode.removeChild(this)}return function(d,k){var p=M.doc.createElement("img"),b=M.doc.body;p.style.cssText="position:absolute;left:-9999em;top:-9999em";p.onload=function(){k.call(p);p.onload=p.onerror=null;b.removeChild(p)};p.onerror=a;
b.appendChild(p);p.src=d}}();w.image=function(f,n,k,p,b){var q=this.el("image");if(z(f,"object")&&"src"in f)q.attr(f);else if(null!=f){var e={"xlink:href":f,preserveAspectRatio:"none"};null!=n&&null!=k&&(e.x=n,e.y=k);null!=p&&null!=b?(e.width=p,e.height=b):d(f,function(){a._.$(q.node,{width:this.offsetWidth,height:this.offsetHeight})});a._.$(q.node,e)}return q};w.ellipse=function(a,d,k,p){var b;z(a,"object")&&"[object Object]"==a?b=a:null!=a&&(b={cx:a,cy:d,rx:k,ry:p});return this.el("ellipse",b)};
w.path=function(a){var d;z(a,"object")&&!z(a,"array")?d=a:a&&(d={d:a});return this.el("path",d)};w.group=w.g=function(a){var d=this.el("g");1==arguments.length&&a&&!a.type?d.attr(a):arguments.length&&d.add(Array.prototype.slice.call(arguments,0));return d};w.svg=function(a,d,k,p,b,q,e,l){var r={};z(a,"object")&&null==d?r=a:(null!=a&&(r.x=a),null!=d&&(r.y=d),null!=k&&(r.width=k),null!=p&&(r.height=p),null!=b&&null!=q&&null!=e&&null!=l&&(r.viewBox=[b,q,e,l]));return this.el("svg",r)};w.mask=function(a){var d=
this.el("mask");1==arguments.length&&a&&!a.type?d.attr(a):arguments.length&&d.add(Array.prototype.slice.call(arguments,0));return d};w.ptrn=function(a,d,k,p,b,q,e,l){if(z(a,"object"))var r=a;else arguments.length?(r={},null!=a&&(r.x=a),null!=d&&(r.y=d),null!=k&&(r.width=k),null!=p&&(r.height=p),null!=b&&null!=q&&null!=e&&null!=l&&(r.viewBox=[b,q,e,l])):r={patternUnits:"userSpaceOnUse"};return this.el("pattern",r)};w.use=function(a){return null!=a?(make("use",this.node),a instanceof v&&(a.attr("id")||
a.attr({id:ID()}),a=a.attr("id")),this.el("use",{"xlink:href":a})):v.prototype.use.call(this)};w.text=function(a,d,k){var p={};z(a,"object")?p=a:null!=a&&(p={x:a,y:d,text:k||""});return this.el("text",p)};w.line=function(a,d,k,p){var b={};z(a,"object")?b=a:null!=a&&(b={x1:a,x2:k,y1:d,y2:p});return this.el("line",b)};w.polyline=function(a){1<arguments.length&&(a=Array.prototype.slice.call(arguments,0));var d={};z(a,"object")&&!z(a,"array")?d=a:null!=a&&(d={points:a});return this.el("polyline",d)};
w.polygon=function(a){1<arguments.length&&(a=Array.prototype.slice.call(arguments,0));var d={};z(a,"object")&&!z(a,"array")?d=a:null!=a&&(d={points:a});return this.el("polygon",d)};(function(){function d(){return this.selectAll("stop")}function n(b,d){var f=e("stop"),k={offset:+d+"%"};b=a.color(b);k["stop-color"]=b.hex;1>b.opacity&&(k["stop-opacity"]=b.opacity);e(f,k);this.node.appendChild(f);return this}function u(){if("linearGradient"==this.type){var b=e(this.node,"x1")||0,d=e(this.node,"x2")||
1,f=e(this.node,"y1")||0,k=e(this.node,"y2")||0;return a._.box(b,f,math.abs(d-b),math.abs(k-f))}b=this.node.r||0;return a._.box((this.node.cx||0.5)-b,(this.node.cy||0.5)-b,2*b,2*b)}function p(a,d){function f(a,b){for(var d=(b-u)/(a-w),e=w;e<a;e++)h[e].offset=+(+u+d*(e-w)).toFixed(2);w=a;u=b}var n=k("snap.util.grad.parse",null,d).firstDefined(),p;if(!n)return null;n.params.unshift(a);p="l"==n.type.toLowerCase()?b.apply(0,n.params):q.apply(0,n.params);n.type!=n.type.toLowerCase()&&e(p.node,{gradientUnits:"userSpaceOnUse"});
var h=n.stops,n=h.length,u=0,w=0;n--;for(var v=0;v<n;v++)"offset"in h[v]&&f(v,h[v].offset);h[n].offset=h[n].offset||100;f(n,h[n].offset);for(v=0;v<=n;v++){var y=h[v];p.addStop(y.color,y.offset)}return p}function b(b,k,p,q,w){b=a._.make("linearGradient",b);b.stops=d;b.addStop=n;b.getBBox=u;null!=k&&e(b.node,{x1:k,y1:p,x2:q,y2:w});return b}function q(b,k,p,q,w,h){b=a._.make("radialGradient",b);b.stops=d;b.addStop=n;b.getBBox=u;null!=k&&e(b.node,{cx:k,cy:p,r:q});null!=w&&null!=h&&e(b.node,{fx:w,fy:h});
return b}var e=a._.$;w.gradient=function(a){return p(this.defs,a)};w.gradientLinear=function(a,d,e,f){return b(this.defs,a,d,e,f)};w.gradientRadial=function(a,b,d,e,f){return q(this.defs,a,b,d,e,f)};w.toString=function(){var b=this.node.ownerDocument,d=b.createDocumentFragment(),b=b.createElement("div"),e=this.node.cloneNode(!0);d.appendChild(b);b.appendChild(e);a._.$(e,{xmlns:"http://www.w3.org/2000/svg"});b=b.innerHTML;d.removeChild(d.firstChild);return b};w.clear=function(){for(var a=this.node.firstChild,
b;a;)b=a.nextSibling,"defs"!=a.tagName?a.parentNode.removeChild(a):w.clear.call({node:a}),a=b}})()});C.plugin(function(a,k,y,M){function A(a){var b=A.ps=A.ps||{};b[a]?b[a].sleep=100:b[a]={sleep:100};setTimeout(function(){for(var d in b)b[L](d)&&d!=a&&(b[d].sleep--,!b[d].sleep&&delete b[d])});return b[a]}function w(a,b,d,e){null==a&&(a=b=d=e=0);null==b&&(b=a.y,d=a.width,e=a.height,a=a.x);return{x:a,y:b,width:d,w:d,height:e,h:e,x2:a+d,y2:b+e,cx:a+d/2,cy:b+e/2,r1:F.min(d,e)/2,r2:F.max(d,e)/2,r0:F.sqrt(d*
d+e*e)/2,path:s(a,b,d,e),vb:[a,b,d,e].join(" ")}}function z(){return this.join(",").replace(N,"$1")}function d(a){a=C(a);a.toString=z;return a}function f(a,b,d,h,f,k,l,n,p){if(null==p)return e(a,b,d,h,f,k,l,n);if(0>p||e(a,b,d,h,f,k,l,n)<p)p=void 0;else{var q=0.5,O=1-q,s;for(s=e(a,b,d,h,f,k,l,n,O);0.01<Z(s-p);)q/=2,O+=(s<p?1:-1)*q,s=e(a,b,d,h,f,k,l,n,O);p=O}return u(a,b,d,h,f,k,l,n,p)}function n(b,d){function e(a){return+(+a).toFixed(3)}return a._.cacher(function(a,h,l){a instanceof k&&(a=a.attr("d"));
a=I(a);for(var n,p,D,q,O="",s={},c=0,t=0,r=a.length;t<r;t++){D=a[t];if("M"==D[0])n=+D[1],p=+D[2];else{q=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6]);if(c+q>h){if(d&&!s.start){n=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6],h-c);O+=["C"+e(n.start.x),e(n.start.y),e(n.m.x),e(n.m.y),e(n.x),e(n.y)];if(l)return O;s.start=O;O=["M"+e(n.x),e(n.y)+"C"+e(n.n.x),e(n.n.y),e(n.end.x),e(n.end.y),e(D[5]),e(D[6])].join();c+=q;n=+D[5];p=+D[6];continue}if(!b&&!d)return n=f(n,p,D[1],D[2],D[3],D[4],D[5],D[6],h-c)}c+=q;n=+D[5];p=+D[6]}O+=
D.shift()+D}s.end=O;return n=b?c:d?s:u(n,p,D[0],D[1],D[2],D[3],D[4],D[5],1)},null,a._.clone)}function u(a,b,d,e,h,f,k,l,n){var p=1-n,q=ma(p,3),s=ma(p,2),c=n*n,t=c*n,r=q*a+3*s*n*d+3*p*n*n*h+t*k,q=q*b+3*s*n*e+3*p*n*n*f+t*l,s=a+2*n*(d-a)+c*(h-2*d+a),t=b+2*n*(e-b)+c*(f-2*e+b),x=d+2*n*(h-d)+c*(k-2*h+d),c=e+2*n*(f-e)+c*(l-2*f+e);a=p*a+n*d;b=p*b+n*e;h=p*h+n*k;f=p*f+n*l;l=90-180*F.atan2(s-x,t-c)/S;return{x:r,y:q,m:{x:s,y:t},n:{x:x,y:c},start:{x:a,y:b},end:{x:h,y:f},alpha:l}}function p(b,d,e,h,f,n,k,l){a.is(b,
"array")||(b=[b,d,e,h,f,n,k,l]);b=U.apply(null,b);return w(b.min.x,b.min.y,b.max.x-b.min.x,b.max.y-b.min.y)}function b(a,b,d){return b>=a.x&&b<=a.x+a.width&&d>=a.y&&d<=a.y+a.height}function q(a,d){a=w(a);d=w(d);return b(d,a.x,a.y)||b(d,a.x2,a.y)||b(d,a.x,a.y2)||b(d,a.x2,a.y2)||b(a,d.x,d.y)||b(a,d.x2,d.y)||b(a,d.x,d.y2)||b(a,d.x2,d.y2)||(a.x<d.x2&&a.x>d.x||d.x<a.x2&&d.x>a.x)&&(a.y<d.y2&&a.y>d.y||d.y<a.y2&&d.y>a.y)}function e(a,b,d,e,h,f,n,k,l){null==l&&(l=1);l=(1<l?1:0>l?0:l)/2;for(var p=[-0.1252,
0.1252,-0.3678,0.3678,-0.5873,0.5873,-0.7699,0.7699,-0.9041,0.9041,-0.9816,0.9816],q=[0.2491,0.2491,0.2335,0.2335,0.2032,0.2032,0.1601,0.1601,0.1069,0.1069,0.0472,0.0472],s=0,c=0;12>c;c++)var t=l*p[c]+l,r=t*(t*(-3*a+9*d-9*h+3*n)+6*a-12*d+6*h)-3*a+3*d,t=t*(t*(-3*b+9*e-9*f+3*k)+6*b-12*e+6*f)-3*b+3*e,s=s+q[c]*F.sqrt(r*r+t*t);return l*s}function l(a,b,d){a=I(a);b=I(b);for(var h,f,l,n,k,s,r,O,x,c,t=d?0:[],w=0,v=a.length;w<v;w++)if(x=a[w],"M"==x[0])h=k=x[1],f=s=x[2];else{"C"==x[0]?(x=[h,f].concat(x.slice(1)),
h=x[6],f=x[7]):(x=[h,f,h,f,k,s,k,s],h=k,f=s);for(var G=0,y=b.length;G<y;G++)if(c=b[G],"M"==c[0])l=r=c[1],n=O=c[2];else{"C"==c[0]?(c=[l,n].concat(c.slice(1)),l=c[6],n=c[7]):(c=[l,n,l,n,r,O,r,O],l=r,n=O);var z;var K=x,B=c;z=d;var H=p(K),J=p(B);if(q(H,J)){for(var H=e.apply(0,K),J=e.apply(0,B),H=~~(H/8),J=~~(J/8),U=[],A=[],F={},M=z?0:[],P=0;P<H+1;P++){var C=u.apply(0,K.concat(P/H));U.push({x:C.x,y:C.y,t:P/H})}for(P=0;P<J+1;P++)C=u.apply(0,B.concat(P/J)),A.push({x:C.x,y:C.y,t:P/J});for(P=0;P<H;P++)for(K=
0;K<J;K++){var Q=U[P],L=U[P+1],B=A[K],C=A[K+1],N=0.001>Z(L.x-Q.x)?"y":"x",S=0.001>Z(C.x-B.x)?"y":"x",R;R=Q.x;var Y=Q.y,V=L.x,ea=L.y,fa=B.x,ga=B.y,ha=C.x,ia=C.y;if(W(R,V)<X(fa,ha)||X(R,V)>W(fa,ha)||W(Y,ea)<X(ga,ia)||X(Y,ea)>W(ga,ia))R=void 0;else{var $=(R*ea-Y*V)*(fa-ha)-(R-V)*(fa*ia-ga*ha),aa=(R*ea-Y*V)*(ga-ia)-(Y-ea)*(fa*ia-ga*ha),ja=(R-V)*(ga-ia)-(Y-ea)*(fa-ha);if(ja){var $=$/ja,aa=aa/ja,ja=+$.toFixed(2),ba=+aa.toFixed(2);R=ja<+X(R,V).toFixed(2)||ja>+W(R,V).toFixed(2)||ja<+X(fa,ha).toFixed(2)||
ja>+W(fa,ha).toFixed(2)||ba<+X(Y,ea).toFixed(2)||ba>+W(Y,ea).toFixed(2)||ba<+X(ga,ia).toFixed(2)||ba>+W(ga,ia).toFixed(2)?void 0:{x:$,y:aa}}else R=void 0}R&&F[R.x.toFixed(4)]!=R.y.toFixed(4)&&(F[R.x.toFixed(4)]=R.y.toFixed(4),Q=Q.t+Z((R[N]-Q[N])/(L[N]-Q[N]))*(L.t-Q.t),B=B.t+Z((R[S]-B[S])/(C[S]-B[S]))*(C.t-B.t),0<=Q&&1>=Q&&0<=B&&1>=B&&(z?M++:M.push({x:R.x,y:R.y,t1:Q,t2:B})))}z=M}else z=z?0:[];if(d)t+=z;else{H=0;for(J=z.length;H<J;H++)z[H].segment1=w,z[H].segment2=G,z[H].bez1=x,z[H].bez2=c;t=t.concat(z)}}}return t}
function r(a){var b=A(a);if(b.bbox)return C(b.bbox);if(!a)return w();a=I(a);for(var d=0,e=0,h=[],f=[],l,n=0,k=a.length;n<k;n++)l=a[n],"M"==l[0]?(d=l[1],e=l[2],h.push(d),f.push(e)):(d=U(d,e,l[1],l[2],l[3],l[4],l[5],l[6]),h=h.concat(d.min.x,d.max.x),f=f.concat(d.min.y,d.max.y),d=l[5],e=l[6]);a=X.apply(0,h);l=X.apply(0,f);h=W.apply(0,h);f=W.apply(0,f);f=w(a,l,h-a,f-l);b.bbox=C(f);return f}function s(a,b,d,e,h){if(h)return[["M",+a+ +h,b],["l",d-2*h,0],["a",h,h,0,0,1,h,h],["l",0,e-2*h],["a",h,h,0,0,1,
-h,h],["l",2*h-d,0],["a",h,h,0,0,1,-h,-h],["l",0,2*h-e],["a",h,h,0,0,1,h,-h],["z"] ];a=[["M",a,b],["l",d,0],["l",0,e],["l",-d,0],["z"] ];a.toString=z;return a}function x(a,b,d,e,h){null==h&&null==e&&(e=d);a=+a;b=+b;d=+d;e=+e;if(null!=h){var f=Math.PI/180,l=a+d*Math.cos(-e*f);a+=d*Math.cos(-h*f);var n=b+d*Math.sin(-e*f);b+=d*Math.sin(-h*f);d=[["M",l,n],["A",d,d,0,+(180<h-e),0,a,b] ]}else d=[["M",a,b],["m",0,-e],["a",d,e,0,1,1,0,2*e],["a",d,e,0,1,1,0,-2*e],["z"] ];d.toString=z;return d}function G(b){var e=
A(b);if(e.abs)return d(e.abs);Q(b,"array")&&Q(b&&b[0],"array")||(b=a.parsePathString(b));if(!b||!b.length)return[["M",0,0] ];var h=[],f=0,l=0,n=0,k=0,p=0;"M"==b[0][0]&&(f=+b[0][1],l=+b[0][2],n=f,k=l,p++,h[0]=["M",f,l]);for(var q=3==b.length&&"M"==b[0][0]&&"R"==b[1][0].toUpperCase()&&"Z"==b[2][0].toUpperCase(),s,r,w=p,c=b.length;w<c;w++){h.push(s=[]);r=b[w];p=r[0];if(p!=p.toUpperCase())switch(s[0]=p.toUpperCase(),s[0]){case "A":s[1]=r[1];s[2]=r[2];s[3]=r[3];s[4]=r[4];s[5]=r[5];s[6]=+r[6]+f;s[7]=+r[7]+
l;break;case "V":s[1]=+r[1]+l;break;case "H":s[1]=+r[1]+f;break;case "R":for(var t=[f,l].concat(r.slice(1)),u=2,v=t.length;u<v;u++)t[u]=+t[u]+f,t[++u]=+t[u]+l;h.pop();h=h.concat(P(t,q));break;case "O":h.pop();t=x(f,l,r[1],r[2]);t.push(t[0]);h=h.concat(t);break;case "U":h.pop();h=h.concat(x(f,l,r[1],r[2],r[3]));s=["U"].concat(h[h.length-1].slice(-2));break;case "M":n=+r[1]+f,k=+r[2]+l;default:for(u=1,v=r.length;u<v;u++)s[u]=+r[u]+(u%2?f:l)}else if("R"==p)t=[f,l].concat(r.slice(1)),h.pop(),h=h.concat(P(t,
q)),s=["R"].concat(r.slice(-2));else if("O"==p)h.pop(),t=x(f,l,r[1],r[2]),t.push(t[0]),h=h.concat(t);else if("U"==p)h.pop(),h=h.concat(x(f,l,r[1],r[2],r[3])),s=["U"].concat(h[h.length-1].slice(-2));else for(t=0,u=r.length;t<u;t++)s[t]=r[t];p=p.toUpperCase();if("O"!=p)switch(s[0]){case "Z":f=+n;l=+k;break;case "H":f=s[1];break;case "V":l=s[1];break;case "M":n=s[s.length-2],k=s[s.length-1];default:f=s[s.length-2],l=s[s.length-1]}}h.toString=z;e.abs=d(h);return h}function h(a,b,d,e){return[a,b,d,e,d,
e]}function J(a,b,d,e,h,f){var l=1/3,n=2/3;return[l*a+n*d,l*b+n*e,l*h+n*d,l*f+n*e,h,f]}function K(b,d,e,h,f,l,n,k,p,s){var r=120*S/180,q=S/180*(+f||0),c=[],t,x=a._.cacher(function(a,b,c){var d=a*F.cos(c)-b*F.sin(c);a=a*F.sin(c)+b*F.cos(c);return{x:d,y:a}});if(s)v=s[0],t=s[1],l=s[2],u=s[3];else{t=x(b,d,-q);b=t.x;d=t.y;t=x(k,p,-q);k=t.x;p=t.y;F.cos(S/180*f);F.sin(S/180*f);t=(b-k)/2;v=(d-p)/2;u=t*t/(e*e)+v*v/(h*h);1<u&&(u=F.sqrt(u),e*=u,h*=u);var u=e*e,w=h*h,u=(l==n?-1:1)*F.sqrt(Z((u*w-u*v*v-w*t*t)/
(u*v*v+w*t*t)));l=u*e*v/h+(b+k)/2;var u=u*-h*t/e+(d+p)/2,v=F.asin(((d-u)/h).toFixed(9));t=F.asin(((p-u)/h).toFixed(9));v=b<l?S-v:v;t=k<l?S-t:t;0>v&&(v=2*S+v);0>t&&(t=2*S+t);n&&v>t&&(v-=2*S);!n&&t>v&&(t-=2*S)}if(Z(t-v)>r){var c=t,w=k,G=p;t=v+r*(n&&t>v?1:-1);k=l+e*F.cos(t);p=u+h*F.sin(t);c=K(k,p,e,h,f,0,n,w,G,[t,c,l,u])}l=t-v;f=F.cos(v);r=F.sin(v);n=F.cos(t);t=F.sin(t);l=F.tan(l/4);e=4/3*e*l;l*=4/3*h;h=[b,d];b=[b+e*r,d-l*f];d=[k+e*t,p-l*n];k=[k,p];b[0]=2*h[0]-b[0];b[1]=2*h[1]-b[1];if(s)return[b,d,k].concat(c);
c=[b,d,k].concat(c).join().split(",");s=[];k=0;for(p=c.length;k<p;k++)s[k]=k%2?x(c[k-1],c[k],q).y:x(c[k],c[k+1],q).x;return s}function U(a,b,d,e,h,f,l,k){for(var n=[],p=[[],[] ],s,r,c,t,q=0;2>q;++q)0==q?(r=6*a-12*d+6*h,s=-3*a+9*d-9*h+3*l,c=3*d-3*a):(r=6*b-12*e+6*f,s=-3*b+9*e-9*f+3*k,c=3*e-3*b),1E-12>Z(s)?1E-12>Z(r)||(s=-c/r,0<s&&1>s&&n.push(s)):(t=r*r-4*c*s,c=F.sqrt(t),0>t||(t=(-r+c)/(2*s),0<t&&1>t&&n.push(t),s=(-r-c)/(2*s),0<s&&1>s&&n.push(s)));for(r=q=n.length;q--;)s=n[q],c=1-s,p[0][q]=c*c*c*a+3*
c*c*s*d+3*c*s*s*h+s*s*s*l,p[1][q]=c*c*c*b+3*c*c*s*e+3*c*s*s*f+s*s*s*k;p[0][r]=a;p[1][r]=b;p[0][r+1]=l;p[1][r+1]=k;p[0].length=p[1].length=r+2;return{min:{x:X.apply(0,p[0]),y:X.apply(0,p[1])},max:{x:W.apply(0,p[0]),y:W.apply(0,p[1])}}}function I(a,b){var e=!b&&A(a);if(!b&&e.curve)return d(e.curve);var f=G(a),l=b&&G(b),n={x:0,y:0,bx:0,by:0,X:0,Y:0,qx:null,qy:null},k={x:0,y:0,bx:0,by:0,X:0,Y:0,qx:null,qy:null},p=function(a,b,c){if(!a)return["C",b.x,b.y,b.x,b.y,b.x,b.y];a[0]in{T:1,Q:1}||(b.qx=b.qy=null);
switch(a[0]){case "M":b.X=a[1];b.Y=a[2];break;case "A":a=["C"].concat(K.apply(0,[b.x,b.y].concat(a.slice(1))));break;case "S":"C"==c||"S"==c?(c=2*b.x-b.bx,b=2*b.y-b.by):(c=b.x,b=b.y);a=["C",c,b].concat(a.slice(1));break;case "T":"Q"==c||"T"==c?(b.qx=2*b.x-b.qx,b.qy=2*b.y-b.qy):(b.qx=b.x,b.qy=b.y);a=["C"].concat(J(b.x,b.y,b.qx,b.qy,a[1],a[2]));break;case "Q":b.qx=a[1];b.qy=a[2];a=["C"].concat(J(b.x,b.y,a[1],a[2],a[3],a[4]));break;case "L":a=["C"].concat(h(b.x,b.y,a[1],a[2]));break;case "H":a=["C"].concat(h(b.x,
b.y,a[1],b.y));break;case "V":a=["C"].concat(h(b.x,b.y,b.x,a[1]));break;case "Z":a=["C"].concat(h(b.x,b.y,b.X,b.Y))}return a},s=function(a,b){if(7<a[b].length){a[b].shift();for(var c=a[b];c.length;)q[b]="A",l&&(u[b]="A"),a.splice(b++,0,["C"].concat(c.splice(0,6)));a.splice(b,1);v=W(f.length,l&&l.length||0)}},r=function(a,b,c,d,e){a&&b&&"M"==a[e][0]&&"M"!=b[e][0]&&(b.splice(e,0,["M",d.x,d.y]),c.bx=0,c.by=0,c.x=a[e][1],c.y=a[e][2],v=W(f.length,l&&l.length||0))},q=[],u=[],c="",t="",x=0,v=W(f.length,
l&&l.length||0);for(;x<v;x++){f[x]&&(c=f[x][0]);"C"!=c&&(q[x]=c,x&&(t=q[x-1]));f[x]=p(f[x],n,t);"A"!=q[x]&&"C"==c&&(q[x]="C");s(f,x);l&&(l[x]&&(c=l[x][0]),"C"!=c&&(u[x]=c,x&&(t=u[x-1])),l[x]=p(l[x],k,t),"A"!=u[x]&&"C"==c&&(u[x]="C"),s(l,x));r(f,l,n,k,x);r(l,f,k,n,x);var w=f[x],z=l&&l[x],y=w.length,U=l&&z.length;n.x=w[y-2];n.y=w[y-1];n.bx=$(w[y-4])||n.x;n.by=$(w[y-3])||n.y;k.bx=l&&($(z[U-4])||k.x);k.by=l&&($(z[U-3])||k.y);k.x=l&&z[U-2];k.y=l&&z[U-1]}l||(e.curve=d(f));return l?[f,l]:f}function P(a,
b){for(var d=[],e=0,h=a.length;h-2*!b>e;e+=2){var f=[{x:+a[e-2],y:+a[e-1]},{x:+a[e],y:+a[e+1]},{x:+a[e+2],y:+a[e+3]},{x:+a[e+4],y:+a[e+5]}];b?e?h-4==e?f[3]={x:+a[0],y:+a[1]}:h-2==e&&(f[2]={x:+a[0],y:+a[1]},f[3]={x:+a[2],y:+a[3]}):f[0]={x:+a[h-2],y:+a[h-1]}:h-4==e?f[3]=f[2]:e||(f[0]={x:+a[e],y:+a[e+1]});d.push(["C",(-f[0].x+6*f[1].x+f[2].x)/6,(-f[0].y+6*f[1].y+f[2].y)/6,(f[1].x+6*f[2].x-f[3].x)/6,(f[1].y+6*f[2].y-f[3].y)/6,f[2].x,f[2].y])}return d}y=k.prototype;var Q=a.is,C=a._.clone,L="hasOwnProperty",
N=/,?([a-z]),?/gi,$=parseFloat,F=Math,S=F.PI,X=F.min,W=F.max,ma=F.pow,Z=F.abs;M=n(1);var na=n(),ba=n(0,1),V=a._unit2px;a.path=A;a.path.getTotalLength=M;a.path.getPointAtLength=na;a.path.getSubpath=function(a,b,d){if(1E-6>this.getTotalLength(a)-d)return ba(a,b).end;a=ba(a,d,1);return b?ba(a,b).end:a};y.getTotalLength=function(){if(this.node.getTotalLength)return this.node.getTotalLength()};y.getPointAtLength=function(a){return na(this.attr("d"),a)};y.getSubpath=function(b,d){return a.path.getSubpath(this.attr("d"),
b,d)};a._.box=w;a.path.findDotsAtSegment=u;a.path.bezierBBox=p;a.path.isPointInsideBBox=b;a.path.isBBoxIntersect=q;a.path.intersection=function(a,b){return l(a,b)};a.path.intersectionNumber=function(a,b){return l(a,b,1)};a.path.isPointInside=function(a,d,e){var h=r(a);return b(h,d,e)&&1==l(a,[["M",d,e],["H",h.x2+10] ],1)%2};a.path.getBBox=r;a.path.get={path:function(a){return a.attr("path")},circle:function(a){a=V(a);return x(a.cx,a.cy,a.r)},ellipse:function(a){a=V(a);return x(a.cx||0,a.cy||0,a.rx,
a.ry)},rect:function(a){a=V(a);return s(a.x||0,a.y||0,a.width,a.height,a.rx,a.ry)},image:function(a){a=V(a);return s(a.x||0,a.y||0,a.width,a.height)},line:function(a){return"M"+[a.attr("x1")||0,a.attr("y1")||0,a.attr("x2"),a.attr("y2")]},polyline:function(a){return"M"+a.attr("points")},polygon:function(a){return"M"+a.attr("points")+"z"},deflt:function(a){a=a.node.getBBox();return s(a.x,a.y,a.width,a.height)}};a.path.toRelative=function(b){var e=A(b),h=String.prototype.toLowerCase;if(e.rel)return d(e.rel);
a.is(b,"array")&&a.is(b&&b[0],"array")||(b=a.parsePathString(b));var f=[],l=0,n=0,k=0,p=0,s=0;"M"==b[0][0]&&(l=b[0][1],n=b[0][2],k=l,p=n,s++,f.push(["M",l,n]));for(var r=b.length;s<r;s++){var q=f[s]=[],x=b[s];if(x[0]!=h.call(x[0]))switch(q[0]=h.call(x[0]),q[0]){case "a":q[1]=x[1];q[2]=x[2];q[3]=x[3];q[4]=x[4];q[5]=x[5];q[6]=+(x[6]-l).toFixed(3);q[7]=+(x[7]-n).toFixed(3);break;case "v":q[1]=+(x[1]-n).toFixed(3);break;case "m":k=x[1],p=x[2];default:for(var c=1,t=x.length;c<t;c++)q[c]=+(x[c]-(c%2?l:
n)).toFixed(3)}else for(f[s]=[],"m"==x[0]&&(k=x[1]+l,p=x[2]+n),q=0,c=x.length;q<c;q++)f[s][q]=x[q];x=f[s].length;switch(f[s][0]){case "z":l=k;n=p;break;case "h":l+=+f[s][x-1];break;case "v":n+=+f[s][x-1];break;default:l+=+f[s][x-2],n+=+f[s][x-1]}}f.toString=z;e.rel=d(f);return f};a.path.toAbsolute=G;a.path.toCubic=I;a.path.map=function(a,b){if(!b)return a;var d,e,h,f,l,n,k;a=I(a);h=0;for(l=a.length;h<l;h++)for(k=a[h],f=1,n=k.length;f<n;f+=2)d=b.x(k[f],k[f+1]),e=b.y(k[f],k[f+1]),k[f]=d,k[f+1]=e;return a};
a.path.toString=z;a.path.clone=d});C.plugin(function(a,v,y,C){var A=Math.max,w=Math.min,z=function(a){this.items=[];this.bindings={};this.length=0;this.type="set";if(a)for(var f=0,n=a.length;f<n;f++)a[f]&&(this[this.items.length]=this.items[this.items.length]=a[f],this.length++)};v=z.prototype;v.push=function(){for(var a,f,n=0,k=arguments.length;n<k;n++)if(a=arguments[n])f=this.items.length,this[f]=this.items[f]=a,this.length++;return this};v.pop=function(){this.length&&delete this[this.length--];
return this.items.pop()};v.forEach=function(a,f){for(var n=0,k=this.items.length;n<k&&!1!==a.call(f,this.items[n],n);n++);return this};v.animate=function(d,f,n,u){"function"!=typeof n||n.length||(u=n,n=L.linear);d instanceof a._.Animation&&(u=d.callback,n=d.easing,f=n.dur,d=d.attr);var p=arguments;if(a.is(d,"array")&&a.is(p[p.length-1],"array"))var b=!0;var q,e=function(){q?this.b=q:q=this.b},l=0,r=u&&function(){l++==this.length&&u.call(this)};return this.forEach(function(a,l){k.once("snap.animcreated."+
a.id,e);b?p[l]&&a.animate.apply(a,p[l]):a.animate(d,f,n,r)})};v.remove=function(){for(;this.length;)this.pop().remove();return this};v.bind=function(a,f,k){var u={};if("function"==typeof f)this.bindings[a]=f;else{var p=k||a;this.bindings[a]=function(a){u[p]=a;f.attr(u)}}return this};v.attr=function(a){var f={},k;for(k in a)if(this.bindings[k])this.bindings[k](a[k]);else f[k]=a[k];a=0;for(k=this.items.length;a<k;a++)this.items[a].attr(f);return this};v.clear=function(){for(;this.length;)this.pop()};
v.splice=function(a,f,k){a=0>a?A(this.length+a,0):a;f=A(0,w(this.length-a,f));var u=[],p=[],b=[],q;for(q=2;q<arguments.length;q++)b.push(arguments[q]);for(q=0;q<f;q++)p.push(this[a+q]);for(;q<this.length-a;q++)u.push(this[a+q]);var e=b.length;for(q=0;q<e+u.length;q++)this.items[a+q]=this[a+q]=q<e?b[q]:u[q-e];for(q=this.items.length=this.length-=f-e;this[q];)delete this[q++];return new z(p)};v.exclude=function(a){for(var f=0,k=this.length;f<k;f++)if(this[f]==a)return this.splice(f,1),!0;return!1};
v.insertAfter=function(a){for(var f=this.items.length;f--;)this.items[f].insertAfter(a);return this};v.getBBox=function(){for(var a=[],f=[],k=[],u=[],p=this.items.length;p--;)if(!this.items[p].removed){var b=this.items[p].getBBox();a.push(b.x);f.push(b.y);k.push(b.x+b.width);u.push(b.y+b.height)}a=w.apply(0,a);f=w.apply(0,f);k=A.apply(0,k);u=A.apply(0,u);return{x:a,y:f,x2:k,y2:u,width:k-a,height:u-f,cx:a+(k-a)/2,cy:f+(u-f)/2}};v.clone=function(a){a=new z;for(var f=0,k=this.items.length;f<k;f++)a.push(this.items[f].clone());
return a};v.toString=function(){return"Snap\u2018s set"};v.type="set";a.set=function(){var a=new z;arguments.length&&a.push.apply(a,Array.prototype.slice.call(arguments,0));return a}});C.plugin(function(a,v,y,C){function A(a){var b=a[0];switch(b.toLowerCase()){case "t":return[b,0,0];case "m":return[b,1,0,0,1,0,0];case "r":return 4==a.length?[b,0,a[2],a[3] ]:[b,0];case "s":return 5==a.length?[b,1,1,a[3],a[4] ]:3==a.length?[b,1,1]:[b,1]}}function w(b,d,f){d=q(d).replace(/\.{3}|\u2026/g,b);b=a.parseTransformString(b)||
[];d=a.parseTransformString(d)||[];for(var k=Math.max(b.length,d.length),p=[],v=[],h=0,w,z,y,I;h<k;h++){y=b[h]||A(d[h]);I=d[h]||A(y);if(y[0]!=I[0]||"r"==y[0].toLowerCase()&&(y[2]!=I[2]||y[3]!=I[3])||"s"==y[0].toLowerCase()&&(y[3]!=I[3]||y[4]!=I[4])){b=a._.transform2matrix(b,f());d=a._.transform2matrix(d,f());p=[["m",b.a,b.b,b.c,b.d,b.e,b.f] ];v=[["m",d.a,d.b,d.c,d.d,d.e,d.f] ];break}p[h]=[];v[h]=[];w=0;for(z=Math.max(y.length,I.length);w<z;w++)w in y&&(p[h][w]=y[w]),w in I&&(v[h][w]=I[w])}return{from:u(p),
to:u(v),f:n(p)}}function z(a){return a}function d(a){return function(b){return+b.toFixed(3)+a}}function f(b){return a.rgb(b[0],b[1],b[2])}function n(a){var b=0,d,f,k,n,h,p,q=[];d=0;for(f=a.length;d<f;d++){h="[";p=['"'+a[d][0]+'"'];k=1;for(n=a[d].length;k<n;k++)p[k]="val["+b++ +"]";h+=p+"]";q[d]=h}return Function("val","return Snap.path.toString.call(["+q+"])")}function u(a){for(var b=[],d=0,f=a.length;d<f;d++)for(var k=1,n=a[d].length;k<n;k++)b.push(a[d][k]);return b}var p={},b=/[a-z]+$/i,q=String;
p.stroke=p.fill="colour";v.prototype.equal=function(a,b){return k("snap.util.equal",this,a,b).firstDefined()};k.on("snap.util.equal",function(e,k){var r,s;r=q(this.attr(e)||"");var x=this;if(r==+r&&k==+k)return{from:+r,to:+k,f:z};if("colour"==p[e])return r=a.color(r),s=a.color(k),{from:[r.r,r.g,r.b,r.opacity],to:[s.r,s.g,s.b,s.opacity],f:f};if("transform"==e||"gradientTransform"==e||"patternTransform"==e)return k instanceof a.Matrix&&(k=k.toTransformString()),a._.rgTransform.test(k)||(k=a._.svgTransform2string(k)),
w(r,k,function(){return x.getBBox(1)});if("d"==e||"path"==e)return r=a.path.toCubic(r,k),{from:u(r[0]),to:u(r[1]),f:n(r[0])};if("points"==e)return r=q(r).split(a._.separator),s=q(k).split(a._.separator),{from:r,to:s,f:function(a){return a}};aUnit=r.match(b);s=q(k).match(b);return aUnit&&aUnit==s?{from:parseFloat(r),to:parseFloat(k),f:d(aUnit)}:{from:this.asPX(e),to:this.asPX(e,k),f:z}})});C.plugin(function(a,v,y,C){var A=v.prototype,w="createTouch"in C.doc;v="click dblclick mousedown mousemove mouseout mouseover mouseup touchstart touchmove touchend touchcancel".split(" ");
var z={mousedown:"touchstart",mousemove:"touchmove",mouseup:"touchend"},d=function(a,b){var d="y"==a?"scrollTop":"scrollLeft",e=b&&b.node?b.node.ownerDocument:C.doc;return e[d in e.documentElement?"documentElement":"body"][d]},f=function(){this.returnValue=!1},n=function(){return this.originalEvent.preventDefault()},u=function(){this.cancelBubble=!0},p=function(){return this.originalEvent.stopPropagation()},b=function(){if(C.doc.addEventListener)return function(a,b,e,f){var k=w&&z[b]?z[b]:b,l=function(k){var l=
d("y",f),q=d("x",f);if(w&&z.hasOwnProperty(b))for(var r=0,u=k.targetTouches&&k.targetTouches.length;r<u;r++)if(k.targetTouches[r].target==a||a.contains(k.targetTouches[r].target)){u=k;k=k.targetTouches[r];k.originalEvent=u;k.preventDefault=n;k.stopPropagation=p;break}return e.call(f,k,k.clientX+q,k.clientY+l)};b!==k&&a.addEventListener(b,l,!1);a.addEventListener(k,l,!1);return function(){b!==k&&a.removeEventListener(b,l,!1);a.removeEventListener(k,l,!1);return!0}};if(C.doc.attachEvent)return function(a,
b,e,h){var k=function(a){a=a||h.node.ownerDocument.window.event;var b=d("y",h),k=d("x",h),k=a.clientX+k,b=a.clientY+b;a.preventDefault=a.preventDefault||f;a.stopPropagation=a.stopPropagation||u;return e.call(h,a,k,b)};a.attachEvent("on"+b,k);return function(){a.detachEvent("on"+b,k);return!0}}}(),q=[],e=function(a){for(var b=a.clientX,e=a.clientY,f=d("y"),l=d("x"),n,p=q.length;p--;){n=q[p];if(w)for(var r=a.touches&&a.touches.length,u;r--;){if(u=a.touches[r],u.identifier==n.el._drag.id||n.el.node.contains(u.target)){b=
u.clientX;e=u.clientY;(a.originalEvent?a.originalEvent:a).preventDefault();break}}else a.preventDefault();b+=l;e+=f;k("snap.drag.move."+n.el.id,n.move_scope||n.el,b-n.el._drag.x,e-n.el._drag.y,b,e,a)}},l=function(b){a.unmousemove(e).unmouseup(l);for(var d=q.length,f;d--;)f=q[d],f.el._drag={},k("snap.drag.end."+f.el.id,f.end_scope||f.start_scope||f.move_scope||f.el,b);q=[]};for(y=v.length;y--;)(function(d){a[d]=A[d]=function(e,f){a.is(e,"function")&&(this.events=this.events||[],this.events.push({name:d,
f:e,unbind:b(this.node||document,d,e,f||this)}));return this};a["un"+d]=A["un"+d]=function(a){for(var b=this.events||[],e=b.length;e--;)if(b[e].name==d&&(b[e].f==a||!a)){b[e].unbind();b.splice(e,1);!b.length&&delete this.events;break}return this}})(v[y]);A.hover=function(a,b,d,e){return this.mouseover(a,d).mouseout(b,e||d)};A.unhover=function(a,b){return this.unmouseover(a).unmouseout(b)};var r=[];A.drag=function(b,d,f,h,n,p){function u(r,v,w){(r.originalEvent||r).preventDefault();this._drag.x=v;
this._drag.y=w;this._drag.id=r.identifier;!q.length&&a.mousemove(e).mouseup(l);q.push({el:this,move_scope:h,start_scope:n,end_scope:p});d&&k.on("snap.drag.start."+this.id,d);b&&k.on("snap.drag.move."+this.id,b);f&&k.on("snap.drag.end."+this.id,f);k("snap.drag.start."+this.id,n||h||this,v,w,r)}if(!arguments.length){var v;return this.drag(function(a,b){this.attr({transform:v+(v?"T":"t")+[a,b]})},function(){v=this.transform().local})}this._drag={};r.push({el:this,start:u});this.mousedown(u);return this};
A.undrag=function(){for(var b=r.length;b--;)r[b].el==this&&(this.unmousedown(r[b].start),r.splice(b,1),k.unbind("snap.drag.*."+this.id));!r.length&&a.unmousemove(e).unmouseup(l);return this}});C.plugin(function(a,v,y,C){y=y.prototype;var A=/^\s*url\((.+)\)/,w=String,z=a._.$;a.filter={};y.filter=function(d){var f=this;"svg"!=f.type&&(f=f.paper);d=a.parse(w(d));var k=a._.id(),u=z("filter");z(u,{id:k,filterUnits:"userSpaceOnUse"});u.appendChild(d.node);f.defs.appendChild(u);return new v(u)};k.on("snap.util.getattr.filter",
function(){k.stop();var d=z(this.node,"filter");if(d)return(d=w(d).match(A))&&a.select(d[1])});k.on("snap.util.attr.filter",function(d){if(d instanceof v&&"filter"==d.type){k.stop();var f=d.node.id;f||(z(d.node,{id:d.id}),f=d.id);z(this.node,{filter:a.url(f)})}d&&"none"!=d||(k.stop(),this.node.removeAttribute("filter"))});a.filter.blur=function(d,f){null==d&&(d=2);return a.format('<feGaussianBlur stdDeviation="{def}"/>',{def:null==f?d:[d,f]})};a.filter.blur.toString=function(){return this()};a.filter.shadow=
function(d,f,k,u,p){"string"==typeof k&&(p=u=k,k=4);"string"!=typeof u&&(p=u,u="#000");null==k&&(k=4);null==p&&(p=1);null==d&&(d=0,f=2);null==f&&(f=d);u=a.color(u||"#000");return a.format('<feGaussianBlur in="SourceAlpha" stdDeviation="{blur}"/><feOffset dx="{dx}" dy="{dy}" result="offsetblur"/><feFlood flood-color="{color}"/><feComposite in2="offsetblur" operator="in"/><feComponentTransfer><feFuncA type="linear" slope="{opacity}"/></feComponentTransfer><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>',
{color:u,dx:d,dy:f,blur:k,opacity:p})};a.filter.shadow.toString=function(){return this()};a.filter.grayscale=function(d){null==d&&(d=1);return a.format('<feColorMatrix type="matrix" values="{a} {b} {c} 0 0 {d} {e} {f} 0 0 {g} {b} {h} 0 0 0 0 0 1 0"/>',{a:0.2126+0.7874*(1-d),b:0.7152-0.7152*(1-d),c:0.0722-0.0722*(1-d),d:0.2126-0.2126*(1-d),e:0.7152+0.2848*(1-d),f:0.0722-0.0722*(1-d),g:0.2126-0.2126*(1-d),h:0.0722+0.9278*(1-d)})};a.filter.grayscale.toString=function(){return this()};a.filter.sepia=
function(d){null==d&&(d=1);return a.format('<feColorMatrix type="matrix" values="{a} {b} {c} 0 0 {d} {e} {f} 0 0 {g} {h} {i} 0 0 0 0 0 1 0"/>',{a:0.393+0.607*(1-d),b:0.769-0.769*(1-d),c:0.189-0.189*(1-d),d:0.349-0.349*(1-d),e:0.686+0.314*(1-d),f:0.168-0.168*(1-d),g:0.272-0.272*(1-d),h:0.534-0.534*(1-d),i:0.131+0.869*(1-d)})};a.filter.sepia.toString=function(){return this()};a.filter.saturate=function(d){null==d&&(d=1);return a.format('<feColorMatrix type="saturate" values="{amount}"/>',{amount:1-
d})};a.filter.saturate.toString=function(){return this()};a.filter.hueRotate=function(d){return a.format('<feColorMatrix type="hueRotate" values="{angle}"/>',{angle:d||0})};a.filter.hueRotate.toString=function(){return this()};a.filter.invert=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="table" tableValues="{amount} {amount2}"/><feFuncG type="table" tableValues="{amount} {amount2}"/><feFuncB type="table" tableValues="{amount} {amount2}"/></feComponentTransfer>',{amount:d,
amount2:1-d})};a.filter.invert.toString=function(){return this()};a.filter.brightness=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="linear" slope="{amount}"/><feFuncG type="linear" slope="{amount}"/><feFuncB type="linear" slope="{amount}"/></feComponentTransfer>',{amount:d})};a.filter.brightness.toString=function(){return this()};a.filter.contrast=function(d){null==d&&(d=1);return a.format('<feComponentTransfer><feFuncR type="linear" slope="{amount}" intercept="{amount2}"/><feFuncG type="linear" slope="{amount}" intercept="{amount2}"/><feFuncB type="linear" slope="{amount}" intercept="{amount2}"/></feComponentTransfer>',
{amount:d,amount2:0.5-d/2})};a.filter.contrast.toString=function(){return this()}});return C});

]]> </script>
<script> <![CDATA[

(function (glob, factory) {
    // AMD support
    if (typeof define === "function" && define.amd) {
        // Define as an anonymous module
        define("Gadfly", ["Snap.svg"], function (Snap) {
            return factory(Snap);
        });
    } else {
        // Browser globals (glob is window)
        // Snap adds itself to window
        glob.Gadfly = factory(glob.Snap);
    }
}(this, function (Snap) {

var Gadfly = {};

// Get an x/y coordinate value in pixels
var xPX = function(fig, x) {
    var client_box = fig.node.getBoundingClientRect();
    return x * fig.node.viewBox.baseVal.width / client_box.width;
};

var yPX = function(fig, y) {
    var client_box = fig.node.getBoundingClientRect();
    return y * fig.node.viewBox.baseVal.height / client_box.height;
};


Snap.plugin(function (Snap, Element, Paper, global) {
    // Traverse upwards from a snap element to find and return the first
    // note with the "plotroot" class.
    Element.prototype.plotroot = function () {
        var element = this;
        while (!element.hasClass("plotroot") && element.parent() != null) {
            element = element.parent();
        }
        return element;
    };

    Element.prototype.svgroot = function () {
        var element = this;
        while (element.node.nodeName != "svg" && element.parent() != null) {
            element = element.parent();
        }
        return element;
    };

    Element.prototype.plotbounds = function () {
        var root = this.plotroot()
        var bbox = root.select(".guide.background").node.getBBox();
        return {
            x0: bbox.x,
            x1: bbox.x + bbox.width,
            y0: bbox.y,
            y1: bbox.y + bbox.height
        };
    };

    Element.prototype.plotcenter = function () {
        var root = this.plotroot()
        var bbox = root.select(".guide.background").node.getBBox();
        return {
            x: bbox.x + bbox.width / 2,
            y: bbox.y + bbox.height / 2
        };
    };

    // Emulate IE style mouseenter/mouseleave events, since Microsoft always
    // does everything right.
    // See: http://www.dynamic-tools.net/toolbox/isMouseLeaveOrEnter/
    var events = ["mouseenter", "mouseleave"];

    for (i in events) {
        (function (event_name) {
            var event_name = events[i];
            Element.prototype[event_name] = function (fn, scope) {
                if (Snap.is(fn, "function")) {
                    var fn2 = function (event) {
                        if (event.type != "mouseover" && event.type != "mouseout") {
                            return;
                        }

                        var reltg = event.relatedTarget ? event.relatedTarget :
                            event.type == "mouseout" ? event.toElement : event.fromElement;
                        while (reltg && reltg != this.node) reltg = reltg.parentNode;

                        if (reltg != this.node) {
                            return fn.apply(this, event);
                        }
                    };

                    if (event_name == "mouseenter") {
                        this.mouseover(fn2, scope);
                    } else {
                        this.mouseout(fn2, scope);
                    }
                }
                return this;
            };
        })(events[i]);
    }


    Element.prototype.mousewheel = function (fn, scope) {
        if (Snap.is(fn, "function")) {
            var el = this;
            var fn2 = function (event) {
                fn.apply(el, [event]);
            };
        }

        this.node.addEventListener(
            /Firefox/i.test(navigator.userAgent) ? "DOMMouseScroll" : "mousewheel",
            fn2);

        return this;
    };


    // Snap's attr function can be too slow for things like panning/zooming.
    // This is a function to directly update element attributes without going
    // through eve.
    Element.prototype.attribute = function(key, val) {
        if (val === undefined) {
            return this.node.getAttribute(key);
        } else {
            this.node.setAttribute(key, val);
            return this;
        }
    };
});


// When the plot is moused over, emphasize the grid lines.
Gadfly.plot_mouseover = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);

    var xgridlines = root.select(".xgridlines"),
        ygridlines = root.select(".ygridlines");

    xgridlines.data("unfocused_strokedash",
                    xgridlines.attribute("stroke-dasharray").replace(/(\d)(,|$)/g, "$1mm$2"));
    ygridlines.data("unfocused_strokedash",
                    ygridlines.attribute("stroke-dasharray").replace(/(\d)(,|$)/g, "$1mm$2"));

    // emphasize grid lines
    var destcolor = root.data("focused_xgrid_color");
    xgridlines.attribute("stroke-dasharray", "none")
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    destcolor = root.data("focused_ygrid_color");
    ygridlines.attribute("stroke-dasharray", "none")
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    // reveal zoom slider
    root.select(".zoomslider")
        .animate({opacity: 1.0}, 250);
};


// Unemphasize grid lines on mouse out.
Gadfly.plot_mouseout = function(event) {
    var root = this.plotroot();
    var xgridlines = root.select(".xgridlines"),
        ygridlines = root.select(".ygridlines");

    var destcolor = root.data("unfocused_xgrid_color");

    xgridlines.attribute("stroke-dasharray", xgridlines.data("unfocused_strokedash"))
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    destcolor = root.data("unfocused_ygrid_color");
    ygridlines.attribute("stroke-dasharray", ygridlines.data("unfocused_strokedash"))
              .selectAll("path")
              .animate({stroke: destcolor}, 250);

    // hide zoom slider
    root.select(".zoomslider")
        .animate({opacity: 0.0}, 250);
};


var set_geometry_transform = function(root, tx, ty, scale) {
    var xscalable = root.hasClass("xscalable"),
        yscalable = root.hasClass("yscalable");

    var old_scale = root.data("scale");

    var xscale = xscalable ? scale : 1.0,
        yscale = yscalable ? scale : 1.0;

    tx = xscalable ? tx : 0.0;
    ty = yscalable ? ty : 0.0;

    var t = new Snap.Matrix().translate(tx, ty).scale(xscale, yscale);

    root.selectAll(".geometry, image")
        .forEach(function (element, i) {
            element.transform(t);
        });

    bounds = root.plotbounds();

    if (yscalable) {
        var xfixed_t = new Snap.Matrix().translate(0, ty).scale(1.0, yscale);
        root.selectAll(".xfixed")
            .forEach(function (element, i) {
                element.transform(xfixed_t);
            });

        root.select(".ylabels")
            .transform(xfixed_t)
            .selectAll("text")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var cx = element.asPX("x"),
                        cy = element.asPX("y");
                    var st = element.data("static_transform");
                    unscale_t = new Snap.Matrix();
                    unscale_t.scale(1, 1/scale, cx, cy).add(st);
                    element.transform(unscale_t);

                    var y = cy * scale + ty;
                    element.attr("visibility",
                        bounds.y0 <= y && y <= bounds.y1 ? "visible" : "hidden");
                }
            });
    }

    if (xscalable) {
        var yfixed_t = new Snap.Matrix().translate(tx, 0).scale(xscale, 1.0);
        var xtrans = new Snap.Matrix().translate(tx, 0);
        root.selectAll(".yfixed")
            .forEach(function (element, i) {
                element.transform(yfixed_t);
            });

        root.select(".xlabels")
            .transform(yfixed_t)
            .selectAll("text")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var cx = element.asPX("x"),
                        cy = element.asPX("y");
                    var st = element.data("static_transform");
                    unscale_t = new Snap.Matrix();
                    unscale_t.scale(1/scale, 1, cx, cy).add(st);

                    element.transform(unscale_t);

                    var x = cx * scale + tx;
                    element.attr("visibility",
                        bounds.x0 <= x && x <= bounds.x1 ? "visible" : "hidden");
                    }
            });
    }

    // we must unscale anything that is scale invariance: widths, raiduses, etc.
    var size_attribs = ["font-size"];
    var unscaled_selection = ".geometry, .geometry *";
    if (xscalable) {
        size_attribs.push("rx");
        unscaled_selection += ", .xgridlines";
    }
    if (yscalable) {
        size_attribs.push("ry");
        unscaled_selection += ", .ygridlines";
    }

    root.selectAll(unscaled_selection)
        .forEach(function (element, i) {
            // circle need special help
            if (element.node.nodeName == "circle") {
                var cx = element.attribute("cx"),
                    cy = element.attribute("cy");
                unscale_t = new Snap.Matrix().scale(1/xscale, 1/yscale,
                                                        cx, cy);
                element.transform(unscale_t);
                return;
            }

            for (i in size_attribs) {
                var key = size_attribs[i];
                var val = parseFloat(element.attribute(key));
                if (val !== undefined && val != 0 && !isNaN(val)) {
                    element.attribute(key, val * old_scale / scale);
                }
            }
        });
};


// Find the most appropriate tick scale and update label visibility.
var update_tickscale = function(root, scale, axis) {
    if (!root.hasClass(axis + "scalable")) return;

    var tickscales = root.data(axis + "tickscales");
    var best_tickscale = 1.0;
    var best_tickscale_dist = Infinity;
    for (tickscale in tickscales) {
        var dist = Math.abs(Math.log(tickscale) - Math.log(scale));
        if (dist < best_tickscale_dist) {
            best_tickscale_dist = dist;
            best_tickscale = tickscale;
        }
    }

    if (best_tickscale != root.data(axis + "tickscale")) {
        root.data(axis + "tickscale", best_tickscale);
        var mark_inscale_gridlines = function (element, i) {
            var inscale = element.attr("gadfly:scale") == best_tickscale;
            element.attribute("gadfly:inscale", inscale);
            element.attr("visibility", inscale ? "visible" : "hidden");
        };

        var mark_inscale_labels = function (element, i) {
            var inscale = element.attr("gadfly:scale") == best_tickscale;
            element.attribute("gadfly:inscale", inscale);
            element.attr("visibility", inscale ? "visible" : "hidden");
        };

        root.select("." + axis + "gridlines").selectAll("path").forEach(mark_inscale_gridlines);
        root.select("." + axis + "labels").selectAll("text").forEach(mark_inscale_labels);
    }
};


var set_plot_pan_zoom = function(root, tx, ty, scale) {
    var old_scale = root.data("scale");
    var bounds = root.plotbounds();

    var width = bounds.x1 - bounds.x0,
        height = bounds.y1 - bounds.y0;

    // compute the viewport derived from tx, ty, and scale
    var x_min = -width * scale - (scale * width - width),
        x_max = width * scale,
        y_min = -height * scale - (scale * height - height),
        y_max = height * scale;

    var x0 = bounds.x0 - scale * bounds.x0,
        y0 = bounds.y0 - scale * bounds.y0;

    var tx = Math.max(Math.min(tx - x0, x_max), x_min),
        ty = Math.max(Math.min(ty - y0, y_max), y_min);

    tx += x0;
    ty += y0;

    // when the scale change, we may need to alter which set of
    // ticks is being displayed
    if (scale != old_scale) {
        update_tickscale(root, scale, "x");
        update_tickscale(root, scale, "y");
    }

    set_geometry_transform(root, tx, ty, scale);

    root.data("scale", scale);
    root.data("tx", tx);
    root.data("ty", ty);
};


var scale_centered_translation = function(root, scale) {
    var bounds = root.plotbounds();

    var width = bounds.x1 - bounds.x0,
        height = bounds.y1 - bounds.y0;

    var tx0 = root.data("tx"),
        ty0 = root.data("ty");

    var scale0 = root.data("scale");

    // how off from center the current view is
    var xoff = tx0 - (bounds.x0 * (1 - scale0) + (width * (1 - scale0)) / 2),
        yoff = ty0 - (bounds.y0 * (1 - scale0) + (height * (1 - scale0)) / 2);

    // rescale offsets
    xoff = xoff * scale / scale0;
    yoff = yoff * scale / scale0;

    // adjust for the panel position being scaled
    var x_edge_adjust = bounds.x0 * (1 - scale),
        y_edge_adjust = bounds.y0 * (1 - scale);

    return {
        x: xoff + x_edge_adjust + (width - width * scale) / 2,
        y: yoff + y_edge_adjust + (height - height * scale) / 2
    };
};


// Initialize data for panning zooming if it isn't already.
var init_pan_zoom = function(root) {
    if (root.data("zoompan-ready")) {
        return;
    }

    // The non-scaling-stroke trick. Rather than try to correct for the
    // stroke-width when zooming, we force it to a fixed value.
    var px_per_mm = root.node.getCTM().a;

    // Drag events report deltas in pixels, which we'd like to convert to
    // millimeters.
    root.data("px_per_mm", px_per_mm);

    root.selectAll("path")
        .forEach(function (element, i) {
        sw = element.asPX("stroke-width") * px_per_mm;
        if (sw > 0) {
            element.attribute("stroke-width", sw);
            element.attribute("vector-effect", "non-scaling-stroke");
        }
    });

    // Store ticks labels original tranformation
    root.selectAll(".xlabels > text, .ylabels > text")
        .forEach(function (element, i) {
            var lm = element.transform().localMatrix;
            element.data("static_transform",
                new Snap.Matrix(lm.a, lm.b, lm.c, lm.d, lm.e, lm.f));
        });

    var xgridlines = root.select(".xgridlines");
    var ygridlines = root.select(".ygridlines");
    var xlabels = root.select(".xlabels");
    var ylabels = root.select(".ylabels");

    if (root.data("tx") === undefined) root.data("tx", 0);
    if (root.data("ty") === undefined) root.data("ty", 0);
    if (root.data("scale") === undefined) root.data("scale", 1.0);
    if (root.data("xtickscales") === undefined) {

        // index all the tick scales that are listed
        var xtickscales = {};
        var ytickscales = {};
        var add_x_tick_scales = function (element, i) {
            xtickscales[element.attribute("gadfly:scale")] = true;
        };
        var add_y_tick_scales = function (element, i) {
            ytickscales[element.attribute("gadfly:scale")] = true;
        };

        if (xgridlines) xgridlines.selectAll("path").forEach(add_x_tick_scales);
        if (ygridlines) ygridlines.selectAll("path").forEach(add_y_tick_scales);
        if (xlabels) xlabels.selectAll("text").forEach(add_x_tick_scales);
        if (ylabels) ylabels.selectAll("text").forEach(add_y_tick_scales);

        root.data("xtickscales", xtickscales);
        root.data("ytickscales", ytickscales);
        root.data("xtickscale", 1.0);
    }

    var min_scale = 1.0, max_scale = 1.0;
    for (scale in xtickscales) {
        min_scale = Math.min(min_scale, scale);
        max_scale = Math.max(max_scale, scale);
    }
    for (scale in ytickscales) {
        min_scale = Math.min(min_scale, scale);
        max_scale = Math.max(max_scale, scale);
    }
    root.data("min_scale", min_scale);
    root.data("max_scale", max_scale);

    // store the original positions of labels
    if (xlabels) {
        xlabels.selectAll("text")
               .forEach(function (element, i) {
                   element.data("x", element.asPX("x"));
               });
    }

    if (ylabels) {
        ylabels.selectAll("text")
               .forEach(function (element, i) {
                   element.data("y", element.asPX("y"));
               });
    }

    // mark grid lines and ticks as in or out of scale.
    var mark_inscale = function (element, i) {
        element.attribute("gadfly:inscale", element.attribute("gadfly:scale") == 1.0);
    };

    if (xgridlines) xgridlines.selectAll("path").forEach(mark_inscale);
    if (ygridlines) ygridlines.selectAll("path").forEach(mark_inscale);
    if (xlabels) xlabels.selectAll("text").forEach(mark_inscale);
    if (ylabels) ylabels.selectAll("text").forEach(mark_inscale);

    // figure out the upper ond lower bounds on panning using the maximum
    // and minum grid lines
    var bounds = root.plotbounds();
    var pan_bounds = {
        x0: 0.0,
        y0: 0.0,
        x1: 0.0,
        y1: 0.0
    };

    if (xgridlines) {
        xgridlines
            .selectAll("path")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var bbox = element.node.getBBox();
                    if (bounds.x1 - bbox.x < pan_bounds.x0) {
                        pan_bounds.x0 = bounds.x1 - bbox.x;
                    }
                    if (bounds.x0 - bbox.x > pan_bounds.x1) {
                        pan_bounds.x1 = bounds.x0 - bbox.x;
                    }
                    element.attr("visibility", "visible");
                }
            });
    }

    if (ygridlines) {
        ygridlines
            .selectAll("path")
            .forEach(function (element, i) {
                if (element.attribute("gadfly:inscale") == "true") {
                    var bbox = element.node.getBBox();
                    if (bounds.y1 - bbox.y < pan_bounds.y0) {
                        pan_bounds.y0 = bounds.y1 - bbox.y;
                    }
                    if (bounds.y0 - bbox.y > pan_bounds.y1) {
                        pan_bounds.y1 = bounds.y0 - bbox.y;
                    }
                    element.attr("visibility", "visible");
                }
            });
    }

    // nudge these values a little
    pan_bounds.x0 -= 5;
    pan_bounds.x1 += 5;
    pan_bounds.y0 -= 5;
    pan_bounds.y1 += 5;
    root.data("pan_bounds", pan_bounds);

    root.data("zoompan-ready", true)
};


// Panning
Gadfly.guide_background_drag_onmove = function(dx, dy, x, y, event) {
    var root = this.plotroot();
    var px_per_mm = root.data("px_per_mm");
    dx /= px_per_mm;
    dy /= px_per_mm;

    var tx0 = root.data("tx"),
        ty0 = root.data("ty");

    var dx0 = root.data("dx"),
        dy0 = root.data("dy");

    root.data("dx", dx);
    root.data("dy", dy);

    dx = dx - dx0;
    dy = dy - dy0;

    var tx = tx0 + dx,
        ty = ty0 + dy;

    set_plot_pan_zoom(root, tx, ty, root.data("scale"));
};


Gadfly.guide_background_drag_onstart = function(x, y, event) {
    var root = this.plotroot();
    root.data("dx", 0);
    root.data("dy", 0);
    init_pan_zoom(root);
};


Gadfly.guide_background_drag_onend = function(event) {
    var root = this.plotroot();
};


Gadfly.guide_background_scroll = function(event) {
    if (event.shiftKey) {
        var root = this.plotroot();
        init_pan_zoom(root);
        var new_scale = root.data("scale") * Math.pow(2, 0.002 * event.wheelDelta);
        new_scale = Math.max(
            root.data("min_scale"),
            Math.min(root.data("max_scale"), new_scale))
        update_plot_scale(root, new_scale);
        event.stopPropagation();
    }
};


Gadfly.zoomslider_button_mouseover = function(event) {
    this.select(".button_logo")
         .animate({fill: this.data("mouseover_color")}, 100);
};


Gadfly.zoomslider_button_mouseout = function(event) {
     this.select(".button_logo")
         .animate({fill: this.data("mouseout_color")}, 100);
};


Gadfly.zoomslider_zoomout_click = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);
    var min_scale = root.data("min_scale"),
        scale = root.data("scale");
    Snap.animate(
        scale,
        Math.max(min_scale, scale / 1.5),
        function (new_scale) {
            update_plot_scale(root, new_scale);
        },
        200);
};


Gadfly.zoomslider_zoomin_click = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);
    var max_scale = root.data("max_scale"),
        scale = root.data("scale");

    Snap.animate(
        scale,
        Math.min(max_scale, scale * 1.5),
        function (new_scale) {
            update_plot_scale(root, new_scale);
        },
        200);
};


Gadfly.zoomslider_track_click = function(event) {
    // TODO
};


Gadfly.zoomslider_thumb_mousedown = function(event) {
    this.animate({fill: this.data("mouseover_color")}, 100);
};


Gadfly.zoomslider_thumb_mouseup = function(event) {
    this.animate({fill: this.data("mouseout_color")}, 100);
};


// compute the position in [0, 1] of the zoom slider thumb from the current scale
var slider_position_from_scale = function(scale, min_scale, max_scale) {
    if (scale >= 1.0) {
        return 0.5 + 0.5 * (Math.log(scale) / Math.log(max_scale));
    }
    else {
        return 0.5 * (Math.log(scale) - Math.log(min_scale)) / (0 - Math.log(min_scale));
    }
}


var update_plot_scale = function(root, new_scale) {
    var trans = scale_centered_translation(root, new_scale);
    set_plot_pan_zoom(root, trans.x, trans.y, new_scale);

    root.selectAll(".zoomslider_thumb")
        .forEach(function (element, i) {
            var min_pos = element.data("min_pos"),
                max_pos = element.data("max_pos"),
                min_scale = root.data("min_scale"),
                max_scale = root.data("max_scale");
            var xmid = (min_pos + max_pos) / 2;
            var xpos = slider_position_from_scale(new_scale, min_scale, max_scale);
            element.transform(new Snap.Matrix().translate(
                Math.max(min_pos, Math.min(
                         max_pos, min_pos + (max_pos - min_pos) * xpos)) - xmid, 0));
    });
};


Gadfly.zoomslider_thumb_dragmove = function(dx, dy, x, y) {
    var root = this.plotroot();
    var min_pos = this.data("min_pos"),
        max_pos = this.data("max_pos"),
        min_scale = root.data("min_scale"),
        max_scale = root.data("max_scale"),
        old_scale = root.data("old_scale");

    var px_per_mm = root.data("px_per_mm");
    dx /= px_per_mm;
    dy /= px_per_mm;

    var xmid = (min_pos + max_pos) / 2;
    var xpos = slider_position_from_scale(old_scale, min_scale, max_scale) +
                   dx / (max_pos - min_pos);

    // compute the new scale
    var new_scale;
    if (xpos >= 0.5) {
        new_scale = Math.exp(2.0 * (xpos - 0.5) * Math.log(max_scale));
    }
    else {
        new_scale = Math.exp(2.0 * xpos * (0 - Math.log(min_scale)) +
                        Math.log(min_scale));
    }
    new_scale = Math.min(max_scale, Math.max(min_scale, new_scale));

    update_plot_scale(root, new_scale);
};


Gadfly.zoomslider_thumb_dragstart = function(event) {
    var root = this.plotroot();
    init_pan_zoom(root);

    // keep track of what the scale was when we started dragging
    root.data("old_scale", root.data("scale"));
};


Gadfly.zoomslider_thumb_dragend = function(event) {
};


var toggle_color_class = function(root, color_class, ison) {
    var guides = root.selectAll(".guide." + color_class + ",.guide ." + color_class);
    var geoms = root.selectAll(".geometry." + color_class + ",.geometry ." + color_class);
    if (ison) {
        guides.animate({opacity: 0.5}, 250);
        geoms.animate({opacity: 0.0}, 250);
    } else {
        guides.animate({opacity: 1.0}, 250);
        geoms.animate({opacity: 1.0}, 250);
    }
};


Gadfly.colorkey_swatch_click = function(event) {
    var root = this.plotroot();
    var color_class = this.data("color_class");

    if (event.shiftKey) {
        root.selectAll(".colorkey text")
            .forEach(function (element) {
                var other_color_class = element.data("color_class");
                if (other_color_class != color_class) {
                    toggle_color_class(root, other_color_class,
                                       element.attr("opacity") == 1.0);
                }
            });
    } else {
        toggle_color_class(root, color_class, this.attr("opacity") == 1.0);
    }
};


return Gadfly;

}));


//@ sourceURL=gadfly.js

(function (glob, factory) {
    // AMD support
      if (typeof require === "function" && typeof define === "function" && define.amd) {
        require(["Snap.svg", "Gadfly"], function (Snap, Gadfly) {
            factory(Snap, Gadfly);
        });
      } else {
          factory(glob.Snap, glob.Gadfly);
      }
})(window, function (Snap, Gadfly) {
    var fig = Snap("#fig-80d265d38d6f4718b346a3ae93949088");
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-4")
   .mouseenter(Gadfly.plot_mouseover)
.mouseleave(Gadfly.plot_mouseout)
.mousewheel(Gadfly.guide_background_scroll)
.drag(Gadfly.guide_background_drag_onmove,
      Gadfly.guide_background_drag_onstart,
      Gadfly.guide_background_drag_onend)
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-7")
   .plotroot().data("unfocused_ygrid_color", "#D0D0E0")
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-7")
   .plotroot().data("focused_ygrid_color", "#A0A0A0")
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-8")
   .plotroot().data("unfocused_xgrid_color", "#D0D0E0")
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-8")
   .plotroot().data("focused_xgrid_color", "#A0A0A0")
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-14")
   .data("mouseover_color", "#cd5c5c")
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-14")
   .data("mouseout_color", "#6a6a6a")
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-14")
   .click(Gadfly.zoomslider_zoomin_click)
.mouseenter(Gadfly.zoomslider_button_mouseover)
.mouseleave(Gadfly.zoomslider_button_mouseout)
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-16")
   .data("max_pos", 120.42)
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-16")
   .data("min_pos", 103.42)
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-16")
   .click(Gadfly.zoomslider_track_click);
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-17")
   .data("max_pos", 120.42)
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-17")
   .data("min_pos", 103.42)
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-17")
   .data("mouseover_color", "#cd5c5c")
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-17")
   .data("mouseout_color", "#6a6a6a")
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-17")
   .drag(Gadfly.zoomslider_thumb_dragmove,
     Gadfly.zoomslider_thumb_dragstart,
     Gadfly.zoomslider_thumb_dragend)
.mousedown(Gadfly.zoomslider_thumb_mousedown)
.mouseup(Gadfly.zoomslider_thumb_mouseup)
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-18")
   .data("mouseover_color", "#cd5c5c")
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-18")
   .data("mouseout_color", "#6a6a6a")
;
fig.select("#fig-80d265d38d6f4718b346a3ae93949088-element-18")
   .click(Gadfly.zoomslider_zoomout_click)
.mouseenter(Gadfly.zoomslider_button_mouseover)
.mouseleave(Gadfly.zoomslider_button_mouseout)
;
    });
]]> </script>
</svg>



