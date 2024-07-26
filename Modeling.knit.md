---
title: "ST558 Final Project Model Fitting - Lee Worthington"
format: html
editor: visual
---


# Intro

> ### Data Overview
> The dataset used for this analysis is derived from the Behavioral Risk Factor Surveillance System (BRFSS) 2015, a health-related telephone survey conducted annually by the Centers for Disease Control and Prevention (CDC). The BRFSS collects data from over 400,000 Americans on health-related risk behaviors, chronic health conditions, and the use of preventative services.
>
> For this project Ill focus on the diabetes_binary_health_indicators_BRFSS2015.csv file that contains 253680 survey responses, the main goal is to predict Diabetes_binary which has the below levels.
> - 0: No diabetes
> - 1: Prediabetes or diabetes
>
> There are 21 potential predictors in the data, since I dont know anything about diabetes I will be using all of the predictors when modelling rather than making assumptions about what may be relevant
>
> - **HighBP**: High blood pressure (N/Y)
> - **HighChol**: High cholesterol (N/Y)
> - **CholCheck**: Cholesterol check within the past five years (N/Y)
> - **BMI**: Body mass index (numeric)
> - **Smoker**: Smoker status (N/Y)
> - **Stroke**: History of stroke (N/Y)
> - **HeartDiseaseorAttack**: Coronary heart disease or myocardial infarction (N/Y)
> - **PhysActivity**: Physical activity in the past 30 days (N/Y)
> - **Fruits**: Consumption of fruits at least once per day (N/Y)
> - **Veggies**: Consumption of vegetables at least once per day (N/Y)
> - **HvyAlcoholConsump**: Heavy alcohol consumption (N/Y)
> - **AnyHealthcare**: Access to healthcare coverage (N/Y)
> - **NoDocbcCost**: Inability to see a doctor due to cost (N/Y)
> - **GenHlth**: General health status (Excellent/VGood/Good/Fair/Poor)
> - **MentHlth**: Days in the past 30 days when mental health was not good (numeric)
> - **PhysHlth**: Days in the past 30 days when physical health was not good (numeric)
> - **DiffWalk**: Difficulty walking or climbing stairs (N/Y)
> - **Sex**: Gender (F/M)
> - **Age**: Age categories (18-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70-74, 75-79, 80+)
> - **Education**: Education level (No School, Elem, Some HS, HS Grad, Some College, College Grad)
> - **Income**: Income categories (<$10k, $10-15k, $15-20k, $20-25k, $25-35k, $35-50k, $50-75k, >$75k)

> ### Purpose of EDA and Modeling Goal
> The main goal of the EDA here is to get a better idea of the data and to spot potential relationships in the data, to hopefully build a model that can accurately predict diabetes based on the available predictors.
> Since most of the predictors are categorical and there are a large number of them, I will mainly focus on understanding the following in the EDA:
>
> 1. **Understanding the Distribution**: Primarily how the distributions for each predictor vary based on having or not having diabetes, in order to identify potentially relevant predictors
> 2. **Checking Data Quality**: ID any potential issues with the data, that may require adjustment. That being said my general approach is to not clean or remove data unless im certain its an error, which I cannot be with this data

# Setup environment and read data

::: {.cell}

```{.r .cell-code}
# load libraries
library(tidyverse)
library(GGally)
library(caret)
library(corrplot)
library(reshape2)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(scales)
library(knitr)  # For rendering tables in Quarto

# set seed
set.seed(1)  
```
:::

::: {.cell}

```{.r .cell-code}
# Read in the data https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/
input_data <- read_csv(
  'diabetes_binary_health_indicators_BRFSS2015.csv',
  show_col_types = FALSE
)
```
:::


# Data prep

::: {.cell}

```{.r .cell-code}
# Data cleanup
df <- input_data |>
  mutate(
    Diabetes_binary = factor(Diabetes_binary, levels = c(0, 1), labels = c("No", "Yes")), # slightly inconsistent here, but caret seems to expect yes = 1
    
    HighBP = factor(HighBP, levels = c(0, 1), labels = c("N", "Y")),
    
    HighChol = factor(HighChol, levels = c(0, 1), labels = c("N", "Y")),
    
    CholCheck = factor(CholCheck, levels = c(0, 1), labels = c("N", "Y")),
    
    Smoker = factor(Smoker, levels = c(0, 1), labels = c("N", "Y")),
    
    Stroke = factor(Stroke, levels = c(0, 1), labels = c("N", "Y")),
    
    HeartDiseaseorAttack = factor(HeartDiseaseorAttack, levels = c(0, 1), labels = c("N", "Y")),
    
    PhysActivity = factor(PhysActivity, levels = c(0, 1), labels = c("N", "Y")),
    
    Fruits = factor(Fruits, levels = c(0, 1), labels = c("N", "Y")),
    
    Veggies = factor(Veggies, levels = c(0, 1), labels = c("N", "Y")),
    
    HvyAlcoholConsump = factor(HvyAlcoholConsump, levels = c(0, 1), labels = c("N", "Y")),
    
    AnyHealthcare = factor(AnyHealthcare, levels = c(0, 1), labels = c("N", "Y")),
    
    NoDocbcCost = factor(NoDocbcCost, levels = c(0, 1), labels = c("N", "Y")),
    
    GenHlth = factor(GenHlth, levels = c(1, 2, 3, 4, 5), labels = c("Exc", "VGood", "Good", "Fair", "Poor")),
    
    DiffWalk = factor(DiffWalk, levels = c(0, 1), labels = c("N", "Y")),
    
    Sex = factor(Sex, levels = c(0, 1), labels = c("F", "M")),
    
    Age = factor(
      Age, 
      levels = 1:13, 
      labels = c("18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+")
    ),
    
    Education = factor(
      Education, 
      levels = 1:6, 
      labels = c(
        "None/Kinder", "Elem", "Some HS", 
        "HS Grad", "Some College", "College Grad"
      )
    ),
    
    Income = factor(
      Income, 
      levels = 1:8, 
      labels = c(
        "<$10k", "$10-15k", "$15-20k", "$20-25k", 
        "$25-35k", "$35-50k", "$50-75k", ">$75k"
      )
    )
  )

# Split dummy data and drop original fields
train_index <- createDataPartition(df$Diabetes_binary, p = 0.7, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# check results
dim(df)
```

::: {.cell-output .cell-output-stdout}
```
[1] 253680     22
```
:::

```{.r .cell-code}
dim(train_data)
```

::: {.cell-output .cell-output-stdout}
```
[1] 177577     22
```
:::

```{.r .cell-code}
dim(test_data)
```

::: {.cell-output .cell-output-stdout}
```
[1] 76103    22
```
:::
:::

::: {.cell}

```{.r .cell-code}
# Fit logistic model with every predictor
logistic_model <- train(
  Diabetes_binary ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  metric = "logLoss",
  preProcess = c("center", "scale"),
  trControl = trainControl(method = "cv", number = 5)
)

# Print training model fit
logistic_model
```

::: {.cell-output .cell-output-stdout}
```
Generalized Linear Model 

177577 samples
    21 predictor
     2 classes: 'No', 'Yes' 

Pre-processing: centered (45), scaled (45) 
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 142062, 142063, 142061, 142061, 142061 
Resampling results:

  Accuracy   Kappa    
  0.8651571  0.1912069
```
:::

```{.r .cell-code}
summary(logistic_model)
```

::: {.cell-output .cell-output-stdout}
```

Call:
NULL

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.5563  -0.5382  -0.3068  -0.1536   3.6301  

Coefficients:
                         Estimate Std. Error  z value Pr(>|z|)    
(Intercept)             -2.499792   0.011959 -209.028  < 2e-16 ***
HighBPY                  0.352708   0.008722   40.441  < 2e-16 ***
HighCholY                0.261730   0.008047   32.527  < 2e-16 ***
CholCheckY               0.226773   0.015440   14.687  < 2e-16 ***
BMI                      0.381605   0.007225   52.819  < 2e-16 ***
SmokerY                 -0.023837   0.007906   -3.015 0.002569 ** 
StrokeY                  0.027816   0.005918    4.700 2.60e-06 ***
HeartDiseaseorAttackY    0.075230   0.006206   12.121  < 2e-16 ***
PhysActivityY           -0.022712   0.007393   -3.072 0.002126 ** 
FruitsY                 -0.008297   0.007901   -1.050 0.293643    
VeggiesY                -0.013835   0.007437   -1.860 0.062829 .  
HvyAlcoholConsumpY      -0.181739   0.010726  -16.944  < 2e-16 ***
AnyHealthcareY           0.019350   0.008666    2.233 0.025551 *  
NoDocbcCostY             0.001925   0.007656    0.252 0.801425    
GenHlthVGood             0.343304   0.019241   17.842  < 2e-16 ***
GenHlthGood              0.659610   0.018022   36.601  < 2e-16 ***
GenHlthFair              0.617502   0.014058   43.926  < 2e-16 ***
GenHlthPoor              0.440976   0.010831   40.715  < 2e-16 ***
MentHlth                -0.015783   0.007527   -2.097 0.036000 *  
PhysHlth                -0.032258   0.008398   -3.841 0.000122 ***
DiffWalkY                0.048412   0.007580    6.387 1.69e-10 ***
SexM                     0.128121   0.008017   15.981  < 2e-16 ***
`Age25-29`               0.041121   0.031215    1.317 0.187725    
`Age30-34`               0.126094   0.034092    3.699 0.000217 ***
`Age35-39`               0.237921   0.035859    6.635 3.25e-11 ***
`Age40-44`               0.323149   0.037948    8.516  < 2e-16 ***
`Age45-49`               0.393194   0.041143    9.557  < 2e-16 ***
`Age50-54`               0.514946   0.046343   11.112  < 2e-16 ***
`Age55-59`               0.584471   0.049482   11.812  < 2e-16 ***
`Age60-64`               0.676783   0.050993   13.272  < 2e-16 ***
`Age65-69`               0.720518   0.050305   14.323  < 2e-16 ***
`Age70-74`               0.641152   0.043909   14.602  < 2e-16 ***
`Age75-79`               0.522055   0.037130   14.060  < 2e-16 ***
`Age80+`                 0.489432   0.038448   12.730  < 2e-16 ***
EducationElem            0.010902   0.030599    0.356 0.721634    
`EducationSome HS`      -0.002309   0.046285   -0.050 0.960208    
`EducationHS Grad`      -0.037840   0.104375   -0.363 0.716947    
`EducationSome College` -0.024833   0.108208   -0.229 0.818483    
`EducationCollege Grad` -0.069766   0.119647   -0.583 0.559826    
`Income$10-15k`         -0.009085   0.008913   -1.019 0.308072    
`Income$15-20k`         -0.017543   0.009941   -1.765 0.077616 .  
`Income$20-25k`         -0.028525   0.010809   -2.639 0.008318 ** 
`Income$25-35k`         -0.052392   0.011979   -4.374 1.22e-05 ***
`Income$35-50k`         -0.084993   0.013656   -6.224 4.85e-10 ***
`Income$50-75k`         -0.100975   0.014805   -6.820 9.08e-12 ***
`Income>$75k`           -0.200728   0.018742  -10.710  < 2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 143396  on 177576  degrees of freedom
Residual deviance: 112487  on 177531  degrees of freedom
AIC: 112579

Number of Fisher Scoring iterations: 7
```
:::
:::

