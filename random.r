---
title: "ST558 Final Project EDA - Lee Worthington"
format: html
editor: visual
---

# Setup environment
```{r}
#| eval: true
#| warning: false

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

# Load df
```{r}
#| eval: true
#| warning: false

# Read in the data https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/
df <- read_csv(
  'C:\\Users\\lawor\\OneDrive\\Desktop\\School\\ST 558\\Projects\\ST558_FinalProject\\diabetes_binary_health_indicators_BRFSS2015.csv',
  show_col_types = FALSE
)

summary(df)
```

> Based on the dataset descriptions and their data types, these are the variables:
>
> -   **Diabetes_binary**: Indicator of diabetes status (0: No diabetes, 1: Prediabetes, 2: Diabetes) - Factor
> -   **HighBP**: High blood pressure (0: No, 1: Yes) - Factor
> -   **HighChol**: High cholesterol (0: No, 1: Yes) - Factor
> -   **CholCheck**: Cholesterol check within the past five years (0: No, 1: Yes) - Factor
> -   **BMI**: Body mass index - Numeric
> -   **Smoker**: Smoker status (0: No, 1: Yes) - Factor
> -   **Stroke**: History of stroke (0: No, 1: Yes) - Factor
> -   **HeartDiseaseorAttack**: Coronary heart disease or myocardial infarction (0: No, 1: Yes) - Factor
> -   **PhysActivity**: Physical activity in past 30 days (0: No, 1: Yes) - Factor
> -   **Fruits**: Consumption of fruits at least once per day (0: No, 1: Yes) - Factor
> -   **Veggies**: Consumption of vegetables at least once per day (0: No, 1: Yes) - Factor
> -   **HvyAlcoholConsump**: Heavy alcohol consumption (0: No, 1: Yes) - Factor
> -   **AnyHealthcare**: Access to healthcare coverage (0: No, 1: Yes) - Factor
> -   **NoDocbcCost**: Inability to see a doctor due to cost (0: No, 1: Yes) - Factor
> -   **GenHlth**: General health status (1: Excellent, 2: Very good, 3: Good, 4: Fair, 5: Poor) - Factor with 5 levels
> -   **MentHlth**: Days in the past 30 days when mental health was not good - Numeric
> -   **PhysHlth**: Days in the past 30 days when physical health was not good - Numeric
> -   **DiffWalk**: Difficulty walking or climbing stairs (0: No, 1: Yes) - Factor
> -   **Sex**: Gender (0: Female, 1: Male) - Factor with 2 levels
> -   **Age**: Age categories (1: 18-24, 2: 25-29, 3: 30-34, 4: 35-39, 5: 40-44, 6: 45-49, 7: 50-54, 8: 55-59, 9: 60-64, 10: 65-69, 11: 70-74, 12: 75-79, 13: 80 or older) - Factor with 13 levels
> -   **Education**: Education level (1: Never attended school or only kindergarten, 2: Grades 1 through 8 (Elementary), 3: Grades 9 through 11 (Some high school), 4: Grade 12 or GED (High school graduate), 5: College 1 year to 3 years (Some college or technical school), 6: College 4 years or more (College graduate)) - Factor with 6 levels
> -   **Income**: Income categories (1: Less than $10,000, 2: $10,000 to less than $15,000, 3: $15,000 to less than $20,000, 4: $20,000 to less than $25,000, 5: $25,000 to less than $35,000, 6: $35,000 to less than $50,000, 7: $50,000 to less than $75,000, 8: $75,000 or more) - Factor with 8 levels

# Data Prep
```{r}
#| eval: true
#| warning: false

# Set df types
df <- df |>
  mutate(
    Diabetes_binary = as.factor(Diabetes_binary),
    HighBP = as.factor(HighBP),
    HighChol = as.factor(HighChol),
    CholCheck = as.factor(CholCheck),
    Smoker = as.factor(Smoker),
    Stroke = as.factor(Stroke),
    HeartDiseaseorAttack = as.factor(HeartDiseaseorAttack),
    PhysActivity = as.factor(PhysActivity),
    Fruits = as.factor(Fruits),
    Veggies = as.factor(Veggies),
    HvyAlcoholConsump = as.factor(HvyAlcoholConsump),
    AnyHealthcare = as.factor(AnyHealthcare),
    NoDocbcCost = as.factor(NoDocbcCost),
    GenHlth = as.factor(GenHlth),
    DiffWalk = as.factor(DiffWalk),
    Sex = as.factor(Sex),
    Age = as.factor(Age),
    Education = as.factor(Education),
    Income = as.factor(Income)
  )

# plot pairs
# Generate pair plots in chunks so this is readable
#GGally::ggpairs(df, columns = c(1, 2, 3, 4))
#GGally::ggpairs(df, columns = c(1, 5, 6, 7))
#GGally::ggpairs(df, columns = c(1, 8, 9, 10))
#GGally::ggpairs(df, columns = c(1, 11, 12, 13))
#GGally::ggpairs(df, columns = c(1, 14, 15, 16))
#GGally::ggpairs(df, columns = c(1, 17, 18, 19))

```

# EDA
```{r}
#| eval: true
#| warning: false

# Function to generate heatmap for confusion matrix of factor variables
generate_heatmap <- function(var) {
  confusion <- table(df$Diabetes_binary, df[[var]])
  confusion_melted <- melt(confusion)
  confusion_melted$percentage <- confusion_melted$value / sum(confusion_melted$value) * 100
  confusion_melted$label <- paste(comma(confusion_melted$value), sprintf("(%.1f%%)", confusion_melted$percentage))
  
  ggplot(confusion_melted, aes(Var1, Var2, fill = value)) +
    geom_tile(alpha = 0.5) +
    geom_text(aes(label = label), color = "black") +
    scale_fill_gradient(low = "lightgreen", high = "darkgreen") +
    labs(title = paste("Confusion Matrix for", var),
         x = "Diabetes_binary",
         y = var,
         fill = "Count") +
    theme_minimal() +
    scale_x_discrete(labels = c("0", "1")) +
    scale_y_discrete(labels = c("0", "1"))
}

# Function to generate histogram for numeric variables with log scale on x-axis
generate_histogram_numeric <- function(var) {
  ggplot(df, aes_string(x = var, fill = "Diabetes_binary")) +
    geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
    #scale_x_log10() +
    labs(title = paste("Histogram of", var, "partitioned by Diabetes_binary (Log Scale)"), x = var, y = "Count") +
    theme_minimal()
}

# Function to generate boxplot for numeric variables
generate_boxplot_numeric <- function(var) {
  ggplot(df, aes(x = Diabetes_binary, y = df[[var]], fill = Diabetes_binary)) +
    geom_boxplot(alpha = 0.5) +
    scale_y_log10() +
    labs(title = paste("Boxplot of", var, "partitioned by Diabetes_binary"), x = "Diabetes_binary", y = var) +
    theme_minimal()
}

# Function to generate bar plot for factor variables with more than 3 levels
generate_histogram_factor <- function(var) {
  df |>
    group_by(!!sym(var), Diabetes_binary) |>
    summarise(count = n(), .groups = "drop") |>
    group_by(!!sym(var)) |>
    mutate(percentage = count / sum(count) * 100,
           label = paste0(sprintf("%.1f", percentage), "%")) |>
    ggplot(aes_string(x = var, y = "count", fill = "Diabetes_binary")) +
    geom_bar(position = "stack", stat = "identity", alpha = 0.5) +
    geom_text(aes(label = label), position = position_stack(vjust = 0.5), color = "black") +
    labs(title = paste("Bar Plot of", var, "partitioned by Diabetes_binary"), x = var, y = "Count") +
    theme_minimal()
}

# List of factor variables with up to 3 levels
factor_vars_heatmap <- c(
  "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
  "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
  "NoDocbcCost", "DiffWalk", "Sex"
  )

# List of factor variables with more than 3 levels
factor_vars_histogram <- c("GenHlth", "Age", "Education", "Income")

# List of numeric variables
numeric_vars <- c("BMI", "MentHlth", "PhysHlth")

# Generate heatmaps for factor variables with up to 3 levels
for (var in factor_vars_heatmap) {
  print(generate_heatmap(var))
}

# Generate histograms for factor variables with more than 3 levels
for (var in factor_vars_histogram) {
  print(generate_histogram_factor(var))
}

# Generate histograms and boxplots for numeric variables with log scale on x-axis
for (var in numeric_vars) {
  print(generate_histogram_numeric(var))
  print(generate_boxplot_numeric(var))
}

```


```{r}
#| eval: true
#| warning: false

# Function to perform chi-square test for a categorical variable against Diabetes_binary
perform_chi_square_test <- function(var) {
  tbl <- table(df[[var]], df$Diabetes_binary)
  chi_square_result <- chisq.test(tbl)
  return(list(
    statistic = round(chi_square_result$statistic, 2), 
    p_value = formatC(chi_square_result$p.value, format = "e", digits = 10),
    df = chi_square_result$parameter,
    expected = round(chi_square_result$expected, 2)
  ))
}

# Initialize a data frame to store the results
results <- data.frame(
  Variable1 = character(),
  Variable2 = character(),
  Chi_Square_Statistic = numeric(),
  Degrees_of_Freedom = numeric(),
  P_Value = character(),
  stringsAsFactors = FALSE
)

# Perform chi-square test for all categorical variables
for (var in c(factor_vars_heatmap, factor_vars_histogram)) {
  chi_square_result <- perform_chi_square_test(var)
  
  results <- rbind(results, data.frame(
    Variable1 = var,
    Variable2 = "Diabetes_binary",
    Chi_Square_Statistic = chi_square_result$statistic,
    Degrees_of_Freedom = chi_square_result$df,
    P_Value = chi_square_result$p_value
  ))
}

# Sort the results by Chi-Square statistic in descending order
results <- results |> arrange(desc(Chi_Square_Statistic))

# Render the table using knitr::kable for Quarto
results


```