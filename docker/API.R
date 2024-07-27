# --------------------------------------------------------------------
# Load data and fit model
# --------------------------------------------------------------------

# Setup environment
library(plumber)
library(tidyverse)
library(caret)
library(ranger)
library(Metrics) # for logloss
set.seed(1)

# Load data
input_data <- read_csv(
  'diabetes_binary_health_indicators_BRFSS2015.csv',
  show_col_types = FALSE
)

# Data cleanup
df <- input_data |>
  mutate(
    Diabetes_binary = factor(Diabetes_binary, levels = c(0, 1), labels = c("N", "Y")),
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

# Split the data into training and testing sets
train_index <- createDataPartition(df$Diabetes_binary, p = 0.7, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Fit the logistic regression model
logistic_model <- train(
  Diabetes_binary ~ .,
  data = df, # 7/27/2024 doc mentions to train your final model on the full data
  method = "glm",
  family = "binomial",
  metric = "logLoss",
  preProcess = c("center", "scale"),
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    summaryFunction = mnLogLoss, 
    classProbs = TRUE,
    verboseIter = TRUE
  )
)

# --------------------------------------------------------------------
# Define the /pred endpoint
# --------------------------------------------------------------------

#* @param HighBP High blood pressure (Y/N)
#* @param HighChol High cholesterol (Y/N)
#* @param CholCheck Cholesterol check (Y/N)
#* @param BMI Body Mass Index (numeric)
#* @param Smoker Smoking status (Y/N)
#* @param Stroke Stroke history (Y/N)
#* @param HeartDiseaseorAttack History of heart disease or attack (Y/N)
#* @param PhysActivity Physical activity (Y/N)
#* @param Fruits Fruit consumption (Y/N)
#* @param Veggies Vegetable consumption (Y/N)
#* @param HvyAlcoholConsump Heavy alcohol consumption (Y/N)
#* @param AnyHealthcare Access to any healthcare (Y/N)
#* @param NoDocbcCost No doctor visit due to cost (Y/N)
#* @param GenHlth General health status (Exc/VGood/Good/Fair/Poor)
#* @param MentHlth Mental health (numeric, number of days)
#* @param PhysHlth Physical health (numeric, number of days)
#* @param DiffWalk Difficulty walking (Y/N)
#* @param Sex Sex (F/M)
#* @param Age Age group (18-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70-74, 75-79, 80+)
#* @param Education Education level (None/Kinder, Elem, Some HS, HS Grad, Some College, College Grad)
#* @param Income Income level (<$10k, $10-15k, $15-20k, $20-25k, $25-35k, $35-50k, $50-75k, >$75k)
#* @get /pred
function(
    HighBP = 'N', HighChol = 'N', CholCheck = 'Y', BMI = 28.38, Smoker = 'N', Stroke = 'N', 
    HeartDiseaseorAttack = 'N', PhysActivity = 'Y', Fruits = 'Y', Veggies = 'Y', 
    HvyAlcoholConsump = 'N', AnyHealthcare = 'Y', NoDocbcCost = 'N', GenHlth = 'VGood', 
    MentHlth = 3.185, PhysHlth = 4.242, DiffWalk = 'N', Sex = 'F', Age = '60-64', Education = 'College Grad', Income = '>$75k'
) {
  new_data <- data.frame(
    HighBP = factor(HighBP, levels = c("N", "Y")),
    HighChol = factor(HighChol, levels = c("N", "Y")),
    CholCheck = factor(CholCheck, levels = c("N", "Y")),
    BMI = as.numeric(BMI),
    Smoker = factor(Smoker, levels = c("N", "Y")),
    Stroke = factor(Stroke, levels = c("N", "Y")),
    HeartDiseaseorAttack = factor(HeartDiseaseorAttack, levels = c("N", "Y")),
    PhysActivity = factor(PhysActivity, levels = c("N", "Y")),
    Fruits = factor(Fruits, levels = c("N", "Y")),
    Veggies = factor(Veggies, levels = c("N", "Y")),
    HvyAlcoholConsump = factor(HvyAlcoholConsump, levels = c("N", "Y")),
    AnyHealthcare = factor(AnyHealthcare, levels = c("N", "Y")),
    NoDocbcCost = factor(NoDocbcCost, levels = c("N", "Y")),
    GenHlth = factor(GenHlth, levels = c("Exc", "VGood", "Good", "Fair", "Poor")),
    MentHlth = as.numeric(MentHlth),
    PhysHlth = as.numeric(PhysHlth),
    DiffWalk = factor(DiffWalk, levels = c("N", "Y")),
    Sex = factor(Sex, levels = c("F", "M")),
    Age = factor(Age, levels = c("18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+")),
    Education = factor(Education, levels = c("None/Kinder", "Elem", "Some HS", "HS Grad", "Some College", "College Grad")),
    Income = factor(Income, levels = c("<$10k", "$10-15k", "$15-20k", "$20-25k", "$25-35k", "$35-50k", "$50-75k", ">$75k"))
  )
  predict(logistic_model, new_data, type = "prob")
}

# --------------------------------------------------------------------
# Define the /info endpoint
# --------------------------------------------------------------------

#* @get /info
function() {
  list(
    name = "Lee Worthington",
    github_page = "https://github.com/law5097/ST558_FinalProject"
  )
}

# http://localhost:8000/pred?HighBP=Y&HighChol=N&CholCheck=Y&BMI=30&Smoker=Y&Stroke=N&HeartDiseaseorAttack=N&PhysActivity=Y&Fruits=Y&Veggies=Y&HvyAlcoholConsump=N&AnyHealthcare=Y&NoDocbcCost=N&GenHlth=Good&MentHlth=0&PhysHlth=0&DiffWalk=N&Sex=M&Age=40-44&Education=College%20Grad&Income=%3E$75k
# http://localhost:8000/pred?HighBP=N&HighChol=N&CholCheck=Y&BMI=25&Smoker=N&Stroke=N&HeartDiseaseorAttack=N&PhysActivity=Y&Fruits=Y&Veggies=Y&HvyAlcoholConsump=N&AnyHealthcare=Y&NoDocbcCost=N&GenHlth=VGood&MentHlth=0&PhysHlth=0&DiffWalk=N&Sex=F&Age=30-34&Education=Some%20College&Income=$35-50k
# http://localhost:8000/pred?HighBP=Y&HighChol=Y&CholCheck=Y&BMI=35&Smoker=Y&Stroke=Y&HeartDiseaseorAttack=Y&PhysActivity=N&Fruits=N&Veggies=N&HvyAlcoholConsump=Y&AnyHealthcare=Y&NoDocbcCost=Y&GenHlth=Poor&MentHlth=30&PhysHlth=30&DiffWalk=Y&Sex=M&Age=60-64&Education=HS%20Grad&Income=$10-15k


