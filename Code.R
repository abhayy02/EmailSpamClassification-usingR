library(dplyr) 
library(ggplot2)
library(corrplot) 
library(caret)
library(corrr)
library(factoextra)
library(randomForest)
library(e1071)
library(ROCR)
library(pROC)
library(DMwR2)
library(smotefamily)
library(ROSE)
library(xgboost)

#IMPORTING THE DATA
getwd()
setwd("C:\\Users\\kevin\\OneDrive - aus.edu\\Desktop\\intro to data mining")
email=read.csv("emails.csv")

#DATA CLEANING
Prediction = as.factor(email$Prediction)
sum(duplicated(email))
email <- unique(email)
sum(duplicated(email))

sum(is.na(email))
dim(email)

#######################################

###########################
###########################
# Load required libraries
library(e1071)
library(caret)
library(readr)
library(dplyr)
library(randomForest)
# Set a seed for reproducibility
set.seed(125)

# Step 2: Preprocess the data
email$Prediction = as.factor(email$Prediction)
sum(duplicated(email))
email <- unique(email)
sum(duplicated(email))
sum(is.na(email))
dim(email)

email.new = email[,-1] 

# Convert non-numeric columns to numeric
email.new <- email.new %>%
  mutate(across(everything(), as.numeric))

# Remove columns with zero variance
email.new <- email.new[, sapply(email.new, function(x) var(x, na.rm = TRUE) != 0)]

# Step 3: Dimensionality Reduction using PCA
# Normalize the data
email.new_normalized <- scale(email.new)

# Apply PCA
pca <- prcomp(email.new_normalized, center = TRUE, scale. = TRUE)
# Select the number of principal components that explain 95% of the variance
explained_variance <- cumsum(pca$sdev^2 / sum(pca$sdev^2))
num_components <- which(explained_variance >= 0.95)[1]

# Transform the data
email.new_pca <- data.frame(pca$x[, 1:num_components])

#Scree plot
scree_plot <- plot(pca, type = "l")

#Summary of Principal Components
summary(pca)

# Calculate the variance explained by each principal component
variance_explained <- pca$sdev^2 / sum(pca$sdev^2)

# Get the number of principal components to retain
num_components <- which(cumsum(variance_explained) >= 0.95)[1]

# Get the variable loadings on the retained principal components
loadings <- pca$rotation[, 1:num_components]

# Plot the variable loadings
biplot(pca, choices = c(1, 2), scale = 0, cex = 0.7)

# Add labels and title
title("Variables - PCA")


# Step 4: Model Training
# Split the data into training and testing sets

trainIndex <- createDataPartition(email.new$Prediction, p = .7, list = FALSE, times = 1)
train_email.new <- email.new_pca[trainIndex, ]
test_email.new <- email.new_pca[-trainIndex, ]
train_Prediction <- email.new$Prediction[trainIndex]
test_Prediction <- email.new$Prediction[-trainIndex]

########################################################
# Train the SVM model
svm_model.linear <- svm(train_email.new, train_Prediction, kernel = "linear")

# Step 5: Model Evaluation
# Make predictions
predictions <- predict(svm_model.linear, test_email.new)

# Calculate accuracy
accuracy <- mean(predictions == test_Prediction)
print(paste("Accuracy: ", accuracy))

# Display confusion matrix
conf_matrix <- confusionMatrix(predictions, test_Prediction)
print(conf_matrix)



############################################################Rand

# Step 4: Model Training
# Train the random forest model
rf_model <- randomForest(email.new$Prediction ~ ., data = cbind(train_email.new, Prediction = train_Prediction),
                         mtry = sqrt(ncol(train_email.new)), importance = TRUE)

# Step 5: Model Evaluation

predictions <- predict(rf_model, test_email.new)

# Calculate accuracy
accuracy <- mean(predictions == test_Prediction)
print(paste("Accuracy:", accuracy))

# Display confusion matrix
conf_matrix <- confusionMatrix(predictions, test_Prediction)
print(conf_matrix)

# Calculate precision, recall, and F1-score
true_positives <- conf_matrix$table[2, 2]
true_negatives <- conf_matrix$table[1, 1]
false_positives <- conf_matrix$table[1, 2]
false_negatives <- conf_matrix$table[2, 1]

precision <- true_positives / (true_positives + false_positives)
recall <- true_positives / (true_positives + false_negatives)
f1_score <- 2 * precision * recall / (precision + recall)


cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-score:", f1_score, "\n")


# Generate probability predictions for the positive class
prob_predictions <- predict(rf_model, test_email.new, type = "prob")[,2]

# Plot ROC curve and calculate AUC
roc_obj <- roc(test_Prediction, prob_predictions)
plot(roc_obj, main = "ROC Curve")
auc <- auc(roc_obj)
cat("AUC:", auc, "\n")

##########################################
##########################################
# xgboost model

  library(xgboost)

# Split the data into training and testing sets
trainIndex <- createDataPartition(email.new$Prediction, p = .7, list = FALSE, times = 1)
train_email.new <- email.new_pca[trainIndex, ]
test_email.new <- email.new_pca[-trainIndex, ]
train_Prediction_0 <- email.new$Prediction[trainIndex]
test_Prediction_0 <- email.new$Prediction[-trainIndex]
########################################################
# Install XGBoost package if not installed
train_Prediction = train_Prediction_0-1
test_Prediction = test_Prediction_0 -1
test_Prediction
train_Prediction
# Convert data to DMatrix format required by XGBoost
train_data <- xgb.DMatrix(data = as.matrix(train_email.new), label = train_Prediction)
test_data <- xgb.DMatrix(data = as.matrix(test_email.new), label = test_Prediction)


# Set XGBoost parameters
xgb_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "error",
  eta = 0.3,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 1,
  lambda = 1,
  alpha = 0.5
)


# Train the XGBoost model
xgb_model <- xgb.train(
  params = xgb_params,
  data = train_data,
  nrounds = 100,
  watchlist = list(train = train_data),
  early_stopping_rounds = 10,
  verbose = 1
)

# Make predictions on the test set
test_predictions <- predict(xgb_model, test_data)

# Convert predictions to binary labels
test_predictions_labels <- ifelse(test_predictions > 0.5, 1, 0)

# Evaluate model performance
conf_matrix = confusionMatrix(data = factor(test_predictions_labels), reference = factor(test_Prediction))
# Calculate precision, recall, and F1-score
true_positives <- conf_matrix$table[2, 2]
true_negatives <- conf_matrix$table[1, 1]
false_positives <- conf_matrix$table[1, 2]
false_negatives <- conf_matrix$table[2, 1]

precision <- true_positives / (true_positives + false_positives)
recall <- true_positives / (true_positives + false_negatives)
f1_score <- 2 * precision * recall / (precision + recall)


cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-score:", f1_score, "\n")
conf_matrix

