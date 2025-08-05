#Goal: Build Two Classifiers to identify "mdeSI" among adolescents
#Your Full Name: Jonathan Ma

#adding packages
install.packages(c("caret", "pROC", "randomForest"))
library(dplyr)
library(ggplot2)
library(caret) #Confusion Matrix
library(pROC) #ROC
library(randomForest) #random forest

#Importation
da = read.csv("/Users/jonathanma/Desktop/R/depression.csv", header = T) 
dim(da) 
str(da)

##### Setting Up #####
da$mdeSI = factor(da$mdeSI)
da$income = factor(da$income)
da$gender = factor(da$gender, levels = c("Male", "Female")) #Male is the reference group
da$age = factor(da$age)
da$race = factor(da$race, levels = c("White", "Hispanic", "Black", "Asian/NHPIs", "Other")) #white is the reference
da$insurance = factor(da$insurance, levels = c("Yes", "No")) #"Yes" is the reference group
da$siblingU18 = factor(da$siblingU18, levels = c("Yes", "No"))
da$fatherInHH = factor(da$fatherInHH)
da$motherInHH = factor(da$motherInHH)
da$parentInv = factor(da$parentInv)
da$schoolExp = factor(da$schoolExp, levels = c("good school experiences", "bad school experiences"))

#Splitting and dropping Year variable
(n = dim(da)[1])
set.seed(2024) 
index = sample(1:n, 4500) #75% of training and 25% of test data
train = da[index,] 
test = da[-index,]
dim(train)
dim(test)
da$year <- NULL
#####

##### Data Exploration and Features Selection #####
summary(da)

#Use of chi square to see impactful variables
chi_square_results <- lapply(da, function(x) {
  chisq.test(table(x, da$mdeSI))
})
chi_square_results

# Visualize the relationship between categorical variables and the target variable
for (col in names(da)) {
  print(ggplot(da, aes(x=col, fill='mdeSI')) +
          geom_bar(position="fill") +
          labs(title=paste("Distribution of", col, "by Depression Status"), x=col, y="Proportion") +
          scale_fill_discrete(name = "Depression Status", labels = c("No", "Yes")))
}
#####

##### Logistic Classifier #####
model <- glm(mdeSI ~ gender + age + race + income, data = train, family = "binomial")
test$predicted_prob <- predict(model, test, type = "response")

##### Random Forest #####
rf_model <- randomForest(mdeSI ~ gender + age + race + income, data = train, ntree = 500, importance = TRUE)
test$predicted_prob2 <- predict(rf_model, newdata = test, type = "prob")[,2]


##### Threshold Testing #####
# Possible thresholds from 0 to 1, at intervals, e.g., 0.01
thresholds <- seq(0, 1, by = 0.01)

# Initialize vectors to store metrics
accuracies <- numeric(length(thresholds))
recalls <- numeric(length(thresholds))

# Calculate metrics for each threshold
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  predicted_classes <- ifelse(test$predicted_prob > threshold, "Yes", "No")  #LC One
  #predicted_classes <- ifelse(test$predicted_prob2 > threshold, "Yes", "No") #RF One
  actual_classes <- test$mdeSI  
  accuracies[i] <- sum(predicted_classes == actual_classes) / length(actual_classes)
  tp <- sum(predicted_classes == "Yes" & actual_classes == "Yes")
  fn <- sum(predicted_classes == "No" & actual_classes == "Yes")
  recalls[i] <- tp / (tp + fn)
}

#Finding A/R Tradeoff
optimal_index <- which.max(sqrt(accuracies*recalls))
optimal_threshold <- thresholds[optimal_index]

# Print the optimal threshold
print(paste("Optimal Threshold:", optimal_threshold))
print(paste("Optimal Accuracy:", accuracies[optimal_index]))
print(paste("Optimal Recall:", recalls[optimal_index]))

#Plot of A/R at each threshold
library(ggplot2)
data_to_plot <- data.frame(Thresholds = thresholds, Accuracy = accuracies, Recall = recalls)
ggplot(data_to_plot, aes(x = Thresholds)) +
  geom_line(aes(y = Accuracy, colour = "Accuracy")) +
  geom_line(aes(y = Recall, colour = "Recall")) +
  labs(title = "Accuracy and Recall by Threshold", y = "Metric Value") +
  scale_colour_manual("", 
                      breaks = c("Accuracy", "Recall"),
                      values = c("blue", "red"))
#####

##### Predicting LC with optimal Threshold #####
test$predicted_class <- ifelse(test$predicted_prob > 0.32, "Yes", "No") #Chosen threshold based on 50/50 split 

#Confusion Matrix, A&R
table_pred <- table(Predicted = test$predicted_class, Actual = test$mdeSI)
accuracy <- sum(table_pred[2,2]+table_pred[1,1]) / sum(table_pred)
recall <- table_pred[2,2] / sum(table_pred[2,2]+table_pred[1,2])
accuracy
recall

#ROC
roc_result <- roc(response = test$mdeSI, predictor = test$predicted_prob)
auc_value <- auc(roc_result)
plot(roc_result, main = paste("ROC Curve, AUC =", auc_value))
#####

##### Predicting RF with optimal Threshold #####
test$predicted_class2 <- ifelse(test$predicted_prob2 > 0.02, "Yes", "No")

#Confusion Matrix, A&R
table_pred2 <- table(Predicted = test$predicted_class2, Actual = test$mdeSI)
accuracy2 <- sum(table_pred2[2,2]+table_pred2[1,1]) / sum(table_pred2)
recall2 <- table_pred2[2,2] / sum(table_pred2[2,2]+table_pred2[1,2])
accuracy2
recall2

#ROC
roc_result2 <- roc(response = test$mdeSI, predictor = test$predicted_prob2)
auc_value2 <- auc(roc_result2)
plot(roc_result2, main = paste("ROC Curve, AUC =", auc_value2))
#####