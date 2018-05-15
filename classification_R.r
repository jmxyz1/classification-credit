library(readr)
library(rpart)
library(caret)
library(e1071)
library(Metrics)
library(gbm)
#library(pROC)
library(prediction)
library(randomForest)

credit <- read.csv("credit_data.csv")

n<-nrow(credit)
set.seed(123)
n_train<-round(0.8*n)
train_indices<-sample(1:n, n_train)
credit_train<-credit[train_indices,]
credit_test<-credit[-train_indices,]

credit_model <- rpart(formula = default~.,
                      data = credit_train,
                      method = "class")

#print(credit_model)
class_prediction <- predict(object =credit_model,
                        newdata = credit_test,  
                        type = "class")

#A significant test result means that you have rejected this null hypothesis and the two 
#variables are associated. The proportion of one variable changes as 
#the values of the other variable changes.
confusionMatrix(data = class_prediction,     
         reference = credit_test$default) 

credit_model1 <- rpart(formula = default ~ .,
                       data = credit_train,
                       method = "class",
                       parms = list(split = "gini"))
#Train an information-based model
credit_model2 <- rpart(formula = default ~ .,
                       data = credit_train,
                       method = "class",
                       parms = list(split = "information"))

# Generate predictions on the validation set using the gini model
pred1 <- predict(object = credit_model1,
            newdata = credit_train,
             type = "class")    


# Generate predictions on the validation set using the information model
pred2 <- predict(object = credit_model2,
            newdata = credit_train,
             type = "class")

# Compare classification error
ce(actual = credit_test$default,
   predicted = pred1)
ce(actual = credit_test$default,
   predicted = pred2) 

credit_train$default <- ifelse(credit_train$default == "yes", 1, 0)
set.seed(1)

credit_model <- gbm(formula = default ~ .,
                    distribution = "bernoulli",
                    data = credit_train,
                     n.trees = 10000)
print(credit_model)
summary(credit_model)

credit_test$default <- ifelse(credit_test$default == "yes", 1, 0)

#generate predictions on the test set
preds1 <- predict(object = credit_model,
                 newdata = credit_test,
                  n.trees = 10000)

#generate predictions on the test set where type is response
preds2 <- predict(object = credit_model,
                 newdata = credit_test,
                  n.trees = 10000,
                  type = "response")


# Compare the range of the two sets of predictions
range(preds1)
range(preds2)
auc(actual = credit_test$default, predicted = preds2) 

# Optimal ntree estimate based on OOB
ntree_opt_oob <- gbm.perf(object = credit_model,
                          method = "OOB",
                          oobag.curve = TRUE)

# Train a CV GBM model
set.seed(1)
credit_model_cv <- gbm(formula = default ~ .,
                      distribution = "bernoulli",
                      data = credit_train,
                       n.trees = 10000,
                       cv.folds = 2)

# Optimal ntree estimate based on CV
ntree_opt_cv <- gbm.perf(object = credit_model_cv,
                         method = "cv")

# Compare the estimates
print(paste0("Optimal n.trees (OOB Estimate): ", ntree_opt_oob))
print(paste0("Optimal n.trees (CV Estimate): ", ntree_opt_cv))
# Generate predictions on the test set using ntree_opt_oob number of trees
preds1 <- predict(object = credit_model,
                  newdata = credit_test,
                  n.trees = ntree_opt_oob)
                  
# Generate predictions on the test set using ntree_opt_cv number of trees
preds_gbm <- predict(object = credit_model, 
                    newdata = credit_test,
                  n.trees = ntree_opt_cv)   

# Generate the test set AUCs using the two sets of preditions & compare
auc1 <- auc(actual = credit_test$default, predicted = preds1)  #OOB
auc2 <- auc(actual = credit_test$default, predicted = preds_gbm)  #CV 

# Compare AUC 
print(paste0("Test set AUC (OOB): ", auc1))                         
print(paste0("Test set AUC (CV): ", auc2))

#random forest
#train a Random Forest
set.seed(1)  # for reproducibility
credit <- read.csv("credit_data.csv")

n<-nrow(credit)
set.seed(123)
n_train<-round(0.8*n)
train_indices<-sample(1:n, n_train)
credit_train<-credit[train_indices,]
credit_test<-credit[-train_indices,]

credit_model <- randomForest(formula = default ~ .,
                             data = credit_train)
credit <- read.csv("https://assets.datacamp.com/production/course_3022/datasets/credit.csv")

#OOB error matrix
err <- credit_model$err.rate
head(err)

#final OOB error rate
oob_err <- err[nrow(err), "OOB"]
print(oob_err)

plot(credit_model)
legend(x = "right",
       legend = colnames(err),
       fill = 1:ncol(err))

preds_list <- list(preds_gbm, preds_gbm)

m <- length(preds_list)
actuals_list <- rep(list(credit_test$default), m)

#Plot ROC curves
pred <- prediction(preds_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright",
       legend = c("GBM"),
       fill = 1:m)
	   