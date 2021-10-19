library(caret)
library(RANN)
library(randomForest)
library(dplyr)
library(factoextra)
require(caTools)
library(neuralnet)
library(ggfortify)
library(ggplot2)
library(e1071)
library(rpart)
library(rpart.plot)
library(gbm)

#------------------------------
#-------- Process data --------
#------------------------------
#dataset in cvs fomat saved as "master" in r studio
Master <- read.csv ("E:\\UNI\\ADVANCED ANALYTICS\\CW 2\\Phishing-Dataset.csv")

#Remove NA's
Master <- na.omit(Master)

#checks fro NA's
anyNA(Master)

#scales data
MasterScaled <- scale(Master)

#------------------------------
#------------ PCA -------------
#------------------------------
#perform PCA and view results
PCA <- prcomp(Master)
summary(PCA)

#scree plot
fviz_eig(PCA)

#view attribute importance on autoplot
autoplot(PCA)

#cos2 visual
fviz_cos2(PCA, choice = "var", axes = 1:2)

## Color by cos2 values: quality on the factor map
#too many varibales which makes the visual a little busy
fviz_pca_var(
  PCA,
  col.var = "cos2",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel = TRUE # Avoid text overlapping
)

#PCA importants vis:
# Plot PCA With labels and importance for visual identification of correct variables
varImportance = NULL
Rotat <- abs(PCA$rotation)

# NV = number of variables , NPC = number of PCs you want to use
NV = 49
NPC = 6

#loops per variable and calculates importance
for (i in 1:NV) {
  varImportance[i] <- sum(Rotat[i, 1:NPC])
}

#get variable names and sound to 2 digits , then store in data frame
varNames = names(Master)
varImportance <- round(varImportance, digits = 2)
vImportance = data.frame(varImportance, varNames)

#plot importance
ggplot(
  vImportance,
  aes(
    x = as.factor(varNames),
    y = varImportance,
    color =
      as.factor(varImportance),
    size = varImportance
  )
) +
  geom_point() +
  labs(title = "Attribute importance of Principle Components",
       y = "Attribute importance",
       x = "Attribute names") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)
  )



#----------------------------------------------------------
#----- Data splitting for subset and training / testing -----
#----------------------------------------------------------

#99% of variance  / proportion with 5 variables so i am going to subset these 5 + the target variable
SubMaster <-
  subset.matrix(
    Master ,
    select = c(
      NumDash ,
      NumNumericChars ,
      PathLength,
      QueryLength,
      UrlLength ,
      CLASS_LABEL
    )
  )

#set as data frame
SubMaster <- as.data.frame(SubMaster)

#-- test & train data --
# Random sampling
samplesize = 0.70 * nrow(SubMaster)
set.seed(80)
index = sample(seq_len (nrow (SubMaster)), size = samplesize)

# Create training and test set
datatrain = SubMaster[index, ]
datatest = SubMaster[-index, ]

#check dimensions of training and testing data
dim(datatrain)

dim(datatest)


#-----------------------------------
#-----Artificial Neural Network-----
#-----------------------------------
# Scale data for neural network
max = apply(SubMaster , 2 , max)
min = apply(SubMaster, 2 , min)

# fit neural network
set.seed(2)
NN = neuralnet(
  CLASS_LABEL ~ NumDash + NumNumericChars + PathLength + QueryLength + UrlLength ,
  datatrain,
  hidden = c(6, 2, 3) ,
  3,
  rep = 1 ,
  linear.output = T
)

#PlotNN
NN$result.matrix
plot(NN)

# Prediction using neural network
predict_testNN = compute(NN, datatest[, c(1:5)])
results <-
  data.frame(actual = datatest$CLASS_LABEL,
             prediction = predict_testNN$net.result)

#round off results to get whole numbers
roundedresults <- sapply(results, round, digits = 0)
roundedresultsdf = data.frame(roundedresults)
attach(roundedresultsdf)
table(actual, prediction)

#fix "x must be atomic"
#datatest$CLASS_LABEL <- unlist(datatest$CLASS_LABEL)
ANN_ConfusionMatrix = confusionMatrix(as.factor(roundedresultsdf$actual),
                                      as.factor(roundedresultsdf$prediction))
ANN_ConfusionMatrix



#---------------------------------
#---------- Decision tree --------
#---------------------------------
#fit model
Fit_DT <- rpart(CLASS_LABEL ~ ., data = datatrain, method = 'class')

#plot model visual
rpart.plot(Fit_DT, extra = 107)

#predict
TestDT <- predict(Fit_DT, datatest, type = 'class')

tableDT <- table(datatest$CLASS_LABEL, TestDT)
tableDT

#confusion matrix
DecisionTree_ConfusionMatrix <-
  confusionMatrix(as.factor(datatest$CLASS_LABEL), TestDT)
DecisionTree_ConfusionMatrix



#-----------------------------------
#---------- Random forest ----------
#-----------------------------------
# Set random seed to make results reproducible:
set.seed(17)

#train model
modfit.rf <-
  randomForest(
    CLASS_LABEL ~ .,
    data = datatrain,
    ntree = 100,
    mtry = 5,
    importance = T
  )

#predict using train model
predictions <- predict(modfit.rf, datatest, type = "class")

#Calculate the confusion matrix and statistics
predictions <- round(predictions)

#confusion matrix
RandomForest_ConfusionMatrix <-
  confusionMatrix(as.factor(datatest$CLASS_LABEL), as.factor(predictions))
RandomForest_ConfusionMatrix



#---------------------------------
#----- SUPPORT VECTOR MACHINE ----
#---------------------------------
#factorize
datatrain[["CLASS_LABEL"]] = factor(datatrain[["CLASS_LABEL"]])

#using traincontrol() method to train data. it has 10 sampling iterations
#and it will cross validate 3 times
trctrl <-
  trainControl(method = "repeatedcv",
               number = 10,
               repeats = 3)

#fit model
svm_Linear <-
  train(
    CLASS_LABEL ~ .,
    data = datatrain,
    method = "svmLinear",
    trControl = trctrl,
    preProcess = c("center"),
    tuneLength = 10
  )

svm_Linear

#predict() method will predict results. i am passing 2 arguments, the first is the trained model and the
#second "newdata" is the test data frame. the Predicts() returns a list we are saving into a data frame
test_pred <- predict(svm_Linear, newdata = datatest)
test_pred

#confusion matrix predicts the accuracy of the model
SVM_ConfusionMatrix <-
  confusionMatrix(table(test_pred, datatest$CLASS_LABEL))
SVM_ConfusionMatrix

# #tuning of an SVM classifier with different values of Cost ("C") , we can enter different values of cost
# #and use the tune grid parameter in the train method to test the different C values
# #---!!!WARNING THE BELOW MAY CRASH R STUDIO on a standard laptop!!!----
# grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5))
# svm_Linear_Grid <- train(CLASS_LABEL ~., data = SubMaster, method = "svmLinear",
#                          trControl = trctrl,
#                          preProcess = c("center"),
#                          tuneGrid = grid,
#                          tuneLength = 10)
#
# #view results of cost tuning test
# svm_Linear_Grid
# plot(svm_Linear_Grid)
#
# #retest SVM using best cost value
# test_pred_grid <- predict(svm_Linear_Grid, newdata = datatest)
# test_pred_grid
#
# #view results in confusion matrix
# confusionMatrix(table(test_pred_grid, datatest$CLASS_LABEL))



#-----------------------------
#--------Ensemble Models------
#-----------------------------
#~~~~~~~~~~~~~~~~
# ~~~ BAGGING ~~~
#~~~~~~~~~~~~~~~~
#Spliting training set into two parts based on outcome: 75% and 25%
index <-
  createDataPartition(SubMaster$CLASS_LABEL, p = 0.75, list = FALSE)
trainSet <- SubMaster[index, ]
testSet <- SubMaster[-index, ]

trainSet$CLASS_LABEL <- as.factor(trainSet$CLASS_LABEL)
testSet$CLASS_LABEL <- as.factor(testSet$CLASS_LABEL)

levels(trainSet$CLASS_LABEL) <- c("phishing", "genuine")
levels(testSet$CLASS_LABEL) <- c("phishing", "genuine")

#Defining the training controls for multiple models
fitControl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  classProbs = T
)

#Defining the predictors and outcome
predictors <-
  c('NumDash',
    'NumNumericChars',
    'PathLength',
    'QueryLength',
    'UrlLength')
outcomeName <- 'CLASS_LABEL'

#Random forest ---
#-- Training and testing of 3 models (RF, KNN, LR). Compare 3 confusion matrix --
#Training the random forest model
model_rf <-
  train(
    trainSet[, predictors],
    trainSet[, outcomeName],
    method = 'rf',
    trControl = fitControl,
    tuneLength = 2
  )

#Predicting using random forest model
testSet$pred_rf <- predict(object = model_rf, testSet[, predictors])

#Checking the accuracy of the random forest model
#testSet$CLASS_LABEL <- as.factor(testSet$CLASS_LABEL)
testSet$pred_rf <- as.factor(testSet$pred_rf)
confusionMatrix(testSet$CLASS_LABEL, testSet$pred_rf)

#KNN ---
#Training the knn model
model_knn <-
  train(
    trainSet[, predictors],
    trainSet[, outcomeName],
    method = 'knn',
    trControl = fitControl,
    tuneLength = 3
  )

#Predicting using knn model
testSet$pred_knn <- predict(object = model_knn, testSet[, predictors])

#Checking the accuracy of the random forest model
confusionMatrix(testSet$CLASS_LABEL, testSet$pred_knn)

#linear regression ---
#Training the Logistic regression model
model_lr <-
  train(
    trainSet[, predictors],
    trainSet[, outcomeName],
    method = 'glm',
    trControl = fitControl,
    tuneLength = 3
  )

#Predicting using knn model
testSet$pred_lr <- predict(object = model_lr, testSet[, predictors])

#Checking the accuracy of the random forest model
confusionMatrix(testSet$CLASS_LABEL, testSet$pred_lr)

#avaraging ---
#Predicting the probabilities
testSet$pred_rf_prob <-
  predict(object = model_rf, testSet[, predictors], type = 'prob')
testSet$pred_knn_prob <-
  predict(object = model_knn, testSet[, predictors], type = 'prob')
testSet$pred_lr_prob <-
  predict(object = model_lr, testSet[, predictors], type = 'prob')

#Taking average of predictions
testSet$pred_avg <-
  (
    testSet$pred_rf_prob$phishing + testSet$pred_knn_prob$phishing + testSet$pred_lr_prob$phishing
  ) / 3

#Splitting into binary classes at 0.5
testSet$pred_avg <-
  as.factor(ifelse(testSet$pred_avg > 0.5, 'phishing', 'genuine'))

#confusion matrix
BaggingAvaraging_CM <-
  confusionMatrix(testSet$CLASS_LABEL, testSet$pred_avg)
BaggingAvaraging_CM

#majority vote ---
#The majority vote
testSet$pred_majority <-
  as.factor(
    ifelse(
      testSet$pred_rf == 'phishing' & testSet$pred_knn == 'phishing',
      'phishing',
      ifelse(
        testSet$pred_rf == 'phishing' & testSet$pred_lr == 'phishing',
        'phishing',
        ifelse(
          testSet$pred_knn == 'phishing' & testSet$pred_lr == 'phishing',
          'phishing',
          'genuine'
        )
      )
    )
  )

MajorityVote_CM <-
  confusionMatrix(testSet$CLASS_LABEL, testSet$pred_majority)
MajorityVote_CM

#weighted avarage ---
#Taking weighted average of predictions
testSet$pred_weighted_avg <-
  (testSet$pred_rf_prob$phishing * 0.50) +
  (testSet$pred_knn_prob$phishing * 0.25) +
  (testSet$pred_lr_prob$phishing * 0.25)

#Splitting into binary classes at 0.5
testSet$pred_weighted_avg <-
  as.factor(ifelse(testSet$pred_weighted_avg > 0.5, 'phishing', 'genuine'))

WeightedAvrange_CM <-
  confusionMatrix(testSet$CLASS_LABEL, testSet$pred_weighted_avg)
WeightedAvrange_CM

#~~~~~~~~~~~~~~~~~
# ~~~ Stacking ~~~
#~~~~~~~~~~~~~~~~~
#Defining the training control
fitControl <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = 'final',
  # To save out of fold predictions for best parameter combinantions
  classProbs = T # To save the class probabilities of the out of fold predictions
)

#Defining the predictors and outcome
predictors <-
  c('NumDash',
    'NumNumericChars',
    'PathLength',
    'QueryLength',
    'UrlLength')
outcomeName <- 'CLASS_LABEL'

#training models ---
#Training the random forest model
model_rf <-
  train(
    trainSet[, predictors],
    trainSet[, outcomeName],
    method = 'rf',
    trControl = fitControl,
    tuneLength = 3
  )

#Training the knn model
model_knn <-
  train(
    trainSet[, predictors],
    trainSet[, outcomeName],
    method = 'knn',
    trControl = fitControl,
    tuneLength = 3
  )

#Training the Logistic regression model
model_lr <-
  train(
    trainSet[, predictors],
    trainSet[, outcomeName],
    method = 'glm',
    trControl = fitControl,
    tuneLength = 3
  )

#predicting models ---
#Predicting the out of fold prediction probabilities for training data
trainSet$OOF_pred_rf <-
  model_rf$pred$phishing[order(model_rf$pred$rowIndex)]
trainSet$OOF_pred_knn <-
  model_knn$pred$phishing[order(model_knn$pred$rowIndex)]
trainSet$OOF_pred_lr <-
  model_lr$pred$phishing[order(model_lr$pred$rowIndex)]

#Predicting probabilities for the test data
testSet$OOF_pred_rf <-
  predict(model_rf, testSet[predictors], type = 'prob')$phishing
testSet$OOF_pred_knn <-
  predict(model_knn, testSet[predictors], type = 'prob')$phishing
testSet$OOF_pred_lr <-
  predict(model_lr, testSet[predictors], type = 'prob')$phishing

#Predictors for top layer models
predictors_top <- c('OOF_pred_rf', 'OOF_pred_knn', 'OOF_pred_lr')

#trialing diffrent top models ---
#GBM as top layer model
model_gbm <-
  train(
    trainSet[, predictors_top],
    trainSet[, outcomeName],
    method = 'gbm',
    trControl = fitControl,
    tuneLength = 3
  )

#predict using GBM top layer model
testSet$gbm_stacked <- predict(model_gbm, testSet[, predictors_top])
confusionMatrix(testSet$CLASS_LABEL, testSet$gbm_stacked)

#---

#Logistic regression as top layer model
model_glm <-
  train(
    trainSet[, predictors_top],
    trainSet[, outcomeName],
    method = 'glm',
    trControl = fitControl,
    tuneLength = 3
  )

#predict using logictic regression top layer model
testSet$glm_stacked <- predict(model_glm, testSet[, predictors_top])
confusionMatrix(testSet$CLASS_LABEL, testSet$glm_stacked)
