###############################################################################################################################################
                                                     #Breast Cancer Classification 
###############################################################################################################################################
###############################################################################################################################################
#Installing required Packages
install.packages("corrplot")
install.packages("randomForest")
install.packages("caret")
install.packages("e1071")
install.packages("party")
install.packages("ggplot2")
install.packages("PerformanceAnalytics")
install.packages("scatterplot3d")
###############################################################################################################################################
#Libraries required
library(corrplot) #For plotting correlation plots
library(PerformanceAnalytics) #For checking the performance Analysis of the desired features
library(randomForest) #For Random forest ML Algorithm
library(caret)   #(short for Classification And REgression Training, confusion Matrix) 
                 #contains functions to streamline the model training process for complex regression and classification problems.
library(e1071)   #For SVM for prediction, plots, tuning the hyperparameters 
library(party)   #For Decision Tree ML Algorithm
library(ggplot2) #For Data Visualization 

###############################################################################################################################################
###############################################################################################################################################
#Reading the dataset
data <- read.csv("data_with_na.csv")
View(data)

#Finding the null values
summary(data) #Summary of Data
sum(is.na(data))#sum of Na values in the dataset
colSums(is.na(data)) #checking for na values columnwise
rowSums(is.na(data)) #Row wise

# percentage of data missing in each column
perc = function(x){sum(is.na(x))/length(x)*100}
apply(data,2,perc)

#Columns having NA Values
#For Concavity_mean
sum(is.na(data$concavity_mean))
data$concavity_mean[is.na(data$concavity_mean)]=mean(data$concavity_mean,na.rm=T)
sum(is.na(data$concavity_mean))
View(data$concavity_mean)
sum(is.na(data)) #78-13= 65
View(data)

#For Concave.points_mean
sum(is.na(data$concave.points_mean))
data$concave.points_mean[is.na(data$concave.points_mean)]=mean(data$concave.points_mean,na.rm=T)
sum(is.na(data$concavity_mean))
View(data$concavity_mean)
sum(is.na(data)) #65-13=52
View(data)

#For Concavity_se
sum(is.na(data$concavity_se))
data$concavity_se[is.na(data$concavity_se)]=mean(data$concavity_se,na.rm=T)
sum(is.na(data$concavity_se))
View(data$concavity_se)
sum(is.na(data)) #52-13=39
View(data)

#For Concave.points_se
sum(is.na(data$concave.points_se))
data$concave.points_se[is.na(data$concave.points_se)]=mean(data$concave.points_se,na.rm=T)
sum(is.na(data$concave.points_se))
View(data$concave.points_se)
sum(is.na(data)) #39-13=26
View(data)

#For concavity_worst
sum(is.na(data$concavity_worst))
data$concavity_worst[is.na(data$concavity_worst)]=mean(data$concavity_worst,na.rm=T)
sum(is.na(data$concavity_worst))
View(data$concavity_worst)
sum(is.na(data)) #26-13=13
View(data)

#For concave.points_worst
sum(is.na(data$concave.points_worst))
data$concave.points_worst[is.na(data$concave.points_worst)]=mean(data$concave.points_worst,na.rm=T)
sum(is.na(data$concave.points_worst))
View(data$concave.points_worst)
sum(is.na(data)) #13-13=0
View(data)

#Removing unwanted features from the df
data$id <- NULL

data$diagnosis <- factor(ifelse(data$diagnosis=="B","Benign","Malignant")) #categorical Variables 
colnames(data)
data

#Data Exploration
dim(data) #checking of Dimensions (569-Rows and 31-columns)
str(data) #Structure of Dataset
summary(data) #Summary of dataset
attributes(data) #Checking the attributes in dataset 

quantile(data$radius_mean) #tells you how much of your data lies below a certain value.
quantile(data$radius_mean, c(.1,.3,.7))
var(data$radius_mean)  #variance 
hist(data$radius_mean, col= "cyan") #plotting a histogram

plot(density(data$radius_mean),col = 'green') #plotting a graph of data$radius_mean
t = table(data$diagnosis) 
t #No of Benign and Malignant
pie(t)
barplot(t,col = c('green','red')) #plotting a bargraph of two categories benign, malignant

#Correlation between the features
cor(data$radius_mean, data$radius_se)
aggregate(data$radius_mean ~ diagnosis,summary,data = data)
sapply(data, class)

#Let's check for correlations. For an anlysis to be robust it is good to remove mutlicollinearity ( remove highly correlated predictors)
library(corrplot)
corr<- cor(data[,c(2:31)])
corrplot(corr, method="circle")
#Positive correlations are displayed in blue and negative correlations in red color. 
#Color intensity and the size of the circle are proportional to the correlation coefficients.

#Conclusion from the plot: 
            #The highest correlations are between:perimeter_mean and radius_worst; area_worst 
            #and radius_worst; perimeter_worst and radius_worst, perimeter_mean, area_worst, area_mean, radius_mean; 
            #texture_mean and texture_worst;

library(scatterplot3d)
scatterplot3d(data$radius_mean,data$area_mean,data$perimeter_mean)#Plotting 3D scatterplot 

#Analyzing correlation between variables
library(PerformanceAnalytics)
#Mean
chart.Correlation(data[,c(3:12)],histogram=TRUE, col="grey10", pch=1, main="Data Mean") #pch=1 default empty circle symbol 
#Standard Error
chart.Correlation(data[,c(13:22)],histogram=TRUE, col="grey10", pch=1, main="Data SE")
#Worst
chart.Correlation(data[,c(23:31)],histogram=TRUE, col="grey10", pch=1, main="Data Worst")

###############################################################################################################################################
#Building ML Algorithms
#Division of training and testing data
set.seed(2412) #function sets the starting number used to generate a sequence of random numbers.
               #It ensures that you get the same result if you start with that same seed each time you run the same process. 
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7,0.3)) ##Randomly selecting 70% of the data for training and 30% for testing.
trainData <- data[ind==1,]
testData <- data[ind==2,]

#check proportion of benign and malignant cases in both datasets
#Train dataset
prop.table(table(trainData$diagnosis)) #Probability Value of trainData
#Test dataset
prop.table(table(testData$diagnosis)) #Probability Value of testData

#########################################################################################################################################################
#1.Random forest
library(randomForest)
library(caret)

rf <- randomForest(diagnosis~.,data=trainData,ntree = 100,proximity = TRUE)
rf

predTrain <- predict(rf, trainData, type = "class") #Predicting the Train data
table(predTrain, trainData$diagnosis)

predValid <- predict(rf, testData, type = "class") #Predicting the Test data
predValid
mean(predValid == testData$diagnosis)   #Finding the Mean                 
table(predValid,testData$diagnosis) 

library('e1071')
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid") 
#method = "cv": The method used to resample the dataset.
#number = n: Number of folders to create
#search = "grid": Use the search grid method. For randomized method, use "grid"

set.seed(1234)
rf_default <- train(diagnosis~.,
                    data = trainData,
                    method = "rf",
                    metric = "Accuracy",
                    trControl = trControl)
# Print the results
print(rf_default) 
plot(rf_default)
#Printing the results
print(rf_default)
attributes(rf_default)
plot(rf_default)  
importance(rf)
#Variance Importance Plot
varImpPlot(rf)

prediction <- predict(rf_default,newdata = testData) #Prediction on test data
table(prediction,testData$diagnosis) #Confusion Matrix
 
rf_CM = confusionMatrix(prediction,testData$diagnosis) #confusion Matrix with accuracy scores
rf_CM

###############################################################################################################################################
#2. Decision Tree
library(party)
myFormula <- diagnosis ~ radius_mean + texture_mean + radius_se + texture_se #Formula 
diagnosis_ctree <- ctree(myFormula,data=trainData) #model
table(predict(diagnosis_ctree), trainData$diagnosis)
print(diagnosis_ctree)
plot(diagnosis_ctree) #plot of DT
plot(diagnosis_ctree,type='simple') #Plot Simple 
testpredict <- predict(diagnosis_ctree, newdata = testData) #prediction on Test Data
table(testpredict,testData$diagnosis) #Confusion Matrix

DT_CM = confusionMatrix(testpredict,testData$diagnosis) #confusion Matrix with accuracy scores
DT_CM

################################################################################################################################################
#3. KMeans
data2 <- data
data2$diagnosis <- NULL
(kmeans.result <- kmeans(data2, 2)) #Result of Kmeans
table(data$diagnosis, kmeans.result$cluster) #Confusion Matrix
plot(data2[c("radius_mean", "texture_mean")], col = kmeans.result$cluster) #plot for Kmeans

###############################################################################################################################################
#4.SVM 
library(e1071)
svmodel = svm(diagnosis~.,data=trainData,kernel ="linear", cost = 0.1, scale=F) #Model 
svmpred = predict(svmodel,testData) #prediction on test Data
svm_CM = confusionMatrix(svmpred,testData$diagnosis) #confusion Matrix with Accuracy scores
svm_CM
f = ggplot(testData, aes( radius_mean ,texture_mean, colour = diagnosis))
f + geom_jitter() + scale_colour_hue() + theme(legend.position = "bottom")
###############################################################################################################################################
#5.SVM Tuned
library(e1071)
gamma = seq(0,0.1,0.005)  #gamma parameter defines how far the influence of a single training example reaches,
    #with low values meaning 'far' and high values meaning 'close'. 
cost = 2^(0:5)#hyperparameter 
param = expand.grid(cost=cost, gamma=gamma)    

acc = numeric()
acc1 = NULL; acc2 = NULL

for(i in 1:NROW(param)){        
  svmodel2 = svm(diagnosis~., data=trainData, gamma=param$gamma[i], cost=param$cost[i])
  svm_pred2 = predict(svmodel2, testData)
  acc1 = confusionMatrix(svm_pred2, testData$diagnosis)
  acc2[i] = acc1$overall[1]
} #Model Building 

acc_list = data.frame(p= seq(1,NROW(param)), accuracy = acc2) #Accuracy 

tune_p = subset(acc_list, accuracy==max(accuracy))[1,] #Tuning
subs = paste("Optimal cost =", param$cost[tune_p$p],"and optimal gamma =", param$gamma[tune_p$p], "(Accuracy :", round(tune_p$accuracy,4),") in SVM")
subs #Optimal cost,optimal gamma and accuracy in SVM  

library(ggplot2)
ggplot(acc_list, aes(x=p,y= accuracy)) +
  geom_line()+labs(title = "Accuracy with changing gamma and cost parameter options", subtitle = subs)

svm_tune = svm(diagnosis~., data=trainData, cost=param$cost[tune_p$p], gamma=param$gamma[tune_p$p]) #Tuning on TrainData
svm_tunepred = predict(svm_tune, testData) #Predicting on TestData
svm_CMtune = confusionMatrix(svm_tunepred, testData$diagnosis) #Confusion Matrix and Accuracy Score 
svm_CMtune

###############################################################################################################################################
#Comparing The models
model_cmp = data.frame(Model = c("Random Forest","Decision Tree","SVM","SVM Tuned"), Accuracy = c(rf_CM$overall[1],DT_CM$overall[1],
                                                                                                  svm_CM$overall[1],svm_CMtune$overall[1]))

library(ggplot2) #Visualizing the Accuracy of Different models .
ggplot(model_cmp, aes(x=Model,y= Accuracy,color= Model, label= Accuracy)) +
  geom_point()+labs(title = "Accuracy comparison of models")+ geom_text(aes(label= round(Accuracy,4)),vjust=2)

###############################################################################################################################################
#In Conclusion:
         #For the given dataset the Machine Learning model with high accuracy is for SVM Tuned Model and SVM.
###############################################################################################################################################






