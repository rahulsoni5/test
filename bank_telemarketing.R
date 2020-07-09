# Loading Packages
library(dplyr)
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(pROC)
library(ggplot2)
library(caTools)
library(mltools)


# Read Data
setwd("E:/NCI Projects/Machine Learning Final Project/Datasets/Bank/Code")
data <- read.csv("bank_tele_data.csv")

# Changing colnames post reading the additional names file to make it more specific to understand

colnames(data)[5] <- "default_credit"
colnames(data)[6] <- "housing_loan"
colnames(data)[7] <- "personal_loan"
colnames(data)[8] <- "contact_type"
colnames(data)[12] <- "current_campaign_contact_count"
colnames(data)[13] <- "days_passed"
colnames(data)[14] <- "previous_campaign_contact_count"
colnames(data)[15] <- "previous_campaign_outcome"


# Structure and summary of datset
str(data)
summary(data)


## Check missing values
sum(is.na(data)) # 0 NA values


# Boxplot and quantile plot

boxplot(data$age) # it shows there are nuber of outliers
boxplot(data$duration) # there are number of outliers in duration

quantile(data$age) # there is number of outliers in the upper part
quantile(data$duration) # number of outliers in the upper part



## education vs output variable
ggplot(data, aes(education)) + geom_bar(aes(fill=y)) # people with university degree subscribe the most


## Job title vs the output variable

ggplot(data, aes(job)) + geom_bar(aes(fill=y))
# Admin and techinican job title have subscribed the most


## Marital status vs output variable
ggplot(data, aes(marital)) + geom_bar(aes(fill=y)) #married people subscribe the most



# Default_credit vs output variable
ggplot(data, aes(default_credit)) + geom_bar(aes(fill=y)) # People with no default credit history subscribes the term deposit



# Housing loan vs output variable
ggplot(data, aes(housing_loan)) + geom_bar(aes(fill=y))
# people with housing loan have higher proportion to subscribe the term deposit


# Personal loan vs output variable
ggplot(data, aes(personal_loan)) + geom_bar(aes(fill=y))
# People with no personal loan history have higher proportion of buying term



# Correalation Matrix
for (col in colnames(data)){
  print(col)
  #res <- cor.test(bankData$y, bankData[,col], method = "pearson")
  print(cor(data[,col], data$y))
}


# Taking significant factors in dataframe by using correlation matrix
data <- data[,c(1,2,3,4,5,6,7,11,21)]


# Dividing the data into training and test data using caTools package
set.seed(101)  # random numbers are generated
sample <- sample.split(data, SplitRatio = 0.7)
training <- subset(data, sample==T) 
testing <- subset(data, sample==F)



## Modelling
# Naive Bayes

nb_model <- naiveBayes(training, training$y)
nb_model

# Predictions
nb_predict <- predict(nb_model, testing)
nb_predict


#Confusion Matrix
confusionMatrix(table(testing$y, nb_predict)) # confusion matrix , accuracy 96.11





## KNN

library(data.table)
# Preprocessing for KNN algorithm
# One hot encoding making data suitable for KNN 

data$job<- one_hot(as.data.table(data$job))
data$marital <-one_hot(as.data.table(data$marital))
data$education <- one_hot(as.data.table(data$education))
data$default_credit <-one_hot(as.data.table(data$default_credit))      
data$housing_loan <-one_hot(as.data.table(data$housing_loan))       
data$personal_loan <-one_hot(as.data.table(data$personal_loan))       
       

data$age <- as.integer(data$age)
data$duration <- as.integer(data$duration)

#Recoding output variable to 1 and 0
data$y <- recode(data$y, "yes" = 1, "no" = 0)
data$y <- as.integer(data$y)


# checking structure of data
str(data)


# Normalize
normaliize <- function(x) {
  return((x-min(x))/(max(x)- min(x)))
}

data.n <- as.data.frame(lapply(data, normaliize))


# Dividing data into train and test
set.seed(123)

data.d <- sample(1:nrow(data.n),size=nrow(data.n)*0.7, replace=FALSE)

training <- data[data.d,] 
testing <- data[-data.d,]


# Creating labels for 'y' which is our target variable
training_labels <- data[data.d,9]
testing_labels <- data[-data.d,9]



nrow(training)  # number of observations
sqrt(28831)     # square root to find optimal value


#Modelling
library(class)

knn_model <- knn(train=training,test=testing,cl=training_labels, k=28)

confusionMatrix(table(knn_model, testing_labels)) #confusion matrix


#Find optimal value of K which gives the best accuracy
i=1
k.optimal=1 
for (i in 1:200) {
  knn.mod <- knn(train=training,test=testing,cl=training_labels, k=i )
  k.optimal[i] <- 100*sum(testing_labels==knn.mod)/NROW(testing_labels)
  k=i
  cat(k,"=", k.optimal[i], '\n')  # to print accuracy
}

# Plotiing k value(Elbow graph)
plot(k.optimal, type= "b", xlab=" K value", ylab = "Accuracy levels") # plotting K - Values



