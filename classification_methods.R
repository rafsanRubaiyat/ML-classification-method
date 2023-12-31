---
title: "STAT617_Final_Project_Analysis"
author: "Burke Gray, Ryan McAllister, Rafsan Siddiqui"
date: "2023-11-22"
output: html_document
---


#Read Data & Merge
```{r}
d1=read.table("student-mat.csv",sep=";",header=TRUE)
d2=read.table("student-por.csv",sep=";",header=TRUE)

student=merge(d1,d2, all=TRUE)
print(nrow(student))

```

#Data Cleaning
```{r}
duplicated(student) #check for duplicates
str(student) #examine variable names & types
#head(student)
any(is.na(student)) #check for missing values
#colnames(student)

#Remove unwanted variables by column index
student <- student[ ,-c(1,11,12,13,20,27,31,32)]
student

#Rename 'paid' variable as 'tutoring' to align with definition
colnames(student)[colnames(student) == "paid"] <- "tutoring"

#str(student) #Correct variables were removed & paid is renamed to tutoring

#Making Categorical Variables Factors
student$sex <- as.factor(student$sex)
student$address <- as.factor(student$address)
student$famsize <- as.factor(student$famsize)
student$Pstatus <- as.factor(student$Pstatus)
student$Medu <- as.factor(student$Medu)
student$Fedu <- as.factor(student$Fedu)
student$Mjob <- as.factor(student$Mjob)
student$Fjob <- as.factor(student$Fjob)
student$studytime <- as.factor(student$studytime)
student$schoolsup <- as.factor(student$schoolsup)
student$famsup <- as.factor(student$famsup)
student$tutoring <- as.factor(student$tutoring)
student$activities <- as.factor(student$activities)
student$internet <- as.factor(student$internet)
student$romantic <- as.factor(student$romantic)
student$famrel <- as.factor(student$famrel)
student$freetime <- as.factor(student$freetime)
student$goout <- as.factor(student$goout)
student$Walc <- as.factor(student$Walc)
student$health <- as.factor(student$health)

student$higher <- ifelse(student$higher == "no", 0 , 1) #no - 0; yes - 1
student$higher <- as.factor(student$higher)

#Resolve undersampling
#?sample
set.seed(202312)

indices_to_delete <- which(student$higher == 1)
indices_to_delete <- sample(indices_to_delete, size = floor(0.8 * length(indices_to_delete)))

student <- student[-indices_to_delete,]

#str(student) #checking..

#Renaming Levels
levels(student$Medu) <- c('none','elementary','middle','high','college')
levels(student$Fedu) <- c('none','elementary','middle','high','college')
levels(student$Mjob) <- c('teacher','health','civilserv','stayhome','other')
levels(student$Fjob) <- c('teacher','health','civilserv','stayhome','other')
levels(student$studytime) <- c('<2 hrs','2-5 hrs','5-10 hrs','>10 hrs')
levels(student$famrel) <- c('very bad','bad','average','good','excellent')
levels(student$health) <- c('very bad','bad','average','good','excellent')
levels(student$freetime) <- c('very low','low','average','high','very high')
levels(student$goout) <- c('very low','low','average','high','very high')
levels(student$Walc) <- c('very low','low','average','high','very high')

#str(student)
#head(student)

#make dummy variable dataset for methods that don't work with categorical variables directly.
#install.packages("fastDummies")
library(fastDummies)

fastDummies1 <- data.frame(numbers = 1:dim(student)[[1]],
                   sex = c(student$sex),
                   address = c(student$address),
                   famsize = c(student$famsize),
                   Pstatus = c(student$Pstatus),
                   Medu = c(student$Medu),
                   Fedu = c(student$Fedu),
                   Mjob = c(student$Mjob),
                   Fjob = c(student$Fjob),
                   studytime = c(student$studytime),
                   schoolsup = c(student$schoolsup),
                   famsup = c(student$famsup),
                   tutoring = c(student$tutoring), 
                   activities = c(student$activities),
                   internet = c(student$internet),
                   romantic = c(student$romantic),
                   famrel = c(student$famrel),
                   freetime = c(student$freetime),
                   goout = c(student$goout),
                   Walc = c(student$Walc),
                   health = c(student$health),
                   stringsAsFactors = F)

results <- fastDummies::dummy_cols(fastDummies1)
results <- results[, -c(1:21)]
print(results)
str(results)

#str(student)

dumstud <- cbind(results,student[,c(2,11,16,24,25)], all=TRUE) #bind numeric variables with dummy variables.
dumstud <- dumstud[,-75] #remove automatically added 'all' variable that came from binding
dim(dumstud)
str(dumstud)
head(dumstud)
```

#Exploratory Analysis
```{r}
library(mosaic)

favstats(student$age)
favstats(student$failures)
favstats(student$absences)
favstats(student$G3)

boxplot(G3 ~ higher, student)

barplot(table(student$Medu), col = rainbow(6), border = NA, cex.names = 0.8, xlab = "Mother's Education Level", ylab = "Count", ylim = c(0,160))
barplot(table(student$Fedu), col = rainbow(6), border = NA, cex.names = 0.8, xlab = "Father's Education Level", ylab = "Count", ylim = c(0,160))
barplot(table(student$Mjob), col = rainbow(6), border = NA, cex.names = 0.8, xlab = "Mother's Occupation", ylab = "Count", ylim = c(0,160))
barplot(table(student$Fjob), col = rainbow(6), border = NA, cex.names = 0.8, xlab = "Father's Occupation", ylab = "Count", ylim = c(0,160))
barplot(table(student$Walc), col = rainbow(6), border = NA, cex.names = 0.8, xlab = "Weekend Alcohol Consumption", ylab = "Count", ylim = c(0,100))
barplot(table(student$famrel), col = rainbow(6), border = NA, cex.names = 0.8, xlab = "Quality of Family Relationship", ylab = "Count", ylim = c(0,150))

?pie
pie(table(student$higher), main = "Response Variable - Do you want to go to college after high school?", labels = c('no: 89','yes: 191'))
#table(student$higher)

#additional explanatory analysis for paper
#numerical variables
student[which(student$age == 22),] 
nrow(student[which(student$age == 15 & student$higher == 1),])/nrow(student[which(student$age == 15),])
student[which(student$G3 == 19),]

#categorical variables
nrow(student[which(student$Medu == "college"),])
nrow(student[which(student$Fedu == "college"),])
nrow(student[which(student$Mjob == "teacher"),])
nrow(student[which(student$Fjob == "teacher"),])
```

#Discriminant Analysis
```{r}
numstud <- student[,c(2,11,16,24,25)] # dataset with just numeric indep. variables
head(numstud)
nrow(numstud)

install.packages("MVN")
library(MVN)

# Assuming your data is in a data frame called 'mydata'
# and the variables are named var1, var2, var3, var4

# Subset the data to include only the variables of interest
variables_of_interest <- numstud[, c("age", "failures", "absences", "G3")]

mardia_results <- mvn(variables_of_interest, mvnTest = "mardia")

# View the results
print(mardia_results$multivariateNormality)

library(biotools)
library(MASS)
set.seed(202312)
boxM(numstud[,-3],numstud[,3])
# Since our p-value < 0.05, we reject H0. We have convincing evidence to suggest that the covariance 
# matricies differ between the two populations. We will use QDA to continue

set.seed(202312)
qdafit <- qda(higher~., data=numstud, method='mle')
qdafit
#predict(qdafit)$class
tab5 = table(predict(qdafit)$class,numstud$higher)
tab5
print(calc <- (tab5[2,1]+tab5[1,2])/(tab5[2,1]+tab5[1,2]+tab5[1,1]+tab5[2,2]))

# With LOOCV
qdafit2 <- qda(higher~., data=numstud, method='mle', CV=TRUE)
qdafit2
tab6 = table(qdafit2$class,numstud$higher)
tab6
print(calc <- (tab6[2,1]+tab6[1,2])/(tab6[2,1]+tab6[1,2]+tab6[1,1]+tab6[2,2]))

# With 10 fold CV
library(ipred)
ip.qda = function(object, newdata) { #define prediction method
predict(object, newdata = newdata)$class
}
qdafit3 = errorest(higher ~ ., data=numstud, model=qda, estimator="cv",
est.para=control.errorest(k=10, predictions = TRUE), predict=ip.qda)
qdafit3
```

#Logistic Regression
```{r}
library(car)

set.seed(202312) #for reproducible results

#70/30 Testing/Training Split
#?sample
train = sample(1:nrow(student), nrow(student)*.7) #creates a sequence of numbers from 1 to the number of rows in student, and then randomly samples 70% of them. This gives us the indices of our training observations.

traindata = student[train,]
traindata

testdata = student[-train,]
testdata

#?glm
log.reg.fit <- glm(higher ~ ., family = binomial, data = traindata)
summary(log.reg.fit)

vif(log.reg.fit) #multicollinearity assumption check
# Assumption violated, there are multiple variables with VIF values above 5.
# We will remove them and see if the assumption can be met.

# Old model with whole data set
# log.reg.fit2 <- glm(higher ~ sex+age+address+famsize+Pstatus+failures+schoolsup+famsup+tutoring+activities+internet+romantic+absences+G3, family = binomial, data = student)
# summary(log.reg.fit2)

# Model using training data set
log.reg.fit2 <- glm(higher ~ sex+address+famsize+Pstatus+schoolsup+famsup+tutoring+internet+romantic+G3, family = binomial, data = traindata)
summary(log.reg.fit2)

vif(log.reg.fit2)
# Now we see that all of the VIF values are close to 1 so the multicollinearity
# assumption is met. 

#using stepwise by AIC criteria
#?step
step.log.reg.fit <- step(log.reg.fit2, direction = "both")

# Model on whole dataset
# log.reg.fit3 <- glm(higher ~ sex + age + address + failures + tutoring + activities + G3, family = binomial, data = student)

# Model on training data set
log.reg.fit3 <- glm(higher ~ address + famsup + tutoring + G3, family = binomial, data = traindata)

summary(log.reg.fit3)

log.reg.pred <- predict(log.reg.fit3, newdata=testdata, type="response")
log.reg.pred2 <- ifelse(log.reg.pred > 0.5,1,0)
tablog <- table(log.reg.pred2, student[-train,"higher"])
print(calc <- (tablog[2,1]+tablog[1,2])/(tablog[2,1]+tablog[1,2]+tablog[1,1]+tablog[2,2]))

# Heteroscedasticity not an issue b/c no linear relationship assumed
# and logistic models are robust to heteroscedasticity. 
#resid <- resid(log.reg.fit3)
#plot(fitted(log.reg.fit3),resid)

# Check variable importance of logistic model
library(caret)
varImp(log.reg.fit3)
# The most important variable is G3, then tutoring, famsup, then address

#McFadden's Rsq
#install.packages('pscl')
library(pscl)
pscl::pR2(log.reg.fit3)["McFadden"]
```

#Classification Trees
```{r}
library(tree)

set.seed(202312) #for reproducible results

#70/30 Testing/Training Split
#?sample
#train = sample(1:nrow(student), nrow(student)*.7) #creates a sequence of numbers from 1 to the number of rows in student, and then randomly samples 70% of them. This gives us the indices of our training observations.

#traindata = student[train,]
#traindata

#testdata = student[-train,]
#testdata

tree.student = tree(higher ~., student)
summary(tree.student)
plot(tree.student)
text(tree.student,pretty=0)

#Cannot calculate MSE for dichotomous response variable
#yhat=predict(tree.student,newdata=student[-train,])
#yhat
#class(yhat)
#nrow(yhat)
student.test=student[-train,"higher"] #we still will need this below
#student.test
#length(student.test)
#mean((yhat-student.test)^2)

set.seed(202312)
cv.student=cv.tree(tree.student)
plot(cv.student$size,cv.student$dev,type='b')
cv.student

prune.student=prune.tree(tree.student,best=4)
plot(prune.student)
text(prune.student,pretty=0)

#Same comment as above re MSE
#yhat2=predict(prune.student,newdata=student[-train,])
#mean((yhat2-student.test)^2)

#confusion matrix - NON PRUNED TREE
set.seed(202312)
tree.pred=predict(tree.student, testdata, type = "class") #the argument type="class" instructs R to return the  class prediction
tab1 <- table(tree.pred, student[-train,"higher"])
print(calc <- (tab1[2,1]+tab1[1,2])/(tab1[2,1]+tab1[1,2]+tab1[1,1]+tab1[2,2]))

#confusion matrix - PRUNED TREE
set.seed(202312)
tree.pred2=predict(prune.student, testdata, type = "class") #the argument type="class" instructs R to return the  class prediction
tab2 <- table(tree.pred2, student[-train,"higher"])
print(calc <- (tab2[2,1]+tab2[1,2])/(tab2[2,1]+tab2[1,2]+tab2[1,1]+tab2[2,2]))
```

#Random Forests
Bagging (ntrees = 500)
```{r}
library(randomForest)
set.seed(202312)
bag.student=randomForest(higher~.,data=student,subset=train,mtry=24,importance=TRUE, proximity = T) #bagging
bag.student
yhat.bag = predict(bag.student,newdata=student[-train,])
tab.rf2 <- table(yhat.bag, student[-train,"higher"])
print(calc <- (tab.rf2[2,1]+tab.rf2[1,2])/(tab.rf2[2,1]+tab.rf2[1,2]+tab.rf2[1,1]+tab.rf2[2,2]))
plot(yhat.bag, student.test)
importance(bag.student) 
varImpPlot(bag.student)
```

m = M/3 = 8 (ntrees = 500)
```{r}
set.seed(202312)
rf1.student=randomForest(higher~.,data=student,subset=train,mtry=8,importance=TRUE, proximity = T) 
rf1.student
yhat.rf8 = predict(rf1.student,newdata=student[-train,])
tab.rf8 <- table(yhat.rf8, student[-train,"higher"])
print(calc <- (tab.rf8[2,1]+tab.rf8[1,2])/(tab.rf8[2,1]+tab.rf8[1,2]+tab.rf8[1,1]+tab.rf8[2,2]))
importance(rf1.student) 
varImpPlot(rf1.student)
```

m = sqrt(M) = 5 (ntrees = 500)
```{r}
set.seed(202312)
rf1.student=randomForest(higher~.,data=student,subset=train,mtry=5,importance=TRUE, proximity = T) 
rf1.student
yhat.rf5 = predict(rf1.student,newdata=student[-train,])
tab.rf5 <- table(yhat.rf5, student[-train,"higher"])
print(calc <- (tab.rf5[2,1]+tab.rf5[1,2])/(tab.rf5[2,1]+tab.rf5[1,2]+tab.rf5[1,1]+tab.rf5[2,2]))
importance(rf1.student) 
varImpPlot(rf1.student)
```

Comparing different values of mtry for varying ntrees - NOT VALID FOR CLASSIFICATION PROB
```{r, eval = F}
set.seed(202312)
rf24.student=randomForest(higher~.,data=student,subset=train,mtry=24,importance=TRUE, proximity=TRUE) #bag
rf16.student=randomForest(higher~.,data=student,subset=train,mtry=16,importance=TRUE, proximity=TRUE) # midpoint btwn M and M/3
rf12.student=randomForest(higher~.,data=student,subset=train,mtry=12,importance=TRUE, proximity=TRUE) #M/2
rf8.student=randomForest(higher~.,data=student,subset=train,mtry=8,importance=TRUE, proximity=TRUE) #M/3
rf5.student=randomForest(higher~.,data=student,subset=train,mtry=5,importance=TRUE, proximity=TRUE) #sqrt(M)

?plot
plot(1:rf24.student$ntree, rf24.student$err.rate[,1], type = "l", col = "blue", xlab = "Number of Trees", ylab = "Error", main = "OOB (Out-Of-Bag) Error by mtry & ntree", xlim = c(0,500))
legend("topright", legend = c("mtry = 24 (bagging)", "mtry = 16 (2M/3)", "mtry = 12 (M/2)", "mtry = 8 (M/3)", "mtry = 5 (√M)"), lty = 1, lwd = 2, col = c("blue","red","green","purple","orange"))
lines(1:rf16.student$ntree, rf16.student$err.rate[,1], type = "l", col = "red")
lines(1:rf12.student$ntree, rf12.student$err.rate[,1], type = "l", col = "green")
lines(1:rf8.student$ntree, rf8.student$err.rate[,1], type = "l", col = "purple")
lines(1:rf5.student$ntree, rf5.student$err.rate[,1], type = "l", col = "orange")
```

m = sqrt(M) = 8 (ntrees = 95)
```{r}
set.seed(202312)
rf8.student=randomForest(higher~.,data=student,subset=train,mtry=8, ntree = 95, importance=TRUE, proximity = T) 
rf8.student
yhat.rf8 = predict(rf8.student,newdata=student[-train,])
tab.rf8 <- table(yhat.rf8, student[-train,"higher"])
print(calc <- (tab.rf8[2,1]+tab.rf8[1,2])/(tab.rf8[2,1]+tab.rf8[1,2]+tab.rf8[1,1]+tab.rf8[2,2]))
plot(yhat.rf8, student.test)
importance(rf8.student) 
varImpPlot(rf8.student)
```

#Support Vector Machines (SVM)
https://myweb.uiowa.edu/pbreheny/uk/764/notes/11-1.pdf 
[See pg. 24 for why Kernel Density Classification will work for our data but Descrim Analysis won't]

```{r}
set.seed(202312)
#70/30 Testing/Training Split
#?sample
dumtrain = sample(1:nrow(dumstud), nrow(dumstud)*.7) #creates a sequence of numbers from 1 to the number of rows in student, and then randomly samples 70% of them. This gives us the indices of our training observations.

dumtraindata = dumstud[dumtrain,]
dumtraindata

dumtestdata = dumstud[-dumtrain,]
dumtestdata

#install.packages("e1071")
library(e1071)

#class(dumstud)

set.seed(202312)
#?tune
tune.out=tune(svm,higher~., data=dumtraindata, kernel="radial", ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)

set.seed(202312)
svmfit=svm(higher~., data=dumtraindata, kernel="radial", cost=5, scale=FALSE)
print(svmfit)
svmfit$index
summary(svmfit)

ypred=predict(svmfit,dumtestdata)
tab3 <- table(predict=ypred, truth=dumtestdata$higher)
print(calc <- (tab3[2,1]+tab3[1,2])/(tab3[2,1]+tab3[1,2]+tab3[1,1]+tab3[2,2]))
```


#KDC
##INCLUDE AS SOURCE:
###https://stats.libretexts.org/Bookshelves/Computing_and_Modeling/RTG%3A_Classification_Methods/4%3A_Numerical_Experiments_and_Real_Data_Analysis/Preprocessing_of_categorical_predictors_in_SVM%2C_KNN_and_KDC_(contributed_by_Xi_Cheng)
```{r}
set.seed(202312)

#Just for checking purpose
print(rows <- nrow(student))
print(cols <- ncol(student))
print(numTrain <- floor((.7) * rows))
print(numTest <- rows - numTrain)

### KDC ###
library(MASS)
#install.packages('klaR')
library(klaR)
nb1 <- NaiveBayes(as.factor(higher) ~.,data=traindata, usekernel=T) 
nb1
#head(testdata)
#testdata[,-16]

#Suppressed Warning due to mismatch in size of training and testing sets.
p1 <- suppressWarnings(predict(nb1, testdata[,-16]))
print(p1)

table(true = testdata$higher, predict = p1$class)

p2 <- suppressWarnings(predict(nb1, traindata[,-16]))
tab3 <- table(true = traindata$higher,predict = p2$class)
print(calc <- (tab3[2,1]+tab3[1,2])/(tab3[2,1]+tab3[1,2]+tab3[1,1]+tab3[2,2])) #mis-classification rate
```
Comparison: The misclassification rate for SVM (using Kernel Density Classification) is actually slightly worse than the rate for the unpruned classification tree. 

#Neural Networks
```{r}
#install.packages('neuralnet')
library(neuralnet)

# Create a dataset with the numeric variables normalized between 0 and 1
dumstudscaled <- dumstud
dumstudscaled[, c(70,71,73,74)] <- scale(dumstud[, c(70,71,73,74)], center = FALSE, scale = apply(dumstud[, c(70,71,73,74)], 2, function(x) diff(range(x))))

y = as.numeric(as.matrix(dumstudscaled[,72]))

#dumstudscaled$higher[which(dumstudscaled$higher == 0)]
#dumstudscaled$higher[which(dumstudscaled$higher == 1)]

x = as.numeric(as.matrix(dumstudscaled[,-72]))
x = matrix(as.numeric(x), ncol = 73)
dim(x)

df = data.frame(cbind(x,y))
dim(df)
str(df)

# To change neural network size, use the hidden option. Ex 3 hidden layers, 8 nodes each -> hidden = c(8,8,8)
# Tried hidden = c(8,8,8), c(8,8), c(4,4,4), c(4,4), c(8), & c(4). 
set.seed(202312)
nn1 = neuralnet(y~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15 + V16
 + V17 + V18 + V19 + V20 + V21 + V22 + V23 + V24 + V25 + V26 + V27 + V28 + V29 + V30 + V31 + V32 + V33
 + V34 + V35 + V36 + V37 + V38 + V39 + V40 + V41 + V42 + V43 + V44 + V45 + V46 + V47 + V48 + V49 + V50
 + V51 + V52 + V53 + V54 + V55 + V56 + V57 + V58 + V59 + V60 + V61 + V62 + V63 + V64 + V65 + V66 + V67
 + V68 + V69 + V70 + V71 + V72 + V73, data = df, hidden = c(4))

plot(nn1)

yy1 = nn1$net.result[[1]]

yhat = matrix(0,length(y),1)
yhat[which(yy1 > mean(yy1))] = 1
yhat[which(yy1 <= mean(yy1))] = 0
print(tab4 <- table(y,yhat))
print(calc <- (tab4[2,1]+tab4[1,2])/(tab4[2,1]+tab4[1,2]+tab4[1,1]+tab4[2,2])) #mis-classification rate

# Having 1 hidden layer with 4 nodes seems to do a good job with only a 2.5% misclassification rate.
# All of the more complex models have perfect prediction, which leads me to believe this could be 
# due to overfitting. 
```

