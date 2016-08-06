data <- read.csv("cs-training.csv")
data <- data[,-1]
train.data <- data[1:100000,] # training data
test.data <- data[100001:150000,] # test data
###### Initialization ######
library(DMwR) ## K-NN imputation, SMOTE
library(caret) ##  nearZeroVar,downSample
library(GGally)
library(Hmisc)
library(MASS)
library(gbm) ## GBM
library(glmnet) ## Logistic
library(gam) ## gam
library(klaR) ## Naive Bayes
library(kernlab) ## SVM
library(randomForest)
library(e1071) 
library(corrplot) ## Correlation Matrix
library(pROC) ## Receiver Operating Characteristic Curves

###########################
###### Data Analysis ######
###########################

names(train.data)
summary(train.data)
str(train.data)
table(train.data$age,train.data$SeriousDlqin2yrs)
table(train.data$NumberOfDependents,train.data$SeriousDlqin2yrs)
table(train.data$NumberOfTime30.59DaysPastDueNotWorse,train.data$SeriousDlqin2yrs)
table(train.data$NumberOfTime60.89DaysPastDueNotWorse,train.data$SeriousDlqin2yrs)
table(train.data$NumberOfTimes90DaysLate,train.data$SeriousDlqin2yrs)
describe(train.data)

# 96:others 98:refused to say

#################################
###### Data Pre-processing ######
#################################

## train data
# Replace coded values "96" and "98"  with NA
train.data$NumberOfTime30.59DaysPastDueNotWorse[train.data$NumberOfTime30.59DaysPastDueNotWorse==96] <- NA
train.data$NumberOfTime30.59DaysPastDueNotWorse[train.data$NumberOfTime30.59DaysPastDueNotWorse==98] <- NA
train.data$NumberOfTime60.89DaysPastDueNotWorse[train.data$NumberOfTime60.89DaysPastDueNotWorse==96] <- NA
train.data$NumberOfTime60.89DaysPastDueNotWorse[train.data$NumberOfTime60.89DaysPastDueNotWorse==98] <- NA
train.data$NumberOfTimes90DaysLate[train.data$NumberOfTimes90DaysLate==96] <- NA
train.data$NumberOfTimes90DaysLate[train.data$NumberOfTimes90DaysLate==98] <- NA
# Dealing with NA values
# Replace NA in NumberOfDependents with 0
train.data$NumberOfDependents[is.na(train.data$NumberOfDependents)] <- 0

# test data
# Replace coded values "96" and "98"  with NA
test.data$NumberOfTime30.59DaysPastDueNotWorse[test.data$NumberOfTime30.59DaysPastDueNotWorse==96] <- NA
test.data$NumberOfTime30.59DaysPastDueNotWorse[test.data$NumberOfTime30.59DaysPastDueNotWorse==98] <- NA
test.data$NumberOfTime60.89DaysPastDueNotWorse[test.data$NumberOfTime60.89DaysPastDueNotWorse==96] <- NA
test.data$NumberOfTime60.89DaysPastDueNotWorse[test.data$NumberOfTime60.89DaysPastDueNotWorse==98] <- NA
test.data$NumberOfTimes90DaysLate[test.data$NumberOfTimes90DaysLate==96] <- NA
test.data$NumberOfTimes90DaysLate[test.data$NumberOfTimes90DaysLate==98] <- NA
# Dealing with NA values
# Replace NA in NumberOfDependents with 0
test.data$NumberOfDependents[is.na(test.data$NumberOfDependents)] <- 0

###### K-NN for Missing Values ######

# Use k-NN to deal with NA in training data
# For each case with any NA value it will search for its k most similar cases 
# and use the values of these cases to fill in the unknowns
set.seed(123)
dd <- knnImputation(train.data[,-1],k=5)
train.data[,2:11] <- dd

set.seed(123)
dt <- knnImputation(test.data[,-1],k=5)
test.data[,2:11] <- dt

##### SMOTE for Imbalanced Data #####

# Use SMOTE to re-sampling the training data
# The minority class is over-sampled by taking each minority class sample and 
# introducing synthetic examples along the line segments joining any/all of the 
# k minority class nearest neighbors. 
set.seed(123)
train.data$SeriousDlqin2yrs <- as.factor(train.data$SeriousDlqin2yrs)
train.data <- train.data[,c(2:11,1)] ; test.data <- test.data[,c(2:11,1)]
train.smote <- SMOTE(SeriousDlqin2yrs~.,data=train.data,perc.over=500,perc.under=120)

################################
####### Feature Analysis #######
################################

describe(train.smote)
colnames(train.smote) <- make.names(c("Utilization","Age","30_59days","DebtRatio","Income",
                           "OpenCredit","90days","RealEstate","60_89days",
                           "Dependents","Dlqin2yrs"))
colnames(test.data) <- make.names(c("Utilization","Age","30_59days","DebtRatio","Income",
                           "OpenCredit","90days","RealEstate","60_89days",
                           "Dependents","Dlqin2yrs"))
                        
###### Scatter Plot Matrix ######

pairs(train.smote[,c(1,2,4,5,6,7,8)],col=c("seagreen","indianred")[unclass(train.smote$Dlqin2yrs)])


nearZeroVar(train.smote) # no zero variance predictors

my.theme <- trellis.par.get()
mycolors <- c("seagreen","indianred")
trellis.par.set(superpose.symbol = list(col = mycolors),
                superpose.line= list(col = mycolors))
featurePlot(x = train.smote[,1:5],y = train.smote$Dlqin2yrs,plot = "density",
            ## Pass in options to xyplot() to make it prettier
            scales = list(x = list(relation="free"),y = list(relation="free")),
            adjust = 1.5, pch = "|", layout = c(5, 1),
            auto.key = list(columns = 2))
featurePlot(x = train.smote[,6:10],y = train.smote$Dlqin2yrs,plot = "density",
            ## Pass in options to xyplot() to make it prettier
            scales = list(x = list(relation="free"),y = list(relation="free")),
            adjust = 1.5, pch = "|", layout = c(5, 1),
            auto.key = list(columns = 2))

####### Correlation Matrix #######
corr.matrix <- cor(train.smote[,-11])  ## Calculate the correlation matrix
corrplot(corr.matrix, main="\nCorrelation Matrix")  ## Correlation Matrix

###### Skewness ######
skewValues <- apply(train.smote[,-11], 2, skewness);skewValues # skewness

###### Remove outliers #####
find.outlier = which(outlier(train.smote[,-11],logical=T)==TRUE,arr.ind=TRUE)
train.new <- train.smote[-find.outlier[,1],]

###### Centering and Scaling #####
trans <- preProcess(train.smote,method = c("center", "scale")); trans
train.centered <- predict(trans,train.smote)
test.centered <- predict(trans, test.data)
####### PCA ########
train.pca <- prcomp(train.smote[,-11], center=TRUE, scale. = TRUE) 
plot(train.pca,typ="l",col="indianred",lwd=2)
train.pca$rotation
biplot(train.pca)
plot(train.pca$x[,1],train.pca$x[,2],
     col=c("seagreen","indianred")[unclass(train.smote[,11])],
     xlab="PC1",ylab="PC2",main="PC1 vs PC2")

# library(devtools)
# install_github("vqv/ggbiplot")
# 
# library(ggbiplot)
# g <- ggbiplot(train.pca, obs.scale = 1, var.scale = 1, 
#               groups = train.smote$Dlqin2yrs, ellipse = TRUE, 
#               circle = TRUE)
# g <- g + scale_color_discrete(name = '')
# g <- g + theme(legend.direction = 'horizontal', 
#                legend.position = 'top')
# print(g)

####### LDA(FDA) #######
ldaMdl <- lda(x=train.centered[,-11], grouping = train.centered$Dlqin2yrs)
plot(unclass(train.centered$Dlqin2yrs)~(as.matrix(train.centered[,-11])%*%ldaMdl$scaling),
     xlab="Predictor",ylab="")

###################################
######## Modeling Training ########
###################################

####### Logistic regression #######

logisticMdl <- glmnet(as.matrix(train.smote[,-11]),train.smote$Dlqin2yrs,
                      family=c("binomial"),lambda=0)
pred.logistic.train <- predict(logisticMdl,as.matrix(train.smote[,-11]),type="class")
mean(pred.logistic.train!=train.smote$Dlqin2yrs) #[1] 0.2605011

prob.logistic.test <- predict(logisticMdl,as.matrix(test.data[,-11]),type="response")
pred.logistic.test <- predict(logisticMdl,as.matrix(test.data[,-11]),type="class")
mean(pred.logistic.test!=test.data$Dlqin2yrs) #[1] 0.1505

confusionMatrix(pred.logistic.test,test.data$Dlqin2yrs)

##          Reference
##  Prediction     0     1
##            0 40348  1233
##            1  6292  2127
###### ROC ######
roc.logistic <- roc(test.data$Dlqin2yrs,as.numeric(prob.logistic.test))
auc(roc.logistic) #Area under the curve: 0.8214
plot(roc.logistic,legacy.axes=T,main="Logistic Regression")
ci.logistic <- ci.auc(roc.logistic) # 95% CI: 0.8134-0.8294 (DeLong)

######### Naive Bayes ###########

nBayesfit <- NaiveBayes(Dlqin2yrs~.,data=train.smote,usekernel=T)  
nBayes.pred.train <- predict(nBayesfit,newdata=train.smote[,-11]) 
mean(nBayes.pred.train$class!=train.smote$Dlqin2yrs) #[1] 0.3813381

nBayes.pred.test <- predict(nBayesfit,newdata=test.data[,-11]) 
mean(nBayes.pred.test$class!=test.data$Dlqin2yrs) #[1] 0.06552
nBayes.prob <- nBayes.pred.test$posterior[,2]

confusionMatrix(nBayes.pred.test$class,test.data$Dlqin2yrs)
##          Reference
##  Prediction     0     1
##            0 46302  2938
##            1   338   422

###### ROC ######
roc.nBayes <- roc(test.data$Dlqin2yrs,as.numeric(nBayes.prob))
auc(roc.nBayes) #Area under the curve: 0.7943
plot(roc.nBayes,legacy.axes=T,main="Naive Bayes Classifier",col="1")
ci.nBayes <- ci.auc(roc.nBayes) # 95% CI: 0.7856-0.8031 (DeLong)

######### Generalized Additive Model(GAM) ##########

form <- "Dlqin2yrs ~ s(Utilization,5)+s(Age,5)+s(X30_59days,5)+s(DebtRatio,5)+
        s(Income,5)+ s(OpenCredit,5)+s(X90days,5)+s(RealEstate,5)+
        s(X60_89days,5)+s(Dependents,5)" #use smoothing splines
form <- formula(form)
gamMdl <- gam(form,data=train.smote,family=binomial) 
par(mfrow=c(2,5))
plot(gamMdl, se=T, col="royalblue3")
gamMdl.train.prob <- predict(gamMdl,train.smote,type="response")
gamMdl.test.prob <- predict(gamMdl,newdata=test.data,type="response")
gamMdl.train.err <- mean(((gamMdl.train.prob>0.5)*1)!=train.smote$Dlqin2yrs)#[1] 0.209571
gamMdl.train.auc <- auc(roc(train.smote$Dlqin2yrs,gamMdl.train.prob))# 0.8733
gamMdl.test.err <- mean(((gamMdl.test.prob>0.5))!=test.data$Dlqin2yrs)#[1] 0.20588
gamMdl.roc <- roc(test.data$Dlqin2yrs,gamMdl.test.prob)
gamMdl.test.auc <- auc(gamMdl.roc)# 0.857
ci.gamMdl <- ci.roc(gamMdl.roc) # 95% CI: 0.8502-0.8637 (DeLong)
plot(gamMdl.roc,legacy.axes=T,main="GAM",col="1",cex.main=1.6,cex.lab=1.3,cex.axis=1.3) 

## Change smoothing spline to linear function  
gam.mdl <-  gam(Dlqin2yrs ~ Utilization+s(Age,5)+s(X30_59days,5)+DebtRatio+
                 Income+ s(OpenCredit,5)+s(X90days,5)+RealEstate+
                 s(X60_89days,5)+Dependents,data=train.smote,family=binomial)
plot(gam.mdl, se=T, col="royalblue3")
gam.mdl.train.prob <- predict(gam.mdl,train.smote,type="response")
gam.mdl.test.prob <- predict(gam.mdl,newdata=test.data,type="response")
gam.mdl.train.err <- mean(((gam.mdl.train.prob>0.5)*1)!=train.smote$Dlqin2yrs)
#[1] 0.2469497
gam.mdl.train.auc <- auc(roc(train.smote$Dlqin2yrs,gam.mdl.train.prob))# 0.8254
gam.mdl.test.err <- mean(((gam.mdl.test.prob>0.5))!=test.data$Dlqin2yrs)
#[1] 0.16652
gam.mdl.roc <- roc(test.data$Dlqin2yrs,gam.mdl.test.prob)
gam.mdl.test.auc <- auc(gam.mdl.roc)# 0.8306
plot(gam.mdl.roc,legacy.axes=T,main="GAM",add=T,col="2") 
legend("bottomright",legend=c("GAM with all smooth splines","GAM with some linear functions"),
       col=c(1:2),lty=1,lwd=2,cex=.8)
par(mfrow=c(1,1))

######### SVM ###########
# set.seed(123)
# # fitControl <- trainControl(method = "repeatedcv", number = 5,
# #                            repeats = 5, classProbs = TRUE,
# #                            summaryFunction = twoClassSummary)
# radial.grid <- expand.grid(gamma = 10^(seq(-6:0)), C = 10^(seq(-3,3)))
# # colnames(train.smote) <- make.names(names(train.smote),unique = T)
# nfolds <- 5
# n.train <- nrow(train.smote)
# s <- split(sample(n.train),rep(1:nfolds,length=n.train))
# n <- dim(radial.grid)[1]
# tune.summary <- data.frame(gamma=rep(NA,n),cost=rep(NA,n),
#                            CV.err=rep(NA,n),AUC=rep(NA,n))
# for(i in n){
#   random.pred <- rep(NA,n.train) 
#   auc.pred <- rep(NA,nfolds)
#   for(j in seq(nfolds)){
#     random.temp <- svm(Dlqin2yrs~.,data=train.smote[-s[[j]],],kernel="radial",
#                        gamma=radial.grid[i,1],cost=radial.grid[i,2],scale=T,
#                        probability=T)
#     random.pred[s[[j]]] <- predict(random.temp,newdata=train[s[[j]],],type="Class")
#   }
#   
# }
# svm.temp <- svm(Dlqin2yrs~.,data=train.smote,kernel="radial",
#                    gamma=10^-5,cost=1,scale=T,
#                    probability=T)

############ Boosting(GBM) ############
train.boost <- train.smote
train.boost$Dlqin2yrs <- unclass(train.smote$Dlqin2yrs)-1
set.seed(123)
###### CV ######
nfolds <- 5
n.train <- nrow(train.smote)
s <- split(sample(n.train),rep(1:nfolds,length=n.train))
boost.grid <- c(3000,4000,5000,6000,7000,8000,9000)
nn <- length(boost.grid)
boost.cv.auc <- rep(NA,nn)
boost.cv.err <- rep(NA,nn)

for(i in 1:nn){ 
  boost.auc.temp <- rep(NA,5)
  boost.err.temp <- rep(NA,5)
  for(j in seq(nfolds)){
    boost.temp <- gbm(Dlqin2yrs~ .,distribution = "adaboost",
                         data = train.boost[-s[[j]],],n.trees=boost.grid[i],
                         interaction.depth = 4,
                         shrinkage=0.01, verbose=F)
    boost.prob <- predict.gbm(boost.temp, newdata=train.boost[s[[j]],], 
                               n.trees=boost.grid[i],type="response")
    boost.err.temp[j] <- mean(((boost.prob>0.5))!=train.boost[s[[j]],]$Dlqin2yrs)
    boost.roc.temp <- roc(train.boost[s[[j]],]$Dlqin2yrs,boost.prob)
    boost.auc.temp[j] <- auc(boost.roc.temp)
  }
  boost.cv.err[i] <- mean(boost.err.temp)
  boost.cv.auc[i] <- mean(boost.auc.temp)
}



################### Train and Test #######################
boost.train.err <- rep(NA,nn)
boost.train.auc <- rep(NA,nn)
boost.test.err <- rep(NA,nn)
boost.test.auc <- rep(NA,nn)
for(i in 1:nn){
boost.temp <- gbm(Dlqin2yrs ~ .,distribution = "adaboost",data = train.boost,
                  n.trees = boost.grid[i],interaction.depth=4,
                  shrinkage = 0.01, verbose = F) 
boost.train.prob <- predict.gbm(boost.temp, train.boost, n.trees=boost.grid[i],
                                type="response")
boost.test.prob <-  predict.gbm(boost.temp,newdata=test.data,n.trees=boost.grid[i],
                                type="response")
# train
boost.train.err[i] <- mean((boost.train.prob>0.5)!=train.boost$Dlqin2yrs)
boost.roc.train <- roc(train.boost$Dlqin2yrs,boost.train.prob)
boost.train.auc[i] <- auc(boost.roc.train)
# test
boost.test.err[i] <- mean((boost.test.prob>0.5)!=test.data$Dlqin2yrs)
boost.roc.test <- roc(test.data$Dlqin2yrs,boost.test.prob)
boost.test.auc[i] <- auc(boost.roc.test)
}
###########
best.boost <- boost.temp
best.boost.cv.err <- min(boost.cv.err) 
best.boost.cv.auc <- max(boost.cv.auc)
best.boost.test.err <- boost.test.err[which.min(boost.cv.auc)]
best.boost.test.auc <- boost.test.auc[which.min(boost.cv.auc)]
best.boost.test.roc <- boost.roc.test 
ci.boost <- ci.auc(best.boost.test.roc)
boost.table <- cbind(boost.grid,boost.cv.err,boost.cv.auc,boost.test.err,boost.test.auc)
colnames(boost.table) <- c("n.trees","cv err","cv auc","eval err","eval auc")
boost.table
par(mfrow=c(1,1),mar=c(4.5,4.5,2,1),oma=c(0,0,2,0))
plot(x=boost.grid,y=boost.cv.auc,xlab="Boosting Iterations",ylab="AUC",
     col=2,pch=19,ylim=c(0.85,1),main="GBM ROC Curve AUC")
points(x=boost.grid,y=boost.train.auc,col=3,pch=19)
points(x=boost.grid,y=boost.test.auc,col=4,pch=19)
lines(x=boost.grid,y=boost.cv.auc,col=2)
lines(x=boost.grid,y=boost.train.auc,col=3)
lines(x=boost.grid,y=boost.test.auc,col=4)
points(x=boost.grid[which.max(boost.test.auc)],y=max(boost.test.auc),col=4,cex=2,pch=19)
legend("topleft",legend=c("CV AUC","Train AUC","Evaluation AUC"),
       col=c(2:4),lty=1,lwd=2,cex=1,text.width=1200)

plot(x=boost.grid,y=boost.cv.err,xlab="Boosting Iterations",ylab="Errors",
     col=2,pch=19,ylim=c(0,0.2),main="GBM Misclassification Error")
points(x=boost.grid,y=boost.train.err,col=3,pch=19)
points(x=boost.grid,y=boost.test.err,col=4,pch=19)
lines(x=boost.grid,y=boost.cv.err,col=2)
lines(x=boost.grid,y=boost.train.err,col=3)
lines(x=boost.grid,y=boost.test.err,col=4)
legend("topleft",legend=c("CV Error","Train Error","Evaluation Error"),
       col=c(2:4),lty=1,lwd=2,cex=1,text.width=1000)

########## Random Forest ########

nfolds <- 5
n.train <- nrow(train.smote)
s <- split(sample(n.train),rep(1:nfolds,length=n.train))
floor_n=1
top_n=6
m<-floor_n:top_n
m_n=top_n-floor_n+1
random.cv.err <- rep(NA,m_n)
random.train.err <- rep(NA,m_n)
random.test.err <- rep(NA,m_n)
random.auc.train.cv <- rep(NA,m_n)
random.auc.train <- rep(NA,m_n)
random.auc.test <- rep(NA,m_n)


for(i in 1:m_n)
{
  random.pred <- rep(NA,n.train) 
  random.auc.train.cv.temp <- rep(NA,nfolds)
  for(j in 1:5)
  { 
    cat("m =",m[i],"cross validation=",j,"\n") 
    random.temp <- randomForest(Dlqin2yrs~.,data=train.smote[-s[[j]],],ntree=250,mtry=m[i],importance=TRUE)
    random.pred[s[[j]]] <- predict(random.temp,newdata=train.smote[s[[j]],])
    random.prob.train <-predict(random.temp,newdata=train.smote[s[[j]],],type="prob")
    random.roc <- roc(train.smote[s[[j]],]$Dlqin2yrs,as.numeric(random.prob.train[,2]))
    random.auc.train.cv.temp[j]<-auc(random.roc)
  }
  ## AUC
  random.cv.err[i] <- mean(random.pred!=train.smote[,"Dlqin2yrs"]) ## misclassification error
  random.auc.train.cv[i] <- mean(random.auc.train.cv.temp)
}
for(i in 1:m_n)
{
  cat("m =",m[i],"\n")
  #train error and auc
  random.temp <- randomForest(Dlqin2yrs~.,data=train.smote,ntree=250,mtry=m[i],importance=TRUE)
  random.pred.train <- predict(random.temp)
  random.prob.train <- predict(random.temp,type="prob")
  random.roc <- roc(train.smote$Dlqin2yrs,as.numeric(random.prob.train[,2]))
  random.auc.train[i] <- auc(random.roc)
  random.train.err[i] <- mean(random.pred.train!=train.smote[,"Dlqin2yrs"])
  #test error and auc
  random.pred.test <- predict(random.temp,newdata=test.data)
  random.prob.test<- predict(random.temp,newdata=test.data,type="prob")
  random.roc <- roc(test.data$Dlqin2yrs,as.numeric(random.prob.test[,2]))
  random.auc.test[i] <- auc(random.roc)
  random.test.err[i] <- mean(random.pred.test!=test.data[,"Dlqin2yrs"])
}
# Plot
plot(floor_n:top_n,random.cv.err,pch=19,col=2,ylim=c(0.01,1.2),ylab="Errors",xlab="m",main="Random Forest")
points(floor_n:top_n,random.train.err,pch=19,col=3)
points(floor_n:top_n,random.test.err,pch=19,col=4)
legend("topright",legend=c("CV error","train error","test error"),col=c(2:4),lty=1,lwd=2,cex=.8)
random.train.err[which.min(random.cv.err)]
random.test.err[which.min(random.cv.err)]
#plot auc 
plot(floor_n:top_n,random.auc.train.cv,pch=19,col=2,ylim=c(0.83,1.3),ylab="auc",xlab="m",main="Random Forest AUC")
points(floor_n:top_n,random.auc.train,pch=19,col=3)
points(floor_n:top_n,random.auc.test,pch=19,col=4)
legend("topright",legend=c("CV auc","train auc","test auc"),col=c(2:4),lty=1,lwd=2,cex=.8)
which.max(random.auc.train.cv)

nfolds <- 5
n.train <- nrow(train.smote)
s <- split(sample(n.train),rep(1:nfolds,length=n.train))
floor_n=2
top_n=6
m<-floor_n:top_n
m_n=top_n-floor_n+1
random.cv.err.500 <- rep(NA,m_n)
random.train.err.500 <- rep(NA,m_n)
random.test.err.500 <- rep(NA,m_n)
random.auc.train.cv.500 <- rep(NA,m_n)
random.auc.train.500 <- rep(NA,m_n)
random.auc.test.500 <- rep(NA,m_n)

ptm<-proc.time()
for(i in 1:m_n)
{
  random.pred <- rep(NA,n.train) 
  random.auc.train.cv.temp <- rep(NA,nfolds)
  for(j in 1:5)
  { 
    cat("m =",m[i],"cross validation=",j,"\n") 
    random.temp <- randomForest(Dlqin2yrs~.,data=train.smote[-s[[j]],],ntree=500,mtry=m[i],importance=TRUE)
    random.pred[s[[j]]] <- predict(random.temp,newdata=train.smote[s[[j]],])
    random.prob.train <-predict(random.temp,newdata=train.smote[s[[j]],],type="prob")
    random.roc <- roc(train.smote[s[[j]],]$Dlqin2yrs,as.numeric(random.prob.train[,2]))
    random.auc.train.cv.temp[j]<-auc(random.roc)
  }
  ## AUC
  random.cv.err.500[i] <- mean(random.pred!=train.smote[,"Dlqin2yrs"]) ## misclassification error
  random.auc.train.cv.500[i] <- mean(random.auc.train.cv.temp)
}

for(i in 1:m_n)
{
  cat("m =",m[i],"\n")
  #train error and auc
  random.temp <- randomForest(Dlqin2yrs~.,data=train.smote,ntree=500,mtry=m[i],importance=TRUE)
  random.pred.train <- predict(random.temp)
  random.prob.train <- predict(random.temp,type="prob")
  random.roc <- roc(train.smote$Dlqin2yrs,as.numeric(random.prob.train[,2]))
  random.auc.train.500[i] <- auc(random.roc)
  random.train.err.500[i] <- mean(random.pred.train!=train.smote[,"Dlqin2yrs"])
  #test error and auc
  random.pred.test <- predict(random.temp,newdata=test.data)
  random.prob.test<- predict(random.temp,newdata=test.data,type="prob")
  random.roc <- roc(test.data$Dlqin2yrs,as.numeric(random.prob.test[,2]))
  random.auc.test.500[i] <- auc(random.roc)
  random.test.err.500[i] <- mean(random.pred.test!=test.data[,"Dlqin2yrs"])
}
#plot auc 
plot(floor_n:top_n,random.auc.train.cv.500,pch=19,col=2,ylim=c(0.83,1.1),
     ylab="AUC",xlab="m",main="Random Forest ROC Curve AUC (ntree=500)")
lines(floor_n:top_n,random.auc.train.cv.500,col=2)
lines(floor_n:top_n,random.auc.train.500,col=3)
lines(floor_n:top_n,random.auc.test.500,col=4)
points(floor_n:top_n,random.auc.train.500,pch=19,col=3)
points(floor_n:top_n,random.auc.test.500,pch=19,col=4)
legend("topleft",legend=c("CV AUC","Train AUC","Evaluation AUC"),col=c(2:4)
       ,lty=1,lwd=2,cex=1,text.width=1.3)
points(x=1,y=max(random.auc.test.500),pch=19,col=4,cex=2)
# Plot errors
plot(floor_n:top_n,random.cv.err.500,pch=19,col=2,ylim=c(0.01,1.2),
     ylab="Errors",xlab="m",main="Random Forest Missclassification Error (ntree=500)")
lines(floor_n:top_n,random.cv.err.500,col=2)
lines(floor_n:top_n,random.train.err.500,col=3)
points(floor_n:top_n,random.train.err.500,pch=19,col=3)
lines(floor_n:top_n,random.test.err.500,col=4)
points(floor_n:top_n,random.test.err.500,pch=19,col=4)
legend("topright",legend=c("CV Error","Train Error","Evaluation Error"),col=c(2:4),
       lty=1,lwd=2,cex=.8)
which.max(random.auc.train.cv)

best.rf <- randomForest(Dlqin2yrs~.,data=train.smote,ntree=500,mtry=1,importance=TRUE)
best.rf.prob <- predict(best.rf,newdata=test.data,type="prob")
best.rf.roc <- roc(test.data$Dlqin2yrs,as.numeric(best.rf.prob[,2]))
ci.best.rf <- ci.auc(best.rf.roc)
auc.table <- rbind(random.cv.err,random.test.err,random.auc.train.cv,
                   random.auc.test,random.cv.err.500,random.test.err.500,
                   random.auc.train.cv.500,random.auc.test.500)
auc.table <- t(auc.table)
colnames(auc.table) <- c("cv err_250","eval err_250","cv auc_250","eval auc_250",
                         "cv err_500","eval err_500","cv auc_500","eval auc_500")
rownames(auc.table) <- c("m=1","m=2","m=3","m=4","m=5","m=6")
auc.table

############ Summary #############
par(mfrow=c(1,1),mar=c(4.5,4.5,2.5,1),oma=c(0,6,1,0))
boxplot(ci.nBayes,ci.logistic,ci.best.rf,ci.gamMdl,ci.boost,horizontal=T,
        names=c("Naive Bayes","Logistic regression","Random Forest","GAM","GBM"),
        las=1,border=F,xlab="AUC",main="Evaluation set ROC Curve AUC")

abline(h=5,col="grey")
lines(x=c(ci.boost[1],ci.boost[3]),y=c(5,5),col="red")
points(x=ci.boost[2],y=5,pch=19,col="red")
points(x=c(ci.boost[1],ci.boost[3]),y=c(5,5),pch="|",col="red")

abline(h=4,col="grey")
lines(x=c(ci.gamMdl[1],ci.gamMdl[3]),y=c(4,4))
points(x=ci.gamMdl[2],y=4,pch=19)
points(x=c(ci.gamMdl[1],ci.gamMdl[3]),y=c(4,4),pch="|")

abline(h=3,col="grey")
lines(x=c(ci.best.rf[1],ci.best.rf[3]),y=c(3,3))
points(x=ci.best.rf[2],y=3,pch=19)
points(x=c(ci.best.rf[1],ci.best.rf[3]),y=c(3,3),pch="|")

abline(h=2,col="grey")
lines(x=c(ci.logistic[1],ci.logistic[3]),y=c(2,2))
points(x=ci.logistic[2],y=2,pch=19)
points(x=c(ci.logistic[1],ci.logistic[3]),y=c(2,2),pch="|")

abline(h=1,col="grey")
lines(x=c(ci.nBayes[1],ci.nBayes[3]),y=c(1,1))
points(x=ci.nBayes[2],y=1,pch=19)
points(x=c(ci.nBayes[1],ci.nBayes[3]),y=c(1,1),pch="|")

AUC <- c(ci.boost[2],ci.gamMdl[2],ci.best.rf[2],ci.logistic[2],ci.nBayes[2])
Thres.boost <- coords(best.boost.test.roc, x = "best", best.method = "closest.topleft")
Thres.gam <- coords(gamMdl.roc, x = "best", best.method = "closest.topleft")
Thres.rf <- coords(best.rf.roc, x = "best", best.method = "closest.topleft")
Thres.logistic <- coords(roc.logistic, x = "best", best.method = "closest.topleft")
Thres.nBayes <- coords(roc.nBayes, x = "best", best.method = "closest.topleft")
Thres.all <- rbind(Thres.boost,Thres.gam,Thres.rf,Thres.logistic,Thres.nBayes)
AUC.sum <- cbind(AUC,Thres.all)
row.names(AUC.sum) <- c("GBM","GAM","Random Forest","Logistic Regression"
                        ,"Naive Bayes")

##### Predictor variable importance #####
gbmImp <- varImp(boost.temp,numTrees=10000)
gbm.Relative.Imp <- data.frame(gbmImp[(order(gbmImp)),])
gbm.Relative.Imp <- data.frame((gbm.Relative.Imp[,1]/gbm.Relative.Imp[10,1])*100)
row.names(gbm.Relative.Imp) <- row.names(gbmImp)[(order(gbmImp))]
colnames(gbm.Relative.Imp) <- c("Relative Importance")
gbm.Relative.Imp
par(mfrow=c(1,1),mar=c(4.5,4.5,2.5,1),oma=c(0,6,1,0),bg="NA")
barplot(gbm.Relative.Imp[,1],names=row.names(gbm.Relative.Imp),horiz=T,las=1,
        main="Relative Importance From GBM",col="red",border="NA")

############################################
############ Plots for Powerpoint ##########
############################################

par(bg="NA",bty="n")
# plot(roc.logistic,legacy.axes=T,col="steelblue1",main="ROC Curve",col.lab="orange",
#      col.sub="orange",col.main="orange",col.axis="orange",fg="orange",
#      cex.main=1.7,cex.lab=1.3,cex.axis=1.3)
# 
# plot(roc.nBayes,legacy.axes=T,col="gold",add=T)
# 
# plot(best.rf.roc,legacy.axes=T,col="springgreen1",add=T)
# plot(gamMdl.roc,legacy.axes=T,col="greenyellow",add=T) 
# plot(best.boost.test.roc,legacy.axes=T,col="firebrick1",add=T)
# legend(x="bottomright",legend=c("GBM","GAM","Random Forest","Logistic","Naive Bayes"),
#        col=c("firebrick1","greenyellow","springgreen1","steelblue1","gold"),
#        lty=1,border="NA")

plot(roc.logistic,legacy.axes=T,col="steelblue1",main="ROC Curve",
     cex.main=1.6,cex.lab=1.3,cex.axis=1.3)

plot(roc.nBayes,legacy.axes=T,col="gold",add=T)

plot(best.rf.roc,legacy.axes=T,col="springgreen1",add=T)
plot(gamMdl.roc,legacy.axes=T,col="seagreen",add=T) 
plot(best.boost.test.roc,legacy.axes=T,col="firebrick1",add=T)
legend(x="bottomright",legend=c("GBM","GAM","Random Forest","Logistic","Naive Bayes"),
       col=c("firebrick1","seagreen","springgreen1","steelblue1","gold"),
       lty=1,border="NA")

# Boosting Plot
par(bg="NA",bty="n")
plot(x=boost.grid,y=boost.cv.auc,xlab="Boosting Iterations",ylab="AUC",
     col.lab="white",col.sub="white",col.main="white",col.axis="white",
     fg="white", cex.main=1.6,cex.lab=1.3,cex.axis=1.3,
     col="brown1",pch=19,ylim=c(0.85,0.97),main="GBM ROC Curve AUC")
points(x=boost.grid,y=boost.train.auc,col="greenyellow",pch=19)
points(x=boost.grid,y=boost.test.auc,col="lightskyblue",pch=19)
lines(x=boost.grid,y=boost.cv.auc,col="brown1",lwd=2)
lines(x=boost.grid,y=boost.train.auc,col="greenyellow",lwd=2)
lines(x=boost.grid,y=boost.test.auc,col="lightskyblue",lwd=2)
points(x=boost.grid[which.max(boost.test.auc)],y=max(boost.test.auc),
       pch=19,col="orange",cex=2)
# Random Plot
plot(floor_n:top_n,random.auc.train.cv.500,pch=19,col="brown1",
     ylim=c(0.83,1), col.lab="white", col.sub="white",col.main="white",
     col.axis="white",fg="white",cex.main=1.6,cex.lab=1.3,cex.axis=1.3,
     ylab="AUC",xlab="m",main="Random Forest ROC Curve AUC (ntree=500)")
lines(floor_n:top_n,random.auc.train.cv.500,col="brown1",lwd=2)
lines(floor_n:top_n,random.auc.train.500,col="greenyellow",lwd=2)
lines(floor_n:top_n,random.auc.test.500,col="lightskyblue",lwd=2)
points(floor_n:top_n,random.auc.train.500,pch=19,col="greenyellow")
points(floor_n:top_n,random.auc.test.500,pch=19,col="lightskyblue")
points(x=2,y=max(random.auc.test.500),pch=19,col="orange",cex=2)

par(mfrow=c(1,1),mar=c(4.5,4.5,2.5,1),oma=c(0,9,1,0),bty="l",bg="NA",fg="white",
    col.lab="white",col.sub="white",col.main="white", col.axis="white")
boxplot(ci.nBayes,ci.logistic,ci.best.rf,ci.gamMdl,ci.boost,horizontal=T,
        names=c("Naive Bayes","Logistic regression","Random Forest","GAM","GBM"),
        las=1,border=F,xlab="AUC",main="Evaluation set ROC Curve AUC",
        cex.main=1.6,cex.lab=1.3,cex.axis=1.3)

abline(h=5,col="grey")
lines(x=c(ci.boost[1],ci.boost[3]),y=c(5,5),col="plum1",lwd=3)
points(x=ci.boost[2],y=5,pch=19,col="plum1")
points(x=c(ci.boost[1],ci.boost[3]),y=c(5,5),pch="|",col="plum1")

abline(h=4,col="grey")
lines(x=c(ci.gamMdl[1],ci.gamMdl[3]),y=c(4,4),col="lightskyblue1",lwd=3)
points(x=ci.gamMdl[2],y=4,pch=19,col="lightskyblue1")
points(x=c(ci.gamMdl[1],ci.gamMdl[3]),y=c(4,4),pch="|",col="lightskyblue1")

abline(h=3,col="grey")
lines(x=c(ci.best.rf[1],ci.best.rf[3]),y=c(3,3),col="lightskyblue1",lwd=3)
points(x=ci.best.rf[2],y=3,pch=19,col="lightskyblue1")
points(x=c(ci.best.rf[1],ci.best.rf[3]),y=c(3,3),pch="|",col="lightskyblue1")

abline(h=2,col="grey")
lines(x=c(ci.logistic[1],ci.logistic[3]),y=c(2,2),col="lightskyblue1",lwd=3)
points(x=ci.logistic[2],y=2,pch=19,col="lightskyblue1")
points(x=c(ci.logistic[1],ci.logistic[3]),y=c(2,2),pch="|",col="lightskyblue1")

abline(h=1,col="grey")
lines(x=c(ci.nBayes[1],ci.nBayes[3]),y=c(1,1),col="lightskyblue1",lwd=3)
points(x=ci.nBayes[2],y=1,pch=19,col="lightskyblue1")
points(x=c(ci.nBayes[1],ci.nBayes[3]),y=c(1,1),pch="|",col="lightskyblue1")


par(mfrow=c(2,5),mar=c(4.5,4.5,1,1),oma=c(0,0,0,0),col.lab="white",
    col.sub="white",col.main="white",col.axis="white",fg="white",bg="NA")
plot(gamMdl,se=T, col="darkseagreen1",lwd=2)


par(mfrow=c(1,1),mar=c(4.5,4.5,2.5,1),oma=c(0,6,1,0),bg="NA",col.lab="white",
    col.sub="white",col.main="white",col.axis="white",fg="white",bg="NA")
barplot(gbm.Relative.Imp[,1],names=row.names(gbm.Relative.Imp),horiz=T,las=1,
        main="Relative Importance From GBM",col="khaki1",border="NA",cex.axis=1,
        cex.names=1.2,cex.main=1.5)
####################
par.default <- par()
par(par.default)

