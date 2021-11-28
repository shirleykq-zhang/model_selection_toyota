toyota.df <- read.csv("ToyotaCorolla.csv") 

head(toyota.df[,"Cylinders"])

dim(toyota.df)

set.seed(1)  # set seed for reproducing the partition
drops <- c("Model","Id", "Mfg_Year", "Cylinders")
toyota.df <- toyota.df[ , !(names(toyota.df) %in% drops)]

X = model.matrix(toyota.df$Price~., toyota.df)[,-1]
y = array(toyota.df$Price, dim = c(dim(toyota.df)[1], 1))

train.index <- sample(dim(toyota.df)[1], dim(toyota.df)[1]*2/3)


X.train <- as.data.frame(X[train.index, ])
y.train <- as.data.frame(y[train.index, ])
X.test <- as.data.frame(X[-train.index, ])
y.test <- as.data.frame(y[-train.index, ])

train.df <- data.frame(X.train, y.train)
names(train.df)[ncol(train.df)] <- 'Price'

valid.df <- data.frame(X.test, y.test)
names(valid.df)[ncol(valid.df)] <- 'Price'

dim(train.df)
#train.df

dim(valid.df)

toyota.lm <- lm(Price ~ ., data = train.df)

summary(toyota.lm)

if (!require(leaps)) install.packages("leaps")
if (!require(leaps)) install.packages("forecast")

toyota.lm.pred <- predict(toyota.lm, valid.df)

library(leaps)
regfit.full= regsubsets(Price~.,train.df)
summary(regfit.full)

regfit.full= regsubsets(Price~.,train.df,nvmax=42)
reg.summary = summary(regfit.full)

reg.summary$rsq 
plot(reg.summary$rsq)

par(mfrow=c(2,2))
plot(reg.summary$rss, xlab="number of Varaibles",ylab="RSS",type="l")
plot(reg.summary$adjr2, xlab="number of Varaibles",ylab="Rsq Adjusted",type="l")
ind<-which.max(reg.summary$adjr2)
points(ind,reg.summary$adjr2[ind],col="red",cex=2,pch=20) #plot a red dot at maximum Adjusted R2

plot(reg.summary$cp, xlab="number of Varaibles",ylab="Cp",type="l")
ind<-which.min(reg.summary$cp) # find the location of the minimum Cp
points(ind,reg.summary$cp[ind],col="red",cex=2,pch=20) #plot a red dot at minimum Cp

plot(reg.summary$bic, xlab="number of Varaibles",ylab="BIC",type="l")
ind<-which.min(reg.summary$bic) # find the location of the minimum BIC
points(ind,reg.summary$bic[ind],col="red",cex=2,pch=20) #plot a red dot at minimum BIC

# the best model options
which.min(reg.summary$bic)
which.min(reg.summary$cp)
which.max(reg.summary$adjr2)

best_num = which.min(reg.summary$bic)
best_num

# the coefficients associated with the lowest rss based on bic - with 18 variables
coef(regfit.full,best_num)

regfit.fwd=regsubsets(Price~.,data=toyota.df,nvmax=42,method="forward")
fwd.summary=summary(regfit.fwd)
regfit.bwd=regsubsets(Price~.,data=toyota.df,nvmax=42,method="backward")
bwd.summary=summary(regfit.fwd)

#the best model options for forward selection
which.min(fwd.summary$bic)
which.min(fwd.summary$cp)
which.max(fwd.summary$adjr2)

#the best model options for backward selection
which.min(bwd.summary$bic)
which.min(bwd.summary$cp)
which.max(bwd.summary$adjr2)

coef(regfit.fwd,which.min(fwd.summary$bic))

coef(regfit.bwd,which.min(bwd.summary$bic))

x=scale(model.matrix(Price~.,toyota.df)[,-1]) #this creates a matrix of predictors, and converts all the categorical variables to dummy variables
# the [,-1] indexing removes the intercept from the matrix. We do this because it will be added back automatically in later methods
y=toyota.df$Price #create the response variable

head(x)

if (!require(glmnet)) install.packages("glmnet")
library(glmnet)

grid=10^seq(10,-2,length=100) #create grid values for lambda from 10^10 to 10^-2
ridge.mod=glmnet(x,y,alpha=0,lambda=grid) #implements ridge regression when alpha=0

dim(coef(ridge.mod))

predict(ridge.mod,s=100,type="coefficients")[1:43,] 

set.seed(1)
train=sample(1:nrow(x),nrow(x)/2) #generate a random index of half of the data, to use as training set
test=(-train) 
y.test=y[test] 

ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid,thresh=1e-12) #estimate ridge regression on training data
ridge.pred=predict(ridge.mod,s=4,newx=x[test,]) #generate predicted values on test data using lambda=4 (note this use the predict funciton we wrote above)

ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,])

set.seed(1) # the choice of folds is random, so we want to set the seed to get reproduible results
cv.out=cv.glmnet(x[train,],y[train],alpha=0) #performs cross-validation, default is 10-fold
plot(cv.out)
bestlam=cv.out$lambda.min #find lambda with lowest MSE (cross-validation)
bestlam

#predicting using the best lambda
ridge.pred=predict(ridge.mod,s=bestlam,newx=x[test,])

out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:43,]

lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=grid)
plot(lasso.mod)

set.seed(1)
cv.out=cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min

#predicting using the best lambda with min sq error
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])

out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:43,]
lasso.coef

install.packages('pls')
library(pls)

set.seed(1)
train=sample(1:nrow(toyota.df),nrow(toyota.df)/5)
test=(-train)

pcr.fit=pcr(Price~.,data=toyota.df,subset=train,scale=TRUE,validation="CV") 
validationplot(pcr.fit,val.type="MSEP")

summary(pcr.fit)

#choose 35 components because it gives least squares
pcr.fit=pcr(Price~.,data=toyota.df,scale=TRUE,ncomp=35) 
summary(pcr.fit)

set.seed(1)
pls.fit=plsr(Price~.,data=toyota.df,subset=train,scale=TRUE,validation="CV") 
summary(pls.fit)

validationplot(pls.fit,val.type="MSEP")

#using m=2 according to the graph
pls.fit=plsr(Price~.,data=toyota.df,scale=TRUE,ncomp=2) 
summary(pls.fit)


