# (1)    INTRODUCTION    #
# The goal of this project is to build a categorical machine learning algorithm
# with 85 variables (socio-demographic and product ownership
# variables) to learn from, and on the strength of what it has learnt, predict 
# which customer will buy a caravan insurance policy in the future. This model 
# should possess minimal error.

#IMPORTING LIBRARIES#
library('pls')
library(MASS)
library(glmnet)
library('dplyr')
library(ROSE)
install.packages(randomForest)
library(randomForest)
library(e1071)

#IMPORTING DATA#
#Here, we import 3 data sets, the first we will use to train our model, the second
#we will use for the prediction and the last is the actual values we will use
#to compare with the predicted values to find out the error margin and model
#performance

#DATA IMPORT AND TRANSFORMATION CODE#
FIRST = read.csv('C:/Users/Ifeanyi Uzowuru/Documents/DATA.TRAIN.csv',header=F)
FIRST=FIRST[,-c(87)]
SECOND=read.csv('C:/Users/Ifeanyi Uzowuru/Documents/DATA.TEST.csv',header=F)
THIRD=read.csv('C:/Users/Ifeanyi Uzowuru/Documents/DATA.TARGET.csv',header=F)

#INSERTING THEIR RESPECTIVE Column names
colnames(FIRST)<-c("MOSTYPE",  "MAANTHUI" ,"MGEMOMV" , "MGEMLEEF", "MOSHOOFD" ,"MGODRK",
                      "MGODPR" ,"MGODOV" ,  "MGODGE"   ,"MRELGE"   ,"MRELSA"  , "MRELOV"  ,
                      "MFALLEEN" ,"MFGEKIND" ,"MFWEKIND", "MOPLHOOG", "MOPLMIDD", "MOPLLAAG"
                      ,"MBERHOOG", "MBERZELF", "MBERBOER" ,"MBERMIDD", "MBERARBG","MBERARBO",
                      "MSKA",     "MSKB1",   "MSKB2"    ,"MSKC"  ,   "MSKD"     ,"MHHUUR",
                      "MHKOOP", "MAUT1"   , "MAUT2"   , "MAUT0"   , "MZFONDS",  "MZPART",
                      "MINKM30"  ,"MINK3045", "MINK4575", "MINK7512" ,"MINK123M", "MINKGEM",
                      "MKOOPKLA" ,"PWAPART"  ,"PWABEDR" , "PWALAND" , "PPERSAUT", "PBESAUT",
                      "PMOTSCO" , "PVRAAUT",  "PAANHANG" ,"PTRACTOR", "PWERKT" ,  "PBROM",
                      "PLEVEN"  ,"PPERSONG", "PGEZONG",  "PWAOREG",  "PBRAND"   ,"PZEILPL"
                      ,"PPLEZIER" ,"PFIETS"  , "PINBOED" , "PBYSTAND" ,"AWAPART" , "AWABEDR"
                      , "AWALAND"  ,"APERSAUT", "ABESAUT" , "AMOTSCO",  "AVRAAUT","AAANHANG",
                      "ATRACTOR" ,"AWERKT"  , "ABROM"    ,"ALEVEN"  , "APERSONG" ,"AGEZONG" ,
                      "AWAOREG", "ABRAND",   "AZEILPL",  "APLEZIER", "AFIETS"   ,"AINBOED"
                      ,"ABYSTAND" ,"CARAVAN")
colnames(SECOND)<-c("MOSTYPE",  "MAANTHUI" ,"MGEMOMV" , "MGEMLEEF", "MOSHOOFD" ,"MGODRK",
                       "MGODPR" ,"MGODOV" ,  "MGODGE"   ,"MRELGE"   ,"MRELSA"  , "MRELOV"  ,
                       "MFALLEEN" ,"MFGEKIND" ,"MFWEKIND", "MOPLHOOG", "MOPLMIDD", "MOPLLAAG"
                       ,"MBERHOOG", "MBERZELF", "MBERBOER" ,"MBERMIDD", "MBERARBG","MBERARBO",
                       "MSKA",     "MSKB1",   "MSKB2"    ,"MSKC"  ,   "MSKD"     ,"MHHUUR",
                       "MHKOOP", "MAUT1"   , "MAUT2"   , "MAUT0"   , "MZFONDS",  "MZPART",
                       "MINKM30"  ,"MINK3045", "MINK4575", "MINK7512" ,"MINK123M", "MINKGEM",
                       "MKOOPKLA" ,"PWAPART"  ,"PWABEDR" , "PWALAND" , "PPERSAUT", "PBESAUT",
                       "PMOTSCO" , "PVRAAUT",  "PAANHANG" ,"PTRACTOR", "PWERKT" ,  "PBROM",
                       "PLEVEN"  ,"PPERSONG", "PGEZONG",  "PWAOREG",  "PBRAND"   ,"PZEILPL"
                       ,"PPLEZIER" ,"PFIETS"  , "PINBOED" , "PBYSTAND" ,"AWAPART" , "AWABEDR"
                       , "AWALAND"  ,"APERSAUT", "ABESAUT" , "AMOTSCO",  "AVRAAUT","AAANHANG",
                       "ATRACTOR" ,"AWERKT"  , "ABROM"    ,"ALEVEN"  , "APERSONG" ,"AGEZONG" ,
                       "AWAOREG", "ABRAND",   "AZEILPL",  "APLEZIER", "AFIETS"   ,"AINBOED"
                       ,"ABYSTAND")
colnames(THIRD)<-"CARAVAN"



# (2)    DATA EXPLORATION #
#Here, we try to understand the structure,type of data sets,distribution and 
#check out for missing values


#Checking Dimension#
dim(FIRST)  #First data has 5822 rows and 86 columns 
dim(SECOND)   #Second data has 4000 rows and 85 columns 
dim(THIRD) #Third data has 4000 rows and 1 column 

#STRUCTURE OF DATA SET#
str(FIRST) #Data type for 'FIRST data' are all integers, second and third will 
#have similar data type since they are the test and validation data set, so we 
#dont have to check.
#However we need to change the structure of the target class (CARAVAN) to factor
#in the three data sets
FIRST$CARAVAN=as.factor(FIRST$CARAVAN)
THIRD$CARAVAN=as.factor(THIRD$CARAVAN)
str(FIRST$CARAVAN)

#MISSING DATA#
sum(is.na(FIRST))
sum(is.na(SECOND))
sum(is.na(THIRD))
#There are no missing values in the three data set#

#SPLITTING FIRST DATA INTO TRAIN AND TEST (70:30)#
set.seed(1)
train.index=sample(c(1:dim(FIRST)[1]),dim(FIRST)[1]*0.7)
test.index=(-train.index)
train.df=FIRST[train.index,]
test.df=FIRST[test.index,]
test.y=test.df$CARAVAN






# (3)   IDENTIFYING BEST MODEL   #

#FITTING LOGISTICS REGRESSION ON TRAIN DATA #
logReg=glm(CARAVAN~.,data=train.df,family=binomial)
logstart=glm(CARAVAN~1,data=train.df,family=binomial)
summary(logReg)
#The logReg Model identified 3 variables out of 85 as being significant which
#makes the rest insignificant. We thus reject the null hypothesis of 
#insignificance for these 3 variables. They include include; 'PPERSAUT','ALEVEN'
#'APLEZIER'.

#PREDICTING TEST DATA WITH logReg#
logprob=predict(logReg,test.df,type='response')
predLog=ifelse(logprob>0.5,1,0)
table(predLog,test.y)
model.accuracy=mean(predLog==test.y)*100
project.accuracy=(table(predLog,test.y)[2,2]/sum(table(predLog,test.y)[,2]))*100
model.accuracy
project.accuracy
#COMMENT:Out of 1627 customers 'who will not buy a caravan insurance policy' in
#the test data set, the logReg model was able to identify 1619 of those customers 
#correctly. On the other hand, out of 120 customers 'who will buy a caravan 
#insurance policy' in the test data, the logReg model was able to predict just 1 
#of them correctly. This is a massive problem. Though the general model accuracy
#is 92.73%, the project's goal is bent on correctly predicting customers who will
#buy a caravan insurance policy and we are less than 1% close to it.

#SOME OF THE CAUSES OF THE MODEL'S BIASE PERFORMANCE#
# 1. The Distribution of the target data might be highly imbalanced.
# 2.Too many independent variables that can be multicollinear to themselves and
# negatively impact on model's performance


#DIVING INTO THE DISTRIBUTION OF THE DATA SET#
NotBuy0=length(train.df[train.df$CARAVAN==0,'CARAVAN'])/length(train.df$CARAVAN) * 100
WillBuy0=length(train.df[train.df$CARAVAN==1,'CARAVAN'])/length(train.df[train.df$CARAVAN]) * 100  
dist=data.frame(X=c(0,1),Y=c(NotBuy0,WillBuy0))
dist
#COMMENT: We can see that one of the reasons why we had a poor unbiased
#logistics model is as a result of the distribution of data set. This 
#distribution is highly skewed to customers 'who has not subscribed to the 
#caravan insurance policy'.Since the model just have approximately 6% of data set
#to learn 'for customers who has subscribed in the past', it will perform poorly 
#for this sector of customers.

#THE WAY OUT#
#CLASS BALANCING USING ROSE (RANDOMLY OVER SAMPLING EXAMPLES)
#ROSE creates a sample of synthetic data by balancing the feature space of 
#both the majority and minority class. So we attempt to balance the 0's and 1's
#for the CARAVAN column
set.seed(3)
train.df=ROSE(CARAVAN~.,data=train.df)
train.df=train.df$data
train.index=sample(c(1:dim(train.df)[1]),dim(train.df)[1])
table(train.df$CARAVAN)


#COMMENT: As seen,the new train data is now balanced when the ROSE function was
#applied to it. Now, we can rebuild the logistics regression model on the new 
#train data to see if there would be a model improvement when we use this model
#to predict the test data. 
logReg=glm(CARAVAN~.,data=train.df,family=binomial)
logstart=glm(CARAVAN~1,data=train.df,family=binomial)
logprob=predict(logReg,test.df,type='response')
predLog=ifelse(logprob>0.525,1,0)
table(predLog,test.y)
model.accuracyLOG=mean(predLog==test.y)*100
project.accuracyLOG=table(predLog,test.y)[2,2]/sum(table(predLog,test.y)[,2])*100
model.accuracyLOG
project.accuracyLOG

#COMMENT: Well, after applying the SMOTE function and balancing the data set,
#the general model accuracy reduced (from 92.73% to 70.29%) while the project's 
#accuracy increased by a very significant amount (from 0.833% to 64.17%). This
#is great news and good trade-off. We can now work on improving the new model
#performance using several model selection techniques.


#ELIMINATING UNIMPORTANT VARIABLES USING VARIABLE SELECTION TECHNIQUES#
#We see that there are 86 variables which is a lot and there is high chance of
#multicollinearity among these independent variables. for this reason, I would 
#apply some variable selection techniques to fetch fewer variables that
#explains the target variable better and that will eliminate multi-collinearity 

#BACKWARD VARIABLE SELECTION#
bwdMod=stepAIC(logReg,direction='backward')
summary(bwdMod)
length(bwdMod$coefficients)
#The backward stepwise regression selected 48 best variables out of 85. We now 
#do a prediction using the backward logistics regression model and see if it does
#improve the current model's performance.

#Predicting test data and checking accuracy#
bwdpred=predict(bwdMod,test.df,type='response')
bwdpredLog=ifelse(bwdpred>0.525,1,0)
table(bwdpredLog,test.y)
model.accuracyBwd=mean(bwdpredLog==test.y)*100
project.accuracyBwd=table(bwdpredLog,test.y)[2,2]/sum(table(bwdpredLog,test.y)[,2])*100
model.accuracyBwd
project.accuracyBwd

#COMMENT: Comparing this result with the current model performance, the 
#backward selection technique improved slightly on the logistics regression model
#Its performance was slightly better both on the general model's performance and
# the project's goal accuracy (70.7%,65%) respectively.


#FORWARD VARIABLE SELECTION#
fwdMod=stepAIC(logstart,direction='forward',scope=list(upper=logReg))
summary(fwdMod)
length(fwdMod$coefficients)
#The forward stepwise regression selected 48 best variables(same as backward). 
#We now do a prediction using the forward logistics regression model and see if 
#it does improve the model's performance.

#Predicting test data and checking accuracy#
fwdpred=predict(fwdMod,test.df,type='response')
fwdpredLog=ifelse(bwdpred>0.525,1,0)
table(fwdpredLog,test.y)
model.accuracyfwd=mean(fwdpredLog==test.y)*100
project.accuracyfwd=table(fwdpredLog,test.y)[2,2]/sum(table(fwdpredLog,test.y)[,2])*100
model.accuracyfwd
project.accuracyfwd

#COMMENT: The forward and backward regression model had exactly same model 
#performance of (70.7%,65%) respectively.

#LASSO BINOMIAL REGRESSION
#Now, we try to introduce cross validation technique using LASSO regression as
#this will consider several folds of the train data sets and eliminate
#unimportant independent variable based on multiple folds
#Independent Variables
IND=model.matrix(CARAVAN~.,data=train.df)[,-1]
#Dependent variable
DEP=train.df$CARAVAN
IND1=model.matrix(CARAVAN~.,data=test.df)[,-1]

#We now perform cross validation using lasso REGRESSION for binomial and see if 
#it does improve the model's performance
set.seed(123)
cv.lasso <- cv.glmnet(x=IND, y=DEP, alpha = 1,family = "binomial",type.measure = 
                          'class',nlambda=200)
plot(cv.lasso)

#FITTING MODEL
lasso.model <- glmnet(x=IND, y=DEP, alpha = 1,family = "binomial",lambda=
                          cv.lasso$lambda.1se)
length(lasso.model$beta[,1][lasso.model$beta[,1]==0])
lasso.model$beta[,1]
#The lasso regression eliminated 38 variables and left 47 variables as important.

# Making prediction on test data
probabilities=predict(lasso.model,newx=IND1)
LassopredLog <- ifelse(probabilities > 0.17,1,0)
table(LassopredLog,test.y)

#Accuracy Metric for Lasso
model.accuracyLasso=mean(LassopredLog ==test.y)*100
project.accuracyLasso=table(LassopredLog ,test.y)[2,2]/sum(table(LassopredLog,test.y)[,2])*100
model.accuracyLasso
project.accuracyLasso
#COMMENT: The lasso binomial regression had the best general model performance
#so far with accuracy of 73.33%, however the forward and backward regression
#model outperformed lasso in the project's model accuracy with accuracy of 60%




#RANDOM FOREST FOR CLASSIFICATION#
set.seed(224)
randomMod=train(CARAVAN~.,data=train.df,method='rf',TuneLength=5,
trControl=trainControl(method='cv',number=12,classProbs=FALSE))
probability=predict(randomMod,test.df,'prob')
randPred=ifelse(probability$'0'>=probability$'1',0,1)
table(randPred,test.y)

#Accuracy Metric for Random Forest
model.accuracyRand=mean(randPred ==test.y)*100
project.accuracyRand=table(randPred ,test.y)[2,2]/sum(table(randPred,test.y)[,2])*100
model.accuracyRand
project.accuracyRand
#COMMENT: The random forest regression performed poorly in its general
#model accuracy but performed extremely well in correctly classifying customers
#that will buy a caravan insurance policy. It predicted correctly 118 of those
#customers and misclassified just 2 but had 17.06% accuracy in the general
#model performance. Not good trade-off


#SUPPORT VECTOR MACHINE#
set.seed(5)
SVM=svm(CARAVAN~.,data=train.df,kernel='linear',type='C-classification')
SVMPredict=predict(SVM,test.df,type='class')
table(SVMPredict,test.y)

#ACCURACY METRIC FOR SVM
model.accuracySVM=mean(SVMPredict ==test.y)*100
project.accuracySVM=table(SVMPredict,test.y)[2,2]/sum(table(SVMPredict,test.y)[,2])*100
model.accuracySVM
project.accuracySVM
#COMMENT: SVM outperformed backward,forward and logistics model in its project's
#goal model with accuracy of 66.67%, but slightly failed to beat those three 
#model in its general model performance with value of 66.91%. Good results

#SELECTING BEST MODEL
ModelAccuracy=c(model.accuracyLOG,model.accuracyBwd,model.accuracyfwd,
model.accuracyLasso,model.accuracyRand,model.accuracySVM)

ProjectAccuracy=c(project.accuracyLOG,project.accuracyBwd,project.accuracyfwd,
project.accuracyLasso,project.accuracyRand,project.accuracySVM)

ProbabilityTweaked=c('Yes-0.525','yes-0.525','yes-0.525','yes-0.17','NA','NA')

ModelOverview=data.frame(ModelAccuracy,ProjectAccuracy,ProbabilityTweaked)

row.names(ModelOverview)=c('LogisticsReg','LogisticsBwd','LogisticsFwd',
'LassoBinomial','RandomForest','SVM')

ModelOverview

#COMMENT: From the dataframe below, its slightly difficult to say which model
#performs the most as they are almost in same level. LassoBinomial had the best 
#performance in generally predicting both target classes(WillBuy,Will NotBuy).
#However, we are specifically concerned with predicting a customers
#probability to buy a CARAVAN INSURANCE POLICY as it is the goal of project.
#Logistics forward and backward stepwise seems to have produced a good balance
#between model performance and project goal which is why I am opting for the 
#BACKWARD STEPWISE LOGISTICS REGRESSION MODEL. It also performs well when we 
#have lots of independent variable.




# (4)   FITTING CHOSEN MODEL AND FORECASTING TARGET VARIABLE  #
#Now we fit the backward stepwise logistics regression model to the full
#data set. However, we need to apply the ROSE technique again to the full data
#set so as to take care of its imbalance.

set.seed(14)
FIRST1=ROSE(CARAVAN~.,data=FIRST)
FIRST1=FIRST1$data
table(FIRST$CARAVAN)
table(FIRST1$CARAVAN)
#So the recreated full data set is now balanced, and so we can fit our model 
#of choice to the balanced full data set
logReg2=glm(CARAVAN~.,data=FIRST1,family=binomial)
bwdMod2=stepAIC(logReg2,direction='backward')
summary(bwdMod2)
length(bwdMod2$coefficients)
#Applying the backward stepwise regression to the full data set selected 44 best 
#variables out of 85. We now do a prediction on the test data(second data provided).

#Predicting test data and checking accuracy#
bwdpred2=predict(bwdMod2,SECOND,type='response')
bwdpredLog2=ifelse(bwdpred2>0.48,1,0)



# (5)  EVALUATING MODEL PERFORMANCE  #

table(bwdpredLog2,THIRD$CARAVAN)
model.accuracyBwd2=mean(bwdpredLog2==test.y)*100
project.accuracyBwd2=table(bwdpredLog2,THIRD$CARAVAN)[2,2]/sum(table(bwdpredLog2,THIRD$CARAVAN)[,2])*100
model.accuracyBwd2
project.accuracyBwd2





# (6) CONCLUSIONS  #
#Finally, I decided to settle with the performance metric below after applying
#several model optimization technique. The general model performance is approxima
#tely 64% and the accuracy of the project's goal (predicting if a customer Will
#buy a caravan insurance policy) is 64.29%. This is not a bad metric especially
#for the kind of data supplied. Even after balancing the full train data set, the 
#test data set is highly imbalanced which resulted to this moderate model 
#performance. Furthermore, as more data are added to the set, the model's 
#performance would keep improving until it starts to perform absolutely 
#great as expected



