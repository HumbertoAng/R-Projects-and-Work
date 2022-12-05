library(dplyr) 
library(haven)
library(caret) 
library(pROC) 
library(randomForest)

# Pre-processing ----
## Read in data 
mret <- read_sas("C://ucf_classes/FIN_4451/Documents/mret7018.sas7bdat") 
as.data.frame(mret)

mret <- filter(mret, SHRCD %in% c(10,11))


# Construct new variables and Filter out to contain last 5 years only              
mret <-
  mret %>%
  group_by(PERMNO) %>%
  mutate(year = as.numeric(format(DATE, "%Y")),
         month = as.numeric(format(DATE,"%m")),
         PRC = abs(PRC), sz= PRC*SHROUT,
         CRASH = if_else(RET < -0.08,1,0),
         abs_ret = abs(RET),
         turnover = VOL/SHROUT,
         lag1_turnover = lag(turnover, 1),
         lag2_turnover = lag(turnover, 2),
         lag1_absret = lag(abs_ret, 1),
         lag2_absret = lag(abs_ret, 2),
         lag3_absret = lag(abs_ret, 3),
         lag1_PRC = lag(PRC, 1),
         lag2_PRC = lag(PRC, 2),
         lag1_RET = lag(RET, 1),
         lag2_RET = lag(RET, 2),
         lag3_RET = lag(RET, 3),
         lag4_RET = lag(RET, 4),
         lag1_vwretd = lag(VWRETD, 1),
         lag2_vwretd = lag(VWRETD, 2),
         lag3_vwretd = lag(VWRETD, 3),
         mcap = lag(sz, 1),
         lag1_mcap = lag(mcap, 1),
         lag2_mcap = lag(mcap, 2)) %>%
  as.data.frame()
    
mret <- filter(mret, mcap != 'NA')
mret <- filter(mret, CRASH != 'NA')
mret <- filter(mret, lag1_PRC != 'NA')
mret <- filter(mret, lag2_vwretd != 'NA')
mret <- filter(mret, lag3_vwretd != 'NA')
mret <- filter(mret, lag2_absret != 'NA')
mret <- filter(mret, lag3_absret != 'NA')
mret <- filter(mret, lag1_absret != 'NA')
mret <- filter(mret, year >= "2014" & year <= "2018")


# Non-Linear regression
fit <- lm(CRASH ~ lag1_turnover + lag2_turnover + lag1_absret + lag2_absret + lag3_absret + lag1_PRC + lag2_PRC +
            lag1_RET + lag2_RET + lag3_RET + lag4_RET + lag1_mcap + lag2_mcap + 
            mcap + lag1_vwretd + lag2_vwretd + lag3_vwretd, data=mret)
summary(fit)

fit1_5 <- lm(CRASH ~ lag2_absret +  lag3_absret + lag1_absret +lag3_vwretd, data=mret)
summary(fit1_5)

# Logistic regression
fit2 <- glm(CRASH ~ lag2_absret +  lag3_absret + lag1_absret +lag3_vwretd, data=mret)
summary(fit2)

# lag2_absret, lag3_absret, lag1_absret, lag3_vwretd, lag2_RET, lag2_turnover, lag1_turnover 

# ---- Variable Explanations ----
# absret is absolute return
# turnover is VOL/SHROUT
# number after word 'lag' is how many months variables were lagged



# GLM ---------------------------------
mret2 <- 
  mret %>% 
  select(CRASH, lag1_PRC, lag2_PRC, lag1_turnover, lag2_turnover, 
         lag1_absret, lag2_absret, lag3_absret, lag1_RET, lag2_RET, lag3_RET, lag4_RET, 
         lag1_vwretd, lag2_vwretd, lag3_vwretd) %>% 
  mutate(CRASH = as.factor(CRASH))
summary(mret2) 


# Identify variables with limited variation 
nearZeroVar(mret2, freqCut = 70/30, uniqueCut = 10)


# Data splitting 
set.seed(123)

inTrain <- createDataPartition(mret2$CRASH, p = .70, list = FALSE)
inTrain[1:10]

Train <- mret2[ inTrain,]
Test  <- mret2[-inTrain,]
summary(Train)
summary(Test)




# Models ---------------------------


# Model 1
glm <- train(CRASH ~ lag1_PRC+lag2_vwretd+lag3_vwretd+lag2_absret, 
             family=binomial(link='logit'), 
             method = "glm", data=Train)
glm 


# Model 2
glm <- train(CRASH ~ lag2_absret+I(lag2_PRC)+lag1_absret+lag3_vwretd, 
             family=binomial(link='logit'), 
             method = "glm", data=Train)
glm 



# Model 3
glm <- train(CRASH ~ lag2_absret+lag3_absret+lag1_absret+lag3_vwretd, 
             family=binomial(link='logit'), 
             method = "glm", data=Train)
glm 


# Model 4 (Best)
glm <- train(CRASH ~ lag2_absret+lag3_absret+lag1_absret+lag2_PRC, 
             family=binomial(link='logit'), 
             method = "glm", data=Train)
glm 


# Testing ----

# Obtaining details about each model 
getModelInfo("glm") 

varimpGLM = varImp(glm)
varimpGLM
plot(varimpGLM, main = "MRET Variable Importance: GLM") 


### Creating ROC curves 
# Output the probabilities for 1   
fitProb <- predict(glm, Test, type = "prob")
head(fitProb)

fitProb1 <- fitProb$"1"
par(pty = "s") 

roc(Test$CRASH, fitProb1, plot=TRUE, legacy.axes=TRUE, col="#377eb8", 
    print.auc=TRUE)

# AUC 0.672







# Playing with k-fold cross validation
library(boot)
library(caret)
library(ROCR)
library(dplyr)

attach(mret)


# Here I am creating new classification to so that a predict crash = 0 and no crash = 1
# More intuitive that a crash is 0 or negative not a positive number
NO_CRASH <- 0

NO_CRASH[CRASH==1] <- 0
NO_CRASH[CRASH==0] <- 1

check <- data.frame(CRASH, NO_CRASH)

head(check, 20)
summary(check)

rm(check)

mret_data <- data.frame(mret_data, NO_CRASH)


# Functin for looping and finding optimal cutoff
costfunc = function(NO_CRASH, pred_prob){
  weight1 = 1
  weight0 = 1
  c1 <- (NO_CRASH==1)&(pred_prob<optimal_cutoff)
  c0 <- (NO_CRASH==0)&(pred_prob>=optimal_cutoff)
  cost <- mean(weight1*c1 + weight0*c0)
  return(cost)
}


# Testing different models and interactions
model2 <- glm(CRASH ~ lag2_absret+lag3_absret+lag1_absret+lag3_vwretd, data=mret, family=binomial)



prob_seq <- seq(0.01, 1, 0.01)

cv_cost = rep(0, length(prob_seq))
for(i in 1:length(prob_seq)){
  optimal_cutoff = prob_seq[i]
  set.seed(123)
  cv_cost[i] = cv.glm(data=mret_data, glmfit=model2, cost = costfunc, K=10)$delta[2]
}

plot(prob_seq, cv_cost)

optimal_cutoff_cv = prob_seq[which(cv_cost==min(cv_cost))]
optimal_cutoff_cv
min(cv_cost)



# 1. Estimation of the model with the full dataset
model_2 <- glm(CRASH ~ lag1_PRC + lag2_vwretd + lag3_vwretd + lag2_absret, data=mret, family=binomial)

# 2. The predicted probabilites are obtained
pred_prob <- predict.glm(model_2, type=c("response"))

# 3. The generation of the confusion Matrix
class_prediction <- ifelse(pred_prob > optimal_cutoff_cv, 1, 0)
class_prediction <- factor(class_prediction)
NO_CRASH <- factor(NO_CRASH)

confusionMatrix(class_prediction, NO_CRASH, positive="1")

# Plotting of ROC Curve
roc <- performance(prediction(pred_prob, NO_CRASH), "tpr", "fpr")
plot(roc, colorize=TRUE)

# Calculation of AUC
pred <- prediction(pred_prob, NO_CRASH)
unlist(slot(performance(pred, "auc"), "y.values"))






