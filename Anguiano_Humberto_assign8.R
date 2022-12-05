library(dplyr) 
library(haven)
library(ggplot2) 
library(caret) 
library(pROC) 
library(randomForest)
library(varImp)
library(corrplot)
library(ggcorrplot)

# Read in the data
njdata <- read_sas("C://ucf_classes/FIN_4451/Documents/nj2000.sas7bdat") 
as.data.frame(njdata)


# Filter out observations for Philadelpha / METAREA = 616
panj <- filter(njdata, METAREA == 616)


# Using variable OWNERSHIP, construct a variable equal to 1
# for owning a home vs 0 for renting

panj <- panj %>% 
  mutate(OWN = if_else(OWNERSHP == 1, 1, 0))

summary(panj)


# Predict homeownership based on the available variables ------


# Attempting to find variable significance
fit <- lm(OWN ~  HHINCOME + SEX + AGE + MARST + ANCESTR1 + CITIZEN + EDUC + EMPSTAT + OCC1990 + IND1990 + INCTOT + MIGRATE5 + TRANTIME, data=panj)
summary(fit)

                      # AGE, Sex, HHINCOME, MIGRATE5, MARST seem significant

# Correlation matrix to double check significance
ggcorrplot(cor(panj))
ggcorrplot

                      # AGE, Sex, HHINCOME, MIGRATE5, MARST 
                      # seem significant


# Moving forward here to split data
panj2 <- 
  panj %>%
  select(OWN, AGE, SEX, HHINCOME, MIGRATE5, MARST, ANCESTR1, CITIZEN, EDUC, EMPSTAT, OCC1990, IND1990, INCTOT, TRANTIME) %>%
  mutate(OWN = as.factor(OWN))
summary(panj2)


# Simple test split
nearZeroVar(panj2, freqCut = 70/30, uniqueCut = 10)

# Data splitting 
set.seed(123)

inTrain <- createDataPartition(panj2$OWN, p = .70, list = FALSE)
inTrain[1:10]

Train <- panj2[ inTrain,]
Test  <- panj2[-inTrain,]
summary(Train)
summary(Test)




# Models -------- Models are in comment format expect for best model


# Model 1 (AUC 0.830)
#                 glm1 <- train(OWN ~ AGE + SEX + HHINCOME + I(MIGRATE5^2) + MARST, 
#                               family=binomial(link='logit'), 
#                               method = "glm", data=Train)
#                 glm1 


# Model 2 (AUC 0.834)

#                glm2 <- train(OWN ~ AGE + SEX + HHINCOME + MIGRATE5 + (MARST^2),
#                             family=binomial(link='logit'), 
#                             method = "glm", data=Train)
#                glm2 


# Random Forest Model (AUC 0.8322)
#                  rf <- train(OWN ~ AGE + SEX + HHINCOME + MIGRATE5 + MARST, 
#                              method = "rf", data=Train)
#                  rf

#      fitProbRF <- predict(rf, Test, type = "prob")
#       head(fitProbRF)

#      fitProbRF1 <- fitProbRF$"1"

#      par(pty = "s") 
#      roc(Test$OWN, fitProbRF1, plot=TRUE, legacy.axes=TRUE, col="#377eb8",
#          print.auc=TRUE)


# Model (Best)
glm <- train(OWN ~ AGE + SEX + HHINCOME + MIGRATE5 + MARST, 
             family=binomial(link='logit'), 
             method = "glm", data=Train)
glm 




# Testing ----

### Creating ROC curves 

# Output the probabilities for 1   
fitProb <- predict(glm, Test, type = "prob")
head(fitProb)

fitProb1 <- fitProb$"1"
par(pty = "s") 

roc(Test$OWN, fitProb1, plot=TRUE, legacy.axes=TRUE, col="#377eb8", 
    print.auc=TRUE)

# AUC 0.8343










