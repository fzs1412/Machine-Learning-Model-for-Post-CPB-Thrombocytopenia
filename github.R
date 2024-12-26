library(ricu)
library(dplyr)
library(data.table)
library(stringr)
library(gtsummary)
library(lubridate)
library(tidyr)
library(caret)
library(gbm)
library("caretEnsemble")
library(pROC)
library(naivebayes)
     
dat=read.csv("C:/Users/lxqji/OneDrive/R/2024cardiacTP/DataAnalysis/mimic_mice2.csv")

dat=dat%>%
  select( age,sofa_initia,glu_max,glu_min,plt_initial,lac,wbc_initial,gas_calcium,inr,crrt,TP)
colnames(dat)=c("Age","Sofa","Glucose_max","Glucose_min","Platelet","Lactic","WBC","Calcium","INR","RRT","PostComplication")
fwrite(dat,"mimic_dat建模.csv")

dat=read.csv("C:/Users/lxqji/OneDrive/R/2024cardiacTP/DataAnalysis/mimic_dat建模.csv")
dat$PostComplication=ifelse(dat$PostComplication==1,"Yes","No")
dat$PostComplication=as.factor(dat$PostComplication)


set.seed(3456)
trainIndex <- createDataPartition(
  y = dat$PostComplication, p = .7, 
  list = FALSE, 
  times = 1)
datTrain <- dat[trainIndex,]
datTest <- dat[-trainIndex,]

fitControl <- trainControl( method = "repeatedcv",
                            number = 10, repeats = 10)
gbmGrid <-  expand.grid(  interaction.depth = c(1, 5, 9), 
                          n.trees = (1:30)*50,  shrinkage = 0.1, n.minobsinnode = 20)
gbmFit <- caret::train(  PostComplication ~ ., 
                         data = datTrain,   method = "gbm",   trControl = fitControl,
                         tuneGrid = gbmGrid,  verbose = FALSE)


train_control <- trainControl(  method="boot",
                                number=25,  savePredictions="final",  classProbs=TRUE,
                                index=createResample(datTrain$PostComplication, 25),
                                summaryFunction=twoClassSummary)
model_list <- caretList(  
  PostComplication~., data=datTrain,
  trControl=train_control,  metric="ROC",
  tuneList=list( 
    SVM=caretModelSpec( method="svmLinearWeights", 
                        tuneGrid=expand.grid(  cost=seq(0.1,1,0.2),  weight=c(0.5,0.8,1))),
    C5.0=caretModelSpec(  method="C5.0", 
                          tuneGrid=expand.grid( trials=(1:5)*10,
                                                model=c("tree", "rules"),  winnow=c(TRUE, FALSE))),
    Bayes=caretModelSpec( method="naive_bayes"),
    XGboost=caretModelSpec( method="xgbTree",  tuneGrid=expand.grid(
      nrounds=(1:5)*10,  max_depth= 6, eta=c(0.1),gamma= c(0.1),
      colsample_bytree=1,  min_child_weight=c(0.5,0.8,1),
      subsample=c(0.3,0.5,0.8))) ))
#gbm_ensemble <- caretStack(  model_list,  method="gbm",  verbose=FALSE,  tuneLength=10,  metric="ROC",  trControl=trainControl(    method="boot",   number=10,   savePredictions="final",    classProbs=TRUE, summaryFunction=twoClassSummary  ))
#####分开
plot(model_list$C5.0)
plot(model_list$SVM)
plot(model_list$XGboost)

library("PerformanceAnalytics")
dtResample <- resamples(model_list)$values %>% 
  dplyr::select(ends_with("~ROC")) %>% 
  rename_with(~str_replace(.x,"~ROC","")) %>% 
  chart.Correlation() 
library(cowplot)
model_preds <- lapply(
  model_list, predict, 
  newdata=datTest, type="prob")
model_preds <- lapply(
  model_preds, function(x) x[,"Yes"])
model_preds <- data.frame(model_preds)
#model_preds$Ensemble <- 1-predict(  gbm_ensemble, newdata=datTest,  type="prob")
model_roc <- lapply(
  model_preds, function(xx){
    roc(response=datTest$PostComplication,
        direction = "<",predictor = xx)
  })

#Add model performance of the ensemble model
#model_roc$Ensemble <- roc( response=datTest$PostComplication,direction = "<",predictor = model_preds$Ensemble)
model_TextAUC <- lapply(model_roc, function(xx){
  paste("AUC: ",round(pROC::auc(xx),3),
        "[",round(pROC::ci.auc(xx)[1],3),",",
        round(pROC::ci.auc(xx)[3],3),"]",sep = "")
})

names(model_roc) <- paste(names(model_TextAUC),unlist(model_TextAUC))
plotROC <- ggroc(model_roc)+
  theme(legend.position=c(0.6,0.3))+
  guides(color=guide_legend(title="Models and AUCs"))
datCalib <- cbind(model_preds,testSetY=datTest$PostComplication)
#datCalib
#fwrite(datCalib, "datCalib_dry.csv")
#删除Ensemble
#cal_obj <- calibration(relevel(testSetY,ref = "Yes")  ~ SVM + C5.0+XGboost+Bayes+Ensemble, data = datCalib,  cuts = 6)

cal_obj <- calibration(relevel(testSetY,ref = "Yes")  ~ SVM +
                         C5.0+XGboost+Bayes,
                       data = datCalib,
                       cuts = 6)

calplot <- plot(cal_obj, type = "b", auto.key = list(columns = 3,
              lines = TRUE, points = T),xlab="Predicted Event Percentage")

#
ggdraw() +
  draw_plot(calplot, 0,0.5,  1, 0.5) +
  draw_plot(plotROC, 0, 0, 1, 0.5) +
  draw_plot_label(c("A", "B"), 
                  c(0, 0), c(1, 0.5), size = 15)
library(gbm)
ggplot(caret::varImp(gbmFit))
ggplot(caret::varImp(model_list$C5.0))
ggplot(caret::varImp(model_list$SVM))
ggplot(caret::varImp(model_list$Bayes))
ggplot(caret::varImp(model_list$XGboost))

library(lime)
explanation<-lime(datTrain,model_list$XGboost)
exp<-lime::explain(
  datTrain[1:2,], 
  explanation,n_labels = 2,n_features = 10)
plot_explanations(exp)
plot_features(exp, ncol = 2)
##模型XGboost
library("iBreakDown")
library("DALEX")
p_fun <- function(object, newdata){
  1-predict(object, newdata = newdata, 
            type = "prob") }
Ensmeble_la_un <- break_down_uncertainty(
  model_list$XGboost, 
  data = datTrain[,!names(datTrain)%in%"PostComplication"],
  new_observation = datTrain[1,],
  predict_function = p_fun,
  path = "average")
plot(Ensmeble_la_un)





