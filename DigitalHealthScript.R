library(dplyr)
library(tidyr)
library(purrr)
library(randomForest)
library(caret)
library(e1071)
library(psych)
library(VIM)
library(pROC)
library(prcr)
library(factoextra)
library(FeatureImpCluster)
library(flexclust)
library(xgboost)

x1=read.csv("C:\\Users\\Nikos\\Documents\\Python Scripts\\Project\\904cols.csv",header = TRUE)


#Scale but not CC and Sex
x1[, -c(1,2)]<-scale(x1[, -c(1,2)], center = TRUE, scale = TRUE)


smp_siz = floor(0.75*nrow(x1)) # creates a value for dividing the data into train and test
smp_siz  # 

set.seed(1)   # set seed to ensure you always have same random numbers generated
train_ind = sample(seq_len(nrow(x1)),size = smp_siz)  # Randomly identifies therows equal to sample size

train1 =x1[train_ind,] #creates the training dataset with row numbers stored in train_ind
test1=x1[-train_ind,]  # creates the test dataset excluding the row numbers mentioned in train_ind

train1$CC <- as.factor(train1$CC) #as.factor for classification
test1$CC <- as.factor(test1$CC)


######################Random forest Classification 1st Model(Tuned)######################

set.seed(1234)
model2 <- randomForest(CC ~ ., data = train1, importance = TRUE,ntree = 800,maxnodes =60,mtry=346)

#For the train set
table(predict(model2),train1$CC)
confusionMatrix(table(predict(model2),train1$CC))

#Important Features
varImp(model2)
varImpPlot(model2)

#For the Test set
prediction2 <-predict(model2, test1)
confusionMatrix(test1$CC,prediction2)



############# Random forest Classification 2nd Model(Tuned) ###################
set.seed(1234)
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid")
tuneGrid <- expand.grid(.mtry = 346)
fit_rf <- train(CC~.,
                train1,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                ntree = 550,
                maxnodes = 60)



prediction <-predict(fit_rf, test1)

varImp(fit_rf)
confusionMatrix(prediction, test1$CC)

table(predict(fit_rf),train1$CC)
confusionMatrix(table(predict(fit_rf),train1$CC))

############# XGboost ###################
###  (Tuned) Model ###

dtest <- xgb.DMatrix(as.matrix(test1[,-1]), label =as.matrix(test1$CC))
dtrain<-xgb.DMatrix(as.matrix(train1[,-1]), label =as.matrix(train1$CC))

model_tuned <- xgboost(data = as.matrix(train1[,-1]),
                       label=as.matrix(train1[,1]),# the data             
                       max.depth = 2, # the maximum depth of each decision tree
                       nround = 6, # max number of boosting iterations
                       objective = "binary:logistic") # the objective function 


#For the train set
predtrain<- predict(model_tuned, dtrain) 
predtrain <- as.numeric(predtrain > 0.5)
confusionMatrix(table(predtrain , train1$CC )) 

#For the Test set
pred2 <- predict(model_tuned, dtest)
# get & print the classification error
err2 <- mean(as.numeric(pred2 > 0.5) != test1$CC)
print(paste("test-error=", err2))
prediction2 <- as.numeric(pred2 > 0.5)
confusionMatrix(table(prediction2 , test1$CC )) 

# create importance matrix
importance_matrix <- xgb.importance(model = model_tuned)
xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain")

gp = xgb.ggplot.importance(importance_matrix,top_n = 10)
print(gp) 


############# Logistic Regression (Tuned) ###################
#Create Formula from xgboost values
measurevar <- "CC"
groupvars  <- c(importance_matrix$Feature[1:10])
formula<-as.formula(paste(measurevar, paste(groupvars, collapse=" + "), sep=" ~ "))


glm.fit <- glm(formula, data = train1, family =  binomial(link = 'logit'))



predtrain = predict(glm.fit, newdata = train1, type = "response")
predtrain  <- ifelse(predtrain  > 0.52, 1, 0)
test_roc = roc(train1$CC ~ predtrain, plot = TRUE, print.auc = TRUE)

confusionMatrix(table(predtrain,train1$CC)) 

predicted.data<-data.frame(probability.of.cc=glm.fit$fitted.values,cc=train1$CC)
predicted.data<-predicted.data[order(predicted.data$probability.of.cc, decreasing=FALSE), ]
predicted.data$rank<-1:nrow(predicted.data)
# plot results
ggplot(data = predicted.data, aes(x = rank, y = probability.of.cc)) + 
  geom_point(aes(color=cc), alpha=1,shape=4,stroke=2) +
  xlab("Index")+
  ylab("Predicted Probability of Diabetes")


predtest=predict(glm.fit, newdata = test1, type = "response")


predicted.data<-data.frame(probability.of.cc=predtest,cc=test1$CC)
predicted.data<-predicted.data[order(predicted.data$probability.of.cc, decreasing=FALSE), ]
predicted.data$rank<-1:nrow(predicted.data)

# plot results
ggplot(data = predicted.data, aes(x = rank, y = probability.of.cc)) + 
  geom_point(aes(color=cc), alpha=1,shape=4,stroke=2) +
  xlab("Index")+
  ylab("Predicted Probability of Diabetes")


predtest  <- ifelse(predtest  > 0.53, 1, 0)
test_roc = roc(test1$CC ~ predtest, plot = TRUE, print.auc = TRUE)

confusionMatrix(table(predtest,test1$CC)) 



############################## CLUSTERING ########################################


########## PCA ##########
res.pca <- prcomp(x1,center = T)
fviz_eig(res.pca)
PCAdata <- data.frame(res.pca$x[,1:100])
get_eig(res.pca)

########## Elbow Method ##########
wss<-0
for (i in 1:10) {
  km.out <- kmeans(PCAdata, centers = i, nstart = 124)
  # Save total within sum of squares to wss variable
  wss[i] <- km.out$tot.withinss
}

plot(1:10, wss, type = "b", 
     main = "Elbow Method",
     xlab = "Number of Clusters" )
abline(v=3, lty=2)


########## PCA Column Importance in Clustering ##########
set.seed(12)
nr_seeds2 <- 20
seeds_vec2 <- sample(1:1000,nr_seeds2)

savedImp <- data.frame(matrix(0,nr_seeds2,dim(PCAdata[1:10])[2]))
count <- 1
for (s in seeds_vec2) {
  set.seed(s)
  res <- kcca(PCAdata[1:10],k=3)
  set.seed(s)
  FeatureImp_res88 <- FeatureImpCluster(res,as.data.table(PCAdata[1:10]),sub = 1,biter = 1)
  savedImp[count,] <- FeatureImp_res88$featureImp[sort(names(FeatureImp_res88$featureImp))]
  count <- count + 1
}
names(savedImp) <- sort(names(FeatureImp_res88$featureImp))

boxplot(savedImp)
bwplot(res,PCAdata[1:10])

fviz_cos2(res.pca, choice="var", axes = 1,top = 7 )


########## PCA Column Importance in Clustering ##########

df <- PCAdata[1:10]
m3 <- create_profiles_cluster(df,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,n_profiles = 3)


plot_profiles(m3, to_center = TRUE)


########## Blood Metabolites in PCA Columns  ##########
fviz_cos2(res.pca, choice="var", axes =1,top = 30 )


# ################## RUN THIS TO TUNE AN RF MODEL ON NEW DATA ########################################
# # Define the control
# set.seed(1234)
# trControl <- trainControl(method = "cv",
#                           number = 10,
#                           search = "grid")
# 
# set.seed(1234)
# # Run the model
# rf_default <- train(CC~.,
#                     data = train1,
#                     method = "rf",
#                     metric = "Accuracy",
#                     trControl = trControl)
# # Print the results
# print(rf_default)
# plot(rf_default)
# 
# set.seed(1234)
# tuneGrid <- expand.grid(.mtry = c(rf_default$bestTune$mtry : (rf_default$bestTune$mtry+10)))
# rf_mtry <- train(CC~.,
#                  data = train1,
#                  method = "rf",
#                  metric = "Accuracy",
#                  tuneGrid = tuneGrid,
#                  trControl = trControl)
# print(rf_mtry)
# plot(rf_mtry)
# 
# best_mtry <- rf_mtry$bestTune$mtry 
# best_mtry
# 
# store_maxnode <- list()
# tuneGrid <- expand.grid(.mtry = best_mtry)
# for (maxnodes in c(15: 20)) {
#   set.seed(1234)
#   rf_maxnode <- train(CC~.,
#                       data = train1,
#                       method = "rf",
#                       metric = "Accuracy",
#                       tuneGrid = tuneGrid,
#                       trControl = trControl,
#                       importance = TRUE,
#                       maxnodes = maxnodes)
#   current_iteration <- toString(maxnodes)
#   store_maxnode[[current_iteration]] <- rf_maxnode
# }
# results_mtry <- resamples(store_maxnode)
# summary(results_mtry)
# 
# store_maxtrees <- list()
# for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800)) {
#   set.seed(5678)
#   rf_maxtrees <- train(CC~.,
#                        data = train1,
#                        method = "rf",
#                        metric = "Accuracy",
#                        tuneGrid = tuneGrid,
#                        trControl = trControl,
#                        importance = TRUE,
#                        maxnodes = 15,
#                        ntree = ntree)
#   current_iteration <- toString(ntree)
#   store_maxtrees[[current_iteration]] <- rf_maxtrees
# }
# results_tree <- resamples(store_maxtrees)
# summary(results_tree)