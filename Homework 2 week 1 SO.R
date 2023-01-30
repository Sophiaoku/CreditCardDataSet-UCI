#getwd()

#install.packages("ISLR") - not necessary
#install.packages("e1071") - not necessary
#install.packages("catools") - not necessary
#install.packages("kernlab")

df <- read.table(file = "credit_card_df-headers.txt", header = TRUE)
str(df)
head(df)


#sum(is.na(df)) - read the notes and realized the NA's have been removed.

library(kernlab)
model <- ksvm(as.matrix(df[,1:10]),as.factor(df[,11]),type="C-svc",kernel="vanilladot",C=100,scaled=TRUE)
model
#Training error : 0.136086 

model2 <- ksvm(as.matrix(df[,1:10]),as.factor(df[,11]),type="C-svc",kernel="vanilladot",C=150,scaled=TRUE)
model2

model3 <- model2 <- ksvm(as.matrix(df[,1:10]),as.factor(df[,11]),type="C-svc",kernel="vanilladot",C=50,scaled=TRUE)
model3

#Training error : 0.136086  remains the same for model 1- 3.

# calculate a1…am
a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
print(a)

# calculate a0
a0 <- -model@b
print(a0)

# see what the model predicts
pred <- predict(model,df[,1:10])
pred

sum(pred == df[,11]) / nrow(df)

#2) using rbfdot and anovado 
#?ksvm
model4 <- ksvm(as.matrix(df[,1:10]),as.factor(df[,11]),type="C-svc",kernel="anovado",C=50,scaled=TRUE)
model4

model5 <- ksvm(as.matrix(df[,1:10]),as.factor(df[,11]),type="C-svc",kernel="rbfdot",C=50,scaled=TRUE)
model5

#model4 using the Anovado kernel has Training error : 0.11315 
#model5 using the rbfdot kernel  has Training error : 0.062691 
#model using the vanilladot kernel has Training error : 0.136086 

#since we are looking to minimize the training error, model5 will produce the best outcome.

#to calculate the coefficients
a4 <- colSums(model4@xmatrix[[1]] * model4@coef[[1]])
a5 <- colSums(model5@xmatrix[[1]] * model5@coef[[1]])

a4
a5

a40 <- -model4@b
a50 <- -model5@b

a40
a50

# see what the model predicts
pred4 <- predict(model4,df[,1:10])

pred5 <- predict(model5,df[,1:10])

sum(pred4 == df[,11]) / nrow(df)

sum(pred5 == df[,11]) / nrow(df)


#as expected model5 (rbfdot) produced higher prediction correlation in respect to 
#actual classification
#0.9373089 - 94%

#3)

library(kknn)
Accvector <- c()
PredictionKKnn <- c()


# Iterating in a loop for 25 possible k values 
for (K in 1:25) {
  # Use lapply to train a model to run for each value of k indicated as "z"
  kknn_mod1 <- lapply(1:nrow(df), function(z) {
    kknn_mod2 <- kknn(df[-z,11]~., df[-z,1:10],df[z,1:10],k = K,kernel = "optimal",scale = TRUE)
    return(kknn_mod2)
  })
  # Using lapply to make predictions for each model and rounded 
  PredictionKKnn <- lapply(kknn_mod1, function(x) round(fitted(x)))
  # Flatten the predictions to produce a vector
  PredictionKKnn <- unlist(PredictionKKnn)
  # Calculating the accuracy 
  KnnAccy <- sum(PredictionKKnn == df[,11]) / nrow(df) 
  #combining the accuracy for each K into the blank vector
  Accvector <- c(Accvector,KnnAccy) 
}

plot(Accvector)

#install.packages("caret")
library(caret)

#confusion matrix

class(df$R1) #numeric 
class(PredictionKKnn) #integer

confusionMatrix(as.factor(df$R1),as.factor(PredictionKKnn)) 

#accuracy of the model using the optimal scale is @ 84.5% 

which.max(Accvector) 
which.min(Accvector) 
