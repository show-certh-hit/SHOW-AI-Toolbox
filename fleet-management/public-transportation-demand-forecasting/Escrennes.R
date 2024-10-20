#source the training file
source("scripts/training.R")
library(ggplot2)
#load data
data=read.delim("data/Escrennes_2.csv",sep = ',')
data$Date=as.Date(data$Date, format = "%m/%d/%Y")


##################################lag=0##############################################
lag=0
results=list()
x=vector()
#first results
for (i in 1:5)
{
  results[[i]]=validation_step_number(i)
  x[i]=results[[i]]$rmse
}

#create dataset
dataset=data$demand[1:(order(x)[1]+1+lag)]
for (i in 2:(length(data[,1])-order(x)[1]+1))
{
  dataset=data.frame(dataset,data$demand[i:(i+order(x)[1]+lag)])
}
#transpose dataset
dataset=as.data.frame(t(dataset))
#change names of rows and columns
row.names(dataset)=1:(length(data[,1])-order(x)[1]+1)
colnames(dataset)[order(x)[1]+1+lag]="dependent"


if (lag)
  for (i in 1:lag)
    dataset=dataset[,-(ncol(dataset)-1)]

#create trainset by removing 730 rows (2 years) for validation and testing and number of order(x)[1]
trainset=data.frame(dataset[1:(length(data[,1])-order(x)[1]-10-lag),])

#create validation set
validation_set=dataset[(length(data[,1])-order(x)[1]-10+1-lag):(length(data[,1])-order(x)[1]-lag),]

#create test set
test_set=dataset[(length(data[,1])-order(x)[1]-lag+1):(length(data[,1])-order(x)[1]+1),]


#train pca
#pca = prcomp(trainset[,1:order(x)[1]], scale. = FALSE, center = FALSE)
pca = prcomp(trainset[,1:order(x)[1]], scale. = TRUE, center = TRUE)

#cretate pca_trainset
pca_trainset=as.data.frame(pca$x,trainset$dependent)
pca_trainset=data.frame(pca_trainset,dependent=trainset$dependent)
row.names(pca_trainset)=1:length(pca_trainset[,1])

#cretate pca_validation_set
pca_validation_set=matrix(nrow=10,ncol=order(x)[1])
for (i in 1:order(x)[1])
{
  pca_validation_set[,i]=(validation_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  #pca_validation_set[,i]=validation_set[,i]
}
#pca_validation_set=validation_set[,-(order(x)[1]+1)]
pca_validation_set=as.matrix(pca_validation_set)%*%as.matrix(pca$rotation)
pca_validation_set=as.data.frame(pca_validation_set)
pca_validation_set=data.frame(pca_validation_set,dependent=validation_set$dependent)
row.names(pca_validation_set)=(length(pca_trainset[,1])+1):(length(pca_trainset[,1])+10)

pca_results=list()
pca_x=vector()
#pca results
for (i in 1:(ncol(pca_trainset)-1))
{
  pca_results[[i]]=validation(data.frame(pca_trainset[1:i],dependent=pca_trainset$dependent),
                              data.frame(pca_validation_set[1:i],dependent=pca_validation_set$dependent))
  pca_x[i]=pca_results[[i]]$rmse
}



features_results=list()
features_x=vector()



#if PCA improves the previous results
if (min(pca_x)<=min(x))
{
  #choose number of features
  for (i in 1:order(pca_x)[1])
  {
    features_results[[i]]=validation(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                     data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent),
                                     mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #cretate pca_test_set
  pca_test_set=matrix(nrow=lag+1,ncol=order(x)[1])
  for (i in 1:order(x)[1])
  {
    pca_test_set[,i]=(test_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  }
  #pca_test_set=test_set[,-(order(x)[1]+1)]
  pca_test_set=as.matrix(pca_test_set)%*%as.matrix(pca$rotation)
  pca_test_set=as.data.frame(pca_test_set)
  pca_test_set=data.frame(pca_test_set,dependent=test_set$dependent)
  row.names(pca_test_set)=(length(pca_trainset[,1])+length(pca_validation_set[,1])+1):(length(pca_trainset[,1])+length(pca_validation_set[,1])+lag+1)
  #colnames(pca_test_set)[1:(length(colnames(pca_test_set))-1)]=c("PC1")
  
  
  
  
  #final results
  final_results=validation(rbind(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                 data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent)),
                           data.frame(pca_test_set[1:(order(pca_x)[1])],dependent=pca_test_set$dependent),
                           mtry = order(features_x)[1])
  
}else
{
  #choose number of features
  for (i in 1:order(x)[1])
  {
    features_results[[i]]=validation(trainset,validation_set, mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #final results
  final_results=validation(rbind(trainset,validation_set),test_set,mtry = order(features_x)[1])
  
}

print("==================Predictions+1==================")
print(final_results$predictions)


##################################lag=1##############################################
lag=1
results=list()
x=vector()
#first results
for (i in 1:5)
{
  results[[i]]=validation_step_number(i)
  x[i]=results[[i]]$rmse
}

#create dataset
dataset=data$demand[1:(order(x)[1]+1+lag)]
for (i in 2:(length(data[,1])-order(x)[1]+1))
{
  dataset=data.frame(dataset,data$demand[i:(i+order(x)[1]+lag)])
}
#transpose dataset
dataset=as.data.frame(t(dataset))
#change names of rows and columns
row.names(dataset)=1:(length(data[,1])-order(x)[1]+1)
colnames(dataset)[order(x)[1]+1+lag]="dependent"


if (lag)
{
  for (i in 1:lag)
    dataset=dataset[,-(ncol(dataset)-1)]
  
}
  
#create trainset by removing 730 rows (2 years) for validation and testing and number of order(x)[1]
trainset=data.frame(dataset[1:(length(data[,1])-order(x)[1]-10-lag),])

#create validation set
validation_set=dataset[(length(data[,1])-order(x)[1]-10+1-lag):(length(data[,1])-order(x)[1]-lag),]

#create test set
test_set=dataset[(length(data[,1])-order(x)[1]-lag+1):(length(data[,1])-order(x)[1]+1),]


#train pca
#pca = prcomp(trainset[,1:order(x)[1]], scale. = FALSE, center = FALSE)
pca = prcomp(trainset[,1:order(x)[1]], scale. = TRUE, center = TRUE)

#cretate pca_trainset
pca_trainset=as.data.frame(pca$x,trainset$dependent)
pca_trainset=data.frame(pca_trainset,dependent=trainset$dependent)
row.names(pca_trainset)=1:length(pca_trainset[,1])

#cretate pca_validation_set
pca_validation_set=matrix(nrow=10,ncol=order(x)[1])
for (i in 1:order(x)[1])
{
  pca_validation_set[,i]=(validation_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  #pca_validation_set[,i]=validation_set[,i]
}
#pca_validation_set=validation_set[,-(order(x)[1]+1)]
pca_validation_set=as.matrix(pca_validation_set)%*%as.matrix(pca$rotation)
pca_validation_set=as.data.frame(pca_validation_set)
pca_validation_set=data.frame(pca_validation_set,dependent=validation_set$dependent)
row.names(pca_validation_set)=(length(pca_trainset[,1])+1):(length(pca_trainset[,1])+10)

pca_results=list()
pca_x=vector()
#pca results
for (i in 1:(ncol(pca_trainset)-1))
{
  pca_results[[i]]=validation(data.frame(pca_trainset[1:i],dependent=pca_trainset$dependent),
                              data.frame(pca_validation_set[1:i],dependent=pca_validation_set$dependent))
  pca_x[i]=pca_results[[i]]$rmse
}



features_results=list()
features_x=vector()



#if PCA improves the previous results
if (min(pca_x)<=min(x))
{
  #choose number of features
  for (i in 1:order(pca_x)[1])
  {
    features_results[[i]]=validation(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                     data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent),
                                     mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #cretate pca_test_set
  pca_test_set=matrix(nrow=lag+1,ncol=order(x)[1])
  for (i in 1:order(x)[1])
  {
    pca_test_set[,i]=(test_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  }
  #pca_test_set=test_set[,-(order(x)[1]+1)]
  pca_test_set=as.matrix(pca_test_set)%*%as.matrix(pca$rotation)
  pca_test_set=as.data.frame(pca_test_set)
  pca_test_set=data.frame(pca_test_set,dependent=test_set$dependent)
  row.names(pca_test_set)=(length(pca_trainset[,1])+length(pca_validation_set[,1])+1):(length(pca_trainset[,1])+length(pca_validation_set[,1])+lag+1)
  
  #final results
  final_results=validation(rbind(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                 data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent)),
                           data.frame(pca_test_set[1:(order(pca_x)[1])],dependent=pca_test_set$dependent),
                           mtry = order(features_x)[1])
  
}else
{
  #choose number of features
  for (i in 1:order(x)[1])
  {
    features_results[[i]]=validation(trainset,validation_set, mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #final results
  final_results=validation(rbind(trainset,validation_set),test_set,mtry = order(features_x)[1])
  
}

print("==================Predictions+2==================")
print(final_results$predictions)


##################################lag=2##############################################
lag=2
results=list()
x=vector()
#first results
for (i in 1:5)
{
  results[[i]]=validation_step_number(i)
  x[i]=results[[i]]$rmse
}

#create dataset
dataset=data$demand[1:(order(x)[1]+1+lag)]
for (i in 2:(length(data[,1])-order(x)[1]+1))
{
  dataset=data.frame(dataset,data$demand[i:(i+order(x)[1]+lag)])
}
#transpose dataset
dataset=as.data.frame(t(dataset))
#change names of rows and columns
row.names(dataset)=1:(length(data[,1])-order(x)[1]+1)
colnames(dataset)[order(x)[1]+1+lag]="dependent"


if (lag)
{
  for (i in 1:lag)
    dataset=dataset[,-(ncol(dataset)-1)]
  
}

#create trainset by removing 730 rows (2 years) for validation and testing and number of order(x)[1]
trainset=data.frame(dataset[1:(length(data[,1])-order(x)[1]-10-lag),])

#create validation set
validation_set=dataset[(length(data[,1])-order(x)[1]-10+1-lag):(length(data[,1])-order(x)[1]-lag),]

#create test set
test_set=dataset[(length(data[,1])-order(x)[1]-lag+1):(length(data[,1])-order(x)[1]+1),]


#train pca
#pca = prcomp(trainset[,1:order(x)[1]], scale. = FALSE, center = FALSE)
pca = prcomp(trainset[,1:order(x)[1]], scale. = TRUE, center = TRUE)

#cretate pca_trainset
pca_trainset=as.data.frame(pca$x,trainset$dependent)
pca_trainset=data.frame(pca_trainset,dependent=trainset$dependent)
row.names(pca_trainset)=1:length(pca_trainset[,1])

#cretate pca_validation_set
pca_validation_set=matrix(nrow=10,ncol=order(x)[1])
for (i in 1:order(x)[1])
{
  pca_validation_set[,i]=(validation_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  #pca_validation_set[,i]=validation_set[,i]
}
#pca_validation_set=validation_set[,-(order(x)[1]+1)]
pca_validation_set=as.matrix(pca_validation_set)%*%as.matrix(pca$rotation)
pca_validation_set=as.data.frame(pca_validation_set)
pca_validation_set=data.frame(pca_validation_set,dependent=validation_set$dependent)
row.names(pca_validation_set)=(length(pca_trainset[,1])+1):(length(pca_trainset[,1])+10)

pca_results=list()
pca_x=vector()
#pca results
for (i in 1:(ncol(pca_trainset)-1))
{
  pca_results[[i]]=validation(data.frame(pca_trainset[1:i],dependent=pca_trainset$dependent),
                              data.frame(pca_validation_set[1:i],dependent=pca_validation_set$dependent))
  pca_x[i]=pca_results[[i]]$rmse
}



features_results=list()
features_x=vector()



#if PCA improves the previous results
if (min(pca_x)<=min(x))
{
  #choose number of features
  for (i in 1:order(pca_x)[1])
  {
    features_results[[i]]=validation(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                     data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent),
                                     mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #cretate pca_test_set
  pca_test_set=matrix(nrow=lag+1,ncol=order(x)[1])
  for (i in 1:order(x)[1])
  {
    pca_test_set[,i]=(test_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  }
  #pca_test_set=test_set[,-(order(x)[1]+1)]
  pca_test_set=as.matrix(pca_test_set)%*%as.matrix(pca$rotation)
  pca_test_set=as.data.frame(pca_test_set)
  pca_test_set=data.frame(pca_test_set,dependent=test_set$dependent)
  row.names(pca_test_set)=(length(pca_trainset[,1])+length(pca_validation_set[,1])+1):(length(pca_trainset[,1])+length(pca_validation_set[,1])+lag+1)
  #colnames(pca_test_set)[1:(length(colnames(pca_test_set))-1)]=c("PC1")
  
  
  #final results
  final_results=validation(rbind(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                 data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent)),
                           data.frame(pca_test_set[1:(order(pca_x)[1])],dependent=pca_test_set$dependent),
                           mtry = order(features_x)[1])
  
}else
{
  #choose number of features
  for (i in 1:order(x)[1])
  {
    features_results[[i]]=validation(trainset,validation_set, mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #final results
  final_results=validation(rbind(trainset,validation_set),test_set,mtry = order(features_x)[1])
  
}

print("==================Predictions+3==================")
print(final_results$predictions)


##################################lag=3##############################################
lag=3
results=list()
x=vector()
#first results
for (i in 1:5)
{
  results[[i]]=validation_step_number(i)
  x[i]=results[[i]]$rmse
}

#create dataset
dataset=data$demand[1:(order(x)[1]+1+lag)]
for (i in 2:(length(data[,1])-order(x)[1]+1))
{
  dataset=data.frame(dataset,data$demand[i:(i+order(x)[1]+lag)])
}
#transpose dataset
dataset=as.data.frame(t(dataset))
#change names of rows and columns
row.names(dataset)=1:(length(data[,1])-order(x)[1]+1)
colnames(dataset)[order(x)[1]+1+lag]="dependent"


if (lag)
{
  for (i in 1:lag)
    dataset=dataset[,-(ncol(dataset)-1)]
  
}

#create trainset by removing 730 rows (2 years) for validation and testing and number of order(x)[1]
trainset=data.frame(dataset[1:(length(data[,1])-order(x)[1]-10-lag),])

#create validation set
validation_set=dataset[(length(data[,1])-order(x)[1]-10+1-lag):(length(data[,1])-order(x)[1]-lag),]

#create test set
test_set=dataset[(length(data[,1])-order(x)[1]-lag+1):(length(data[,1])-order(x)[1]+1),]


#train pca
#pca = prcomp(trainset[,1:order(x)[1]], scale. = FALSE, center = FALSE)
pca = prcomp(trainset[,1:order(x)[1]], scale. = TRUE, center = TRUE)

#cretate pca_trainset
pca_trainset=as.data.frame(pca$x,trainset$dependent)
pca_trainset=data.frame(pca_trainset,dependent=trainset$dependent)
row.names(pca_trainset)=1:length(pca_trainset[,1])

#cretate pca_validation_set
pca_validation_set=matrix(nrow=10,ncol=order(x)[1])
for (i in 1:order(x)[1])
{
  pca_validation_set[,i]=(validation_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  #pca_validation_set[,i]=validation_set[,i]
}
#pca_validation_set=validation_set[,-(order(x)[1]+1)]
pca_validation_set=as.matrix(pca_validation_set)%*%as.matrix(pca$rotation)
pca_validation_set=as.data.frame(pca_validation_set)
pca_validation_set=data.frame(pca_validation_set,dependent=validation_set$dependent)
row.names(pca_validation_set)=(length(pca_trainset[,1])+1):(length(pca_trainset[,1])+10)

pca_results=list()
pca_x=vector()
#pca results
for (i in 1:(ncol(pca_trainset)-1))
{
  pca_results[[i]]=validation(data.frame(pca_trainset[1:i],dependent=pca_trainset$dependent),
                              data.frame(pca_validation_set[1:i],dependent=pca_validation_set$dependent))
  pca_x[i]=pca_results[[i]]$rmse
}



features_results=list()
features_x=vector()



#if PCA improves the previous results
if (min(pca_x)<=min(x))
{
  #choose number of features
  for (i in 1:order(pca_x)[1])
  {
    features_results[[i]]=validation(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                     data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent),
                                     mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #cretate pca_test_set
  pca_test_set=matrix(nrow=lag+1,ncol=order(x)[1])
  for (i in 1:order(x)[1])
  {
    pca_test_set[,i]=(test_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  }
  #pca_test_set=test_set[,-(order(x)[1]+1)]
  pca_test_set=as.matrix(pca_test_set)%*%as.matrix(pca$rotation)
  pca_test_set=as.data.frame(pca_test_set)
  pca_test_set=data.frame(pca_test_set,dependent=test_set$dependent)
  row.names(pca_test_set)=(length(pca_trainset[,1])+length(pca_validation_set[,1])+1):(length(pca_trainset[,1])+length(pca_validation_set[,1])+lag+1)
  #colnames(pca_test_set)[1:(length(colnames(pca_test_set))-1)]=c("PC1")
  
  
  #final results
  final_results=validation(rbind(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                 data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent)),
                           data.frame(pca_test_set[1:(order(pca_x)[1])],dependent=pca_test_set$dependent),
                           mtry = order(features_x)[1])
  
}else
{
  #choose number of features
  for (i in 1:order(x)[1])
  {
    features_results[[i]]=validation(trainset,validation_set, mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #final results
  final_results=validation(rbind(trainset,validation_set),test_set,mtry = order(features_x)[1])
  
}

print("==================Predictions+4==================")
print(final_results$predictions)


##################################lag=4##############################################
lag=4
results=list()
x=vector()
#first results
for (i in 1:5)
{
  results[[i]]=validation_step_number(i)
  x[i]=results[[i]]$rmse
}

#create dataset
dataset=data$demand[1:(order(x)[1]+1+lag)]
for (i in 2:(length(data[,1])-order(x)[1]+1))
{
  dataset=data.frame(dataset,data$demand[i:(i+order(x)[1]+lag)])
}
#transpose dataset
dataset=as.data.frame(t(dataset))
#change names of rows and columns
row.names(dataset)=1:(length(data[,1])-order(x)[1]+1)
colnames(dataset)[order(x)[1]+1+lag]="dependent"


if (lag)
{
  for (i in 1:lag)
    dataset=dataset[,-(ncol(dataset)-1)]
  
}

#create trainset by removing 730 rows (2 years) for validation and testing and number of order(x)[1]
trainset=data.frame(dataset[1:(length(data[,1])-order(x)[1]-10-lag),])

#create validation set
validation_set=dataset[(length(data[,1])-order(x)[1]-10+1-lag):(length(data[,1])-order(x)[1]-lag),]

#create test set
test_set=dataset[(length(data[,1])-order(x)[1]-lag+1):(length(data[,1])-order(x)[1]+1),]


#train pca
#pca = prcomp(trainset[,1:order(x)[1]], scale. = FALSE, center = FALSE)
pca = prcomp(trainset[,1:order(x)[1]], scale. = TRUE, center = TRUE)

#cretate pca_trainset
pca_trainset=as.data.frame(pca$x,trainset$dependent)
pca_trainset=data.frame(pca_trainset,dependent=trainset$dependent)
row.names(pca_trainset)=1:length(pca_trainset[,1])

#cretate pca_validation_set
pca_validation_set=matrix(nrow=10,ncol=order(x)[1])
for (i in 1:order(x)[1])
{
  pca_validation_set[,i]=(validation_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  #pca_validation_set[,i]=validation_set[,i]
}
#pca_validation_set=validation_set[,-(order(x)[1]+1)]
pca_validation_set=as.matrix(pca_validation_set)%*%as.matrix(pca$rotation)
pca_validation_set=as.data.frame(pca_validation_set)
pca_validation_set=data.frame(pca_validation_set,dependent=validation_set$dependent)
row.names(pca_validation_set)=(length(pca_trainset[,1])+1):(length(pca_trainset[,1])+10
                                                            )

pca_results=list()
pca_x=vector()
#pca results
for (i in 1:(ncol(pca_trainset)-1))
{
  pca_results[[i]]=validation(data.frame(pca_trainset[1:i],dependent=pca_trainset$dependent),
                              data.frame(pca_validation_set[1:i],dependent=pca_validation_set$dependent))
  pca_x[i]=pca_results[[i]]$rmse
}



features_results=list()
features_x=vector()



#if PCA improves the previous results
if (min(pca_x)<=min(x))
{
  #choose number of features
  for (i in 1:order(pca_x)[1])
  {
    features_results[[i]]=validation(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                     data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent),
                                     mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #cretate pca_test_set
  pca_test_set=matrix(nrow=lag+1,ncol=order(x)[1])
  for (i in 1:order(x)[1])
  {
    pca_test_set[,i]=(test_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  }
  #pca_test_set=test_set[,-(order(x)[1]+1)]
  pca_test_set=as.matrix(pca_test_set)%*%as.matrix(pca$rotation)
  pca_test_set=as.data.frame(pca_test_set)
  pca_test_set=data.frame(pca_test_set,dependent=test_set$dependent)
  row.names(pca_test_set)=(length(pca_trainset[,1])+length(pca_validation_set[,1])+1):(length(pca_trainset[,1])+length(pca_validation_set[,1])+lag+1)
  #colnames(pca_test_set)[1:(length(colnames(pca_test_set))-1)]=c("PC1")
  
  
  #final results
  final_results=validation(rbind(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                 data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent)),
                           data.frame(pca_test_set[1:(order(pca_x)[1])],dependent=pca_test_set$dependent),
                           mtry = order(features_x)[1])
  
}else
{
  #choose number of features
  for (i in 1:order(x)[1])
  {
    features_results[[i]]=validation(trainset,validation_set, mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #final results
  final_results=validation(rbind(trainset,validation_set),test_set,mtry = order(features_x)[1])
  
}

print("==================Predictions+5==================")
print(final_results$predictions)


##################################lag=5##############################################
lag=5
results=list()
x=vector()
#first results
for (i in 1:5)
{
  results[[i]]=validation_step_number(i)
  x[i]=results[[i]]$rmse
}

#create dataset
dataset=data$demand[1:(order(x)[1]+1+lag)]
for (i in 2:(length(data[,1])-order(x)[1]+1))
{
  dataset=data.frame(dataset,data$demand[i:(i+order(x)[1]+lag)])
}
#transpose dataset
dataset=as.data.frame(t(dataset))
#change names of rows and columns
row.names(dataset)=1:(length(data[,1])-order(x)[1]+1)
colnames(dataset)[order(x)[1]+1+lag]="dependent"


if (lag)
{
  for (i in 1:lag)
    dataset=dataset[,-(ncol(dataset)-1)]
  
}

#create trainset by removing 730 rows (2 years) for validation and testing and number of order(x)[1]
trainset=data.frame(dataset[1:(length(data[,1])-order(x)[1]-10-lag),])

#create validation set
validation_set=dataset[(length(data[,1])-order(x)[1]-10+1-lag):(length(data[,1])-order(x)[1]-lag),]

#create test set
test_set=dataset[(length(data[,1])-order(x)[1]-lag+1):(length(data[,1])-order(x)[1]+1),]


#train pca
#pca = prcomp(trainset[,1:order(x)[1]], scale. = FALSE, center = FALSE)
pca = prcomp(trainset[,1:order(x)[1]], scale. = TRUE, center = TRUE)

#cretate pca_trainset
pca_trainset=as.data.frame(pca$x,trainset$dependent)
pca_trainset=data.frame(pca_trainset,dependent=trainset$dependent)
row.names(pca_trainset)=1:length(pca_trainset[,1])

#cretate pca_validation_set
pca_validation_set=matrix(nrow=10,ncol=order(x)[1])
for (i in 1:order(x)[1])
{
  pca_validation_set[,i]=(validation_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  #pca_validation_set[,i]=validation_set[,i]
}
#pca_validation_set=validation_set[,-(order(x)[1]+1)]
pca_validation_set=as.matrix(pca_validation_set)%*%as.matrix(pca$rotation)
pca_validation_set=as.data.frame(pca_validation_set)
pca_validation_set=data.frame(pca_validation_set,dependent=validation_set$dependent)
row.names(pca_validation_set)=(length(pca_trainset[,1])+1):(length(pca_trainset[,1])+10)

pca_results=list()
pca_x=vector()
#pca results
for (i in 1:(ncol(pca_trainset)-1))
{
  pca_results[[i]]=validation(data.frame(pca_trainset[1:i],dependent=pca_trainset$dependent),
                              data.frame(pca_validation_set[1:i],dependent=pca_validation_set$dependent))
  pca_x[i]=pca_results[[i]]$rmse
}



features_results=list()
features_x=vector()



#if PCA improves the previous results
if (min(pca_x)<=min(x))
{
  #choose number of features
  for (i in 1:order(pca_x)[1])
  {
    features_results[[i]]=validation(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                     data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent),
                                     mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #cretate pca_test_set
  pca_test_set=matrix(nrow=lag+1,ncol=order(x)[1])
  for (i in 1:order(x)[1])
  {
    pca_test_set[,i]=(test_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  }
  #pca_test_set=test_set[,-(order(x)[1]+1)]
  pca_test_set=as.matrix(pca_test_set)%*%as.matrix(pca$rotation)
  pca_test_set=as.data.frame(pca_test_set)
  pca_test_set=data.frame(pca_test_set,dependent=test_set$dependent)
  row.names(pca_test_set)=(length(pca_trainset[,1])+length(pca_validation_set[,1])+1):(length(pca_trainset[,1])+length(pca_validation_set[,1])+lag+1)
  #colnames(pca_test_set)[1:(length(colnames(pca_test_set))-1)]=c("PC1")
  
  
  #final results
  final_results=validation(rbind(data.frame(pca_trainset[1:(order(pca_x)[1])],dependent=pca_trainset$dependent),
                                 data.frame(pca_validation_set[1:order(pca_x)[1]],dependent=pca_validation_set$dependent)),
                           data.frame(pca_test_set[1:(order(pca_x)[1])],dependent=pca_test_set$dependent),
                           mtry = order(features_x)[1])
  
}else
{
  #choose number of features
  for (i in 1:order(x)[1])
  {
    features_results[[i]]=validation(trainset,validation_set, mtry = i)
    features_x[i]=features_results[[i]]$rmse
  }
  
  
  #final results
  final_results=validation(rbind(trainset,validation_set),test_set,mtry = order(features_x)[1])
  
}

print("==================Predictions +6==================")
print(final_results$predictions)

