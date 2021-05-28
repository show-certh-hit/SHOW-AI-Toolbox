#source the training file
source("scripts/training.R")
#load data
data=read.delim("data/test_data.txt",sep = " ")

results=list()
x=vector()
#first results
for (i in 1:150)
{
  results[[i]]=validation_step_number(i)
  x[i]=results[[i]]$rmse
}

#create dataset
dataset=data$demand[1:(order(x)[1]+1)]
for (i in 2:(length(data[,1])-order(x)[1]))
{
  dataset=data.frame(dataset,data$demand[i:(i+order(x)[1])])
}
#transpose dataset
dataset=as.data.frame(t(dataset))
#change names of rows and columns
row.names(dataset)=1:(length(data[,1])-order(x)[1])
colnames(dataset)[order(x)[1]+1]="dependent"

#create trainset by removing 90 rows for validation and test and number of order(x)[1]
trainset=data.frame(dataset[1:(length(data[,1])-order(x)[1]-90),])

#create validation set
validation_set=dataset[(length(data[,1])-order(x)[1]-90+1):(length(data[,1])-order(x)[1]-45),]

#create test set
test_set=dataset[(length(data[,1])-order(x)[1]-45+1):(length(data[,1])-order(x)[1]),]

#train pca
pca = prcomp(trainset[,1:order(x)[1]], scale. = TRUE, center = TRUE)

#cretate pca_trainset
pca_trainset=as.data.frame(pca$x,trainset$dependent)
pca_trainset=data.frame(pca_trainset,dependent=trainset$dependent)
row.names(pca_trainset)=1:length(pca_trainset[,1])

#cretate pca_validation_set
pca_validation_set=matrix(nrow=45,ncol=order(x)[1])
for (i in 1:order(x)[1])
{
  pca_validation_set[,i]=(validation_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
}
pca_validation_set=pca_validation_set%*%as.matrix(pca$rotation)
pca_validation_set=as.data.frame(pca_validation_set)
pca_validation_set=data.frame(pca_validation_set,dependent=validation_set$dependent)
row.names(pca_validation_set)=(length(pca_trainset[,1])+1):(length(pca_trainset[,1])+45)

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
  pca_test_set=matrix(nrow=45,ncol=order(x)[1])
  for (i in 1:order(x)[1])
  {
    pca_test_set[,i]=(test_set[,i]-as.data.frame(pca$center)[i,])/as.data.frame(pca$scale)[i,]
  }
  pca_test_set=pca_test_set%*%as.matrix(pca$rotation)
  pca_test_set=as.data.frame(pca_test_set)
  pca_test_set=data.frame(pca_test_set,dependent=test_set$dependent)
  row.names(pca_test_set)=(length(pca_trainset[,1])+1):(length(pca_trainset[,1])+45)


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

print("==================Actual==================")
print(test_set$dependent)
print("==================Predictions==================")
print(final_results$predictions)
print("==================Validation metrics==================")
print("MAE: ",final_results$mae,"MdAE: ",final_results$mdae,"RMSE: ",final_results$rmse)
print("Normalized MAE: ",final_results$mae/mean(test_set$dependent),
      "Normalized MdAE: ",final_results$mda/mean(test_set$dependent),
      "Normalized RMSE: ",final_results$rmse/mean(test_set$dependent))
