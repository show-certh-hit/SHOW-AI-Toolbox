#load library for Random Forest
library(ranger)

#for reproduce
set.seed(1)

validation_step_number=function(steps=5, lag=0)
{
  #create dataset
  dataset=data$demand[1:(steps+1+lag)]
  for (i in 2:(length(data[,1])-steps-lag))
  {
    dataset=data.frame(dataset,data$demand[i:(i+steps+lag)])
  }
  #transpose dataset
  dataset=as.data.frame(t(dataset))
  #change names of rows and columns
  row.names(dataset)=1:(length(data[,1])-steps-lag)
  colnames(dataset)[steps+1+lag]="dependent"
  
  if (lag)
    for (i in 1:lag)
      dataset=dataset[,-(ncol(dataset)-1)]
  
  
  #create trainset by removing 730 rows (2 years) for validation and test and number of steps
  trainset=data.frame(dataset[1:(length(data[,1])-steps-10),])
  
  #create validation set
  validation_set=dataset[(length(data[,1])-steps-10+1):(length(data[,1])-steps-5),]
  
  validation(trainset,validation_set)
}


validation = function(trainset,validation_set,mtry)
{
  if(missing(mtry)) 
  {
    #Fit the RF
    model_rf = ranger(dependent~., trainset,num.trees = 2000)
  } 
  else 
  {
    #Fit the RF
    model_rf = ranger(dependent~., trainset,num.trees = 2000, mtry=mtry)
  }
  
  #predict with model
  predictions = predict(model_rf,validation_set)
  mae=(mean(abs(validation_set$dependent-predictions$predictions)))
  mdae=(median(abs(validation_set$dependent-predictions$predictions)))
  rmse=sqrt(mean((validation_set$dependent-predictions$predictions)^2))
  return(list(mae=mae,mdae=mdae,rmse=rmse,predictions=predictions$predictions))
}




