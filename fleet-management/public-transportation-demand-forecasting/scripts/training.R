#load library for Random Forest
library(ranger)

#for reproduce
set.seed(1)

validation_step_number=function(steps=100)
{
  #create dataset
  dataset=data$demand[1:(steps+1)]
  for (i in 2:(length(data[,1])-steps))
  {
    dataset=data.frame(dataset,data$demand[i:(i+steps)])
  }
  #transpose dataset
  dataset=as.data.frame(t(dataset))
  #change names of rows and columns
  row.names(dataset)=1:(length(data[,1])-steps)
  colnames(dataset)[steps+1]="dependent"
  
  
  #create trainset by removing 90 rows for validation and test and number of steps
  trainset=data.frame(dataset[1:(length(data[,1])-steps-90),])
  
  #create validation set
  validation_set=dataset[(length(data[,1])-steps-90+1):(length(data[,1])-steps-45),]
  
  validation(trainset,validation_set)
}


validation = function(trainset,validation_set,mtry)
{
  if(missing(mtry)) 
  {
    #Fit the RF
    model_rf = ranger(dependent~., trainset,num.trees = 20000)
  } 
  else 
  {
    #Fit the RF
    model_rf = ranger(dependent~., trainset,num.trees = 20000, mtry=mtry)
  }
  
  #predict with model
  predictions = predict(model_rf,validation_set)
  mae=(mean(abs(validation_set$dependent-predictions$predictions)))
  mdae=(median(abs(validation_set$dependent-predictions$predictions)))
  rmse=sqrt(mean((validation_set$dependent-predictions$predictions)^2))
  return(list(mae=mae,mdae=mdae,rmse=rmse,predictions=predictions$predictions))
}



