# **Public Transportation Demand Forecasting**
Forecasting of the demand prediction for public transportation in different time layers (hourly,daily,weekly)

# **Description of the folders/files structure**
* Folder data: contains the required datasets
* File data/test_data.txt: the required sample dataset for the testing of the algorithm
* Folder scripts: contains the required scripts
* File scripts/training.R: the R file that contains the functions for the training
* File main.R: the R file that should be called in order to run the algorithm
* File requirements.txt: the file containing the technical specification (version of language, required packages)
* File readme.md the file that describes the functionality of the algorithm and the pre-requisiste steps in order to test the algorithm


# **Input**
The algorithm uses the file "data/test_data.txt" as an input in order to traing the algorithm

# **Run the algorithm**
* Open the command line
* Find the path of the folder "public-transportation-demand-forecasting"
* Execute the following command: Rscript main.R


# **Output**
The algorithm displays the predictions of the daily demand forecasting for the last 45 days, the actual values and the corresponding evaluation metrics(MAE, MdAE, RMSE, Normalized MAE, Normalized MdAE, Normalized RMSE)

# **Contact**
Georgios Spanos, gspanos@iti.gr