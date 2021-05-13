# SHOW AI Toolbox

## AI Toolbox Description

### 1. About
The AI toolbox is an integral part of the WP5 processes. It will contain all the necessary algorithms and methods that will be used to develop and deploy all the services which will be part of the demonstrations in the sites.
In order to collect all the algorithms that reflet the expertise of the partners involved a central space will be deployed. Namely, a git repository system. There are numerous git web services available but to protect the privacy and ensure the safety of the files a private git server could be deployed (e.g. GitLab). In the Git repo mentioned each partner will create an account and could be able to upload all the necessary files that comprise the tools. 
### 2. Components of a tool 
Every tool uploaded in the repository should contain all the components described below. This way the repository will have a structured design and each partner can understand and deploy all the tools as needed. Every component must be placed in a folder named appropriately if possible, unless otherwise stated. 

#### 2.1 Code files
The code files are the core of the tool. Indicative folder names are “src” or the name of the algorithm. It should contain all the code files that are necessary to run the algorithm. 
#### 2.2 Input data
Input data should be provided in order to be able to run each tool. This data can be either a sample or a whole dataset depending on the availability and privacy. Every piece of data should be anonymized according to GDPR to avoid privacy violations. Indicative folder names are “data” or “input data”.
#### 2.3 Example/Test files
An example of the execution of the tool should be included. In case of a ML algorithm a test file (e.g. test.py) where the tool is run using a test set. A non-ML algorithm should be accompanied with an example file (e.g. example.py) where the tool is run using a set of parameters. Indicative folder names are “test” or “example”.
#### 2.4 Description and instructions 
Every tool uploaded to the repository should accompanied with a clear description along with detailed instructions on how to run the tool. The description should state what the tool accomplices and input and anticipated output based on the data provided in 2.2. This should be in the from of “README.md” that git repositories support.
#### 2.5 Requirements 
This file should contain all the libraries’ requirements and dependencies of the tool. 
