# **Ridesharing**

Matching algorithm that matches drivers with passengers into shared trips. 

# **Pre-requisites**
Python 3.8.8

# **Description of the model files**
* matching_model.py : Contain the mattching MIP model. Also contain methods to handle I/O in json (dict) format. 
* matching_utils.py : Helper functions. Use the test_input.json to create the distance matrix between commuters. Additionally, creates 3 possible 
paths for drivers and the nearest distance of each passegner to each path. 
* test.py: test the module 
* test_input.json : Contain the clients. Each one has the starting point coordinates, time-window, the capacity (if it is driver), 
 and role (0: passenger, 1: driver).
* output_test.json : The matching output. In key 'output' the dictionary contains the driver and the passengers along with the meeting point coordinates. 

# **Citing this model**
In progress... 


# **Contact**
Zisis Maleas, zisismaleas@certh.gr

