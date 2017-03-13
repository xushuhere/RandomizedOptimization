The code used in this assignment relies on ABAGAIL. 
Required Software
-----------------

*  ABAGAIL   
*  R 3.1.X  


Data
----

The data set used is from the UCI machine learning data set, which is the breast cancer data set. The whole data set, and the test and train data set are available in the "data" folder:

*  btest.csv : is the validation/test data set  
*  btrain.csv : is the training data set  




ABAGAIL 
------

**Neural Networks Optimization**

To run the code for determining optimal weights for neural networks, set up java in the IDE and run /src/NNTrain.java
Please note a couple of “TODO comment” to change the parameter for optimal results

The results are saved into /Results folder

The data used for the report was stored in /Results/Part1 and /Results/Part2

The detailed data were saved into Dropbox following this link:
https://www.dropbox.com/sh/gnkqb0lv6tn3mwh/AABZKWYIGdttAnPdoywi8pfFa?dl=0

The folder are named based on the running parameters.


**Optimization problems**
-------

The other 3 problems were completed using the ABAGAIL library.
Run the java script of FourPeaksOpt.java, KnapsackOpt.java, ContinuousPeaksOpt.java, TravelingSalesmanOpt.java and OptimizationTest.java under /src folder

The results of OptimizationTest.java is stored under /Optimization_Results folder

The other three java scripts print out the result and manually import into csv for analysis.


R
—————
R was used to plot the data, several R scripts are stored under /R folder.

Plot code is located in "R/" and requires R to run. To run the plots ensure you have the following R packages installed:

```
magrittr
dplyr
ggplot2
gtable
scales
gsubfn
gridExtra
```

The various scripts are described as follows:

*  Plot_csv.Rmd : used to generate the graphs

