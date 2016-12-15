Introduction
------------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively.

These type of devices are part of the quantified self movement - a group
of enthusiasts who take measurements about themselves regularly to
improve their health, to find patterns in their behavior, or because
they are tech geeks.

One thing that people regularly do is quantify how much of a particular
activity they do, but they rarely quantify how well they do it.

In this project, the goal will be to use data from accelerometers on the
belt, forearm, arm, and dumbell of 6 participants. They were asked to
perform barbell lifts correctly and incorrectly in 5 different ways.

More information is available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

Data
----

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source:

<http://groupware.les.inf.puc-rio.br/har>.

### About Data

#### Human Activity Recognition

Human Activity Recognition - HAR - has emerged as a key research area in
the last years and is gaining increasing attention by the pervasive
computing research community, especially for the development of
context-aware systems. There are many potential applications for HAR,
like: elderly monitoring, life log systems for monitoring energy
expenditure and for supporting weight-loss programs, and digital
assistants for weight lifting exercises.

#### Load packages

    library(caret)
    library(rpart)
    library(rpart.plot)
    library(RColorBrewer)
    library(rattle)
    library(randomForest)
    library(AppliedPredictiveModeling)

### Download data

    download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "data_train.csv")

    download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "data_test.csv")

### Read data

    training <- read.csv("data_train.csv", na.strings=c("NA","#DIV/0!",""))
    validation <- read.csv("data_test.csv", na.strings=c("NA","#DIV/0!",""))
    dim(training); dim(validation)

    ## [1] 19622   160

    ## [1]  20 160

    head(training$classe);class(training$classe); summary(training$classe)

    ## [1] A A A A A A
    ## Levels: A B C D E

    ## [1] "factor"

    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

Cleaning Data
-------------

    ### We Remove columns with Near Zero Values
    NZV <- nearZeroVar(training, saveMetrics = TRUE)
    myTraining <- training[, !NZV$nzv]

    ### Remove columns with NA or is empty

    myTraining <- myTraining[, names(myTraining)[sapply(myTraining, function (x)
            ! (any(is.na(x) | x == "")))]]

    ### Removing  columns of the dataset that is unlikely to influence the prediction

    Useless <-grepl("^X|timestamp|user_name", names(myTraining))
    myTraining <- myTraining[, !Useless]
    rm(Useless)

### Separate the data to be used for Cross Validation

Divide training set into 2 parts

    set.seed(1234)
    inTrain <- createDataPartition(y=myTraining$classe, p=0.6, list=FALSE)
    mytraining <- myTraining[inTrain, ]; mytesting <- myTraining[-inTrain, ]
    dim(mytraining); dim(mytesting)

    ## [1] 11776    54

    ## [1] 7846   54

We can test 2 models: Decision Tree Model and Random forest

#### 1. Decision Tree Model

As the outcomes are categorical (nominal), a decision tree was the first
model tested using the method rpart.

    set.seed(1234)
    modFit1<-train(classe ~ .,method="rpart", data=mytraining)
    print(modFit1$finalModel)

    ## n= 11776 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
    ##    2) roll_belt< 130.5 10774 7436 A (0.31 0.21 0.19 0.18 0.11)  
    ##      4) pitch_forearm< -34.55 919    2 A (1 0.0022 0 0 0) *
    ##      5) pitch_forearm>=-34.55 9855 7434 A (0.25 0.23 0.21 0.2 0.12)  
    ##       10) magnet_dumbbell_y< 436.5 8314 5944 A (0.29 0.18 0.24 0.19 0.11)  
    ##         20) roll_forearm< 122.5 5137 3022 A (0.41 0.18 0.18 0.17 0.061) *
    ##         21) roll_forearm>=122.5 3177 2124 C (0.08 0.18 0.33 0.23 0.18) *
    ##       11) magnet_dumbbell_y>=436.5 1541  743 B (0.033 0.52 0.039 0.23 0.18) *
    ##    3) roll_belt>=130.5 1002   10 E (0.01 0 0 0 0.99) *

    plot(modFit1$finalModel, main = "Classification tree")
    text(modFit1$finalModel, use.n = TRUE, all = TRUE, cex = .7)

![](https://github.com/xetaro/Practical-Machine-Learning-Course-Project/blob/master/plot_modFit1.png)



    fancyRpartPlot(modFit1$finalModel,cex=.8,under.cex=0.4)

![](https://github.com/xetaro/Practical-Machine-Learning-Course-Project/blob/master/plot_modFit1_Final_Model.png)

    predict1 <- predict(modFit1,mytesting)
    RpartCM <-confusionMatrix(mytesting$classe, predict1)
    RpartCM

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2029   44  155    0    4
    ##          B  638  505  375    0    0
    ##          C  644   49  675    0    0
    ##          D  567  232  487    0    0
    ##          E  209  211  383    0  639
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4904          
    ##                  95% CI : (0.4793, 0.5016)
    ##     No Information Rate : 0.5209          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.3339          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.4965  0.48511  0.32530       NA  0.99378
    ## Specificity            0.9460  0.85114  0.87992   0.8361  0.88852
    ## Pos Pred Value         0.9091  0.33267  0.49342       NA  0.44313
    ## Neg Pred Value         0.6334  0.91530  0.78388       NA  0.99938
    ## Prevalence             0.5209  0.13268  0.26447   0.0000  0.08195
    ## Detection Rate         0.2586  0.06436  0.08603   0.0000  0.08144
    ## Detection Prevalence   0.2845  0.19347  0.17436   0.1639  0.18379
    ## Balanced Accuracy      0.7212  0.66812  0.60261       NA  0.94115

#### 2. Random forest

    set.seed(1234)
    modFit2 <- randomForest(classe ~ ., data=mytraining)
    predict2 <- predict(modFit2, mytesting,  type = "class")
    RFCM <-confusionMatrix(predict2, mytesting$classe)
    RFCM

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2232    5    0    0    0
    ##          B    0 1510    6    0    0
    ##          C    0    3 1361   10    0
    ##          D    0    0    1 1275    2
    ##          E    0    0    0    1 1440
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9964          
    ##                  95% CI : (0.9948, 0.9976)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9955          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9947   0.9949   0.9914   0.9986
    ## Specificity            0.9991   0.9991   0.9980   0.9995   0.9998
    ## Pos Pred Value         0.9978   0.9960   0.9905   0.9977   0.9993
    ## Neg Pred Value         1.0000   0.9987   0.9989   0.9983   0.9997
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1925   0.1735   0.1625   0.1835
    ## Detection Prevalence   0.2851   0.1932   0.1751   0.1629   0.1837
    ## Balanced Accuracy      0.9996   0.9969   0.9964   0.9955   0.9992

    plot(modFit2, lwd = 3, main = "Prediction model random forest")

![](https://github.com/xetaro/Practical-Machine-Learning-Course-Project/blob/master/plot_modFit2.png)

#### Compare models and choose with the best accuracy (random forest)

    result <- data.frame(RpartCM$overall, RFCM$overall)
    result

    ##                RpartCM.overall RFCM.overall
    ## Accuracy             0.4904410    0.9964313
    ## Kappa                0.3338858    0.9954857
    ## AccuracyLower        0.4793210    0.9948463
    ## AccuracyUpper        0.5015681    0.9976274
    ## AccuracyNull         0.5209024    0.2844762
    ## AccuracyPValue       1.0000000    0.0000000
    ## McnemarPValue              NaN          NaN

Conclusion
----------

Random Forest was a good model for prediction of exercise quality
compared to rpart. The RF model had over 99% accuracy and fitted well to
other subsamples of the data.

In the first model D was the most difficult to predict.

#### Generating Files to submit as answers for the Assignment:

We use Random Forests for our prediction:

    predictions <- predict(modFit2, validation, type = "class")
    predictionsFinal <- data.frame(validation$problem_id, predictions)
    

    #### Function to generate files with predictions to submit for assignment

    pml_write_files = function(x){
      n = length(x)
      for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
      }
    }

    pml_write_files(predictions)
