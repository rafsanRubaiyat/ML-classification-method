Full report can be found <a href="[https://www.w3schools.com](https://drive.google.com/file/d/1eFIUoX1UZ2dowTc9dOvNWFddC-JY86Qq/view?usp=sharing)">HERE</a> <br>

# ML-classification-method
In this study, we have gathered data from high schools in Portugal, which included details about social and school-related features for each student. Our main goal is to implement and compare various ML classification methods to predict whether a student is likely to pursue higher education based on the pertinent factors found in our dataset.  

## Contributors
Burke Grey, Ryan McAllister, Rafsan Siddiqui

## Contents
- Introduction/Cleaning 
- Exploratory Analysis
- Discriminant Analysis
- Logistic Regression
- Classification trees
- Random forests
- SVM 
- Naive Bayes Classification
- Neural Networks -
- Comparisons & Conclusion

## Background:
We obtained a dataset containing information on academic performances, social characteristics, and demographics of students in two Portuguese high schools Again, our objective is to utilize classification methods to predict whether a student would like to go to college based on many of the other features in our dataset Data Source: UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/dataset/320/student+performance)

## Raw Data Characteristics:
1,044 observations; 33 total variables 
Binary Response: ‘higher’ “Yes”: 955 “No”: 89 

### Findings:
#### Method	and their Misclassification Error Rates:  <br>
<table>
  <tr>
    <th>Classification Method</th>
    <th>Misclassification Error Rate</th>
  </tr>
  <tr>
    <td>Ordinary QDA</td>
    <td>0.2643</td>
  </tr>
  <tr>
    <td>QDA With Leave One Out Cross-Validation</td>
    <td>0.2786</td>
  </tr>
    <tr>
    <td>QDA With 10-fold Cross-Validation	 </td> 
    <td>0.2857</td>
  </tr>
    <tr>
    <td>Logistic regression</td>
    <td>0.2738</td>
  </tr>
    <tr>
    <td>Classification tree (unpruned) 	</td>
    <td>0.0833</td>
  </tr>
    <tr>
    <td>Classification tree (pruned, 4 terminal nodes)	</td>
    <td>0.2262</td>
  </tr>
    <tr>
    <td>Random Forest (m=24, ntrees=500)	</td>
    <td>0.1786</td>
  </tr>
    <tr>
    <td>Random Forest (m=8, ntrees=95)	</td>
    <td>0.2143 </td>
  </tr>
    <tr>
    <td>Support Vector Machines (Cost=5) 	</td>
    <td>0.2737</td>
  </tr>
    </tr>
    <tr>
    <td>Naive Bayes Classifier	  	</td>
    <td>0.1428</td>
  </tr>
    </tr>
    <tr>
    <td>Neural Networks (4 Hidden Layers)	</td>
    <td>0.0178</td>
  </tr>
  
</table>
 		 <br>


We implemented the ordinary QDA without cross-validations. Hence, among the three QDA methods, it has the lowest error rate. QDA with leave-one-out cross-validation used almost all the data for testing, resulting in a lower error rate than QDA with 10-fold cross-validation. 

The higher error rate for Logistic regression is logical considering the architecture of the method itself. It assumes a simple linear relationship between predictors and the log-odds of the outcome, hence it performs poorly when the true relationship is non-linear or complex. It’s also sensitive to outliers, and extreme values can influence the model's coefficients. Compared to other methods such as trees or random forests, it’s generally less accurate. 

The unpruned classification tree performed significantly better than our pruned tree. However, this could be because our dataset is largely homogeneous, so although the unpruned tree tends to overfit the data, the performance on the test data was very good. It’s hard to conclude that the performances for the unpruned and the pruned would vary so much for other datasets. Also, when we look at the results from the random forests, the overfitting pattern becomes more evident. In contrast, with a very simple model with only 4 terminal nodes, the pruned tree demonstrated a pretty good performance.   

The random forest with m=24 and ntrees=500 is likely to have less error than our optimal random forest with m=8 and ntrees=95 because it employs a larger number of decision trees and explores a broader set of features at each split, thus resulting in a more robust and accurate model. 

We have picked the optimal cost for our SVM considering the error rate and the margin width. However, the performance of the SVM model is worse than most of the other methods. Tuning the hyperparameters such as regularization parameters, and kernel-specific parameters may boost the performance more.
 
In contrast, the Naive Bayes Classifier has demonstrated impressive performance on our original dataset. The method simply assumes that the features are conditionally independent given the class label (Rish, I., 2001), and both the number of observations and features of our dataset are small. These might have played some role in the performance of this classifier method. 

Finally, with little wonder, our Neural Networks model outperformed every method by a big margin. By design, Neural Networks are inherently non-linear models, can handle complex datasets and complex relationships among variables. It’s proven that Neural Network almost always performs better than most of the classification techniques.

#### Methods and the Most Important Features <br>
Logistic regression 	1) G3; 2) Tutoring; 3) Address; 4) Famsup <br>
Random Forest (m=8, ntrees=95), MDA	1) Studytime; 2) G3; 3) Fedu; 4) Goout; 5) Failures <br>
Random Forest (m=8, ntrees=95), MDG	1) Studytime; 2) G3; 3) Fedu; 4) Goout; 5) Medu <br>

G3 (result) is consistently one of the most important features both in the logistic regression model and the random forest model. This is expected, as the result of the student is understandably a significant determinant for the student to aspire to pursue higher education. We have removed a bunch of features from the logistic regression as they had multicollinearity issues, and then the stepwise model has removed some other features. So, the difference between the logistic regression model and the random forest model in terms of the set of important features is explainable and not surprising. By design, the Mean Decrease Accuracy (MDA) and Mean Decrease Gini (MDG) have different ways of measuring how much a variable contributes to the overall accuracy of the Random Forest model. So, although they agree on the first 4 most important features, they disagree on the fifth one. 

### Directions on further investigation

Firstly, it's important to note that the dataset used in this project was relatively small. To gain a comprehensive understanding and facilitate meaningful comparisons of the methods employed, we need a substantially larger dataset or consider acquiring additional data for this project. 

Hyperparameter tuning for SVM, and other classification methods may provide a better understanding of the algorithms and a better result. We can also investigate ensemble techniques like stacking or boosting, and by combining the strengths of multiple models we can improve predictive performance further. When the dataset is large, we may try different cross-validation techniques, such as stratified sampling, or bootstrapping, to assess model stability and generalization capacity.

L2 regularization can be used to prevent overfitting and improve the generalization ability of the logistic regression model. L2 regularization based on the L2 norm would penalize large values of the coefficients, effectively shrinking them towards zero. This will help prevent overfitting by reducing the model's complexity (Phaisangittisagul, E., 2016).
Finally, in addition to misclassification errors, the ROC (Receiver Operating Characteristic) curve needs to be used for evaluating and comparing the performance of the classification models.  

### Conclusion

The choice of classification method should be considered based on the dataset's nature and complexity. While simpler models like logistic regression may be enough for certain problems, more complex models like Neural Networks can offer substantial performance gains when dealing with complex data and relationships.  


