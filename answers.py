r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. **False**.
Test set is for estimating out-of-sample error.
For estimating in-sample error we have validation subset.

2. **False**
There is a tradeoff between training quality and validation capabilities when we split our data.
For example, if our training portion is too little, our model wont actually "learn" that much.
On the other hand, if the validation portion is too little, our validation wont be that robust.

3. **True**
Test set is meant for us to check whether we can generalize our our prediction.
If we use it for cross validation, it is not a generalization when we test with it, as it was part of the training process.

4. **False**
Generalization error is calculating by testing the model's performance using the test set.
Validation set performance of each fold help determine our in-sample loss and tune the hyper parameters.
"""

part1_q2 = r"""
In doing so, our friend is trying to find the best lambda value for the w he found earlier.
However, he might find a better w-lambda pair if he trains his model to find them all together.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
Increasing k helps us generalize our model to some extent.
If we increase k to much, then we will over fit our model to our data, as it will consider a too many neighbors for each new data point (neighbors which most will be irrelevant).
On the other hand, for k values too little, our model wont use most of the data we trained it with.
"""

part2_q2 = r"""
**Your answer:**
1. Training our model that way will cause us to over fit our model to the training data.
Having no validation portions will prevent us from estimating our generalization capability.

2. This will also cause us to lose generalization capabilities.
The generalization error will be calculated over already seen data (a data leakage is occurring)

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
Selection of delta is arbitrary because it does not affect the relation between loss of different weight matrices.
If we multiply delta in a certain scalar, this will effect all Li(W) components that was already effected by the same amount. for components that were not effected, they will stay zero anyway.
this propagates to the sum, L(W) which will grow by a certain amount (up to delta), but it will still be consistant with other losses.

"""

part3_q2 = r"""
1. In general, linear classifiers divide our data space into classes, with hyperplane dividers.
We can interpret our model learning as fining said hyperplanes for which the outliers of each class will be as close as possible to said hyperplane.
Classification errors will occur when a data point closer to the predicted class' hyperplane, than its true class'.

2. in KNN we dont try to split out world into classes (by hyperplanes).
instead, in each prediction we draw a sphere around the new point which includes exactly k points from the training set.
then we check what class is most common among these point and predict out new point to be of said class.
"""

part3_q3 = r"""
**Your answer:**
Our model's learning rate is too high. This can be induced by the fact that our loss improvement is not consistant (has "ups and downs").
This happens because in some epochs, our model advances too much towards the minimum (in the gradient direction) and "over shoots" the target.
If the learning rate was good, we would have seen a smoother decent.
If the learning rate was too low, the loos curve would have taken a long time to converge towards the minimum.

Given the accuracy graph, we can see that our model is slightly overfitted to the training data.
We are able to generalize it a little, however the validation accuracy is significantly lower that the training accuracy.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
Ideally, we would like all the residuals to be zero.
Given that it is not possible with real life data, we would like to have a zero mean and low std for our graph.

"""

part4_q2 = r"""
1. Adding non linear features to our data will cause the model to be non-linear in feature space.
It will be linear only in the extracted space of linear features.
2. Yes.
However, this will require us to hand craft features (for example by polynomial mapping)
3. Adding non linear features will cause our classifier to draw hyperplanes only in the projected (mapped) space, and not the original feature space.

"""

part4_q3 = r"""
1. using log space enable us to turn our losses to positives only.
2. We fitted the data K(folds) times for each lambda-degree pair.
in our case that means 3 degrees * 20 lambdas * 3 folds = 120 fittings.

"""

# ==============
