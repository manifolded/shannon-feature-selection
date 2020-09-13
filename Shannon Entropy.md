## Hidden Amongst The Disorder

The 2017 StackExchange developer's survey includes a daunting 154 features.  Selecting a small focused group for further analysis requires figuring out which features are relevant and which are just going to pollute your model.

There are a variety of tools available, but I find Shannon entropy to be particularly fascinating.  My training is in physics so I am familiar with the log(N) formulation of entropy from Statistical Mechanics.  To me the Shannon entropy, with its log(1/p), looks very familiar.





By itself Shannon's discovery doesn't provide us much insight for feature selection.  What we really want is a metric for quantifying the degree of coincidence shared by two features in our dataset. Mutual information, an offshoot of Shannon entropy, provides just such a metric.

Because of this correspondence we treat entropy and information as equivalent.


### Towards Mutual Information

As discussed above, the Shannon entropy itself is not terribly useful for feature selection.  If we want to assess the relationship between two features in our dataset, X and Y, we had better compute our Shannon entropy using their joint distribution. 

~~H(X, Y) = - \sum_{i j} p_{i j} log(p_{i j})~~

The last thing we need to do is to subtract off the contributions from each of the individual Shannon entropies.

MI = H(X,Y) - H(X) - H(Y) = - \sum_{i j} p_{i j} log(p_{i j}/p_i p_j)

The quantity that remains is immune to the entropy of each variable separately and instead only indicates the degree to which they are intertwined, i.e. it's exactly what we're looking for.


One of the great strengths of mutual information as a metric for quantifying feature relevance is that it can be computed on any variable that can be binned.  It treats categorical variables exactly the way it treats quantitative variables.

### Implementation

I implemented my code in Python using a Jupyter notebook.  I made heavy use of the pandas data analysis tool and the NumPy scientific computing package.  


We can take the log in any base we desire, but the most common practice is to perform the logs base 2.  This allows us to interpret the results as the number of bits required to encode the feature, or pair of features.


### Assessing Mutual Information As A Practical Tool

The following plot shows a typical result of a computation of the MI for pairs selected from four features from the 2017 StackExchange Developer Survey dataset.  ~~Salary and JobSatisfaction are quantitative features while HomeRemote and Country are categorical.~~  JobSatisfaction is the respondent's rating between 1 and 10 and thus is treated as quantitative.  So is Salary which is in US dollars and is also represented by an integer.  The Country question asks the respondent to select their country of residence from a long list and is thus  categorical; a similar story for HomeRemote.

Our ultimate goal is to construct a model which will accurately predict the salary of the respondent.  Towards that end we wish to select those features which will provide the most influential inputs to our model.  Of the four candidates shown below the strongest off-diagonal MI score with Salary is for the Country feature.  Salary and Country share a strong MI score of 0.77, almost 1.0.  In contrast, both HomeRemote and JobSatisfaction appear to have negligible MI with Salary.

[[[The word "influence" suggests a causal connection which MI cannot address.]]]


![](Images/Four%20Column%20Heatmap.png)

### The Real Story

One problem I encountered is that my binning code is relatively slow and surely inefficient.  Computing the mutual information for four features took an uncomfortable several minutes on my aging MacBook Pro; about the same amount of time required to fit a linear model to all 150 some odd features in the data set.

On the other hand, fitting that linear model required replacing every categorical feature with a profusion of boolean dummy columns; one for every response.  

We used the coefficients from the linear model to judge how important each feature had been in predicting the response; ~~a somewhat circular argument.~~  In contrast, measuring the MI between any pair of features requires the addition of no dummy columns at all and the idea is to only include those features which we anticipate will be influential based on their MI scores.

Finally, in the absence of time restrictions I would love to have gone on to identify all the (pairs of) features in the data set with a significant MI.  Perhaps Salary isn't the most interesting feature to focus on.

And I fully intend to explore some novel approaches to handling NaNs.  It strikes me that a non-answer to a question contains nearly as much information as an answer.  A NaN represents a valid response and should be treated as such.  In the Shannon entropy context we have the opportunity to treat all variables as categorical and we can count NaNs by putting them in just another bin.

Whether either of these two approaches would lead to a marked improvement in real world modeling remains to be seen.

==========================================================

### Bibliography

1. [https://en.wikipedia.org/wiki/Mutual_information](https://en.wikipedia.org/wiki/Mutual_information)
2. [https://www.wolframalpha.com/input/?i=plot+-p+log+p+from+0+to+1](https://www.wolframalpha.com/input/?i=plot+-p+log+p+from+0+to+1)
3. [https://en.wikipedia.org/wiki/Entropy_(information_theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))
4. [https://amethix.com/entropy-in-machine-learning/](https://amethix.com/entropy-in-machine-learning/)
5. [https://www.quora.com/What-is-the-relationship-between-entropy-and-information](https://www.quora.com/What-is-the-relationship-between-entropy-and-information)

==========================================================

### Footnotes

1. NumPy actually includes a function called `histogram2d()` which generates 2-D histograms like the one we want.  I chose not to use histogram2d() and instead to implement my own because I anticipate the need for a more sophisticated approach to handling missing data (NaNs).  More on this in a future blog post!
