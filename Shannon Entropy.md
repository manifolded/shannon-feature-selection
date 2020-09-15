


The core idea of this project is to explore using Shannon entropy, more precisely mutual information (MI), as a tool for feature selection.  The fact that this exploration took place while analyzing the 2017 StackExchange developer survey dataset is merely incidental.

I would be lying if I claimed that this project came about any way other than me deciding at the outset that I wanted to do something with Shannon entropy.  I'm besotted; it's just so sexy.  I figured Shannon entropy had to have some offspring that would be useful for comparing features.  A little research showed that indeed there is; it's called mutual information^2.  I was elated.

I make no claim that this is a thorough or complete investigation.  I'm just reporting on a quick experiment I did for a bootcamp project.  Nevertheless, I think you'll enjoy it.

### Implementation

At the risk of leaving out the best part, I'm going to be brief in my discussion of my implementation.  Suffice it to say that computing the mutual information consists mainly of computing the Shannon entropy, but using a two variable joint probability distribution.  

In case you're interested, MI is computed like this:

![](Images/mutualInfo.png)

Computing MI requires performing a binning procedure as if you were constructing a sort of 2-D histogram.  That makes it sound harder than it is.  Really... it's no big deal.  All you're trying to do is get an estimate for that joint probability thingy.

I wrote my own 2D binning code^3.  There is a `histogram2d()` function included in NumPy but I declined to use it.  That would be no fun.  However, NumPy's code is wildly faster than mine, probably due to some secret vectorization incantations.  At some point I'll have to go take a look at how they did it.

I did take advantage of the existing code, `value_counts()`, to pick out my bins for me.  That's a fiddly job at best and I didn't feel like reinventing the wheel.

### Discussion

A competing method (one we used in class) for evaluating which features were closely connected to each other was to go and fit the model to all the features and then simply examine the coefficients.  There's nothing wrong with that approach, it's just the process gets a little ungainly when applied to categorical features required replacing said feature with handfuls of binary dummy columns.  It wasn't very appealing for an exploratory pass.

With that in mind, here's the big takeaway message.  Mutual information treats categorical and quantitative columns the same way.  No need for dozens of binary dummy columns.  It just works.

Ok.  I've been muttering long enough, let's look at some data.

### Results

Before you even say it, there's no reason the on-diagonal entries below (e.g. comparing Salary with Salary) should be precisely 1.  This isn't correlation data.  That said, knowing all the data in the Salary column tells you a lot about the data in the Salary column.  We shouldn't be too surprised to find that the mutual information between Salary and itself is large.

Just to give a sense of scale, MI is normalized (-ish) so numbers near unity are considered large.  Numbers an order of magnitude smaller are not.  At 0.77 the MI between Salary and Country is big; super big.

![](Images/Four%20Column%20Heatmap.png)

By the way... these results match the coefficient comparison approach perfectly which bolsters my confidence in the mutual information approach.

And did I mention that we have a takeaway message?  Country is a categorical feature and Salary is quantitative.  They were compared as if it was nothing.  Easy breezy.

=============================================================

### Bibliography

1. [https://en.wikipedia.org/wiki/Mutual_information](https://en.wikipedia.org/wiki/Mutual_information)
2. [https://en.wikipedia.org/wiki/Entropy_(information_theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))
3. [https://amethix.com/entropy-in-machine-learning/](https://amethix.com/entropy-in-machine-learning/)
4. [https://www.quora.com/What-is-the-relationship-between-entropy-and-information](https://www.quora.com/What-is-the-relationship-between-entropy-and-information)
5. [https://github.com/manifolded/shannon-feature-selection](https://github.com/manifolded/shannon-feature-selection)

=============================================================

### Footnotes

1. I'm trained as a physicist, and Shannon's entropy looks just like Boltzman's.  Irresistible.
2. Don't let the 'entropy' vs. 'information' dichotomy confuse you.  As counter-intuitive as it might seem, I double checked and in the context of Shannon entropy they're equivalent.  Huzzah.
3. I have a fantasy that I'm going to cook up a super clever scheme for handling NaNs so I decided to cook up my own binning code so I could muck about in total freedom.  Maybe this will lead to another blog post.