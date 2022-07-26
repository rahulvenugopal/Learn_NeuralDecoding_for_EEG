## Mind flaying aka MVPA `Multi Variate Pattern Analysis`
![](https://github.com/rahulvenugopal/Learn_NeuralDecoding_for_EEG/blob/main/images/Prof-xavier.jpg)
---

### Key notes
- We are interested in **decoding over time** from *epochs x channels x time points* EEG data
- This strategy consists in fitting a multivariate predictive model on each time instant and evaluating its performance at the same instant on new epochs
- `X` is the epochs data of shape n_epochs × n_channels × n_times
As the last dimension of `X` is the time, an estimator will be fit on every time instant
- We can retrieve the **spatial filters** and **spatial patterns** if we explicitly use a `LinearModel`
- `get_coef` function can fetch the `patterns`. Make sure we do `inverse_transform=True`

> Extraction filters of backward models may exhibit large weights at channels not at all picking up the signals of interest, as well as small weights at channels containing the signal

> Such “misleading” weights are by no means indications of suboptimal model estimation

> Rather, they are needed to “filter away” noise and thereby to extract the signal with high SNR

> We derived a transformation by which extraction filters of any linear backward model can be turned into activation patterns of a corresponding forward model. By this means, backward models can eventually be made interpretable

---

### Temporal generalisation - an extension of the decoding over time approach
- Evaluating whether the model estimated at a particular time instant accurately predicts any other time instant
- It is analogous to transferring a trained model to a distinct learning problem, where the problems correspond to decoding the patterns of brain activity recorded at distinct time instants
- The diagonal line in the plot below is exactly same as the time-by-time decoding plot
![](https://github.com/rahulvenugopal/Learn_NeuralDecoding_for_EEG/blob/main/images/TemoralGeneralisation.png)

---

### Trivia
- Difference between `scikit-learn` and `mne.decoding.Scaler`

`scikit-learn` scales each *classification feature* (each time point across channels) with mean and standard deviation computed across epochs

`mne.decoding.Scaler` scales each *channel* using mean and standard deviation computed across all of its time points and epochs
- **Vectorizer**
scikit-learn transformers and estimators generally expect 2D data (n_samples * n_features), whereas MNE transformers typically output data
with a higher dimensionality (e.g. n_samples * n_channels * n_frequencies * n_times)

A Vectorizer therefore needs to be applied between the `MNE` and the `scikit-learn` steps

- **Solvers**
Lookup the **LogisticRegression** solvers documentation. `liblinear` is faster than `lbfgs`
`sag` and `saga` are faster for larger datasets
For multiclass problems, only `newton-cg`, `sag`, `saga` and `lbfgs` handle multinomial loss
`liblinear` is limited to one-versus-rest schemes
- [Logistic regression python solvers' definitions](https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions)

- **Crossvalidation**

![](https://github.com/rahulvenugopal/Learn_NeuralDecoding_for_EEG/blob/main/images/CrossValidation.png)

- **On the interpretation of weight vectors of linear models in multivariate neuroimaging** is an awesome paper which explains why `patterns` instead of `filters` in `get_coef` function makes more sense

- Normally when an optimization algorithm **does not** converge, it is usually because training data is not normalised
- Set `max_iter` to a larger value
---

### Points to ponder
- Shuffling the dataset
- StratifiedKFold
- Using features instead of time series
- I think the limitations are from the kind of features that can be computed on a single time point!
- `F1 scores` vs `roc_auc`
- About balancing the classes

---
### Resources and Inspirations
1. [Decreasing alertness modulates perceptual decision-making](https://github.com/SridharJagannathan/decAlertnessDecisionmaking_JNeuroscience2021)
2. [MNE demo from Richard Höchenberger's workshop](https://www.youtube.com/watch?v=t-twhNqgfSY)

---
# [spatio_temporal_cluster_test](https://mne.tools/stable/generated/mne.stats.spatio_temporal_cluster_test.html#mne.stats.spatio_temporal_cluster_test) in `mne-python`
- The data **X** should be of the form observations*time*channels
- All dimensions except the first should match across all groups
- The threshold has to be a dictionary for TFCE method
- To understand what the hell is `TFCE` read up the awesome blogposts from Benedikt Ehinger's blog (checkout resources)
- tail `0` is two tailed test
- provide adjacency info so that `TFCE` knows the neighboring channels
- `t_power` will count locations or weigh each location by its statistical score
![](https://github.com/rahulvenugopal/Learn_NeuralDecoding_for_EEG/blob/main/images/Cluster%20statistics%20_220726_092943_1.jpg)

### Motivation for TFCE
- Threshold might miss broad but `weak` clusters, and focus only on `strong` but peaky clusters
- The intuition of TFCE is that we are going to try out all possible thresholds and see whether a given time-point belongs to a significant cluster under any of our set of cluster-thresholds
- TFCE will be a weighted average between the cluster extend and cluster height
- i.e how many extended samples and how large the t value is / the evidence for an effect
![](https://github.com/rahulvenugopal/Learn_NeuralDecoding_for_EEG/blob/main/images/TFCE_intuition.png)

### Result images looks like below
![](https://github.com/rahulvenugopal/Learn_NeuralDecoding_for_EEG/blob/main/images/TFCE_output.png)

### Resources
1. [Threshold Free Cluster Enhancement explained](https://benediktehinger.de/blog/science/threshold-free-cluster-enhancement-explained/)
2.[Statistics: Cluster Permutation Test](https://benediktehinger.de/blog/science/statistics-cluster-permutation-test/)
3. [Know your neighbours when calculating how channels are connected spatially](https://www.fieldtriptoolbox.org/faq/how_does_ft_prepare_neighbours_work/)

*Read this [post](https://www.fieldtriptoolbox.org/faq/how_not_to_interpret_results_from_a_cluster-based_permutation_test/) before writing TFCE results*
