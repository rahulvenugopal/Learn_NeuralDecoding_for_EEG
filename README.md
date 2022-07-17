## Mind flaying aka MVPA `Multi Variate Pattern Analysis`
![](https://github.com/rahulvenugopal/Learn_NeuralDecoding_for_EEG/blob/main/images/Prof-xavier.jpg)
---
- We are interested in **decoding over time** from *epochs x channels x time points* EEG data
- This strategy consists in fitting a multivariate predictive model on each time instant and evaluating its performance at the same instant on new epochs
- `X` is the epochs data of shape n_epochs × n_channels × n_times
As the last dimension of `X` is the time, an estimator will be fit on every time instant
- We can retrieve the **spatial filters** and **spatial patterns** if we explicitly use a `LinearModel`
- `get_coef` function can fetch the `patterns`. Make sure we do `inverse_transform=True`

>
- Extraction filters of backward models may exhibit large weights at channels not at all picking up the signalsof-interest, as well as small weights at channels containing the signal
- Such “misleading” weights are by no means indications of suboptimal model estimation
- Rather, they are needed to “filter away” noise and thereby to extract the signal with high SNR


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

- **Crossvalidation**
[](https://github.com/rahulvenugopal/Learn_NeuralDecoding_for_EEG/blob/main/images/CrossValidation.png)

- **On the interpretation of weight vectors of linear models in multivariate neuroimaging** is an awesome paper which explains why `patterns instead of `filters` in `get_coef` function



---
### Resources and Inspirations
1. [Decreasing alertness modulates perceptual decision-making](https://github.com/SridharJagannathan/decAlertnessDecisionmaking_JNeuroscience2021)
2. [MNE demo from Richard Höchenberger's workshop]()