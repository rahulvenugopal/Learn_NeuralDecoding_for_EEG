## Mind flaying aka MVPA `Multi Variate Pattern Analysis`
![](https://github.com/rahulvenugopal/Learn_NeuralDecoding_for_EEG/blob/main/images/Prof-xavier.jpg)
---

---
### Trivia
- Difference between `scikit-learn` and `mne.decoding.Scaler`

`scikit-learn` scales each *classification feature* (each time point across channels) with mean and standard deviation computed across epochs.
`mne.decoding.Scaler` scales each *channel* using mean and standard deviation computed across all of its time points and epochs.
- *Vectorizer*

scikit-learn transformers and estimators generally expect 2D data (n_samples * n_features), whereas MNE transformers typically output data
with a higher dimensionality (e.g. n_samples * n_channels * n_frequencies * n_times).
A Vectorizer therefore needs to be applied between the MNE and the scikit-learn steps

---
### Resources and Inspirations
1. [Decreasing alertness modulates perceptual decision-making](https://github.com/SridharJagannathan/decAlertnessDecisionmaking_JNeuroscience2021)
2. [MNE demo from Richard Höchenberger's workshop]()