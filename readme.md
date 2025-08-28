Multiclass classification using EEG data. 
4 Classes - Left, Right, Tongue, Foot.
Data Link : https://www.kaggle.com/datasets/aymanmostafa11/eeg-motor-imagery-bciciv-2a/data

```python meta.py``` to train the model

- Feature Extraction - Load CSV & Pad Epochs ➔ Normalize & Calculate FFT ➔ Create Sliding Windows. Combine Time/Frequency Features ➔ Reshape for Model ➔ Encode Labels.
- Model Description - Hybrid model. Bidirectional Gru extracts temporal features. 1D CNN extracts spatial features from the GRU's output which is pooled and passed to a classifier.
- 5-fold cross validation training achieved average validation accuracy of 93%

Experiments are logged using MLflow in a Metaflow Pipeline.

















