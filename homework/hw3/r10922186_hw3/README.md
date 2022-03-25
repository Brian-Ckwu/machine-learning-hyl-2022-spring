# Machine Learning 2022 Spring HW3

## Data
Put the food11 data folder here (i.e. there should be a "food11" data folder in r10922186_hw3)

## Download the Model
    bash ./download.sh
The model will be in ./models

## Make Prediction on Test Data
    python test.py
The prediction file (.csv) will be in ./preds

## Reproducibility
1. First set the "tfm" field in config.json to "test", and run "python train.py" (~12 hrs)
2. Then set the "tfm" field in config.json to "AfCropHoriPersChoice" and "trained_model" to "model-SampleClassifier_optimizer-Adam_lr-0.001_wd-0.0_bs-256_nepochs-500_tfm-affine", then run "python train.py" (~12 hrs)

The above steps should be able to reproduce the model in ./models/best_model.pth, which will be stored in ./models/model-SampleClassifier_optimizer-Adam_lr-0.001_wd-0.0_bs-256_nepochs-500_tfm-AfCropHoriPersChain/model.pth