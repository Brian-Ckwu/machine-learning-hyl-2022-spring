## Steps to Reproduce the Result

1. Put the data in "./libriphone"
2. Run lstm_train.py (the trained model will be saved in "./models")
3. Run lstm_test.py (the inference .csv file will be saved in "./preds", note that the batch_size of test_loader should not be modified)