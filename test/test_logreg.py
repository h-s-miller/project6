import regression
import numpy as np
from sklearn.preprocessing import StandardScaler

def model_for_testing():
        X_train, X_test, y_train, y_test = regression.loadDataset(features=['Penicillin V Potassium 500 MG',
                                                                                  'Computed tomography of chest and abdomen',
                                                                                  'Plain chest X-ray (procedure)', 'Low Density Lipoprotein Cholesterol',
                                                                                  'Creatinine'], split_percent=0.8)
        lr = regression.LogisticRegression(X_train.shape[1]) #default parameters
        lr.train_model(X_train, y_train, X_test, y_test)
        return lr

def test_updates():
        ### load data for testing ###
        lr=model_for_testing()
        
	### check that loss decreases ###
        loss_hist=lr.loss_history_train
        beginning_avg=np.mean(loss_hist[:5])
        end_avg=np.mean(loss_hist[-5:])
        assert beginning_avg > end_avg, "Loss not getting smaller during training."

        ### check that loss approaches zero ###
        assert loss_hist[-1]<1, "Loss not close enough to zero"  # i got this number from testing in main.py

        ### check loss decreasing another way ###
        deltas=[loss_hist[i+1]-loss_hist[i] for i in range(len(loss_hist)-1)]
        assert np.mean(deltas)<0, "Loss sequence not decreasing"

def test_predict():
        X_train, X_test, y_train, y_test = regression.loadDataset(features=['Penicillin V Potassium 500 MG',
                                                                                  'Computed tomography of chest and abdomen',
                                                                                  'Plain chest X-ray (procedure)', 'Low Density Lipoprotein Cholesterol',
                                                                                  'Creatinine'], split_percent=0.8)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform (X_test)
        

        lr = regression.LogisticRegression(X_train.shape[1]) #default parameters
        lr.train_model(X_train, y_train, X_test, y_test)

        ### test that predictions are somewhat accurate ###
        X_test=np.column_stack((X_test, np.ones(X_test.shape[0]))) ## add in bias
        y_pred=lr.make_prediction(X_test) ## predict the values of the validation set
        MSE=np.mean((y_pred-y_test)**2)
        assert MSE < 0.25 
