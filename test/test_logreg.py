import regression
import numpy as np

def model_for_testing():
        X_train, X_test, y_train, y_test = regression.loadDataset(features=['Penicillin V Potassium 500 MG',
                                                                                  'Computed tomography of chest and abdomen',
                                                                                  'Plain chest X-ray (procedure)', 'Low Density Lipoprotein Cholesterol',
                                                                                  'Creatinine'], split_percent=0.8)
        lr = regression.LogisticRegression(X_train.shape[1])
        lr.train_model(X_train, y_train, X_test, y_test)
        return lr

def test_updates():
        ### load data for testing ###
        lr=model_for_testing()
        
        # Check that your gradient is being calculated correctly
        

	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training
        
	### check that loss decreases ###
        loss_hist=lr.loss_history_train
        beginning_avg=np.mean(loss_hist[:5])
        end_avg=np.mean(loss_hist[-5:])
        assert beginning_avg > end_avg, "Loss not getting smaller during training."

        ### check that loss approaches zero ###
        assert loss_hist[-1]<1, "Loss not close enough to zero"  # i got this number from testing in main.py

        ### check loss decreasing another way ###
        deltas=[loss_hist[i+1]-loss_hist[i] for i in range(len(loss_hist)-1)]
        assert np.mean(deltas)<0, "Loss sequence not decreaseing"

def test_predict():
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

	pass
