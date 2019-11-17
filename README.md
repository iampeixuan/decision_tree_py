# decision_tree_py
- My numpy only Decision Tree models
- Part of this project is homework from NUS CS5228 Knowledge Discovery and Data Mining
- Currently only supports hyperparameters "max_depth" and "min_samples_split" for "MyDecisionTreeRegressor"
- Besides, "MyGradientBoostingRegressor" supports "n_estimators" and "learning_rate"
 
## Python environment setup
- conda create --name py36 python=3.6
- conda activate py36
- pip install numpy pandas jupyter notebook scikit-learn matplotlib
- (or pip install -r requirements.txt)

## Checkout "compare_regressor.ipynb" 
- for how to use "MyDecisionTreeRegressor" and "MyGradientBoostingRegressor"for regression tasks (1d or higher)
- for result comparison with sklean models
