# Advanced-Housing-Price-Prediction
This project, 'Advanced House Price Prediction System', aims to accurately predict house prices using the 'Ames Housing Dataset', fetched from OpenML as 'house_prices'. The primary objective is to build robust predictive models by transforming the target variable, SalePrice.


Summary:

Data Analysis Key Findings


Initial Model Performance Comparison:

Linear Regression achieved the best initial performance with the lowest RMSE (0.131899) and the highest R2 Score (0.906772).

Ridge Regression followed closely with RMSE 0.136200 and R2 Score 0.900593.

Random Forest and Lasso Regression showed moderate performance (RMSE 0.147999, R2 0.882623 for Random Forest; RMSE 0.149388, R2 0.880410 for Lasso).

XGBoost initially had the highest RMSE (0.152282) and lowest R2 Score (0.875732), indicating suboptimal default parameters.


Hyperparameter Tuning with Optuna (XGBoost):
Optuna was used to tune the XGBoost model by minimizing an objective function that calculated RMSE on the test set.
The search space included max_depth (3-9), learning_rate (0.01-0.3), n_estimators (100-500), subsample (0.6-1.0), and colsample_bytree (0.6-1.0).
Tuning significantly improved XGBoost's performance: RMSE decreased from 0.152282 to 0.128498 and R2 Score increased from 0.875732 to 0.911516. This made the tuned XGBoost the best-performing model.


SHAP Explainability:
shap.TreeExplainer was used on the tuned XGBoost model to compute SHAP values, which represent each feature's contribution to a prediction.
The shap.summary_plot revealed that features like num__OverallQual (Overall Material and Finish Quality), num__GrLivArea (Above grade living area), num__TotalBsmtSF (Total basement square feet), and num__GarageCars (Size of garage in car capacity) were the most influential.
The plot showed that higher values of features like num__OverallQual and num__GrLivArea correlate with higher predicted house prices.


Insights or Next Steps
Hyperparameter tuning, even for models like XGBoost, is crucial for unlocking optimal performance, as demonstrated by the significant improvement from an RMSE of 0.152282 to 0.128498, making it the top-performing model.
Integrating model explainability through SHAP values provides transparent insights into feature importance and their directional impact on predictions, which is essential for building trust and understanding the underlying drivers of house prices in this system.
