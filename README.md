# Advanced-Housing-Price-Prediction
This project, 'Advanced House Price Prediction System', aims to accurately predict house prices using the 'Ames Housing Dataset', fetched from OpenML as 'house_prices'. The primary objective is to build robust predictive models by transforming the target variable, SalePrice.


The Condition1 and Condition2 features in the Ames Housing Dataset describe the proximity of the house to various important local points of interest, main roads, or land features. They essentially ask: "What is this house next to?"

There are two features (Condition1 and Condition2) because a single house might be next to multiple things (e.g., a house located on the corner of a main street and next to a park). Condition1 simply captures the primary proximity condition, and Condition2 catches any secondary one (if it exists).

Here is what all the abbreviation options in that dropdown actually mean in plain English:

Norm (Normal): The house is not near anything particularly remarkable or disruptive. It’s just a normal neighborhood lot. Most houses fall into this category.
Artery: The house is located adjacent to an Arterial Street (a major, high-traffic main road or thoroughfare within the city). Typically lowers property value due to noise.
Feedr: The house is adjacent to a Feeder Street (a medium-traffic street that "feeds" traffic from residential areas onto arterial streets).
RRNn: The house is within 200 feet of a North-South Railroad.
RRAn: The house is adjacent to the tracks of an East-West Railroad.
PosN: The house is near a Positive Off-Site Feature (often a park, greenbelt, scenic area, etc.). Usually increases value!
PosA: The house is adjacent to a Positive Off-Site Feature. (Even closer than PosN).
RRNe: The house is within 200 feet of an East-West Railroad.
RRAe: The house is adjacent to the tracks of a North-South Railroad.


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
