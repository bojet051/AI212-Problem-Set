# Hatch Model: Bayesian vs Non-Bayesian Linear Regression

Raw selected shape: 128,066 rows x 16 columns.  
Cleaned data shape: 71,773 rows x 20 columns.  
Modeling shape: 71,773 rows x 20 columns.  

Response: `CLEAN_CHICKS_RATIO` = `CLEAN_CHICKS / EGGSET`.  
Excluded predictor: `EGGSET`, because it is already the response denominator.  
Raw candidate predictors: `HY, LOCATION, WEEK_NO, HATCH_DATE, DATE_SET, FARM_SOURCE, LINE, B, H, STRAIN, PRODN_DATE, AGE, GRP, S.P.`.  
Categorical candidates not encoded: `HY, LOCATION, FARM_SOURCE, LINE, B, H, STRAIN`.  
Date/week candidates not engineered: `WEEK_NO, HATCH_DATE, DATE_SET, PRODN_DATE`.  
Selected raw numeric predictors: `AGE, GRP, S.P.`.  

OLS full model training RMSE: 0.131681 ratio units (13.168 percentage points). R2=0.4485; adjusted R2=0.4485; AIC=-87330.53.  

Bayesian weak-prior training RMSE: 0.131681.  
Bayesian informative-prior training RMSE: 0.131681.  

Best OLS 5-fold CV: k=3, predictors=AGE + GRP + S.P., CV RMSE=0.131687.  
Best OLS 0.632 bootstrap: k=3, predictors=AGE + GRP + S.P., RMSE=0.131677.  
Best OLS AIC: k=3, predictors=AGE + GRP + S.P., AIC=-87330.53.  
Best Bayesian 5-fold CV: k=3, predictors=AGE + GRP + S.P., CV RMSE=0.131687.  
Best Bayesian WAIC: k=3, predictors=AGE + GRP + S.P., WAIC=-87327.59.  

Conclusion: EGGSET was excluded as a predictor because it is the denominator of the response. The fitted sequence uses only selected raw numeric predictors from the raw candidate pool. Categorical and date/week raw candidates are valid predictors for a future encoded or engineered model, but were not transformed in this no-feature-engineering comparison.
