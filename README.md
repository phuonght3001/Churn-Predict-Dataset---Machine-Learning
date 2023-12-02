# Churn-Predict-Dataset---Machine-Learning
Use Python to analyze and find out the best model that predicts churn customes effectively.
## 1. Introduction
One ecommerce company has a project on predicting churned users in order to offer potential promotions.
## 2. Analysis and Choose the best model
### 2.1 Data Preparation
#### Get the data:
![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/1e652623-3efb-48af-bfcb-b326cf003ed1)
#### Handle missing and duplicated values:
- Missing values:
![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/9a859ede-ba12-428d-87a9-e8af58888b4d)
- Fill missing values with median:
![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/afae4502-26c1-4510-969b-03922398dae3)
- Handle features that have the same values but splitted into 2 different names:
  + In feature 'PreferredLoginDevice': Mobile phone and phone are the same.
  + In feature 'PreferedOrderCat': Mobile and Mobile phone are the same.
  + In feature 'PreferredPaymentMode': COD & Cash on Delivery are the same, CC and Credit Card are the same.
### 2.2 EDA
#### 2.2.1 Create heatmap to see correlation of  feature "Churn" and the other features:
![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/c62d484f-8fdc-48dd-a238-fdaef691d2a5)

=> We found that there are not any columns that have correlation with "Churn" column
#### 2.2.2 Visualize to decide the columns:
##### 2.2.2.1 Density of numeric features by Churn:
![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/029bd0ab-8f1c-46d2-a33e-933b731dd40e)

=> Distribution insights of numeric features:
+ Tenure: Customers with longer tenure seem less likely to churn. Makes sense as longer tenure indicates satisfaction.
+ WarehouseToHome: Shorter warehouse to home distances have a lower churn rate. Faster deliveries may improve satisfaction.
+ HourSpendOnApp: More time spent on app correlates with lower churn. App engagement is a good sign.
+ OrderAmountHikeFromLastYear: Big spenders from last year are less likely to churn. Good to retain big customers.
+ CouponUsed: Coupon usage correlates with lower churn. Coupons enhance loyalty.
+ OrderCount: Higher order counts associate with lower churn. Frequent usage builds habits.
+ DaySinceLastOrder: Longer since last order correlates with higher churn. Recency is a good predictor.
+ Cashbackamount: Higher cashback correlates with lower churn. Need more promotion about cashback.
##### 2.2.2.2 Density of category features by Churn:
![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/b64d89bb-e763-467d-b157-e4363402c152)

=> Distribution of category by Churn:
- CityTier: Churn rate looks similar across tiers. City tier does not seem predictive of churn.
- NumberOfDeviceRegistered: More registered devices associates with lower churn. Access across devices improves convenience.
- SatisfactionScore: Higher satisfaction scores strongly associate with lower churn, as expected. Critical driver.
- NumberOfAddress: Slight downward trend in churn as number of addresses increases. More addresses indicates loyalty.
- Complain: More complaints associate with higher churn, though relationship isn't very strong. Complaints hurt satisfaction.
#### 2.2.2.3 Count percentage of each feature with "Churn":
- Tenure:

