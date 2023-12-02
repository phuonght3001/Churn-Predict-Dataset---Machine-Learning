# Churn Predict Dataset - Machine-Learning
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
#### 2.2.2.3 Analyze each feature:
- Tenure: We can see that Churn users are usually New users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/0207edfc-9cc6-4d50-8d28-15f7ef88376f)

- Preferred Login Device: Both churn and not churn users tend to use mobile phone.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/042d68de-1683-4a81-b8ae-2e54d2f99c6b)

- City Tier: Both churn and not churn users live in City tier 1 and 3 tend to churn more.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/4aa7a958-4be4-459d-a775-54e10f0de0e4)

- Warehouse To Home: There is not significantly different about Warehouse to Home between churn and not churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/9e0d8f1f-4b53-4449-a59c-2e1a7b22c0bc)

- Preferred Payment Mode: There is not significantly different about Preferred Payment Mode between churn and not churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/f7503bc8-ec1b-4c14-afdd-d228851064d8)

- Gender: There is not a big difference between the males and the females.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/6fce031e-6805-4e35-89ee-7e0fc91af036)

- Hour Spend On App: There is not significantly different about Hour Spend On App between churn and not churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/58921d0f-f1c4-4564-8b39-b8225a0c5df5)

- Number Of Device Registered: Users using more devices tend to be churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/6d030892-8e5f-4fc7-b068-990ee46ef97f)

- Prefered Order Cat: Churn users prefer to buy Mobile Phone than not churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/6bd0345c-5a2c-451e-8bfa-867b20320347)

- Satisfaction Score: There is not significantly different about Satisfaction Score between churn and not churn users.
![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/932e8c76-63e7-4ec8-8a16-10a8d4a426cc)

- Marital Status: Users have single status that tend to be churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/47142988-5585-4225-9763-9a0164c7434d)

- Number Of Address: There is not significantly different about Number Of Address between churn and not churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/8c2b91c1-8acf-4f02-93a8-a6adc5290d5d)

- Complain: Churn users tend to have more complaints.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/cb75b240-9b48-479c-9579-862078186025)

- Order Amount Hike From Last Year: There is not significantly different about Order Amount Hike From last Year between churn and not churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/c4fc5b19-4fdd-407d-b3b8-6fe063022d9c)

- Coupon Used: There is not significantly different about Coupon used between churn and not churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/b816373e-f46a-4722-adb6-9a9286385fe9)

- Order ount: There is not significantly different about Ordercount between churn and not churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/a2e8be3c-80a6-47b0-a236-b710353763df)

- DaySinceLastOrder: There is not significantly different about Day Since Last Order between churn and not churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/17776fec-5751-4945-b483-f791aa6aa00f)

- Cashback Amount: Churn users receive less cashback amout than not churn users.

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/b2a3e34f-0e7a-496e-a3e0-0f2cabf11ce3)
#### 2.2.2.4 Conclusion
After EDA, we can keep the below columns to the model:
- Tenure
- PreferredLoginDevice
- CityTier
- NumberOfDeviceRegistered
- PreferedOrderCat
- MaritalStatus
- Complain
- CouponUsed
- CashbackAmount
### 2.3 Data Transforming:
#### 2.3.1 Encoding and Normalizing:
- Encoding:

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/32dc3e3f-edbc-4f8e-9d7d-c971885b7d31)

- Normalizing:

![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/a292e38f-3928-4d97-abdf-4109cc737b04)
#### 2.3.2 Apply Model:
##### 2.3.2.1 Model: Logistic Regression:
![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/44613e3b-5de4-4e8c-b056-2ac11c085aae)
##### 2.3.2.2 Model: Decision Tree:
![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/b70b5c23-4094-4d21-a918-1c436f266360)
##### 2.3.2.3 Model: Random Forest:
![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/e84a573e-d73b-4f02-bbbe-31c73c10cc72)

=> Choose the best model: Random Forest Model with highest accuracy both train and test data.
#### 2.3.3 Enhanced Random Forest model:
![image](https://github.com/phuonght3001/Churn-Predict-Dataset---Machine-Learning/assets/150796721/ceadba29-e00a-40f3-82f7-a88895e19cdf)

=> 2 features that impact to churn users most are Cashback Amount and Tenure.

