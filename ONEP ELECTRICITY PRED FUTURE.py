import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# بيانات لشهر 11 (من 1 حتى 30 يوم)
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-11-01', periods=30),
    'Fuel Type': ['Natural Gas', 'Coal', 'Natural Gas', 'Nuclear', 'Coal', 'Natural Gas', 'Coal', 'Nuclear', 
                  'Natural Gas', 'Coal', 'Nuclear', 'Natural Gas', 'Coal', 'Nuclear', 'Coal', 'Natural Gas',
                  'Nuclear', 'Coal', 'Natural Gas', 'Nuclear', 'Coal', 'Natural Gas', 'Nuclear', 'Coal', 
                  'Natural Gas', 'Coal', 'Nuclear', 'Natural Gas', 'Coal', 'Natural Gas'],  # نوع الوقود
    'Fuel Consumption (m³)': [4000, 3500, 4200, 5000, 3800, 4300, 3700, 5200, 
                              4100, 3600, 4900, 4250, 3850, 5050, 3750, 4350,
                              5150, 3950, 4400, 5100, 3900, 4450, 5000, 3750, 
                              4500, 3650, 4900, 4300, 3700, 4600],  # استهلاك الوقود
    'Maintenance Needed': [0, 0, 1, 0, 0, 0, 1, 0, 
                           0, 0, 0, 0, 1, 0, 0, 0,
                           1, 0, 0, 0, 1, 0, 0, 0,
                           0, 1, 0, 0, 0, 0],  # الصيانة الوقائية
    'Temperature (°C)': [25, 18, 20, 22, 19, 21, 23, 20, 
                         22, 19, 24, 21, 18, 23, 19, 22,
                         20, 23, 21, 19, 24, 21, 20, 22,
                         18, 23, 19, 24, 21, 22],  # درجة الحرارة
    'Load (MW)': [1300, 1200, 1100, 1400, 1150, 1350, 1250, 1450,
                  1200, 1300, 1400, 1150, 1350, 1250, 1400, 1200,
                  1100, 1250, 1350, 1200, 1450, 1300, 1400, 1150,
                  1250, 1200, 1450, 1350, 1100, 1300],  # الطلب على الطاقة
    'Humidity (%)': [60, 55, 50, 65, 58, 62, 57, 60, 
                     55, 63, 50, 60, 52, 55, 60, 58,
                     62, 55, 63, 60, 57, 55, 62, 60,
                     65, 60, 58, 55, 60, 63],  # نسبة الرطوبة
    'Wind Speed (km/h)': [15, 10, 12, 18, 14, 16, 13, 17, 
                          14, 12, 16, 15, 10, 17, 13, 16,
                          18, 15, 13, 17, 12, 14, 16, 15,
                          10, 17, 13, 12, 14, 16],  # سرعة الرياح
    'Electricity Production (MW)': [1500, 1400, 1350, 1600, 1450, 1550, 1500, 1650, 
                                    1400, 1500, 1600, 1450, 1550, 1500, 1600, 1450,
                                    1350, 1550, 1500, 1400, 1650, 1500, 1600, 1450,
                                    1500, 1400, 1650, 1550, 1350, 1500]  # إنتاج الطاقة الكهربائية
})

# تحويل نوع الوقود إلى بيانات رقمية
data['Fuel Type'] = data['Fuel Type'].map({'Natural Gas': 1, 'Coal': 2, 'Nuclear': 3})

# تحضير البيانات
X = data[['Fuel Type', 'Fuel Consumption (m³)', 'Maintenance Needed', 'Temperature (°C)', 'Load (MW)', 'Humidity (%)', 'Wind Speed (km/h)']]
y = data['Electricity Production (MW)']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحسين Hyperparameters باستعمال GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

xgboost_model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(estimator=xgboost_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# أفضل Hyperparameters
best_params = grid_search.best_params_
print(f'أفضل المعلمات: {best_params}')

# تدريب النموذج باستعمال أفضل Hyperparameters
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

import numpy as np
# تقييم النموذج
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

print("Predicted values:", y_pred)
print("Actual values:", y_test)



# عرض النتائج
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual Electricity Production')
plt.plot(y_pred, label='Predicted Electricity Production', linestyle='--')
plt.title('Actual vs Predicted Electricity Production')
plt.xlabel('Time')
plt.ylabel('Electricity Production (MW)')
plt.legend()
plt.show()
