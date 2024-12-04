import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# بيانات صناعية مضافة بنوع الوقود والصيانة
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=365),
    'Fuel Type': np.random.choice(['Natural Gas', 'Coal', 'Nuclear'], size=365),  # نوع الوقود
    'Fuel Consumption (m³)': np.random.randint(1000, 5000, size=365),  # استهلاك الوقود
    'Maintenance Needed': np.random.choice([0, 1], size=365, p=[0.9, 0.1]),  # الصيانة الوقائية
    'Temperature (°C)': np.random.uniform(10, 40, size=365),  # درجة الحرارة
    'Load (MW)': np.random.randint(300, 1500, size=365),  # الطلب على الطاقة
    'Humidity (%)': np.random.uniform(30, 80, size=365),  # نسبة الرطوبة
    'Wind Speed (km/h)': np.random.uniform(0, 20, size=365),  # سرعة الرياح
    'Electricity Production (MW)': np.random.randint(500, 2000, size=365)  # إنتاج الطاقة الكهربائية
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

# تقييم النموذج
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

# عرض النتائج
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual Electricity Production')
plt.plot(y_pred, label='Predicted Electricity Production', linestyle='--')
plt.title('Actual vs Predicted Electricity Production')
plt.xlabel('Time')
plt.ylabel('Electricity Production (MW)')
plt.legend()
plt.show()
