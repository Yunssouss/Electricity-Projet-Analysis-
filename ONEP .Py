import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# إنشاء بيانات صناعية كتقرب لواقع الإنتاج الكهربائي
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=365),
    'Fuel Consumption (m³)': np.random.randint(1000, 5000, size=365),  # استهلاك الوقود بالغاز الطبيعي
    'Temperature (°C)': np.random.uniform(10, 40, size=365),  # درجة الحرارة
    'Load (MW)': np.random.randint(300, 1500, size=365),  # الطلب على الطاقة الكهربائية
    'Humidity (%)': np.random.uniform(30, 80, size=365),  # نسبة الرطوبة
    'Wind Speed (km/h)': np.random.uniform(0, 20, size=365),  # سرعة الرياح
    'Electricity Production (MW)': np.random.randint(500, 2000, size=365)  # إنتاج الطاقة الكهربائية
})

# تحضير البيانات
X = data[['Fuel Consumption (m³)', 'Temperature (°C)', 'Load (MW)', 'Humidity (%)', 'Wind Speed (km/h)']]
y = data['Electricity Production (MW)']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

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
