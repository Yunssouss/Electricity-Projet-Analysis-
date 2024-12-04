import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# إنشاء البيانات الصناعية الخاصة بالصيانة التنبؤية
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-11-01', periods=30, freq='D'),  # شهر نونبر
    'Equipment Age (years)': [5]*30,  # عمر المعدات
    'Previous Maintenance': [0]*20 + [1]*10,  # العمليات السابقة ديال الصيانة
    'Sensor Temperature (°C)': [60, 62, 61, 63, 65, 64, 66, 70, 75, 80] * 3,  # درجة الحرارة ديال المستشعر
    'Sensor Vibration (Hz)': [10, 12, 15, 14, 18, 20, 19, 25, 28, 30] * 3,  # الاهتزازات
    'Electricity Demand (MW)': [400, 420, 450, 430, 480, 470, 490, 510, 530, 520] * 3,  # الطلب على الطاقة
    'Hour of Day': list(range(0, 30)),  # الساعة
    'Day of Week': [i % 7 for i in range(30)],  # اليوم فالأسبوع
    'Season': [4]*30,  # موسم: 4 كيُمثّل الخريف
    'Maintenance Needed': [0]*25 + [1]*5  # خاص الصيانة الوقائية (0 = ماخصاش دابا, 1 = خاصها)
})

# تحضير البيانات
X = data[['Equipment Age (years)', 'Previous Maintenance', 'Sensor Temperature (°C)', 'Sensor Vibration (Hz)',
          'Electricity Demand (MW)', 'Hour of Day', 'Day of Week', 'Season']]
y = data['Maintenance Needed']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحسين Hyperparameters باستعمال GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
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
rmse = mse ** 0.5
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء وتدريب النموذج
model = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=50, subsample=1.0)
model.fit(X_train, y_train)

# توقع الوقت المتوقع لحدوث العطب
predicted_failure_time = model.predict(X_test)

# حالة الآلة الحقيقية (مثلا)
actual_status = y_test

# عرض النتائج
print("Predicted failure time:", predicted_failure_time)
print("Actual machine status:", actual_status)





# عرض النتائج
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual Maintenance Needed')
plt.plot(y_pred, label='Predicted Maintenance Needed', linestyle='--')
plt.title('Actual vs Predicted Maintenance Needed')
plt.xlabel('Time')
plt.ylabel('Maintenance Needed')
plt.legend()
plt.show()
