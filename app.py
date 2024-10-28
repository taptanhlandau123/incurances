
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

app = Flask(__name__)

# Đọc dữ liệu từ file CSV
file_path = "insurances.csv"
insurance = pd.read_csv(file_path)

# Mã hóa biến phân loại
insurance['smoker'] = insurance['smoker'].map({'yes': 1, 'no': 0})
insurance['sex'] = insurance['sex'].map({'male': 0, 'female': 1})

# Chọn các cột đầu vào và đầu ra
X = insurance[['age', 'bmi', 'children', 'smoker', 'sex', 'region']]
y = insurance['charges']

# Chuyển đổi các giá trị phân loại của 'region' thành dummy variables
X = pd.get_dummies(X, columns=['region'], drop_first=True)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu cho mô hình MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện các mô hình một lần
model_linear = LinearRegression()
model_lasso = Lasso(max_iter=1000)
model_mlp = MLPRegressor(max_iter=1000)
base_models = [
    ('linear', model_linear),
    ('lasso', model_lasso),
    ('mlp', model_mlp)
]
model_stacking = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# Fit các mô hình
model_linear.fit(X_train, y_train)
model_lasso.fit(X_train, y_train)
model_mlp.fit(X_train_scaled, y_train)
model_stacking.fit(X_train, y_train)
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    rmse_train = None
    mse_train = None
    r2_train = None

    if request.method == 'POST':
        age = int(request.form['age'])
        bmi = float(request.form['BMI'])
        children = int(request.form['children'])
        smoke = 1 if request.form['smoke'] == 'yes' else 0
        area = request.form['area']
        sex = request.form['sex']  # Lấy giới tính

        # Mã hóa biến giới tính
        sex = 1 if sex == 'female' else 0

        # Tạo DataFrame mới
        X_new = pd.DataFrame([[age, bmi, children, smoke, sex, area]], columns=['age', 'bmi', 'children', 'smoker', 'sex', 'region'])
        X_new = pd.get_dummies(X_new, columns=['region'], drop_first=True)

        # Đảm bảo rằng tất cả các đặc trưng cần thiết đều có
        for col in X.columns:
            if col not in X_new.columns:
                X_new[col] = 0  # Thêm cột thiếu và gán giá trị 0

        X_new = X_new[X.columns]  # Sắp xếp lại cột theo thứ tự của X_train

        # Chuẩn hóa đầu vào nếu chọn mô hình MLP
        model_choice = request.form.get('model_choice', 'linear')
        if model_choice == 'mlp':
            X_new = scaler.transform(X_new)

        # Dự đoán dựa trên mô hình đã được huấn luyện
        if model_choice == 'lasso':
            prediction = model_lasso.predict(X_new)[0]
        elif model_choice == 'mlp':
            prediction = model_mlp.predict(X_new)[0]
        elif model_choice == 'stacking':
            prediction = model_stacking.predict(X_new)[0]
        else:  # Mặc định là hồi quy tuyến tính
            prediction = model_linear.predict(X_new)[0]

        # Tính RMSE, MSE và R-squared cho mô hình được chọn
        if model_choice == 'lasso':
            predictions = model_lasso.predict(X_train)
            rmse_train = np.sqrt(np.mean((predictions - y_train) ** 2))
            mse_train = np.mean((predictions - y_train) ** 2)
            r2_train = model_lasso.score(X_train, y_train)
        elif model_choice == 'mlp':
            predictions = model_mlp.predict(X_train_scaled)
            rmse_train = np.sqrt(np.mean((predictions - y_train) ** 2))
            mse_train = np.mean((predictions - y_train) ** 2)
            r2_train = model_mlp.score(X_train_scaled, y_train)
        elif model_choice == 'stacking':
            predictions = model_stacking.predict(X_train)
            rmse_train = np.sqrt(np.mean((predictions - y_train) ** 2))
            mse_train = np.mean((predictions - y_train) ** 2)
            r2_train = model_stacking.score(X_train, y_train)
        else:  # Mặc định là hồi quy tuyến tính
            predictions = model_linear.predict(X_train)
            rmse_train = np.sqrt(np.mean((predictions - y_train) ** 2))
            mse_train = np.mean((predictions - y_train) ** 2)
            r2_train = model_linear.score(X_train, y_train)

    return render_template('giaodien.html', prediction=prediction, rmse_train=rmse_train, mse_train=mse_train, r2_train=r2_train)
if __name__ == '__main__':
    app.run(debug=True)

