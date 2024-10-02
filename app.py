"""import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np

app = Flask(__name__)

# Dummy data for prediction
X_train = np.array([[25, 22.0, 0, 1, 0], [30, 28.0, 1, 0, 1], [22, 24.0, 0, 1, 1]])
y_train = np.array([2000, 3000, 1500])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    rmse_train = None
    r2_train = None
    convergence_warning = None  # Khởi tạo biến với giá trị mặc định

    if request.method == 'POST':
        age = int(request.form['age'])
        bmi = float(request.form['BMI'])
        children = int(request.form['children'])
        smoke = 1 if request.form['smoke'] == 'yes' else 0
        area = request.form['area']
        
        # Prepare input data
        area_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
        area_encoded = area_mapping[area]
        X_new = np.array([[age, bmi, children, smoke, area_encoded]])
        
        # Model selection
        model_choice = request.form.get('model_choice', 'linear')  # Default to linear regression

        # Thêm cross-validator với số splits ít hơn số lượng mẫu
        kf = KFold(n_splits=2)

        if model_choice == 'lasso':
            model = Lasso(max_iter=1000)
        elif model_choice == 'mlp':
            model = MLPRegressor(max_iter=1000)
        elif model_choice == 'stacking':
            base_models = [
                ('linear', LinearRegression()),
                ('lasso', Lasso(max_iter=1000)),
                ('mlp', MLPRegressor(max_iter=1000))  # Thay thế Ridge bằng MLP
            ]
            model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), cv=kf)
        else:  # Default to linear regression
            model = LinearRegression()

        # Fit model
        model.fit(X_train, y_train)

        # Check for convergence
        if hasattr(model, 'n_iter_'):
            if model.n_iter_ >= model.max_iter:
                convergence_warning = "Cảnh báo: Mô hình không hội tụ. Có thể cần điều chỉnh tham số."
            else:
                convergence_warning = None
        else:
            convergence_warning = None

        # Scale input data and make prediction
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_new_scaled = scaler.transform(X_new)
        
        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_new_scaled)[0]

        # RMSE and R-squared can be calculated here, currently using dummy values
        rmse_train = np.sqrt(np.mean((model.predict(X_train_scaled) - y_train) ** 2))
        r2_train = model.score(X_train_scaled, y_train)

    return render_template('giaodien.html', prediction=prediction, rmse_train=rmse_train, r2_train=r2_train, convergence_warning=convergence_warning)

if __name__ == '__main__':
    app.run(debug=True)"""


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
file_path = "C:/Users/fpt/Documents/insurance.csv"  # Đường dẫn đến file CSV
insurance = pd.read_csv(file_path)

# Mã hóa biến phân loại
insurance['smoker'] = insurance['smoker'].map({'yes': 1, 'no': 0})
insurance['sex'] = insurance['sex'].map({'male': 0, 'female': 1})

# Chọn các cột đầu vào và đầu ra
X = insurance[['age', 'bmi', 'children', 'smoker', 'region']]
y = insurance['charges']

# Chuyển đổi các giá trị phân loại của 'region' thành số
X = pd.get_dummies(X, columns=['region'], drop_first=True)  # Mã hóa khu vực

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
model_mlp.fit(X_train, y_train)
model_stacking.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    rmse_train = None
    r2_train = None

    if request.method == 'POST':
        age = int(request.form['age'])
        bmi = float(request.form['BMI'])
        children = int(request.form['children'])
        smoke = 1 if request.form['smoke'] == 'yes' else 0
        area = request.form['area']

        # Mã hóa biến khu vực
        area_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
        area_encoded = area_mapping[area]

        # Tạo DataFrame cho dữ liệu đầu vào
        X_new = pd.DataFrame([[age, bmi, children, smoke, area_encoded]], columns=['age', 'bmi', 'children', 'smoker', 'region'])
        X_new = pd.get_dummies(X_new, columns=['region'], drop_first=True)

        # Đảm bảo rằng tất cả các đặc trưng cần thiết đều có
        for col in X.columns:
            if col not in X_new.columns:
                X_new[col] = 0  # Thêm cột thiếu và gán giá trị 0

        X_new = X_new[X.columns]  # Sắp xếp lại cột theo thứ tự của X_train

        # Lựa chọn mô hình
        model_choice = request.form.get('model_choice', 'linear')

        # Dự đoán dựa trên mô hình đã được huấn luyện
        if model_choice == 'lasso':
            prediction = model_lasso.predict(X_new)[0]
        elif model_choice == 'mlp':
            prediction = model_mlp.predict(X_new)[0]
        elif model_choice == 'stacking':
            prediction = model_stacking.predict(X_new)[0]
        else:  # Mặc định là hồi quy tuyến tính
            prediction = model_linear.predict(X_new)[0]

        # RMSE và R-squared
        rmse_train = np.sqrt(np.mean((model_linear.predict(X_train) - y_train) ** 2))
        r2_train = model_linear.score(X_train, y_train)

    return render_template('giaodien.html', prediction=prediction, rmse_train=rmse_train, r2_train=r2_train)

if __name__ == '__main__':
    app.run(debug=True)

