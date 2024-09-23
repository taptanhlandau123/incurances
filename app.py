
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

app = Flask(__name__)

# Đọc và chuẩn bị dữ liệu
file_path = './insurances.csv'
insurance = pd.read_csv(file_path)
insurance_encoded = pd.get_dummies(insurance, columns=['sex', 'smoker', 'region'], drop_first=True)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
numeric_features = ['age', 'bmi', 'children']
insurance_encoded[numeric_features] = scaler.fit_transform(insurance_encoded[numeric_features])

# Huấn luyện mô hình Lasso
X = insurance_encoded[['age', 'bmi', 'children', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']]
y = insurance_encoded['charges']
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        age = float(request.form['age'])
        bmi = float(request.form['BMI'])
        children = int(request.form['children'])
        smoker = 1 if request.form['smoke'] == 'yes' else 0
        area = request.form['area']

        # One-Hot Encoding cho khu vực sống
        area_encoded = [0, 0, 0]  # Mặc định là tất cả bằng 0 (northeast)
        if area == 'northwest':
            area_encoded = [1, 0, 0]
        elif area == 'southeast':
            area_encoded = [0, 1, 0]
        elif area == 'southwest':
            area_encoded = [0, 0, 1]

        # Chuẩn hóa
        input_data = scaler.transform([[age, bmi, children]])
        input_data = pd.DataFrame(input_data, columns=numeric_features)
        input_data['smoker_yes'] = smoker
        input_data['region_northwest'] = area_encoded[0]
        input_data['region_southeast'] = area_encoded[1]
        input_data['region_southwest'] = area_encoded[2]

        # Dự đoán
        prediction = lasso_model.predict(input_data)[0]

    return render_template('giaodien.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

