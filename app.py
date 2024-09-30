from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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

        if model_choice == 'lasso':
            model = Lasso(max_iter=1000)
        elif model_choice == 'ridge':
            model = Ridge(max_iter=1000)
        elif model_choice == 'mlp':
            model = MLPRegressor(max_iter=1000)
        elif model_choice == 'stacking':
            base_models = [
                ('linear', LinearRegression()),
                ('lasso', Lasso(max_iter=1000)),
                ('ridge', Ridge(max_iter=1000))
            ]
            model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
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
    app.run(debug=True)

