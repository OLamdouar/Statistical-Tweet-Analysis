import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# --------------------------------------
# Load the tweet dataset
# --------------------------------------
df = pd.read_csv("tweets_dataset.csv")

# --------------------------------------
# Data Exploration
# --------------------------------------
print("📊 Dataset Summary:")
print(df.describe())

sns.pairplot(df[['retweet_count', 'sentiment_score', 'tweet_length', 'hashtag_count']])
plt.suptitle("Pairplot of Tweet Features (N=300)", y=1.02)
plt.tight_layout()
plt.show()

# --------------------------------------
# Train-test split
# --------------------------------------
X = df[['sentiment_score', 'tweet_length', 'hashtag_count']]
y = df['retweet_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------
# Linear Regression
# --------------------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)

print("\n🔹 Linear Regression")
print("Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)
print("MSE:", mse_linear)

# --------------------------------------
# Poisson Regression
# --------------------------------------
df_train = X_train.copy()
df_train['retweet_count'] = y_train
poisson_model = smf.glm(formula="retweet_count ~ sentiment_score + tweet_length + hashtag_count",
                        data=df_train,
                        family=sm.families.Poisson()).fit()

df_test = X_test.copy()
df_test['retweet_count'] = y_test
y_pred_poisson = poisson_model.predict(df_test)
mse_poisson = mean_squared_error(y_test, y_pred_poisson)

print("\n🔹 Poisson Regression")
print(poisson_model.summary())
print("MSE:", mse_poisson)

# --------------------------------------
# XGBoost Regression
# --------------------------------------
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

print("\n🔹 XGBoost Regressor")
print("MSE:", mse_xgb)

xgb.plot_importance(xgb_model)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# --------------------------------------
# Sample Prediction
# --------------------------------------
sample_input = pd.DataFrame({
    'sentiment_score': [0.5],
    'tweet_length': [140],
    'hashtag_count': [3]
})

print("\n📌 Prediction for sample tweet:")
print("Linear:", linear_model.predict(sample_input)[0])
print("Poisson:", poisson_model.predict(sample_input).iloc[0])
print("XGBoost:", xgb_model.predict(sample_input)[0])
