
from queue import Empty
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, RidgeClassifier, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np



file_path = "G:\projects\hw5\dielectron.csv"
df =pd.DataFrame(pd.read_csv(file_path))


df.head(), df.columns
df.info(), df.describe()
sns.set(style="whitegrid")


plt.figure(figsize=(16, 12))


plt.subplot(2, 2, 1)
sns.histplot(df['E1'], bins=50, color='blue', kde=True, label='E1')
sns.histplot(df['E2'], bins=50, color='red', kde=True, label='E2', alpha=0.6)
plt.title('Energy Distributions (E1 and E2)')
plt.legend()


plt.subplot(2, 2, 2)
sns.histplot(df['pt1'], bins=50, color='blue', kde=True, label='pt1')
sns.histplot(df['pt2'], bins=50, color='red', kde=True, label='pt2', alpha=0.6)
plt.title('Transverse Momentum Distributions (pt1 and pt2)')
plt.legend()


plt.subplot(2, 2, 3)
sns.histplot(df['eta1'], bins=50, color='blue', kde=True, label='eta1')
sns.histplot(df['eta2'], bins=50, color='red', kde=True, label='eta2', alpha=0.6)
plt.title('Pseudorapidity Distributions (eta1 and eta2)')
plt.legend()


plt.subplot(2, 2, 4)
sns.histplot(df['M'], bins=50, color='green', kde=True)
plt.title('Invariant Mass (M) Distribution')
plt.figure(figsize=(14, 10))
corr_matrix = df.corr()


sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Variables')
plt.tight_layout()
plt.show()

median_value = df['M'].median()
df['M'] = df['M'].fillna(median_value)

df = df.rename(columns={df.columns[3]: 'px1'})

features = df[['E1','px1', 'py1', 'pz1', 'pt1', 'eta1', 'phi1', 'Q1', 'E2', 'px2', 'py2', 'pz2', 'pt2', 'eta2', 'phi2', 'Q2']] 
target = df['M']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'KNeighborsRegressor': KNeighborsRegressor(),
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'LassoRegressor': Lasso(),
    'Ridge' : Ridge()
    
}

mse_results = {}
r2_results={}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)
    mse_results[model_name] = mse
    r2_results[model_name]=r2
    

    
for model_name, mse in mse_results.items():
    print(f'MSE for {model_name}:{mse:.4f}')
    
for model_name, r2 in r2_results.items():
    print(f'R^2 for {model_name}:{r2:.4f}')


    
plt.figure(figsize=(16, 12))

model_names = list(mse_results.keys())
mse_values = list(mse_results.values())
r2_values=list(r2_results.values())

plt.bar(model_names, mse_values, color='lightblue')
plt.xlabel('model')
plt.ylabel(' MSE')
plt.title('mse_metrics')
plt.grid(True)
plt.show()

plt.bar(model_names, r2_values, color='red')
plt.xlabel('model')
plt.ylabel(' R^2')
plt.title('R^2_metrics')
plt.grid(True)


plt.show()

estimators = [
    ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
    ('ridge', Ridge(random_state=42)),
    ('dt', DecisionTreeRegressor(random_state=42)),
    ('Lasso', Lasso(random_state=42)),
    ('ln' , LinearRegression())   
]


meta_model = LinearRegression()


stacking_model = StackingRegressor(
    estimators=estimators, 
    final_estimator=meta_model
)


stacking_model.fit(X_train, y_train)


y_pred = stacking_model.predict(X_test)


print(f"масса эллектрона = : {np.mean(y_pred)}")

mse = mean_squared_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
print(f"Среднеквадратичная ошибка стэкинга (MSE): {mse}" , f"R^2 ошибка стэкинга (r2): {r2}")


anomalies_dict = {}
numeric_columns = df.select_dtypes(include=[np.number]).columns

for column in numeric_columns:
   
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

   
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

  
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
  
    anomalies_dict[column] = anomalies

   
    print(f"Колонка: {column}, Количество аномалий: {len(anomalies)}")
 
for i, column in enumerate(numeric_columns, 1):
   
         plt.subplot(5, 5, i)  
         sns.boxplot(x=df[column])
         plt.title(f'Boxplot для {column}')
         plt.xlabel(column)

plt.tight_layout() 
plt.show()