import pandas as pd

parkinsons_df = pd.read_csv('parkinsons.csv')
parkinsons_df= parkinsons_df.dropna()
parkinsons_df.head()
selected_features = ['PPE', 'DFA']
target= ['status']
x=parkinsons_df[selected_features]
y=parkinsons_df[target]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(x_train, y_train)
y_perd= model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_perd)
