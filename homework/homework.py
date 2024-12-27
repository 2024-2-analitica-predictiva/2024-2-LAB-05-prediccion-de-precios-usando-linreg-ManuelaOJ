#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
import pandas as pd
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import  GridSearchCV 
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
import gzip
import pickle
import os

def load_data(path):
    df = pd.read_csv(path, index_col=False, compression='zip')
    df_copy = df.copy()
    df_copy['Age'] = 2021 - df_copy['Year']
    df_copy.drop(columns=['Year', 'Car_Name'], inplace=True)
    return df_copy

data_train = load_data('files/input/train_data.csv.zip')
data_test = load_data('files/input/test_data.csv.zip')

print(data_train.Owner.unique())
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
x_train, y_train = data_train.drop(columns='Present_Price'), data_train['Present_Price']
x_test, y_test = data_test.drop(columns='Present_Price'), data_test['Present_Price']
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
colum_cat = ['Fuel_Type', 'Selling_type', 'Transmission']
colum_others = [col for col in x_train.columns if col not in colum_cat]


pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), colum_cat),
            ('num', MinMaxScaler(), colum_others)
        ] 
    )),
    ('selectkbest', SelectKBest(score_func=f_classif)),  
    ('model', LinearRegression())
])
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
param_grid = {
    
    "selectkbest__k": range(1, 12),
    
}

model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)
model.fit(x_train, y_train)

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
file_path = "files/models/model.pkl.gz"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Archivo existente eliminado: {file_path}")

# Guardar el modelo
with gzip.open(file_path, "wb") as file:
    pickle.dump(model, file)
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

with gzip.open('files/models/model.pkl.gz', 'rb') as f:
    model = pickle.load(f)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


def metrics (y_true, y_pred, dataset):
    return {
    'type': 'metrics',
    'dataset': dataset,
    'r2': r2_score(y_true, y_pred),
    'mse': mean_squared_error(y_true, y_pred),
    'mad': median_absolute_error(y_true, y_pred),
    }

metrics_train = metrics(y_train, y_train_pred, 'train')
metrics_test = metrics(y_test, y_test_pred, 'test')

output_dir = "files/output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "metrics.json")

metrics = [metrics_train, metrics_test]
pd.DataFrame(metrics).to_json(output_path, orient='records', lines=True)