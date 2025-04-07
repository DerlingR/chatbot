import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Ejemplo de datos
opiniones = ["Me encanta este producto", "Es un desastre", "Muy buen servicio"]
sentimientos = [1, 0, 1]  # 1: positivo, 0: negativo

# Vectorización
vectorizador = TfidfVectorizer()
X = vectorizador.fit_transform(opiniones)

# Entrenamiento del modelo
modelo = RandomForestClassifier()
modelo.fit(X, sentimientos)

# Guardar el modelo y el vectorizador
with open('modelo_entrenado.pkl', 'wb') as model_file:
    pickle.dump(modelo, model_file)

with open('vectorizador.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizador, vectorizer_file)