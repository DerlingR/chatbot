import pickle
from flask import Flask, request, jsonify, render_template

# Cargar el modelo y el vectorizador previamente guardados
with open('modelo_entrenado.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

with open('vectorizador.pkl', 'rb') as vectorizer_file:
    vectorizador = pickle.load(vectorizer_file)

# Crear la aplicación Flask
app = Flask(__name__)

# Ruta principal para mostrar la página web
@app.route('/')
def home():
    return render_template('index.html')  # Servir el archivo HTML

# Ruta para predecir el sentimiento de una opinión
@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtener el texto de la solicitud
    data = request.get_json()
    opinion = data['opinion']

    # Preprocesar el texto usando el vectorizador
    opinion_vectorizada = vectorizador.transform([opinion])

    # Hacer la predicción
    prediccion = modelo.predict(opinion_vectorizada)

    # Responder con el resultado de la predicción
    resultado = {'sentimiento': int(prediccion[0])}
    return jsonify(resultado)

# Iniciar la aplicación
if __name__ == '__main__':
    app.run(debug=True)