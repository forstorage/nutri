from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import re

app = Flask(__name__, static_folder='static')
model = tf.keras.models.load_model('../food_recognition_model.h5')

with open("../pec/meta/classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

API_KEY = 'AIzaSyDsf3Zg0mPPSyb7F5YmXq6YQwCOUV2Mh94'
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        img = Image.open(image_file)
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        
        confidence = float(np.max(prediction)) * 100
        confidence_percent = round(confidence, 2)
        
        return jsonify({
            'food': predicted_class,
            'confidence': confidence_percent
        })

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({'error': 'Failed to process image'}), 500

@app.route('/get-food-info', methods=['POST'])
def get_food_info():
    try:
        data = request.get_json()
        food_name = data.get('foodName')

        if not food_name:
            return jsonify({'error': 'No food name provided'}), 400

        prompt = f"""
            Analyze {food_name} and provide:
            1. **Title**: [Dish Name]
            2. **Ingredients** (with 1-5 health ratings):
               - [Ingredient 1] (X/5): [Description]
               - [Ingredient 2] (Y/5): [Description]
            3. **Recipe**: Numbered steps
            4. **Nutrition** per serving:
               Protein: [XX]g
               Carbohydrates: [XX]g
               Fats: [XX]g
               Calories: [XX]
            5. **Health Assessment**:
               Healthiness: [1-sentence assessment]
               Suggestions: [2 practical improvements]
            Use strict markdown formatting. Separate Healthiness and Suggestions clearly.
        """

        body = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }

        response = requests.post(GEMINI_URL, json=body, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        result = response.json()

        if not result.get('candidates'):
            return jsonify({'error': 'No response from Gemini'}), 500

        text = result['candidates'][0]['content']['parts'][0]['text']
        
        # Parse nutrition data
        nutrition = {}
        try:
            if '**Nutrition**' in text:
                nutrition_block = text.split('**Nutrition**')[1].split('**Health Assessment**')[0]
                for line in nutrition_block.split('\n'):
                    line = line.strip().lower()
                    if 'protein:' in line:
                        nutrition['protein'] = re.findall(r'\d+', line)[0] + 'g'
                    elif 'carbohydrates:' in line or 'carbs:' in line:
                        nutrition['carbs'] = re.findall(r'\d+', line)[0] + 'g'
                    elif 'fats:' in line:
                        nutrition['fats'] = re.findall(r'\d+', line)[0] + 'g'
        except Exception as e:
            print("Nutrition parsing error:", e)

        # Parse ingredients
        ingredients = []
        try:
            if '**Ingredients**' in text:
                ingredients_block = text.split('**Ingredients**')[1].split('**Recipe**')[0]
                for line in ingredients_block.split('\n'):
                    line = line.strip()
                    if re.match(r'^[-*•]', line):
                        clean_line = re.sub(r'^[-*•]\s*', '', line)
                        rating_match = re.search(r'\((\d)/5\)', clean_line)
                        if rating_match:
                            name = re.split(r'\(\d/5\)', clean_line)[0].strip()
                            ingredients.append({
                                'name': name,
                                'rating': int(rating_match.group(1))
                            })
                        else:
                            ingredients.append({
                                'name': clean_line,
                                'rating': 3
                            })
        except Exception as e:
            print("Ingredient parsing error:", e)

        # Parse health assessment
        healthiness = ""
        suggestion = ""
        try:
            if '**Health Assessment**' in text:
                health_block = text.split('**Health Assessment**')[1].strip()
                if 'Suggestions:' in health_block:
                    parts = health_block.split('Suggestions:', 1)
                    healthiness = parts[0].replace('Healthiness:', '').strip()
                    suggestion = parts[1].strip()
                else:
                    healthiness = health_block
        except Exception as e:
            print("Health assessment parsing error:", e)

        return jsonify({
            'result': text,
            'nutrition': nutrition,
            'ingredients': ingredients,
            'healthiness': healthiness,
            'suggestion': suggestion
        })

    except requests.exceptions.RequestException as e:
        print("API Error:", e)
        return jsonify({'error': 'Failed to communicate with Gemini API'}), 500
    except Exception as e:
        print("General Error:", e)
        return jsonify({'error': 'Failed to process food information'}), 500

if __name__ == '__main__':
    app.run(debug=True)