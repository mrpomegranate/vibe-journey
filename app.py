from flask import Flask, render_template, request, jsonify
from crew_engine3 import generate_itinerary
import traceback

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get JSON data from the frontend
        data = request.get_json()
        
        # Validation
        if not data or 'destination' not in data or 'people' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Log reception
        print(f"Received request for {data['destination']} with {len(data['people'])} people.")

        # Call the CrewAI Logic
        # This will block until completion (can take 30-60s)
        result = generate_itinerary(data)
        
        # Return result as JSON
        return jsonify({'status': 'success', 'itinerary': result})

    except Exception as e:
        print("Error generating itinerary:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)