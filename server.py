from flask import Flask,request
from DocumentLoader import admission_chatbot
from flask_cors import CORS  # Import CORS from flask_cors module

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for all routes

@app.route('/')
def hello_world():
    return 'Hello, Worl                                                                                                                                                             d!'

@app.route('/post-example', methods=['POST'])
def post_example():
    # You can access the data sent in the POST request using request.form for form data or request.json for JSON data
    data = request.json
    name1 = data.get('name', 'default_value')  # 'default_value' is a fallback in case 'name' is not present in the data
    print (data)
    result =admission_chatbot(name1)
    return result
    #return f'Received data: {data}'

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Run the server on port 5000
