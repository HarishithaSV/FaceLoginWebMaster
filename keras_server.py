from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import numpy as np
import cv2
from fr_utils import *
from webcam_utils import *

app = Flask(__name__)
CORS(app)

# Load FaceNet model
FRmodel = load_FRmodel()

# Initialize user database
user_db = ini_user_database()

# Endpoint to add a user's image
@app.route('/add_user_img', methods=['POST'])
def add_user_img():
    data = request.get_json()
    email = data['email']
    name = data['name']
    gender = data['gender']
    password = data['password']
    img_path = data['img_path']
    added = add_user_img_path(user_db, FRmodel, email, name, gender, password, img_path)
    if added:
        return jsonify({"message": "User image added successfully"})
    else:
        return jsonify({"message": "Failed to add user image"})

# Endpoint to recognize a face in an image
@app.route('/recognize', methods=['POST'])
def recognize():
    img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_COLOR)
    name = recognize_img_path(img, user_db, FRmodel)
    return jsonify({"name": name})

# Endpoint to delete a user from database
@app.route('/delete_user', methods=['POST'])
def delete_user_route():
    data = request.get_json()
    email = data['email']
    deleted = delete_user(user_db, email)
    if deleted:
        return jsonify({"message": "User deleted successfully"})
    else:
        return jsonify({"message": "Failed to delete user"})

# Endpoint to get details of a user or all users
@app.route('/get_details', methods=['GET', 'POST'])
def get_details_route():
    if request.method == 'GET':
        email = request.args.get('email')
        view_all = request.args.get('view_all', type=bool)
    elif request.method == 'POST':
        data = request.get_json()
        email = data['email']
        view_all = data.get('view_all', False)
    else:
        return jsonify({"message": "Invalid request method"})

    details = get_details(user_db, email, view_all)
    return jsonify({"details": details})

@app.route('/')
def index():
    return render_template('homepage.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
