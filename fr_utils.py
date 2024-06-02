import numpy as np
from keras.models import load_model
from utility import img_to_encoding, resize_img

# Custom loss function for model
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    # Triplet formula components
    pos_dist = np.sum(np.square(np.subtract(y_pred[0], y_pred[1])))
    neg_dist = np.sum(np.square(np.subtract(y_pred[0], y_pred[2])))
    basic_loss = pos_dist - neg_dist + alpha
    
    loss = np.maximum(basic_loss, 0.0)
   
    return loss

# Load the FaceNet trained model
def load_FRmodel():
    return load_model('models/model.h5', custom_objects={'triplet_loss': triplet_loss})

# Load the saved user database
def ini_user_database():
    global user_db
    # Check for existing database
    if os.path.exists('database/user_dict.pickle'):
        with open('database/user_dict.pickle', 'rb') as handle:
            user_db = pickle.load(handle)
    else:
        # Create a new one
        user_db = defaultdict(dict)
    return user_db

# Deletes a registered user from the database
def delete_user(user_db, email):
    popped = user_db.pop(email, None)

    if popped is not None:
        print('User ' + email + ' deleted successfully')
        # Save the database
        with open('database/user_dict.pickle', 'wb') as handle:
                pickle.dump(user_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    else:
        print('No such user !!')
        return False

# Adds a new user face to the database using his/her image stored on disk using the image path
def add_user_img_path(user_db, FRmodel, email, name, gender, password, img_path):
    if email not in user_db:
        user_db[email]['encoding'] = img_to_encoding(img_path, FRmodel)
        user_db[email]['name'] = name
        user_db[email]['gender'] = gender
        user_db[email]['password'] = password
        # Save the database
        with open('database/user_dict.pickle', 'wb') as handle:
                pickle.dump(user_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('User ' + name + ' added successfully')
        return True
    else:
        print('The email is already registered! Try a different email.........')
        return False
