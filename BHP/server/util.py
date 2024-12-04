import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        location = location.lower()
        if location not in __locations:
            raise ValueError(f"Location '{location}' is not available.")

        loc_index = __data_columns.index(location)

    except ValueError as e:
        return f"Error: {str(e)}"

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    try:
        price = __model.predict([x])[0]
        return round(price, 2)
    except Exception as e:
        return f"Error predicting price: {str(e)}"


def load_saved_artifacts():
    global __data_columns
    global __locations
    global __model

    print("Loading saved artifacts... start")

    try:
        with open("./artifacts/columns.json", "r") as f:
            __data_columns = json.load(f)['data_columns']
            __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk
    except FileNotFoundError as e:
        print(f"Error loading columns.json: {str(e)}")
        return

    try:
        if __model is None:
            with open(r'D:\Sumon\Machine Learning\Code\BHP\server\artifacts\banglore_home_price_model.pickle', 'rb') as f:
                __model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading model pickle: {str(e)}")
        return

    print("Loading saved artifacts... done")


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()

    if __locations and __model:
        print(get_location_names())
        print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
        print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    else:
        print("Error loading artifacts, cannot make predictions.")
