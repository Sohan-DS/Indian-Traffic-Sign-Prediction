import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Predefined class names
CLASS_NAMES = [
    "ALL_MOTOR_VECHILE_PROHIBITED",
    "AXLE_LOAD_LIMIT",
    "BARRIER_AHEAD",
    "BULLOCK_AND_HANDCART_PROHIBITED",
    "BULLOCK_PROHIBITED",
    "CATTLE",
    "COMPULSARY_AHEAD",
    "COMPULSARY_AHEAD_OR_TURN_LEFT",
    "COMPULSARY_AHEAD_OR_TURN_RIGHT",
    "COMPULSARY_CYCLE_TRACK",
    "COMPULSARY_KEEP_LEFT",
    "COMPULSARY_KEEP_RIGHT",
    "COMPULSARY_MINIMUM_SPEED",
    "COMPULSARY_SOUND_HORN",
    "COMPULSARY_TURN_LEFT",
    "COMPULSARY_TURN_LEFT_AHEAD",
    "COMPULSARY_TURN_RIGHT",
    "COMPULSARY_TURN_RIGHT_AHEAD",
    "CROSS_ROAD",
    "CYCLE_CROSSING",
    "CYCLE_PROHIBITED",
    "DANGEROUS_DIP",
    "DIRECTION",
    "FALLING_ROCKS",
    "FERRY",
    "GAP_IN_MADIAN",
    "GIVE_WAY",
    "GAURDED_LEVEL_CROSSING",
    "HANDCRAT_PROHIBITED",
    "HEIGHT_LIMIT",
    "HORN_PROHIBITED",
    "HUMP_OR_ROUGH_ROAD",
    "LEFT_HAIR_BIN_BEND",
    "LEFT_HAND_CURVE",
    "LEFT_REVERSE_BEND",
    "LEFT_TURN_PROHIBITED",
    "LENGTH_LIMIT",
    "LOAD_LIMIT",
    "LOOSE_GRAVEL",
    "MEN_AT_WORK",
    "NARROW_BRIDGE",
    "NARROW_ROAD_AHEAD",
    "NO_ENTRY",
    "NO_PARKING",
    "NO_STOPPING_OR_STANDING",
    "OVERTAKING_PROHIBITED",
    "PASS_EITHER_SIDE",
    "PEDESTRIAN_CROSSING",
    "PEDESTRIAN_PROHIBITED",
    "PRIORITY_FOR_ONCOMING_VEHICLES",
    "QUAY_SIDE_OR_RIVER_BANK",
    "RESTRICTION_ENDS",
    "RIGHT_HAIR_PIN_BEND",
    "RIGHT_HAND_CURVE",
    "RIGHT_REVERSE_BEND",
    "RIGHT_TURN_PROHIBITED",
    "ROAD_WIDENS_AHEAD",
    "ROUNDABOUT",
    "SCHOOL_AHEAD",
    "SIDE_ROAD_LEFT",
    "SIDE_ROAD_RIGHT",
    "SLIPPERY_ROAD",
    "SPEED_LIMIT_5",
    "SPEED_LIMIT_15",
    "SPEED_LIMIT_20",
    "SPEED_LIMIT_30",
    "SPEED_LIMIT_40",
    "SPEED_LIMIT_50",
    "SPEED_LIMIT_60",
    "SPEED_LIMIT_70",
    "SPEED_LIMIT_80",
    "STAGGERED_INTERSECTION",
    "STEEP_ASCENT",
    "STEEP_DESCENT",
    "STOP",
    "STRAIGHT_PROHIBITED",
    "T_INTERSECTION",
    "TONGA_PROHIBITED",
    "TRAFFIC_SIGNAL",
    "TRUCK_PROHIBITED",
    "TURN_RIGHT",
    "U_TURN_PROHIBITED",
    "UNGUARDED_LEVEL_CROSSING",
    "WIDTH_LIMIT",
    "Y_INTERSECTION",
]

# Load the trained model
model = tf.keras.models.load_model("E:/DLP/Code/traffic_sign_model.keras")

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to the expected size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("Indian Traffic Sign Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a traffic sign", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Check"):
        # Preprocess the image
        input_image = preprocess_image(image)
        # Predict the class
        predictions = model.predict(input_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        st.write(f"Predicted Traffic Sign: **{predicted_class}**")