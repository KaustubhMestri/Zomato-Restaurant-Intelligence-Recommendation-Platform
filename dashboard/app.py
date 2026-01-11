import streamlit as st
import requests

# ------------------------------
# CONFIG
# ------------------------------
API_URL = "https://zomato-restaurant-intelligence.onrender.com"

st.set_page_config(
    page_title="Zomato Restaurant Intelligence",
    layout="centered"
)

st.title("🍽️ Zomato Restaurant Intelligence")
st.caption("Powered by FastAPI + XGBoost")

# ------------------------------
# INPUT SECTION
# ------------------------------
st.subheader("Restaurant Details")

average_cost_for_two = st.number_input(
    "Average Cost for Two",
    min_value=100,
    max_value=5000,
    value=600,
    step=50
)

votes = st.number_input(
    "Votes",
    min_value=0,
    max_value=50000,
    value=200,
    step=10
)

cuisine_count = st.slider(
    "Cuisine Count",
    min_value=1,
    max_value=6,
    value=2
)

online_delivery = st.selectbox(
    "Online Delivery",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

table_booking = st.selectbox(
    "Table Booking",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

price_range = st.selectbox(
    "Price Range",
    options=[1, 2, 3, 4]
)

# ------------------------------
# PAYLOAD
# ------------------------------
payload = {
    "average_cost_for_two": average_cost_for_two,
    "cost_per_person": average_cost_for_two / 2,
    "votes": votes,
    "cuisine_count": cuisine_count,
    "online_delivery": online_delivery,
    "table_booking": table_booking,
    "price_range": price_range
}

# ------------------------------
# PREDICTION
# ------------------------------
if st.button("🔮 Predict"):
    with st.spinner("Calling ML API..."):
        rating_response = requests.post(
            f"{API_URL}/predict-rating",
            json=payload
        )

        success_response = requests.post(
            f"{API_URL}/predict-success",
            json=payload
        )

    st.divider()

    if rating_response.status_code == 200:
        rating = rating_response.json()["predicted_rating"]
        st.success(f"⭐ Predicted Rating: {rating}")
    else:
        st.error("Rating prediction failed")

    if success_response.status_code == 200:
        success = success_response.json()["success_probability"]
        st.info(f"📈 Success Probability: {success}")
    else:
        st.error("Success prediction failed")
