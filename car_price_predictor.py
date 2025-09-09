import gradio as gr
import pickle
import pandas as pd
import numpy as np

df=pd.read_csv('cleaned_data.csv')
# --- 1. Helper Functions (from your training script) ---
def age(x):
  """Calculates the age of the car."""
  return 2025 - x

def km(x):
  """Applies a log transformation to the kilometers driven."""
  return np.log1p(x)

def inv(x):
  """Reverses the log transformation to get the actual price."""
  return np.expm1(x)

# --- 2. Load Your Trained Model ---
# This line loads the file you uploaded.
with open('carmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# --- 3. Create the Prediction Function ---
def predict_price(year, km_driven, fuel, seller, transmission,owner,brand):
    """Takes user inputs, processes them, and returns the predicted price."""

    # Pre-process the inputs
    car_age = age(year)
    km_log = km(km_driven)

    # Create a DataFrame in the exact format the model expects
    input_data = pd.DataFrame({
        'year': [car_age],
        'km_driven': [km_log],
        'brand': [brand],  # Placeholder for brand feature
        'fuel': [fuel],
        'seller': [seller],
        'transmission': [transmission],
        'owner': [owner]
    })

    # Make a prediction
    predicted_price_log = model.predict(input_data)

    # Inverse transform to get the final price
    final_price = inv(predicted_price_log[0])

    # Format the output for display
    return f"Predicted Selling Price: ₹ {final_price:,.2f}"
brand=df['brand'].unique().tolist()
# --- 4. Define and Launch the Gradio Interface ---
iface = gr.Interface(
    fn=predict_price,
    title="🚗 Used Car Price Predictor",
    description="Enter the car details to get an estimated selling price.",
    inputs=[
        gr.Slider(minimum=2000, maximum=2024, step=1, value=2015, label="Model Year"),
        gr.Number(value=50000, label="Kilometers Driven"),
        gr.Dropdown(['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'], label="Fuel Type", value="Petrol"),
        gr.Dropdown(['Individual', 'Dealer', 'Trustmark Dealer'], label="Seller Type", value="Individual"),
        gr.Dropdown(['Manual', 'Automatic'], label="Transmission Type", value="Manual"),
        gr.Dropdown(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'], label="Owner", value="First Owner"),
        gr.Dropdown(brand, label="Brand", value="Maruti_800")
    ],
    outputs=gr.Label(label="Prediction Result")
)

# Launch the app! A public URL will be generated.
iface.launch(share=True)