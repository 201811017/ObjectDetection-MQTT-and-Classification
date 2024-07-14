import time
from paho.mqtt import client as mqtt_client
import base64
import io 
from io import BytesIO
from PIL import Image
import os
import base64


message_received = False


def read_saved_image(image_url):

    try:
        # Extract the base64-encoded data from the URL
        encoded_data = image_url.split(",")[1]

        # Decode the base64 data
        decoded_data = base64.b64decode(encoded_data)

        # Load the decoded data as a PIL image
        return Image.open(io.BytesIO(decoded_data))
    except Exception as e:
        print(f"Error reading image: {e}")
        return None



def save_image(pil_img, folder_name="received_images"):

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Generate a unique filename 
    filename = f"{time.time()}.jpg"
    filepath = os.path.join(folder_name, filename)

    try:
        # Convert image to RGB mode
        pil_img = pil_img.convert("RGB")
        # Convert image to bytes in JPEG format
        with open(filepath, "wb") as f:
            pil_img.save(f, "JPEG")

        # Generate a base64-encoded URL for the saved image (simulate FileCoordinator behavior)
        with open(filepath, "rb") as f:
            image_data = f.read()
            image_url = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"

        return folder_path, image_url
    except Exception as e:
        print(f"Error saving image: {e}")
        return None


def on_message(client, userdata, message):
    global message_received
    message_received = True
    print("Received a message.")

    try:
        # Decode the message payload as a base64 string
        image_data = base64.b64decode(message.payload)

        # Convert base64-decoded data to a PIL Image
        pil_img = Image.open(io.BytesIO(image_data))
        
        folder_path = r"C:\Users\TFM\ObjectClassification\received_images" #fill with appropiate full path


        folder_path, image_url = save_image(pil_img, folder_path)

        if folder_path and image_url:
            print(f"Image saved successfully: {image_url}")
            pass
        else:
            print("Error saving image.")

    except Exception as e:
        print(f"Error processing message: {e}")


if __name__ == "__main__":
    global folder_path
    folder_path = r"C:\Users\TFM\ObjectClassification\received_images" #fill with appropiate full path


    timeout_duration = 10  # Timeout duration in seconds

    # With username and password. Another option would be to use TLS
    broker_address = "127.0.0.1"  # localhost
    username = "Algorithm"
    password = "tfm"

    # Create an MQTT client instance
    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2)

    print("Created MQTT client instance")


    # Set the username and password
    client.username_pw_set(username, password)

    client.connect(broker_address, 1883, 60)  # Default port is 1883
    print("Connected to MQTT broker")

    # Initialize blob as None
    blob = None

    client.on_message =  on_message
    client.subscribe("ObjectDetected")
    print("Subscribed to topic")

    client.loop_start()

    while not message_received:
        print("Waiting for message...")
        pass

    print("Message received")
    client.loop_forever()  # This keeps the loop running indefinitely


    






