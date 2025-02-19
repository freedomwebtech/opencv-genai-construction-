import base64
import cv2
import time
import threading
import numpy as np
from openai import OpenAI

# Initialize OpenAI client with OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)

# Define Areas of Interest (ROI)
area = [(458, 130), (263, 399), (843, 458), (786, 185)]   # "area"
area1 = [(46, 101), (4, 216), (237, 225), (231, 115)]     # "area1"

# Function to encode image to Base64
def encode_image(image):
    _, img_bytes = cv2.imencode(".jpg", image)
    return base64.b64encode(img_bytes).decode("utf-8")

# Function to crop specific areas
def crop_area(frame, polygon):
    """Crops the frame based on the given polygon (ROI)."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
    cropped = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(np.array(polygon, np.int32))
    return cropped[y:y+h, x:x+w]

# Function to send cropped images to Gemini API
def process_image(image, area_name):
    if image is None or image.size == 0:
        print(f"⚠️ Skipping empty image for {area_name}")
        return "No image available for analysis."

    img_base64 = encode_image(image)

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional for OpenRouter rankings
                "X-Title": "<YOUR_SITE_NAME>",      # Optional for OpenRouter rankings
            },
            model="google/gemini-2.0-flash-lite-preview-02-05:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"""
                        Identify all excavators and dump trucks in the image.
                        Provide a structured table format with the following columns:

                        | Machinery Name | Color  | Status (Active/Inactive) | Company Name (if available) | Area |
                        |---------------|--------|--------------------------|------------------------------|------|
                        |               |        |                          |                              | {area_name} |

                        - If the machinery is moving, mark it as "Active".
                        - If it is stationary, mark it as "Inactive".
                        - Ensure the "Area" column only contains "{area_name}" (either "area" or "area1").
                        """},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ]
        )
        response_text = completion.choices[0].message.content
        save_response_to_file(response_text, area_name)
        return response_text
    except Exception as e:
        print(f"❌ Error processing image for {area_name}: {e}")
        return "Error processing image."

# Function to save response to a text file
def save_response_to_file(response_text, area_name):
    file_name = f"machinery_detection_{area_name}.txt"
    with open(file_name, "a") as file:
        file.write(f"Response for {area_name}:\n")
        file.write(response_text + "\n")
        file.write("=" * 80 + "\n")
    print(f"✅ Response saved to {file_name}")

# Function to process and print the results asynchronously
def process_and_print(frame):
    """Captures, crops, and sends images for analysis."""
    cropped_area = crop_area(frame, area)
    cropped_area1 = crop_area(frame, area1)

    # Run API calls in separate threads for efficiency
    threading.Thread(target=send_to_gemini, args=(cropped_area, "area")).start()
    threading.Thread(target=send_to_gemini, args=(cropped_area1, "area1")).start()

# Function to send image to Gemini AI
def send_to_gemini(image, area_name):
    print(f"📤 Sending image from {area_name} to Gemini...")
    response_text = process_image(image, area_name)
    print(f"✅ {area_name} Response: {response_text}")

# Function to start video processing
def start_video():
    cap = cv2.VideoCapture('f1.mp4')  # Use 0 for webcam or provide a path to a video file
    if not cap.isOpened():
        print("Error: Unable to access video.")
        return

    last_processed_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.resize(frame, (1020, 500))

        # Draw polylines for visual reference
        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 255), 2)

        # Label Areas
        cv2.putText(frame, "area", (area[0][0], area[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "area1", (area1[0][0], area1[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the live video
        cv2.imshow("Machinery Detection", frame)

        # Process every 5 seconds
        if time.time() - last_processed_time > 5:
            last_processed_time = time.time()
            threading.Thread(target=process_and_print, args=(frame,)).start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the script
if __name__ == "__main__":
    start_video()
