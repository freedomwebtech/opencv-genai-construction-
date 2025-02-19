import cv2  # OpenCV for video processing
import numpy as np  # NumPy for numerical operations

# Define Areas of Interest (ROI) as polygons (list of points)
area = [(458, 130), (263, 399), (843, 458), (786, 185)]  # First region "area"
area1 = [(46, 101), (4, 216), (237, 225), (231, 115)]  # Second region "area1"

def crop_area(frame, polygon):
    """Crops the frame based on the given polygon (ROI)."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # Create a black mask same size as the frame
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)  # Fill the polygon with white (255)
    cropped = cv2.bitwise_and(frame, frame, mask=mask)  # Apply the mask to keep only the polygon area
    x, y, w, h = cv2.boundingRect(np.array(polygon, np.int32))  # Get bounding box of the polygon
    return cropped[y:y+h, x:x+w]  # Crop and return only the bounding box region

# Start video processing
def start_video():
    cap = cv2.VideoCapture('f1.mp4')  # Load video file (use 0 for webcam)
    
    if not cap.isOpened():
        print("Error: Unable to access video.")
        return

    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            print("End of video or error reading frame.")
            break  # Stop the loop if no more frames are available

        frame = cv2.resize(frame, (1020, 500))  # Resize frame for consistency

        # Draw polylines to mark the areas on the frame
        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 255), 2)

        # Label the areas on the frame
        cv2.putText(frame, "area", (area[0][0], area[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "area1", (area1[0][0], area1[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Crop the defined areas
        cropped_area = crop_area(frame, area)
        cropped_area1 = crop_area(frame, area1)

        # Show the main video with marked areas
        cv2.imshow("Original Video", frame)

        # Show cropped areas in separate windows
        if cropped_area is not None and cropped_area.size > 0:
            cv2.imshow("Cropped Area", cropped_area)
        if cropped_area1 is not None and cropped_area1.size > 0:
            cv2.imshow("Cropped Area 1", cropped_area1)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()  # Release video resource
    cv2.destroyAllWindows()  # Close OpenCV windows

# Run the video processing function
if __name__ == "__main__":
    start_video()
