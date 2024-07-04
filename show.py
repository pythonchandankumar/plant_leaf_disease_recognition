import cv2
import requests

# URL of the FastAPI prediction endpoint
PREDICTION_API_URL = "http://localhost:8000/predict"

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to JPEG format
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}

    # Make the prediction request
    response = requests.post(PREDICTION_API_URL, files=files)
    response_data = response.json()
    
    # Extract the predicted class and confidence
    predicted_class = response_data.get('predicted_class', 'Unknown')
    confidence = response_data.get('confidence', 0.0)
    
    # Display the predicted class on the frame
    cv2.putText(frame, f"{predicted_class}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Webcam', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
