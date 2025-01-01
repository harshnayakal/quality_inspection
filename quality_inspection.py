import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_defects(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    edges = cv2.Canny(blurred_image, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    
    return image, contours

def simulate_quality_inspection(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found or cannot be loaded.")
        return

    inspected_image, contours = detect_defects(image)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(inspected_image, cv2.COLOR_BGR2RGB))
    plt.title('Inspected Image')
    
    plt.show()

    if len(contours) > 0:
        print(f"Defects detected: {len(contours)}")
    else:
        print("No defects detected.")

if __name__ == "__main__":
    image_path = r"C:\Users\harsh\Downloads\images.jpg"  
    simulate_quality_inspection(image_path)
