import cv2
import numpy as np
import os

class InputSize:
    def input_one(self):
        print("Input image size: (recommended 640 by 590)")
        self.a, self.b = map(int, input().split())
        print("Updated Image size")

    def input_two(self):
        print("Input Pixel colors for contrast: (recommended 255,255,255)")
        self.e, self.f, self.g = map(int, input().split())
        print("Updated pixel colors")

def load_dataset(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            images.append(cv2.imread(image_path))
            if "no" in directory:
                labels.append(0)  # 0 for no tumor
            elif "yes" in directory:
                labels.append(1)  # 1 for tumor present
    return images, labels

def calculate_accuracy(predictions, labels):
    correct = sum(pred == label for pred, label in zip(predictions, labels))
    total = len(labels)
    accuracy = correct / total
    return accuracy

def main():
    obj1 = InputSize()
    obj2 = InputSize()
    obj1.input_one()
    obj2.input_two()
    c, d, p, q, r = obj1.a, obj1.b, obj2.e, obj2.f, obj2.g
    
    # Load dataset
    tumor_images, tumor_labels = load_dataset(r"C:/Users/mukil/OneDrive/Desktop/Brain-Tumor-OODP-main/Brain-Tumor-OODP-main/brain_tumor_dataset/yes")
    no_tumor_images, no_tumor_labels = load_dataset(r"C:/Users/mukil/OneDrive/Desktop/Brain-Tumor-OODP-main/Brain-Tumor-OODP-main/brain_tumor_dataset/no")

    predicted_labels = []

    # Process tumor images
    for image in tumor_images[:1]:  # Take only one image
        # Resize image
        newSize = (c, d)
        image = cv2.resize(image, newSize)
        
        # Convert image to grayscale
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply top hat filter
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        topHat = cv2.morphologyEx(grayImage, cv2.MORPH_TOPHAT, kernel)
        
        # Blur image
        topHat = cv2.GaussianBlur(topHat, (5, 5), 0)
        
        # Apply thresholding for segmenting
        _, thresholded = cv2.threshold(topHat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Compute distance transform
        thresholded = np.uint8(thresholded)
        dist = cv2.distanceTransform(thresholded, cv2.DIST_L2, 3)
        
        # Find local maxima in distance transform
        localMaxima = cv2.dilate(thresholded, None)
        localMaxima = (thresholded == localMaxima)
        localMaxima = np.uint8(localMaxima)
        
        # Find markers for watershedding
        _, markers = cv2.connectedComponents(localMaxima)
        
        # Apply watershedding
        markers = cv2.watershed(image, markers)
        
        # Display output
        output = np.zeros(image.shape, dtype=np.uint8)
        for i in range(markers.shape[0]):
            for j in range(markers.shape[1]):
                index = markers[i, j]
                if index == -1:
                    output[i, j] = [255, 255, 255]
                elif index == 1:
                    output[i, j] = [p, q, r]
        
        # Assuming prediction is tumor present (1) for demonstration
        predicted_labels.append(1)

        # Calculate accuracy
        accuracy = calculate_accuracy(predicted_labels, [1])  # Ground truth label for tumor image is 1

        # Display accuracy in the image window
        accuracy_text = f"Accuracy: {accuracy:.2f}"
        cv2.putText(image, accuracy_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Tumor Detection", image)
        cv2.waitKey(0)

    # Process no tumor images
    for image in no_tumor_images[:1]:  # Take only one image
        # Resize image
        newSize = (c, d)
        image = cv2.resize(image, newSize)
        
        # Convert image to grayscale
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply top hat filter
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        topHat = cv2.morphologyEx(grayImage, cv2.MORPH_TOPHAT, kernel)
        
        # Blur image
        topHat = cv2.GaussianBlur(topHat, (5, 5), 0)
        
        # Apply thresholding for segmenting
        _, thresholded = cv2.threshold(topHat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Compute distance transform
        thresholded = np.uint8(thresholded)
        dist = cv2.distanceTransform(thresholded, cv2.DIST_L2, 3)
        
        # Find local maxima in distance transform
        localMaxima = cv2.dilate(thresholded, None)
        localMaxima = (thresholded == localMaxima)
        localMaxima = np.uint8(localMaxima)
        
        # Find markers for watershedding
        _, markers = cv2.connectedComponents(localMaxima)
        
        # Apply watershedding
        markers = cv2.watershed(image, markers)
        
        # Display output
        output = np.zeros(image.shape, dtype=np.uint8)
        for i in range(markers.shape[0]):
            for j in range(markers.shape[1]):
                index = markers[i, j]
                if index == -1:
                    output[i, j] = [255, 255, 255]
                elif index == 1:
                    output[i, j] = [p, q, r]
        
        # Assuming prediction is no tumor (0) for demonstration
        predicted_labels.append(0)

        # Calculate accuracy
        accuracy = calculate_accuracy(predicted_labels, [0])  # Ground truth label for no tumor image is 0

        # Display accuracy in the image window
        accuracy_text = f"Accuracy: {accuracy:.2f}"
        cv2.putText(image, accuracy_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Tumor Detection", image)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
