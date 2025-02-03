#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


class input_size{
    public:
    int a,b,e,f,g;
    void inputone(){
    cout << "Input image size: \n (We recommend 640 by 590!)" << endl;
    cin >> a >> b;
    cout << "Updated Image size" << endl;
    }

    void inputtwo(){
        cout << "Input Pixel colors for contrast: \n (We recommend 255,255,255!)" << endl;
        cin >> e >> f >> g;
        cout << "Updated pixel colors" << endl;
    }
};

int main() {

    input_size obj1, obj2;
    obj1.inputone();
    obj2.inputtwo();
    int c=obj1.a, d=obj1.b, p=obj2.e, q=obj2.f, r=obj2.g;
    // Load input image
    Mat image = imread("C:/Users/Rishi/OneDrive/Pictures/brain.png");

    // Resize image to a smaller size
    Size newSize(640, 590);
    resize(image, image, newSize);

    // Convert image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Apply top hat filter
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    Mat topHat;
    morphologyEx(grayImage, topHat, MORPH_TOPHAT, kernel);

    // Blur image
    GaussianBlur(topHat, topHat, Size(5, 5), 0);

    // Apply thresholding for segmenting
    Mat thresholded;
    threshold(topHat, thresholded, 0, 255, THRESH_BINARY + THRESH_OTSU);

    // Compute distance transform
    thresholded.convertTo(thresholded, CV_8UC1);

    Mat dist;
    distanceTransform(thresholded, dist, DIST_L2, 3);

    // Find local maxima in distance transform
    Mat localMaxima;
    dilate(thresholded, localMaxima, Mat());
    compare(thresholded, localMaxima, localMaxima, CMP_EQ);
    localMaxima.convertTo(localMaxima, CV_8UC1);

    // Find markers for watershedding
    Mat markers;
    connectedComponents(localMaxima, markers);

    // Apply watershedding
    watershed(image, markers);

    // Display output
    Mat output = Mat::zeros(image.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index == -1) {
                output.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            } else if (index == 1) {
                output.at<Vec3b>(i, j) = Vec3b(p, q, r);
            }
        }
    }
    imshow("Original Image", image);
    imshow("Thresholded Image", thresholded);
    imshow("Output", output);
    waitKey(0);

    string originalFileName = "original.png";
    imwrite(originalFileName, image);

    string thresholdedFileName = "thresholded.png";
    imwrite(thresholdedFileName, thresholded);

    string outputFileName = "output.png";
    imwrite(outputFileName, output);
    return 0;
}
