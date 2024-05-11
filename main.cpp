#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat read_image(){
 // Load an image
    Mat image = imread("camilia.jpeg");

    // Check if the image was loaded successfully
    if (image.empty()) {
        cout << "Error: Could not open or find the image." << endl;
        return -1;
    }
}

int main() {
   

    // Convert the image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Calculate the histogram
    Mat histogram;
    int channels[] = {0}; // Histogram of the grayscale image
    int histSize[] = {256}; // 256 bins
    float range[] = {0, 256}; // Pixel values range from 0 to 255
    const float* histRange[] = {range};
    calcHist(&grayImage, 1, channels, Mat(), histogram, 1, histSize, histRange, true, false);

    // Draw the histogram
    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound((double)histWidth / histSize[0]);
    Mat histImage(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));
    normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize[0]; i++) {
        line(histImage, Point(binWidth * (i - 1), histHeight - cvRound(histogram.at<float>(i - 1))),
                          Point(binWidth * (i), histHeight - cvRound(histogram.at<float>(i))),
                          Scalar(0, 0, 0), 2, 8, 0);
    }

    // Display the original image and its histogram
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    namedWindow("Histogram", WINDOW_AUTOSIZE);
    imshow("Histogram", histImage);

    // Wait for a key press to exit
    waitKey(0);
    return 0;
}
