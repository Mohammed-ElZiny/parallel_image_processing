#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace cv;
using namespace std;

// Function to rotate a portion of the image
void rotateImagePortion(Mat& inputImage, Mat& outputImage, double angle) {
    Point2f center(inputImage.cols / 2.0, inputImage.rows / 2.0);
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(inputImage, outputImage, rotationMatrix, inputImage.size());
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Load the input image on rank 0
    Mat image;
    if (rank == 0) {
        image = imread("camilia.jpeg", IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Failed to load input image." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast image dimensions to all processes
    int width, height;
    if (rank == 0) {
        width = image.cols;
        height = image.rows;
    }
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate portion of the image to process
    int chunkHeight = height / size;
    int startY = rank * chunkHeight;
    int endY = (rank == size - 1) ? height : startY + chunkHeight;

    // Allocate memory for local portion of the image
    Mat localImage(endY - startY, width, CV_8UC3);

    // Scatter image data among processes
    MPI_Scatter(image.data, (endY - startY) * width * 3, MPI_BYTE, localImage.data,
                (endY - startY) * width * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Rotate the local portion of the image
    Mat rotatedImage;
    double angle = 90.0; // Example rotation angle
    rotateImagePortion(localImage, rotatedImage, angle);

    // Gather rotated image data at rank 0
    if (rank == 0) {
        rotatedImage.create(height, width, CV_8UC3);
    }
    MPI_Gather(rotatedImage.data, (endY - startY) * width * 3, MPI_BYTE,
                rotatedImage.data, (endY - startY) * width * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Save or display rotated image (rank 0 only)
    if (rank == 0) {
        imwrite("rotated_test.jpg", rotatedImage);
        // Display or further process the rotated image
    }

    MPI_Finalize();
    return 0;
}
