#include <iostream> 
#include <string> 
#include <opencv2/opencv.hpp> 
#include <mpi.h> 
 
using namespace cv; 
using namespace std; 
 
void applyImageProcessing(const Mat& inputImage, Mat& outputImage, int rank, int processingFunction) { 
    // Perform image processing tasks based on the chosen function 
    switch (processingFunction) { 
        case 1: 
            GaussianBlur(inputImage, outputImage, Size(5, 5), 0);
            cout<<rank<<" has finished"<<endl;
            break;
        case 2:
            Canny(inputImage, outputImage, 100, 200);
            cout<<rank<<" has finished"<<endl;

            break;
        // Add cases for other processing functions
        default:
            cerr << "Unknown image processing function." << endl; 
            MPI_Abort(MPI_COMM_WORLD, 1); 
    } 
} 
 
int main() { 
    MPI_Init(NULL, NULL); 
 
    int rank, size; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
 
    // Load image only in rank 0 
    Mat image; 
    if (rank == 0) { 
        // Load image from file 
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
    cout<<"scatter done "<<rank<<endl;
    // User chooses the image processing function 
    int processingFunction; 
    if (rank == 0) { 
        cout << "Choose image processing function (1: gaussian_blur, 2: edge_detection): "; 
        cin >> processingFunction; 
    } 
    cout<<"choosing done "<<rank<<endl;
    MPI_Bcast(&processingFunction, 1, MPI_INT, 0, MPI_COMM_WORLD); 
 
    // Perform image processing 
    Mat processedImage; 
    applyImageProcessing(localImage, processedImage, rank, processingFunction); 
 
    // Gather processed image data at rank 0 
    if (rank == 0) { 
        processedImage.create(height, width, CV_8UC3); 
    } 
    MPI_Gather(processedImage.data, (endY - startY) * width * 3, MPI_BYTE, 
                processedImage.data, (endY - startY) * width * 3, MPI_BYTE, 0, MPI_COMM_WORLD); 
 
    // Save or display processed image (rank 0 only) 
    if (rank == 0) { 
        imwrite("output_image.jpg", processedImage); 
        // Display or further process the image 
    } 
 
    MPI_Finalize();
    return 0; 
}
