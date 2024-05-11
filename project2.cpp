#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace cv;
using namespace std;
int64 start =0;
double executionTime= 0.0;

void applyImageProcessing(const Mat& inputImage, Mat& outputImage, int rank, int processingFunction) {
            Mat grey_image;
            std::vector<uchar> compressed_buffer;
            std::vector<int> compression_params;

            double angle = 45;
            Point2f center(inputImage.cols / 2.0, inputImage.rows / 2.0);
            Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
    // Perform image processing tasks based on the chosen function
    switch (processingFunction) {
        case 1:
            GaussianBlur(inputImage, outputImage, Size(5, 5), 0);
            cout << rank << " has finished Gaussian blur." << endl;
            break;
        case 2:
            
            Sobel(inputImage, outputImage, CV_8U, 1,0);
            cout << rank << " has finished edge detection." << endl;
            break;
        case 3:
            warpAffine(inputImage, outputImage, rotationMatrix, inputImage.size());
            // imshow("90 rotation",outputImage);
            // waitKey(0);
            cout << rank << " has finished rotation." << endl;
            break;
        case 4:
            resize(inputImage, outputImage, Size(0.25,0.25), 0.5,2);
            cout << rank << " has finished scaling." << endl;
            break;
        case 5:
// greyscale image fisrt
         
             cvtColor(inputImage, grey_image, COLOR_BGR2GRAY);
             
            equalizeHist(grey_image, outputImage);
            cout << rank << " has finished histogram." << endl;
            break;
        case 6:

                cvtColor(inputImage, outputImage, COLOR_BGR2HSV);
                cout << rank << " has finished color space convertion." << endl;
            break;
        case 7:
        
                threshold(inputImage, outputImage, 128, 255, THRESH_BINARY);
                cout << rank << " has finished global thresholding." << endl;
            break;
        case 8:
            cvtColor(inputImage, grey_image, COLOR_BGR2GRAY);
            adaptiveThreshold(grey_image, outputImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);
            cout << rank << " has finished local thresholding." << endl;
            break;
        case 9:

            compression_params.push_back(IMWRITE_JPEG_QUALITY);
            compression_params.push_back(40);  // Adjust the quality (0-100)

    // Write the compressed image to a file
                imencode(".jpg", inputImage, compressed_buffer, compression_params);
                outputImage = imdecode(compressed_buffer, IMREAD_COLOR);

            cout << rank << " has finished compression." << endl;

            break;
        case 10:
            medianBlur(inputImage, outputImage,5);
            cout << rank << " has finished median filter." << endl;
            break;

        default:
            cerr << "Unknown image processing function." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
    }


}

int main() {

    string input_image = "camilia.jpeg";
    string output_image[10]= {
        "gaussian_blur.jpg",
        "edg_detectoin.jpg",
        "image_rotation.jpg",
        "image_scaling.jpg",
        "histogram.jpg",
        "color_space_conv.jpg",
        "global_thresholding.jpg",
        "local_thresholding.jpg",
        "image_compression.jpg",
        "median.jpg",
    };


    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mat image;

    if (rank == 0) {
        // Load image from file on rank 0
        image = imread(input_image, IMREAD_COLOR);
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

    // User chooses the image processing function
    int processingFunction;
    if (rank == 0) {
        cout << "Choose image processing function: \n1: Gaussian blur \n2: Edge detection\n3: Image Rotation\n4: Image Scaling\n5: Histogram Equalization\n6: color space conversion \n7: Global Thresholding\n8: Local Thresholding\n9:#Image Compression \n10: #Median\n ";
        cin>> processingFunction;
        if (processingFunction < 0 && processingFunction > 9 ) {
            cerr << "Invalid image processing function choice." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&processingFunction, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform image processing
    Mat processedImage;
    Mat gatheredImage;
    if(rank==0){start = getTickCount();}
    applyImageProcessing(localImage, processedImage, rank, processingFunction);

    // Gather processed image data at rank 0
    if (rank == 0) {
        if(processingFunction == 5 || processingFunction == 8){
            cout<<"grey pixels created"<<endl;
            gatheredImage.create(height, width, CV_8UC1);
            }
    else{
            gatheredImage.create(height, width, CV_8UC3);
            }
        
    }
    if(processingFunction == 5 || processingFunction == 8){
         MPI_Gather(processedImage.data, (endY - startY) * width , MPI_CHAR,
                gatheredImage.data, (endY - startY) * width , MPI_CHAR, 0, MPI_COMM_WORLD);
    }
     else if(processingFunction == 3){
         MPI_Gather(processedImage.data, (endY - startY) * width * 3, MPI_BYTE,
                gatheredImage.data, (endY - startY) * width * 3, MPI_BYTE, 0, MPI_COMM_WORLD);
     }
    else{
        MPI_Gather(processedImage.data, (endY - startY) * width * 3, MPI_BYTE,
                gatheredImage.data, (endY - startY) * width * 3, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    
    if(rank ==0){  executionTime= (getTickCount() - start) / getTickFrequency();
    
   
    }
    

    // Save or display processed image (rank 0 only)
    if (rank == 0) {
        imwrite(output_image[processingFunction -1], gatheredImage);
        cout << "Processed image saved as '"<<output_image[processingFunction -1]<<"'." << endl;
        // Display or further process the image
       
        cout << "Execution time for function " << processingFunction << ": " << executionTime*1000 << " miliseconds." << endl;   
    }

    MPI_Finalize();
    return 0;
}
