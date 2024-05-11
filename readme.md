# MPI Image Processing

This project demonstrates parallel image processing using MPI (Message Passing Interface). It allows users to apply various image processing functions to an input image and parallelize the processing across multiple MPI processes.

## Features

- Supports multiple image processing functions, including Gaussian blur, edge detection, image rotation, image scaling, histogram equalization, color space conversion, global thresholding, local thresholding, image compression, and median filtering.
- Utilizes MPI for parallel execution, allowing faster processing of large images by distributing the workload across multiple processes.
- Provides a user-friendly interface for selecting the desired image processing function.
- Saves the processed images and displays the execution time for each processing function.

## Prerequisites

- OpenCV library
- MPI implementation (e.g., Open MPI)

## Usage

1. Compile the source code using a C++ compiler with OpenCV and MPI support. For example:
```
mpic++ -o mpi_image_processing mpi_image_processing.cpp `pkg-config --cflags --libs opencv4`
```

2. Run the compiled executable with the desired input image file. For example:
```
mpiexec -n <num_processes> ./mpi_image_processing
```

3. Follow the on-screen instructions to select the image processing function.

4. Once processing is complete, the processed images will be saved in the current directory, and the execution time for each function will be displayed.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

> developed under the supervision of FCAI college assistants:
## eng\ *mahmoud el-badry* && eng\ *abdelrahman salem* :
## developed by :
1. **mohammed ali abdel-tawab**
2. **ahmed mohamed omar**
3. **abdelfatah alaa**
4. **mostafa rady mizar**
5. **mahmoud abdel-tawab shaban**
6. **ali gamil ali**
