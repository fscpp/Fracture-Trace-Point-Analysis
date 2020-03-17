# FTP
Fracture Trace Point analysis

ftp.py is composed of 3 main functions:
1) analyze_image_stack: is the core of the workflow analysis.
2) analyze_profile: it finds FTP points and records: position (x, y, z), CT value of the FTP, Edge Response (ER) size, aperture-related parameters.
3) ftp_orientation_cpu: measure 3D FTPs orientation for cubic and non-cubic voxels (change z_corr argument in analyze_image_stack).

- The images must be ".tif" files.
- The grayscale images must be stored in a folder called "img".
- The binary image of the fractures in a folder called "msk".
- The folders "img" and "msk" must be placed in the same folder.

To run the program:
- open it on Spyder (if you have Anaconda for Python);
- at the bottom of the code set the path to the directory where "img" and "msk" are;
- fill all the arguments of the function "analyze_image_stack";
- RUN the program

All the results will be stored in a new folder called "Minima_Analysis" (created by the program).
This folder will be created in the directory provided. The aperture parameters are stored as integers by multiplying them to 10. Thus, they must be divided by the same number if you want to obtain the measured value (i.e. real FWHM=FWHM/10; MA=MA/10; PH=PH/10).

Computation time: the FTP analysis presents moderate variations in computation time (which is always relatively short) between images of different sizes. For example, for 3D cubic images the processing times calculated on an i7-8750H are {size axis of image (px), computation time (s)}: {51, 5}, {102, 7}, {510, 64}, {1020, 225}, {1275, 404}.

If using the program, please cite: Cappuccio F., Toy V., Mills S., Adam L., Three-dimensional separation and characterization of fractures in X-ray computed tomographic images of rocks.
Test sample and scripts of analysis are present in the repository "Test-sample-and-scripts"
