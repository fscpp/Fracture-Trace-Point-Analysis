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
This folder will be created in the directory provided.
