# Blur_test_python
Demonstrates how to blur an image using serial and parallel methods.

##Detailed Description:
   Script that uses a brute force method for performing a gaussian blur on the provided image.
   The blurring can either be carried out serially or in parallel (using the multiprocess module).

   Numpy to shared ctypes conversion code base on examples from http://briansimulator.org/sharing-numpy-arrays-between-processes/
##Usage:
   gaussBlurTest.py mode image [noshow]
   
Where:

   **mode:** The method of blurring. Options are:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*serial* Sequentially blur pixels in single process.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*multiproc* Blur pixel in a series of processes that can execute in parallel.
               
   **image:** The path to an image file.

   **noshow:** Optional argument; if present, image will not be displayed after blurring.

