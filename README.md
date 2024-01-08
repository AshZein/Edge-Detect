USAGE: python3 main.py FILE_PATH OUTPUT_FILE_NAME
- FILE_PATH is the path to the input image
- OUTPUT_FILE_NAME is optional. If no file name is given, then it will default to output_FILENAME.jpg

This program uses the Canny edge detection method to outline and visualize the edges
present in the input image. The Canny method is a simple process for edge detection,
applying different kernels to the  image via convolution, then using the image gradient to
determine the positions of edges.

The first step for the Canny method is to obtain the grayscale version of the input image.
This program uses OpenCV to convert the input image to grayscale.

Then, noise reduction needs to be performed on the image. A 5x5 Gaussian kernel with some
standard deviation (FINAL STD. DEV TBD) is applied by convolving it with the image. For this
implementation the kernel is applied to each pixel P<sub>i,j</sub> by overlaying the top left
corner of the kernel over P<sub>i,j</sub>. Then the new pixel value is the sum of the product
of each value in the kernel with the pixel it overlays. In other words it is 
P<sub>i,j</sub> x K<sub>1,1</sub> + P<sub>i,j+1</sub> x K<sub>1,2</sub> +...+ P<sub>i+4,j+4</sub> x K<sub>5,5</sub>.
Due to the top left of the kernel being over layed the current pixel, this leads to the bottom and right edge of 
the resulting image to be lacking detail and losing accuracy. There are methods to alleviate this, however
this implementation has not applied any of the fixes yet.

After the noise reduction is completed, two 3x3 Sobel kernels are applied to the image, again by convolution. 
One kernel is used to obtain the image gradient in the horiziontal direction, the other in the vertical direction.
The following formula is used to produce the image's gradient:
ADD FORMULA HERE!!!
Where G<sub>x</sub> is the horizontal gradient and G<sub>y</sub> is the vertical gradient.



SOURCES:
