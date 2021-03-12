
# Deep_painterly_harmonization

## Purpose

Why do we want to make this project ? Because we often see a lot of interesting meme, suddenly thought of making related themes. And recently, we have studied in deep learning related courses, so I would like to implement the Neural network through the project. We think this technology can be used as a way for corporate marketing, media and social media. Finally, we hope to compare the difference between the paper and our implementation.


##  Related Work

### Programming language: 
1. Python
### Environment :
1. Google Colab
### Package:
1. Open CV
2. Numpy
3. Pytorch
4. Pillow
5. Matplotlib
### API:
1. RemoveBg

### Remove background

We once used OpenCV to complete the background removal operation, but the effect was not very satisfactory. There are still many unremoved backgrounds on the edge of the
image. So,in order to complete the image processing, we decided to use the API, just use the program to join the registered account the key of the obtained free API will be input to the picture material to be processed. The back end of remove.bg will have a smart AI that will automatically separate the main body of the photo from the background. You can even input 1000 pictures at a time to remove the background. However, since it is a free service, there will be a limit on the number of times. Only 50 API calls per month can be provided, but it is quite enough for our interim report, and the processed pictures are used for subsequent masking of the pictures. For processing and synthesis, we can get the result of the style conversion we hope through this operation.

<p style="text-align:center;">
    <img src="images/001.png" width="80%" />
</p>

### Mask 

Using mask is to use selected images, graphics or objects to occlude the processed images (total or regional) to control the image processing area or process. In digital image processing, mask is a two-dimensional matrix array, and sometimes multi-valued images are also used. The main purpose is:

1. Extract the area of interest: multiply the pre-made area of interest mask with the image to be processed to obtain the image of the area of interest. The value of the image in the area of interest remains unchanged, while the value of the image outside the area is all 0.
2. Masking function: Use a mask to mask certain areas on the image so that it does not participate in the processing or calculation of the processing parameters, or only the masked area is processed or counted.
3. Structural feature extraction: Use similarity variables or image matching methods to detect and extract structural features similar to Mask in the image.
4. The production of special-shaped images. 
Processing steps:
Step 1: Create a mask image of the same size as the original image, and initialize all pixels to 0, so the entire image becomes an all-black image.

<p style="text-align:center;">
    <img src="images/002.png" width="80%" />
</p>
Step 2: Set all the pixel values of the reserved area in the mask image to 255, that is, the entire reserved area becomes white.


### Deliation

The concept of image expansion is to expand the white area (or highlight) in the image. The calculated result image is larger than the white area of the original image. It can also be imagined to make the object fat , and the width of this circle is determined by the size of the convolution kernel. In fact, the convolution kernel slides and calculates along the shadow. If there is only one pixel value in the range of the convolution kernel mxn, then the new pixel value is 1, otherwise the new pixel value the pixel value of keeps the original pixel value, which means that all pixels scanned by the convolution kernel will be expanded or dilated (to 1), so the white area of the entire image will increase.
Uses of Dilation:
Purpose 1: Dilation image expansion is usually used in conjunction with image erosion. First, the erosion method is used to narrow the lines in the image and also remove the noise, and then the image is expanded back through Dilation.
Purpose 2: Used to connect two very close but separate objects.

<p style="text-align:center;">
    <img src="images/003.png" width="80%" />
</p>

### Deliation

The concept of image expansion is to expand the white area (or highlight) in the image. The calculated result image is larger than the white area of the original image. It can also be imagined to make the object fat , and the width of this circle is determined by the size of the convolution kernel. In fact, the convolution kernel slides and calculates along the shadow. If there is only one pixel value in the range of the convolution kernel mxn, then the new pixel value is 1, otherwise the new pixel value the pixel value of keeps the original pixel value, which means that all pixels scanned by the convolution kernel will be expanded or dilated (to 1), so the white area of the entire image will increase.

Uses of Dilation:
Purpose 1: Dilation image expansion is usually used in conjunction with image erosion. First, the erosion method is used to narrow the lines in the image and also remove the noise, and then the image is expanded back through Dilation.

Purpose 2: Used to connect two very close but separate objects

## Deep Painterly Harmonization Theory

###  Style transfer

**Style transfer is the technique of recomposing one image in the style of another. Two inputs, a content image and a style image are analyzed by a convolutional neural network which is then used to create an output image whose “content” mirrors the content image and whose style resembles that of the style image








