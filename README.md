# Facial-Expression-Detection
Using EigenFaces and Principle component analysis in Python & Open-CV
The goal of the code is to take a labelled set of images collected from the IMFDB data set that contains iamges of various 
bollywood actors from various movies. The training images have been taken a put into 2 expression folders i.e. happy and sad.
This image labelling is done manually.
After that code reads the happy and sad folders in a vectorized form i.e an input matrix of the shape (number of row pixels*
number of coloumn pixels, number of examples). The image while taking input has been converted into GrayScale using opencv, 
this is done to reduce computational strain.

After that general Principle Component Analysis approach is used where we first calculate the avergae of all training examples to
get a mean face which is subtracted from all the images to reduce the noise and center the general face in every image. Also
the benefit of using this IMFDB database is that all images have been cropped before hand such that they have the image of the actor
centred, therefore we don't need to run a feature extracting algorithm like HAAR cascade classifier to get a bounding region for
the face in the image.

After comptuing the mean face we make use of PCACompute() (a built in opencv function) to obtain the eigenfaces or eigen vectors
for all the training examples. To reduce the number of dimensions to compute while calculating the covarience matrix we perform
A(tranpose)*A and not A*A(transpose). After getting our covarience we calculate all the top m eigenfaces (m is number of training
 examples). We use this to compute the weights for all the images and thus now we store only the weights for all the trained images
 thereby achieving dimensionality reduction.
 
 Any new image which is input is converted into weight vector format and then we calculate this eucildian distance of this weight
 vector with all stored images in happy and sad set. The distance to whichever set is smaller, the expression is classified into that
 class/set.
 
 
