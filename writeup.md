# **Traffic Sign Recognition** 

## Writeup for Project 2

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/TypeHistogram.png "Histogram of Traffic Sign Types"
[image2]: ./writeup_images/TypeSequence.png "Traffic Sign Type by Sequence Number"
[image3]: ./writeup_images/Example11144.png "Example 11144"
[image4]: ./writeup_images/Example11144.png "Example 11145"
[image5]: ./writeup_images/PreAugmentation.png "Traffic Sign Before Preprocessing"
[image6]: ./writeup_images/AugmentedData.png "Generated Traffic Sign Using OpenCV"
[image7]: ./writeup_images/NormalizedData.png "Traffic Sign After Normalization"
[image8]: ./test_images/1.JPG "Traffic Sign 1"
[image9]: ./test_images/2.JPG "Traffic Sign 2"
[image10]: ./test_images/3.JPG "Traffic Sign 3"
[image11]: ./test_images/4.JPG "Traffic Sign 4"
[image12]: ./test_images/5.JPG "Traffic Sign 5"
[image13]: ./test_images/6.JPG "Traffic Sign 6"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/aplegendre/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used basic python functions and the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34,799 examples.
* The size of the validation set is 4,410 examples.
* The size of test set is 12,630 examples.
* The shape of a traffic sign image is 32 x 32 x 3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing the count of each Traffic Sign Type. It shows that the majority of examples are speed limits or signs that indicate right of way such as yield or stop signs. There are significantly fewer examples of specialty signs such as those that indicate road hazards or specify crossing types.

![alt text][image1]

Another view of this same information shows that the examples are somewhat ordered. When viewing the Traffic Sign Type by example number, it can be seen that there are long blocks of examples of a single type.

![alt text][image2]

A closer look shows that within these blocks, the images are almost identical. It seems that they are sourced from frames of video, so there are slightly different views of the same exact sign. For example, here are two adjact images of the "No Passing" sign type, which are almost indistinguishable.

![alt text][image3] ![alt text][image4]

With these characteristics, there is significant risk of overfitting on this dataset, especially for the less common sign types. There are relatively few examples for many sign types and the examples that do exist have almost identical features.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Many of my preprocessing and model architecture ideas came from the Sermanet and LeCun article that was provided as a reference for this project. For preprocessing, my two main goals were to normalize the images and to provided additional feature variation to reduce the overfitting risk described above.

As a first step, I shuffled the data so that any additionally generated data is randomly sourced from the training population rather than grabbing multiple images from a similar batch. After shuffling, I used OpenCV to generate randomly rotated, scaled, and translated versions of 20% of the training set and appended them to the training set. This provided examples with different features than the base set to avoid overfitting.

Here is an example of a traffic sign image before and after random rotation, scaling, and translation:

![alt text][image5] ![alt text][image6]

Once I had an augmented training set, I converted all of the examples from RGB to YUV color mapping. This allowed me to retain color data (in the UV channels), but also have a grayscale channel for normalization. This is a technique that was recommended by the Sermanet article and proved to be useful in improving accuracy.

As a final step, I normalized the data using the Equalize Historgram function from OpenCV on the Y-channel to provide a more uniform contrast across the training set. In exploring the data, I noticed that some images had strong contrast while others were quite dark, so this step allowed features within each type of image to appear more similar to each other. I completed the nomalization by dividing the images by 255 so that the range is more consitent with the initialized weights and the suggested initial hyperparameter settings.

Here is what the grayscale channel of the previous image looks like after histogram equalization:
![alt text][image7]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers in series:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5		| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Dropout				| 50% keep probability							|
| Max pooling			| 2x2 stride, outputs 14x14x12 					|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x50 	|
| RELU					|												|
| Dropout				| 50% keep probability							|
| Max pooling			| 2x2 stride, outputs 5x5x50 					|
| Fully connected		| Inputs 1250x1 (5x5x50), outputs 512x1			|
| RELU					|												|
| Dropout				| 50% keep probability							|
| Fully connected		| Inputs 512x1, outputs 128x1					|
| RELU					|												|
| Fully connected		| Inputs 128x1, outputs 43x1					|
| Softmax				| As part of the loss function definition		|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer since most of the network architecture was take from the LeNet lab and the Adam Optimizer was used in that lab. The optimizer sought to minimize the loss, which was defined as the mean cross-entropy of the Softmax of the output from the final layer as compared to the labels.

I used a default batch size of 128, which was used in the LeNet lab and seemed to execute quickly enough on my computer. I used 15 epochs because the validation accuracy would continue to improve after 10 epochs, but typically converged by epoch 15. Additional epochs would not provide much benefit with the learning rate that I selected and may even be detrimental if the network began to overfit.

For the learning rate, I used the default learning rate of 0.001. I tried lower rates to try to improve the loss, but they did not seem to have a large impact on the final validation accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of 98.7%
* Validation set accuracy of 94.9% 
* Test set accuracy of 93.2%

I started with standard LeNet architecture from the LeNet Lab and added dropout layers. At first, I attempted to use a secondary connection of the first convolutional layer output into the first fully connected layer. This was a recommendation from the Sermanet paper, but it was not very effective for me. I believe it increased the amount of overfitting, so I removed it. Instead, I focused my network design on tuning the number of outputs per layer and selecting the number of dropouts to include. The addition of the dropout layers was particularly important because it limitted overfitting.

The parameter that I spent the most time adjusting was the output size of my layers. Initially, I had a very large number of outputs to allow the network to learn around my dropout layers and because I originally had a secondary connection for one of the layers. This led to severe overfitting and low validation accuracy. Ultimately, I chose to have the network grow quickly and then shrink back down the number of classes, so that the maximum number of outputs was roughly in the middle of the network where there should be the greatest feature complexity.

In choosing locations for dropout, I focused mainly on the convolutional layers since they had the most number of outputs in my network. This meant that if one critical output dropped out, then there should be a sufficient number of additional outputs to make up for it. I did not include dropout in the final layer before the classifier because I had only a limited number of inputs into the classifier and wanted to ensure that enough features would always be available for classification.

The main struggle that I had in achieving good test accuracy was related to my data preprocessing. My initial attempts had poor normalization and no data augmentation. Once I added a contrast normalization and created some augmented data, my results became acceptable.

Based on the final results, this model seems to be fairly effective. The training accuracy is near perfect as it should be and the validation and test accuracies are not substantially lower than the training accuracy. If the training accuracy were good and the others were low, it would indicate overfitting, but that doesn't seem to be the case for my final solution.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12] ![alt text][image13]

Based on the training set, the first and the last image should be easy to classify. Both are speed limit signs with good contrast, which is very similar to the bulk of the training set. Images 4 and 5 however are difficult to classify. They are both red triangles with a detailed image in the center and are of a type that has relatively few examples in the training set. In early implementations of my network, both of these signs were classified incorrectly as different hazard or crossing signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)	| Speed Limit (60km/h)							| 
| Turn Right Ahead		| Ahead Only									|
| No Entry				| No Entry										|
| Bicycle Crossing		| Bicycle Crossing				 				|
| Road Work				| Speed Limit (30km/h)							|
| Speed limit (30km/h)	| Speed Limit (30km/h)							| 

The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.7%. This is substantially lower the the test accuracy and is likely due to the lower number of examples tested here. I have trained this same model without modification and have had varying results due to the random data augmentation and random model optimization. In some cases, these 6 images were classified with 100% accuracy, but I think the results here are more typical and better show the limitations of the classifier. This particular example is a bit strange as it predicted a circular speed limit sign for a triangular road work sign. The other error, in which a "Turn Right Ahead" was mis-classigied as "Ahead Only", is more understandable as those images have much more similar features.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is near the end of the Ipython notebook.

For the first image, the model is absolutely sure that this is a speed limit sign, since 4 of the 5 top guesses are speed limit signs. It correctly guessed the limit to be 60km/h, but 50km/h was a close second. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .52         			| Speed Limit (60km/h)							| 
| .32     				| Speed Limit (50km/h) 							|
| .12					| Speed Limit (80km/h)							|
| .01	      			| Speed Limit (30km/h)			 				|
| .01				    | Wild Animals Crossing							|


For the second image, the model is absolutely sure that this is a directional sign, and in particular, that it is "Ahead Only" or "Turn Right Ahead". The two of these options are almost a 50/50 split. Unfortunately, the model guessed incorrectly in this case and its second choice was actually correct. I suppose this could be in part due to my data augmentation where I've randomly rotated images. Perhaps some of the training examples were rotated too far and confused the model.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .53         			| Ahead Only									| 
| .43     				| Turn Right Ahead								|
| .03					| Turn Left Ahead								|
| .00	      			| Go Straight or Left			 				|
| .00				    | No Passing									| 

For the third image, the model is 100% certain that it is a "No Entry" sign, which is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No Entry										| 
| .00     				| Stop											|
| .00					| Speed Limit (20km/h)							|
| .00	      			| No Passing			 						|
| .00				    | Turn Left Ahead								| 

For the fourth image, the model is very uncertain. It guessed correctly that the sign was "Bicycle Crossing", but only with 27% certainty. It almost guessed incorrectly that the sign was "Children Crossing", which looks almost identical due to the same red triangle background with a detailed picture in the middle.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .27         			| Bicycle Crossing								| 
| .26     				| Children Crossing								|
| .08					| Speed Limit (60km/h)							|
| .04	      			| Speed Limit (80km/h)	 						|
| .04				    | Slippery Road									| 

For the fifth image, the model is very uncertain, as it was for the fourth image. Its guess was entirely wrong, but only had 24% certainty. The true sign type, "Road Work", was down at #4 with 9% certainty. 4 of the top 5 guesses were all red triangle signs, so the model was on the right track, but could not match the inner picture. The various red triangle signs split the vote, so that a very wrong answer won. My explanation for why "30km/h" came out on top is that the man looks a little like a 3 and there's a shadow that could look like a 0 after histogram equalization. Additionally there is a red outline in both sign types, so maybe including color data hurts more than it helps.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .24         			| Speed Limit (30km/h)							| 
| .15     				| Wild Animals Crossing							|
| .12					| Double Curve									|
| .09	      			| Road Work				 						|
| .08				    | General Caution								|

For the sixth image, the model is quite certain that it's a speed limit sign, but is surprisingly uncertain about the actual limit when compared to the 60km/h sign's top 5 results. Here, it is essentially a toss-up between 30km/h (37%) and 20km/h (35%), but luckily the model made the right guess. Similar to the issue with the triangle signs, the model seems to have trouble distinguishing the detailed inner contents of the signs, even when there are substantially more training examples as there are for speed limit signs.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .37         			| Speed Limit (30km/h)							| 
| .35     				| Speed Limit (20km/h)							|
| .10					| Speed Limit (50km/h)							|
| .06	      			| Speed Limit (70km/h)	 						|
| .03				    | Speed Limit (100km/h)							| 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I did not yet complete the optional visualization, but my return to this at a later date.
