# **PPG ECE 189 Kadambi**
This project contains all the code and information for the ECE 189 project. The project is about obtaining PPG signals from videos of the face.

## **Table of Contents**
- [**PPG ECE 189 Kadambi**](#ppg-ece-189-kadambi)
  - [**Table of Contents**](#table-of-contents)
  - [**Introduction**](#introduction)
  - [**Model**](#model)
    - [**Training Data**](#training-data)
    - [**Preprocessing**](#preprocessing)
      - [**Skin Segmentation**](#skin-segmentation)
      - [**Linear Color Reduction**](#linear-color-reduction)
    - [**CNN Models**](#cnn-models)
      - [**2D CNN**](#2d-cnn)
      - [**3D CNN**](#3d-cnn)
    - [**Optimizer and Loss Function**](#optimizer-and-loss-function)
  - [**Final Results**](#final-results)
  - [**Limitations and Future Work**](#limitations-and-future-work)

## **Introduction**
Remote photoplethysmography (rPPG) is a non-invasive technique that uses a video camera to virtually measure a person's vital indicators such as heart rate, respiration rate, and blood oxygen saturation. The rPPG technique is based on the fact that hemoglobin absorbs light differently depending on whether it is oxygenated or deoxygenated in the blood. When the heart beats, the blood volume in the face changes, causing tiny variations in skin tone that can be observed by a camera. To record footage of a person's face, a standard RGB camera is usually used. The footage is then processed using algorithms that extract the coloring variations produced by the pulsing blood flow, resulting in a signal that represents the person's vital signs. To train these algorithms and test their accuracy, a person's vital signs are measured using a clinical device such as a pulse oximeter. The rPPG signal is then compared to the clinical signal to determine the accuracy of the algorithm.

This report will highlight some of the techniques that we have used to improve the quality of our rPPG signal and achieve a mean absolute error (MAE) close to clinical standards. We will also discuss the use of 2D and 3D convolutional neural networks (CNNs) as well as other signal and image processing techniques to improve the signal quality, such as skin segmentation and linear color reduction. Additionally, we will talk about future work and improvements that can be made to this project.

## **Model**
### **Training Data**
Our training data consists of 300 videos of length 1600 seconds. While training we only used 512 frames sampled at 20 frames per second. We did not experiment with training and testing split ups much, but our split was the first 80% of the data for training and the last 80% of the data for testing.

### **Preprocessing**
#### **Skin Segmentation**
We hypothesized that skin segmentation could improve the performance of the model by removing non skin pixels. This would allow the model to focus on the skin pixels rather than non skin pixels. We made two attempts at skin segmentation. First, we used SemanticSegmentation by WillBrennan, which took a large amount of time to run. Second, we tried using LBFmodel by kurnianggoro, which ran around 100x faster. 
![SemanticSegmentation](https://i.imgur.com/BVWLo45.png)

The first model is SemanticSegmentation and the second is LBFmodel.

Since the first method was too long we could never implement it with a neural network model, however the second model did not help improve the MAE of the model. We will discuss potential improvements that could be implemented to help skin segmentation improve the deep learning model.

#### **Linear Color Reduction**
When blood pumps on the face, the change in color occurs on a vector in the RGB space. To visualize these changes, here are two plots showing why this observation is useful. The first one shows a 3D scatter of the average RGB value across all pixels of every frame, and the second is a video plot showing how the traversal of this average value across time. Instead of analyzing the video over three color channels, by taking the dot product of this vector by the RGB value of each pixel allows us to reduce the color channels to 1, and observe the RGB change in the direction of the vector. Implementations of the color reduction vary by model and thus its performance will be discussed there.
![3D Scatter](https://i.imgur.com/swArvMw.png)
![Video Plot](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjNiMGE1OTdkZTMxNTAzYWVkMWMzYjY5NzM3YWIyNmEyOGFjMTEzYiZjdD1n/3HdAmXuJW70a3eQZ2K/giphy.gif)

### **CNN Models**
#### **2D CNN**
Our 2D Convolutional Neural Network follows a standard Computer Vision architecture. It has 5 Convolution layers followed by a Dense layer. Each convolution layer consists of a 2D Convolution, a TanH activation and a 2D MaxPool. The model takes in the derivative of the PPG signal as an input, to keep the signal smooth. Each convolution layer extracts features from the image, and after a number of epochs, the model picks up on features that extract the heart rate signal. With this approach we were able to achieve an MAE of 5.6 on the test set. In an attempt to better explain and visualize the model here are some selected filters from each convolution layer. The first image is the time differential of one of the frames. As it progresses through the model, the convolutions apply different filters and the maxpools reduce the resolution. 
![2D CNN](https://i.imgur.com/Lxcz3Wi.png)
While applying the linear color reduction to the model, we found that finding the correct vector for each video was too time consuming, so we added a learnable vector to the model. We found that adding this transformation and reducing the color channels to 1 significantly improved the speed of the model and decreased the MAE of the model by 8 times. Despite that, we found that 3D CNNs performed better.


#### **3D CNN**
Our final model is a 3D Convolutional Neural Network with 6 convolutional layers, each having a 3D CNN followed by a batch normalization, a ReLU activation and a maxpool. Unlike the 2D convolution layer, the 3D CNN keeps a consistent 32 channels throughout the convolutions, which gives better results. The ReLU function was found to improve computational efficiency and counter overfitting by dropping out layers. Furthermore the batch normalization also improved computational efficiency. Additionally, the maxpool extracts the most prominent features of the Neural Network. This time, we found that the linear color reduction decreased our performance. However, concatenating this to the existing 3 colors at the beginning of the model improved the MAE of our model on the test set from 2.65 to 2.18. Here is a visual of 4 filters from each convolution layer.
![3D CNN](https://i.imgur.com/FkQFJ7W.png)
In the first layer we see the RGB filtered image along with the image with the linear color reduction. Similar to the 2D CNN model, as the layers progress, the model applies different filters and the maxpool reduces the resolution.


### **Optimizer and Loss Function**
Our final model used an Adam optimizer with 1E-3 learning rate and 1E-4 weight decay. We tried a number of different loss functions including Neg Pearson, MSE and a custom Peak comparing loss function. We hoped that our color reduced channel magnified the vectors where color change was significant, and that the model would learn to look along those changes, especially in peaks and troughs of the ground truth. To guide the model with this, we developed a loss function to compare the values of the ground truth and estimated ppg signal where the ground truth had peaks and troughs. In this plot, the dots represent the peaks and troughs, where the ground truth and estimated signal are compared.
![Loss Function](https://i.imgur.com/MSHpKuo.png)
While implementing this loss function, we tried combinations of MAE, Peak and Trough and Neg Pearson. The MAE loss function and our custom loss function overfitted the data while simultaneously punishing faulty lighting conditions. The Neg Pearson on the other hand guided our model well without overfitting the data, so we decided to try permutations of each loss function. Ultimately, Neg Pearson outperformed all the other loss functions, since it is efficient at handling multi-channel data with a broad variety of intensities. Neg Pearson loss is capable of handling such data because it normalizes it to have a zero mean and unit variance, reducing the influence of intensity differences between channels. Furthermore, the Neg Pearson loss is affected by the linear connection between the predicted and ground truth signals. This makes it especially helpful in situations where the connection between input and output variables is anticipated to be linear. In the case of 3D-CNNs, the Neg Pearson loss can be used to optimize network parameters in order to learn features that represent such linear correlations in the data.

## **Final Results**
<!-- Table -->
This only shows the results of our final model. We may add the results of other models in the future.
| | Train Set | Test Set | Out of Distribution set |
| --- | --- | --- | --- |
| MAE | 1.76 | 2.18 | 8.8 |


## **Limitations and Future Work**
Even though our model performs quite well on our test and train set, there are some limitations in the models abilities to predict PPG signals. First, there is a relatively significant difference between the MAE on the test set and the MAE on the train set which shows that our model overfits the train data. This is mainly because we did not test our results on a validation set after every epoch to check overfitting. Furthermore, our model was limited to a small window looking at only the face of the person, and without a background, it was difficult to evaluate the PPG signal based on changing lighting conditions.
Although our final model consisted of  6 3DCNN layers with a linear color reduced filter and a Neg Pearson loss, we tested many other techniques that have potential. One such technique would be skin segmentation: If we could find a way to improve the computational efficiency of skin segmentation and perform it on those videos, we believe that would significantly improve the performance of the model. We also wanted to experiment with a more guided model architecture by adding additional filtered channels to the input and creating a more guided loss function that worked with these filters. Additionally we wanted to experiment with sectioning off parts of the face and taking the dot product of vectors based off of the region of the face the pixel fell on. We believe this may improve results since different parts of the face are affected by the blood volume in different intensities. Furthermore we would like to look into changing lighting conditions and experiment with filters that understand those lighting changes based on background information. Finally, one of the biggest challenges in this project was the model over fitting. We would like to add a validation loss check at each epoch, and play around with changing loss functions and or optimizers to address overfitting.
