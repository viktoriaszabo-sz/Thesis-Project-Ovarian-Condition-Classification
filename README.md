# Medical Image Classification 

This repository is for the project , on which I've written my bachelor's thesis, "Convolutional Neural Network for Medical Image Classification"

In the project, I used openly accessible datasets from RoboFlow. These two datasets consist ultrasound images of various types of ovarian cancers, PCOS, and other benign conditions.
The project aims to create a multi-model programme, which is capable of classifying the tumours, and assigning the correct labels to them. Since the datasets consists cases with various malignancies, it was important to choose the right models for the highest accuracy possible, while keeping down the computaitonal cost. With this adjustment, the solution is implementable into low computer resource enviroments, or embedded software solutions. 

The project relies on transfer learning, upon pre-trianed PyTorch models, where only the final layers were retrained on the current dataset. With this, the computational cost rewquired during training was kept to the minimum. 
The three models were ensembled in an additional neural network upon weighted average, where each model contributed differently, depending on their accuracy scores and types of errors made during the training and testing. 
Hyperparamteres were optimized using the Bayesian optimization method, which is commented out in order to not overload the hardware during the testing period. 

The project filed are divided for the best outlook and debugging purposes. The project is highly adaptable and scaleable. 

Future works include experimenting with more models and higher accuracy. Implementation of different Computer Vision exercises, such as segmentation will be considered. 

The final ensembled model is capable of recognizing the ovarian condition on the input image. 
