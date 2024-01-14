The Jupyter Notebook `notebook.ipynb` is a comprehensive project on explainable machine learning, specifically focusing on image classification of animals. The code implements a model using PyTorch and ResNet18 architecture to classify images of animals. The code is divided into several sections, including imports, exploratory data analysis (EDA), model definition, training, evaluation, and Grad-CAM visualization.

1. **Imports**: The code begins by importing necessary libraries and modules, including PyTorch, pandas, matplotlib, seaborn, and others. It also loads environment variables using the `dotenv` package.

2. **Exploratory Data Analysis (EDA)**: The EDA section reads in the image labels from a text file and creates a pandas DataFrame. It then displays some basic statistics about the data and visualizes a few random images along with their labels. The distribution of classes (animal types) in the dataset is also visualized.

3. **Model Definition**: The authors define a custom ResNet18 model for the task. The model is loaded with pre-trained weights from ImageNet and the final fully connected layer is replaced to match the number of classes in the dataset. The model can be loaded from a saved state if available.

4. **Training**: The training section splits the data into training and testing sets, and defines the loss function and optimizer. If the `do_train` flag is set to True, the model is trained for a specified number of epochs and the training loss and accuracy are plotted.

5. **Evaluation**: In the evaluation section, the model is set to evaluation mode and predictions are made on the validation set. The accuracy of the model is calculated for each class and the mean accuracy is printed. The classes with the lowest accuracy are also displayed.

6. **Grad-CAM Visualization**: The final section of the code uses the Grad-CAM technique to visualize the areas of the image that the model focuses on when making a prediction. This is done for both correctly and incorrectly classified images. The gradients are calculated for the true class in the case of incorrect predictions to show the difference between the true class and the predicted class in the heatmap.

In summary, this Jupyter Notebook provides a detailed implementation of an image classification task using a ResNet18 model, with a focus on explainability through Grad-CAM visualizations. The authors have also included comprehensive EDA and model evaluation sections to understand the data and assess the model's performance.

---

On the other hand, the notebook `notebook_anomaly.ipynb` is a comprehensive script for an explainable machine learning project. It implements a model for anomaly detection using PyTorch and ResNet18, a pre-trained model from torchvision. The code can be summarized into the following sections:

1. **Imports**: The script begins by importing necessary libraries and modules, including PyTorch, torchvision, pandas, seaborn, matplotlib, and others.

2. **Data Loading and Preprocessing**: The script loads the data from a specified path, reads labels from text files, and creates dictionaries for training and testing labels. It then creates pandas dataframes from these dictionaries and balances the dataset by sampling the same number of instances from each class.

3. **Model Configuration**: The script sets up the device (CPU or GPU), batch size, learning rate, and other parameters. It also defines a custom ResNet18 model, replacing the final fully connected layer to match the number of classes in the dataset.

4. **Data Visualization**: The script includes code for visualizing random images from the training set and their corresponding labels, as well as the class distribution.

5. **Model Training**: If the training flag is set, the script trains the model for a specified number of epochs, calculating the loss and updating the model parameters at each step. It also plots the training loss and accuracy over time.

6. **Model Evaluation**: The script evaluates the model on the validation set, calculating the accuracy and visualizing the class distribution of the predictions.

7. **Grad-CAM Visualization**: The script includes code for generating Grad-CAM visualizations, which highlight the areas of the image that the model focuses on when making a prediction. It shows these visualizations for both correctly and incorrectly classified images.

8. **Side-by-Side Comparison**: The script also includes code for visualizing the original image and the Grad-CAM visualization side by side, for both correctly and incorrectly classified images.
