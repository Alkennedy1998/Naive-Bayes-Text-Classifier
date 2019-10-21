# Naive Bayes Text Classifier for Symantec Data Science Internship

## Automated the customer support ticket classification process to determine if an issue was, for example, with billing or installation.

## Increased labeling rate by over 450% through implementing a naïve Bayes classifier 

## Cleaned, formated, and analyzed training data with Numpy, Pandas, and Matplotlib. Trained the model and classified data using sklearn

### Process funcitons as follows:

1. Data was cleaned, few rows with missing data remove
2. Converted the text into TF-IDF feature vectors. Basically gave a score to each word depending on how often it appeared muliplied by te inverse of how many texts it appeared in
3. Data was seperated into training and testing sets
4. Gaussian Naive Bayes model was trained used the training data
5. Testing data was classified using the newly trained model
6. How the test data was labeled by the classifer was compared to the real test data labels and accuracy calculated
7. Confusion Matrix plotted for visualization
8. Model ready to use! :)



-All data had to be removed for security reasons
-Also keep in mine this is one of the first coding projects I ever did so yes there is a lot of extra unessessary package imports
