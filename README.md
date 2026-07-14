    this aims to classify text into written and generated categories.
    using Flask to display text box, get input and show result

    lets explore ways to analyze and identify text.
    and find out if a piece of text was written or generated.

    data is read from csv files. preprocessing steps performed include split the data into training and testing, normalizing and tokenizing text.
    currently models implemented for the identificaiton of texts include naive bayes, and svm.

    the dataset is run on naive bayes. with bag of words done by count vectorizer from sklearn. the model achieve accuracy of 0.946 after a split of 0.2 test size.

    the dataset is run on svm. with term frequency and inverse document frequency done by from sklearn. the model achieve accuracy of 0.96 after a split of 0.2 test size.

    the functionality of uploading a text file was added for detection purpose along with the already given text area.

    global explainability by displaying top 20 words choosen from both written and generated text.
