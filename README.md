# **Multiclass Text Classification**
### Python Script Explanation

#### Imports
The script starts by importing necessary libraries and modules including pandas for data handling, numpy for numerical operations, seaborn and matplotlib.pyplot for visualization, sklearn modules for machine learning tasks, gensim for Word2Vec embeddings, re for text cleaning, and pickle for model serialization.

#### Function Definitions

1. **load_data(file_path)**:
   - Purpose: Loads data from a CSV file, selects relevant columns ('Full_Article', 'Article_Type'), and drops any rows with missing values.
   
2. **clean_text(text)**:
   - Purpose: Cleans text by removing HTML tags and non-alphanumeric characters.
   
3. **preprocess_data(df)**:
   - Purpose: Applies the `clean_text` function to clean the 'Full_Article' column in the DataFrame.
   
4. **encode_labels(df)**:
   - Purpose: Encodes categorical labels ('Article_Type') using `LabelEncoder`, maps encoded labels to their original categories, and returns the processed DataFrame, encoder, and mapping.
   
5. **balance_data(X, y)**:
   - Purpose: Uses RandomOverSampler to balance the dataset by oversampling minority classes.
   
6. **get_sentence_embeddings(data_texts, word_vectors)**:
   - Purpose: Generates sentence embeddings using Word2Vec embeddings from `gensim.models.KeyedVectors` for each text in `data_texts`.
   
7. **train_and_evaluate_model(clf, train_embeddings, train_labels, test_embeddings, test_labels, label_mapping, model_name)**:
   - Purpose: Trains a classifier (`clf`) on training embeddings (`train_embeddings` and `train_labels`), evaluates it on test embeddings (`test_embeddings` and `test_labels`), prints classification report, accuracy score, and confusion matrix visualization using seaborn.
   
8. **save_model(clf, file_path)**:
   - Purpose: Saves a trained model (`clf`) to a file using pickle serialization.
   
9. **load_model(file_path)**:
   - Purpose: Loads a trained model from a file using pickle deserialization.

#### Main Execution (`if __name__ == "__main__"`)
   
- **Data Loading and Preprocessing**:
  - Loads data from a CSV file, preprocesses it (cleans text and encodes labels).

- **Data Balancing**:
  - Balances data by oversampling using RandomOverSampler.

- **Embeddings Generation**:
  - Loads Word2Vec embeddings (`word_vectors`) and generates sentence embeddings for balanced data.

- **Train-Test Split**:
  - Splits embeddings and labels into training and test sets.

- **Model Training and Evaluation**:
  - Trains and evaluates a RandomForestClassifier and an SVM classifier (`clf_rf` and `clf_svm`), prints evaluation metrics and displays confusion matrix for each.

- **Model Serialization**:
  - Saves the trained RandomForestClassifier (`clf_rf`) to a file and then loads it back into `loaded_clf_rf`.

#### Conclusion
This script demonstrates a machine learning pipeline for text classification using Word2Vec embeddings, RandomForestClassifier, and SVM. It handles data loading, cleaning, encoding, balancing, embeddings generation, model training, evaluation, and serialization.


## 🔗 Connect with Me

Feel free to connect with me on LinkedIn:

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://parthebhan143.wixsite.com/datainsights)

[![LinkedIn Profile](https://img.shields.io/badge/LinkedIn_Profile-000?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/parthebhan)

[![Kaggle Profile](https://img.shields.io/badge/Kaggle_Profile-000?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/parthebhan)

[![Tableau Profile](https://img.shields.io/badge/Tableau_Profile-000?style=for-the-badge&logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/parthebhan.pari/vizzes)
