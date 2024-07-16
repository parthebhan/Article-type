# **Multiclass Text Classification**

## Purpose:
The Python script uses  a pre-trained DistilBERT model to classify text article into seven categories: Commercial, Executives, Financing, Military,Others, Support & Services and Training

## Libraries Used:
- `FastAPI`: For creating the interactive web application.
- `transformers.DistilBertTokenizer`: Tokenizer for tokenizing input text.
- `transformers.TFDistilBertForSequenceClassification`: Pre-trained DistilBERT model for sequence classification.
- `tensorflow`: TensorFlow framework for model operations.
- `pandas`: For handling data structures.

## Components of the Script:

1. **Loading the Model and Tokenizer**:
   - The script loads a pre-trained DistilBERT model (`TFDistilBertForSequenceClassification`) and its tokenizer (`DistilBertTokenizer`) from the specified directory (`model_path`).

2. **Prediction Mapping**:
   - `prediction_mapping` defines a mapping from numerical predictions to human-readable categories.

3. **Prediction Function (`predict`)**:
   - `predict(input_text)`: Tokenizes the input text using the tokenizer, passes it through the loaded model, and predicts the category using `argmax` on the logits.

4. **Saving the Model**

 The trained model (model) and its tokenizer (tokenizer) are savedto a file named  `model_weights.h5`. 

5. **Validate and Evaluate**

   Evaluating the metrics of the model and plotting the Confusion Matrix



## 🔗 Connect with Me

Feel free to connect with me on LinkedIn:

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://parthebhan143.wixsite.com/datainsights)

[![LinkedIn Profile](https://img.shields.io/badge/LinkedIn_Profile-000?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/parthebhan)

[![Kaggle Profile](https://img.shields.io/badge/Kaggle_Profile-000?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/parthebhan)

[![Tableau Profile](https://img.shields.io/badge/Tableau_Profile-000?style=for-the-badge&logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/parthebhan.pari/vizzes)