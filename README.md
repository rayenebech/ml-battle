# Machine Learning Models' Battle on Text Classification ⚔️ #

In this project, I fine-tuned different classical machine learning models and neural networks on Text Classification and Topic Modeling tasks.
The results of the evaluation are explained in the "reports.pdf" file.  
The architecture of the models is explained in the figures below:
1. Latent Semantic Analysis (LSA):
The objective of LSA is reducing dimension for classification. The idea is that words will occurs in similar pieces of text if they have similar meaning.
The model’s architecture is:
<img width="660" alt="Screen Shot 2023-09-18 at 17 41 01" src="https://github.com/rayenebech/text-classification/assets/34574318/a1d0987a-2448-4242-91bb-c87b750ddf4e">


2. Latent Dirichlet Allocation (LDA):
In LDA, documents are represented as a mixture of topics and a topic is a bunch of words. Those topics reside within a hidden (latent) layer.
<img width="667" alt="Screen Shot 2023-09-18 at 17 41 36" src="https://github.com/rayenebech/text-classification/assets/34574318/93b6f0d7-a871-4acf-b126-ea0031eb4d00">


3. Term frequency Inverse Document Frequency (TF-IDF) and TF-IDF Character Gram:
In this case we applied TF-IDF on character level. Instead of computing the occurrences of the words, the occurrences of the characters are computed. by setting the range (2,6) which means using from the bigrams, trigrams, 4-grams, 5-grams and 6-grams.
<img width="671" alt="Screen Shot 2023-09-18 at 17 42 14" src="https://github.com/rayenebech/text-classification/assets/34574318/cd2593c1-2ba5-4076-8914-75c665083c59">


4. FastText Model:
A very fast model for computing word representations. Each word is represented as a bag of character n-grams in addition to the word itself. For example, for the word “apple”, with n = 3, the fastText representations for the character n-grams
is <ap, app, ppl, ple, le>. This allows to capture meaning for suffixes/prefixes. The n-gram featuers are averaged to form the hidden layer.
<img width="687" alt="Screen Shot 2023-09-18 at 17 42 50" src="https://github.com/rayenebech/text-classification/assets/34574318/369b27a1-0156-43f2-9a9d-25afd31dbd40">


5. BERT Model:
Bidirectional Encoder Representations from Transformers (BERT) is based on the self- attention mechanism of the Transformers models. Every token in the input sequence is related to other tokens. The powerful mathematics behind this concept enables the model to capture context-based features while generating word embeddings.
In this project, the “bert-base-uncased” model was used in all experiments to extract the word embeddings. The embeddings are then fed to a linear classifier to predict the classes. Also a dropout layer was added between the embeddings layer and the classifier function. The model’s architecture can be visualized as follows:
<img width="671" alt="Screen Shot 2023-09-18 at 17 43 22" src="https://github.com/rayenebech/text-classification/assets/34574318/36785936-aab5-4b61-b607-f88e4573fe23">



