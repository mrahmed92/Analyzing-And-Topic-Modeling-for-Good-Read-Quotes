import pandas as pd
import spacy
from gensim import corpora, models
from gensim.models import CoherenceModel

# Load the existing LDA model
lda_model_path = '/Users/muhammadahmed/Desktop/DM/lda_model.gensim'
lda_model = models.LdaModel.load(lda_model_path)

# Load the dictionary used in the model
dictionary_path = '/Users/muhammadahmed/Desktop/DM/lda_model.gensim.id2word'
dictionary = corpora.Dictionary.load(dictionary_path)

# Load your processed documents
data_path = '/Users/muhammadahmed/Desktop/DM/quotes.csv'
df = pd.read_csv(data_path)
documents = df['quote'].fillna('').tolist()  # Ensure non-empty strings
nlp = spacy.load('en_core_web_sm')

# Preprocess the text
def preprocess(text):
    if isinstance(text, str) and text:  # Ensure it's a non-empty string
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return tokens  # Return tokens directly
    return []  # Return empty list for non-string inputs

# Preprocess the documents
processed_docs = [preprocess(doc) for doc in documents]

# Create the corpus (term-document matrix)
corpus = [dictionary.doc2bow(text) for text in processed_docs if text]  # Ensure non-empty texts

# Print the topics found by LDA with top words and save them to a text file
topics_output_path = '/Users/muhammadahmed/Desktop/DM/topics.txt'
with open(topics_output_path, 'w') as f:
    print("LDA Topics and Top Words:")
    f.write("LDA Topics and Top Words:\n")
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        topic_text = f"Topic {topic[0]}: {topic[1]}"
        print(topic_text)
        f.write(topic_text + "\n")


