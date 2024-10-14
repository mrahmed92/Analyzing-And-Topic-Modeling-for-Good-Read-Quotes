import pandas as pd
import spacy
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from nltk import ngrams
import multiprocessing

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

# Load your data
data_path = '/Users/muhammadahmed/Desktop/DM/quotes.csv'
df = pd.read_csv(data_path)

# Check for missing values and handle them
df['quote'] = df['quote'].fillna('')  # Replace NaN with empty string

# Assuming your quotes are in a column named 'quote'
documents = df['quote'].tolist()

# Function to create bigrams or trigrams
def create_ngrams(tokens, n=2):
    return list(ngrams(tokens, n))

# Preprocess the text
def preprocess(text):
    if isinstance(text, str) and text:  # Ensure it's a non-empty string
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        bigrams = create_ngrams(tokens, n=2)  # You can change this to 3 for trigrams
        return [' '.join(bigram) for bigram in bigrams]  # Join bigrams into strings
    return []  # Return empty list for non-string inputs

# Preprocess documents in parallel
def parallel_preprocess(documents):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        processed_docs = pool.map(preprocess, documents)
    return processed_docs

if __name__ == '__main__':
    # Process documents
    processed_docs = parallel_preprocess(documents)

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(processed_docs)

    # Filter extremes to limit vocabulary (further reduced by 50%)
    dictionary.filter_extremes(no_below=1, no_above=0.015625)  # Adjusted values

    # Create a corpus
    corpus = [dictionary.doc2bow(text) for text in processed_docs if text]  # Ensure non-empty texts

    # Use LdaMulticore for better performance on multi-core machines
    num_topics = 5  # Keep the same number of topics
    chunksize = 62  # Reduced from 125 to 62
    passes = 1  # Remain at 1
    iterations = 2  # Reduced from 4 to 2

    # Perform LDA using LdaMulticore
    lda_model = models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary,
                                     passes=passes, iterations=iterations,
                                     chunksize=chunksize, workers=multiprocessing.cpu_count())

    # Save the LDA model
    lda_model.save('/Users/muhammadahmed/Desktop/DM/lda_model.gensim')

    # Print the topics found by LDA
    topics = lda_model.print_topics(num_words=4)
    for topic in topics:
        print(topic)

    # Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}')

    # Visualizing the topics and saving as HTML
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, '/Users/muhammadahmed/Desktop/DM/lda_visualization.html')

    

    