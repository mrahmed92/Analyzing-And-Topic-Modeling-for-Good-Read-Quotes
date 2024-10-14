import pandas as pd
import spacy
from gensim import corpora, models
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import multiprocessing
import os

# Load the Spacy model and disable unnecessary pipeline components
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Load your processed documents
data_path = '/Users/muhammadahmed/Desktop/DM/quotes.csv'
df = pd.read_csv(data_path)
documents = df['quote'].fillna('').tolist()  # Ensure non-empty strings

# Preprocess the text in parallel
def preprocess(text):
    if isinstance(text, str) and text:  # Ensure it's a non-empty string
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return tokens  # Return tokens directly
    return []  # Return empty list for non-string inputs

def parallel_preprocess(docs):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        return pool.map(preprocess, docs)

# Preprocess the documents in parallel
if __name__ == '__main__':
    processed_docs = parallel_preprocess(documents)

    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(text) for text in processed_docs if text]  # Ensure non-empty texts

    # Create LDA using LdaMulticore for better performance
    lda_model = models.LdaMulticore(corpus, num_topics=5, id2word=dictionary, workers=multiprocessing.cpu_count(), passes=5, iterations=50)

    # Print and save topics
    topics_output_path = '/Users/muhammadahmed/Desktop/DM/topics.txt'
    with open(topics_output_path, 'w') as f:
        print("LDA Topics and Top Words:")
        f.write("LDA Topics and Top Words:\n")
        topics = lda_model.print_topics(num_words=5)
        for topic in topics:
            topic_text = f"Topic {topic[0]}: {topic[1]}"
            print(topic_text)
            f.write(topic_text + "\n")

    # Compute and save the Coherence score
    coherence_score_path = '/Users/muhammadahmed/Desktop/DM/coherence_score.txt'
    if not os.path.exists(coherence_score_path):
        coherence_model = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        with open(coherence_score_path, 'w') as f:
            f.write(f'Coherence Score: {coherence_score}')
    else:
        with open(coherence_score_path, 'r') as f:
            coherence_score = float(f.read().split(':')[-1].strip())

    print(f'Coherence Score: {coherence_score}')

    # Save the coherence score to a CSV file
    coherence_data = pd.DataFrame({
        'Coherence_Score': [coherence_score]
    })
    coherence_csv_path = '/Users/muhammadahmed/Desktop/DM/coherence_score.csv'
    coherence_data.to_csv(coherence_csv_path, index=False)

    # Plot the coherence score and save the plot as a PNG file with better readability
    plt.figure(figsize=(8, 6))  # Increased figure size for clarity
    bar = plt.bar(['Coherence Score'], [coherence_score], color='skyblue')
    
    # Add the score as text on the bar
    plt.text(0, coherence_score / 2, f'{coherence_score:.2f}', ha='center', va='center', fontsize=14, color='black')
    
    plt.ylim(0, 1)  # Set y-axis limit for better proportion
    plt.ylabel('Score', fontsize=14)
    plt.title('LDA Model Coherence Score', fontsize=16)
    
    # Increase font sizes for better readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the improved plot
    plt.savefig('/Users/muhammadahmed/Desktop/DM/coherence_score_plot.png', bbox_inches='tight')
    plt.show()
