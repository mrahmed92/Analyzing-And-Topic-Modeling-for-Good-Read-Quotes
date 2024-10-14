import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Load topics from your pre-processed topics file
topics_file_path = '/Users/muhammadahmed/Desktop/DM/topics.txt'

# Load your topics from the file
def load_topics(file_path):
    topics = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            if 'Topic' in line:  # We assume that topic lines start with 'Topic'
                topics.append(line.strip())
    return topics

# Parse the topics to extract words
def parse_topics(topics):
    parsed_topics = {}
    for topic in topics:
        try:
            topic_number = int(topic.split(":")[0].split(" ")[1])  # Extract the topic number
            words = [word.split('*')[1].replace('"', '').strip() for word in topic.split("+")]
            parsed_topics[topic_number] = words
        except (IndexError, ValueError) as e:
            print(f"Skipping invalid line: {topic}")
    return parsed_topics

# Question 1: Common Themes Across Topics
def common_themes_across_topics(parsed_topics):
    all_words = []
    for words in parsed_topics.values():
        all_words.extend(words)
    
    # Count the occurrences of each word
    common_words = Counter(all_words)
    
    # Plotting the top 10 common words
    common_words_df = pd.DataFrame(common_words.most_common(10), columns=["Word", "Count"])
    common_words_df.plot(kind="bar", x="Word", y="Count", color='skyblue', legend=False)
    plt.title("Most Common Themes Across Topics")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('/Users/muhammadahmed/Desktop/DM/common_themes_across_topics.png')
    plt.show()

# Question 2: Most Common Themes or Genres in Quotes
def most_common_themes(parsed_topics):
    all_words = []
    for words in parsed_topics.values():
        all_words.extend(words)

    # Count the occurrences of each word
    word_count = Counter(all_words)
    
    # Plotting the top 10 most common themes
    common_themes_df = pd.DataFrame(word_count.most_common(10), columns=["Theme", "Count"])
    common_themes_df.plot(kind="bar", x="Theme", y="Count", color='lightgreen', legend=False)
    plt.title("Most Common Themes or Genres in Quotes")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('/Users/muhammadahmed/Desktop/DM/most_common_themes.png')
    plt.show()

# Question 3: Which Topics Contain the Most Diverse Range of Themes?
def diverse_themes_by_topic(parsed_topics):
    diversity_by_topic = {topic: len(set(words)) for topic, words in parsed_topics.items()}
    
    # Plotting the topic diversity
    diversity_df = pd.DataFrame(list(diversity_by_topic.items()), columns=["Topic", "Diverse Themes Count"])
    diversity_df.plot(kind="bar", x="Topic", y="Diverse Themes Count", color='lightcoral', legend=False)
    plt.title("Diversity of Themes by Topic")
    plt.ylabel("Unique Theme Count")
    plt.tight_layout()
    plt.savefig('/Users/muhammadahmed/Desktop/DM/diverse_themes_by_topic.png')
    plt.show()

# Question 4: Most Positive or Uplifting Themes in Quotes
def positive_themes(parsed_topics, positive_words):
    all_words = []
    for words in parsed_topics.values():
        all_words.extend(words)
    
    positive_themes = [word for word in all_words if word in positive_words]
    positive_word_count = Counter(positive_themes)
    
    # Plotting the top positive themes
    positive_df = pd.DataFrame(positive_word_count.most_common(10), columns=["Theme", "Count"])
    positive_df.plot(kind="bar", x="Theme", y="Count", color='lightblue', legend=False)
    plt.title("Most Positive or Uplifting Themes in Quotes")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('/Users/muhammadahmed/Desktop/DM/positive_themes.png')
    plt.show()

# Question 5: Most Negative or Critical Themes in Quotes
def negative_themes(parsed_topics, negative_words):
    all_words = []
    for words in parsed_topics.values():
        all_words.extend(words)
    
    negative_themes = [word for word in all_words if word in negative_words]
    negative_word_count = Counter(negative_themes)
    
    # Plotting the top negative themes
    negative_df = pd.DataFrame(negative_word_count.most_common(10), columns=["Theme", "Count"])
    negative_df.plot(kind="bar", x="Theme", y="Count", color='lightpink', legend=False)
    plt.title("Most Negative or Critical Themes in Quotes")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('/Users/muhammadahmed/Desktop/DM/negative_themes.png')
    plt.show()

if __name__ == '__main__':
    # Load the topics
    topics = load_topics(topics_file_path)
    parsed_topics = parse_topics(topics)

    # Define some positive and negative words for filtering
    positive_words = ['love', 'hope', 'joy', 'happiness', 'inspire', 'peace', 'motivation', 'success', 'compassion']
    negative_words = ['hate', 'sadness', 'anger', 'fear', 'pain', 'failure', 'loss', 'criticism', 'struggle']
    
    # Common Themes Across Topics
    common_themes_across_topics(parsed_topics)
    
    # Most Common Themes or Genres in Quotes
    most_common_themes(parsed_topics)
    
    # Diversity of Themes by Topic
    diverse_themes_by_topic(parsed_topics)
    
    # Positive Themes in Quotes
    positive_themes(parsed_topics, positive_words)
    
    # Negative Themes in Quotes
    negative_themes(parsed_topics, negative_words)
