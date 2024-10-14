import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set the path to your CSV file
file_path = '/Users/muhammadahmed/Desktop/DM/quotes.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Check the data structure
print(df.info())

# Generate a word cloud for the quotes
quote_text = ' '.join(df['quote'].astype(str).tolist())
quote_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(quote_text)

# Plot and save the word cloud for quotes
plt.figure(figsize=(10, 5))
plt.imshow(quote_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Quotes')
plt.savefig('/Users/muhammadahmed/Desktop/DM/wordcloud_quotes.png')  # Save the word cloud for quotes in specified location
plt.show()

# Generate a word cloud for the tags
tags_text = ' '.join(df['category'].dropna().astype(str).tolist())
tags_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tags_text)

# Plot and save the word cloud for tags
plt.figure(figsize=(10, 5))
plt.imshow(tags_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Tags')
plt.savefig('/Users/muhammadahmed/Desktop/DM/wordcloud_tags.png')  # Save the word cloud for tags in specified location
plt.show()

# Create a mock 'likes' column if not present (remove this if you have the 'likes' column)
df['likes'] = pd.Series(range(1, len(df) + 1)).sample(frac=1).reset_index(drop=True)

# Most liked authors
most_liked_authors = df.groupby('author')['likes'].sum().nlargest(10)

plt.figure(figsize=(10, 5))
most_liked_authors.plot(kind='bar', color='skyblue')
plt.title('Top 10 Most Liked Authors')
plt.xlabel('Authors')
plt.ylabel('Total Likes')
plt.xticks(rotation=45)
plt.savefig('/Users/muhammadahmed/Desktop/DM/most_liked_authors.png')  # Save the bar chart for most liked authors in specified location
plt.show()

# Most liked categories
most_liked_categories = df.groupby('category')['likes'].sum().nlargest(10)

plt.figure(figsize=(10, 5))
most_liked_categories.plot(kind='bar', color='lightgreen')
plt.title('Top 10 Most Liked Categories')
plt.xlabel('Categories')
plt.ylabel('Total Likes')
plt.xticks(rotation=45)
plt.savefig('/Users/muhammadahmed/Desktop/DM/most_liked_categories.png')  # Save the bar chart for most liked categories in specified location
plt.show()

# Most liked quotes
most_liked_quotes = df.nlargest(10, 'likes')[['quote', 'likes']]

plt.figure(figsize=(10, 5))
plt.barh(most_liked_quotes['quote'], most_liked_quotes['likes'], color='salmon')
plt.title('Top 10 Most Liked Quotes')
plt.xlabel('Total Likes')
plt.ylabel('Quotes')
plt.savefig('/Users/muhammadahmed/Desktop/DM/most_liked_quotes.png')  # Save the bar chart for most liked quotes in specified location
plt.show()
 