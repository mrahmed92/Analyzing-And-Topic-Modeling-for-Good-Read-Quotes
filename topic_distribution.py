import matplotlib.pyplot as plt

# Path to the topics file
topics_file = '/Users/muhammadahmed/Desktop/DM/topics.txt'

# Function to plot topic distribution
def plot_topic_distribution():
    topic_weights = {}
    
    with open(topics_file, 'r') as f:
        for line in f.readlines():
            if line.startswith("Topic"):
                # Extract the topic number and topic weights only
                topic_id = int(line.split()[1].replace(":", ""))
                weight_parts = line.split(':')[-1].strip().split('+')
                topic_weight = sum([float(w.split('*')[0]) for w in weight_parts if '*' in w])
                topic_weights[topic_id] = topic_weight

    # Plotting the distribution of topic weights
    plt.figure(figsize=(8, 5))
    plt.bar(topic_weights.keys(), topic_weights.values(), color='skyblue')
    plt.title("Topic Distribution")
    plt.xlabel("Topic")
    plt.ylabel("Weight (Summed)")
    plt.savefig('/Users/muhammadahmed/Desktop/DM/topic_distribution_plot.png')
    plt.show()

# Call the function to plot the topic distribution
if __name__ == '__main__':
    plot_topic_distribution()
