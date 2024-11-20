import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet  # Import wordnet for part-of-speech tagging

nltk.download('wordnet') # Download WordNet resources if you haven't already
nltk.download('averaged_perceptron_tagger') # Download POS tagger resources

lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "runner", "better", "easily", "fairly", "lying"]


def get_wordnet_pos(word):  # Function to get WordNet POS tag
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)  # Default to Noun if not found


for word in words:
    lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(word)) # Use POS tag
    print(f"{word} -> {lemmatized_word}")
