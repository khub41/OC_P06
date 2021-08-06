import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import udf

data = pd.read_csv('data/Flipkart/flipkart_com-ecommerce_sample_1050.csv')

# Normalizing capital letters:

data['description'] = data['description'].apply(lambda x : x.lower())
data['product_name'] = data['product_name'].apply(lambda x : x.lower())
# Tokenizing name and description distinctly

data['description_tok'] = data['description'].apply(nltk.word_tokenize)
data['name_tok'] = data['product_name'].apply(nltk.word_tokenize)

# Comparing tokenized data

data['simil_rate_texts'] = data.apply(udf.short_in_long_rate, axis=1)
print(data['simil_rate_texts'].describe(),
      '\nWe can use only the description as very few data from the name is not present in it')

# Exploring data


all_words = udf.get_all_words(data.description_tok)
# all_words.value_counts().head(10).plot(kind='pie', title='10 most common "words"')

print("Size of the repertory : ", len(all_words.unique()))

# text cleaning
# Exclude non alphanumerical text

data.description_tok = data.description.apply(udf.clean_non_alphanum)

all_words = udf.get_all_words(data.description_tok)
# all_words.value_counts().head(10).plot(kind='pie', title='10 most common "words"')
print("Size of the repertory after alphanumeric filter : ", len(all_words.unique()))

# Deleting most frequent words

# print(all_words.value_counts().head(40))
data.description_tok = data.description_tok.apply(udf.delete_words,
                                                  frequent_words=all_words.value_counts().head(40).index)

all_words = udf.get_all_words(data.description_tok)
print("Size of the repertory after most common words filter : ", len(all_words.unique()))

# Deleting stopwords
data.description_tok = data.description_tok.apply(udf.delete_words,
                                                  frequent_words=set(stopwords.words('english')))
all_words = udf.get_all_words(data.description_tok)
print("Size of the repertory after stop words filter : ", len(all_words.unique()))

# Stemming
stemmer_eng = SnowballStemmer("english")

data.description_tok = data.description_tok.apply(udf.stem_string,
                                                  stemmer=stemmer_eng)
all_words = udf.get_all_words(data.description_tok)
print("Size of the repertory after stemming : ", len(all_words.unique()))

