import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

data.description_tok = data.description.apply(udf.tokenize_custom)

all_words = udf.get_all_words(data.description_tok)
# all_words.value_counts().head(10).plot(kind='pie', title='10 most common "words"')
print("Size of the repertory after alphanumeric filter : ", len(all_words.unique()))

# Deleting most frequent words

# We'll use tf and idf to penalize most frequent tokens. Deleting the X most frequent tokens is too arbitrary

all_words = udf.get_all_words(data.description_tok)
print("Size of the repertory after most common words filter : ", len(all_words.unique()))

# Deleting stopwords
data.description_tok = data.description_tok.apply(udf.delete_words,
                                                  frequent_words=set(stopwords.words('english')))
all_words = udf.get_all_words(data.description_tok)
print("Size of the repertory after stop words filter : ", len(all_words.unique()))

# TODO Attention aux stopwords par défaut

# Stemming

lemmatizer_eng = WordNetLemmatizer()

data.description_tok = data.description_tok.apply(udf.lem_string,
                                                  lemmatizer=lemmatizer_eng)
all_words = udf.get_all_words(data.description_tok)
print("Size of the repertory after lemmatization : ", len(all_words.unique()))

#TODO test stemm when connection is better

# Convert token sequences into bigram sequences

data['description_bigram'] = data.description_tok.apply(udf.get_bigrams)


#TODO evaluer les catégories

data['label_categ_0'] = data.product_category_tree.apply(udf.process_categs,
                                                         level=0)
# 7 categs with uniform distribution is yummmmy.
# When going one level deeper in the tree we get 62 categs with lot having one item


def custom_tokenize(text):
    tokens = udf.tokenize_custom(text)

    lemmatizer_eng = WordNetLemmatizer()
    tokens_stem = udf.lem_string(tokens, lemmatizer=lemmatizer_eng)
    return tokens_stem

# TODO faire un transfert learning? VGGNET/RESNET images. BERT : texte

sw = [
 'yourselves',
 "should've",


 'yours',
 'to',


 'those',
 'myself',
 's',
 "you're",
 'ourselves',

 've',

 "hadn't",
 'here',
 'of',






 'on',
 're',

 'y',
 'my',



 'that',

 'doing',


 'd',
 "she's",
 'theirs',
 'you',
 'having',

 'has',

 'been',
 "shouldn't",
 'through',
 'your',
 'all',
 'couldn',
 "didn't",
 'this',
 'i',
 'these',



 'needn',
 'why',
 'm',
 'was',


 'we',
 'just',
 'now',
 'how',
 'as',

 'more',
 "won't",
 'are',
 'does',
 'if',
 'his',
 'ain',
 'himself',
 'be',
 'shouldn',
 'they',
 'me',
 'aren',
 'which',
 'during',

 'such',
 "couldn't",
 "mustn't",
 "you'd",

 'then',
 'will',

 'he',
 "that'll",
 'and',
 'but',
 "needn't",
 'being',
 'from',
 'll',

 'hadn',
 'o',
 'themselves',
 'or',

 'the',



 'haven',

 'for',
 'what',
 'ma',
 'their',
 "doesn't",
 'shan',
 'our',

 "hasn't",
 'yourself',



 't',
 "you'll",
 'am',
 'were',
 'while',
 'who',
 "haven't",
 'mightn',
 'down',
 'she',
 "wasn't",
 'again',
 'itself',
 'isn',
 'mustn',
 "it's",
 "weren't",
 "don't",


 'further',

 'hasn',
 'nor',
 "you've",
 'it',
 'wasn',
 'them',
 'can',
 'have',

 'a',
 "mightn't",
 'until',

 'weren',
 'by',
 'whom',

 "wouldn't",
 'when',
 'is',
 'an',
 'don',
 'herself',

 'its']
tfidf = TfidfVectorizer(tokenizer=custom_tokenize, stop_words=sw, ngram_range=(1,2))
values = tfidf.fit_transform(data.description)

dense = values.todense()
feature_names = tfidf.get_feature_names()
denselist = dense.tolist()
data_raw = pd.DataFrame(denselist, columns=feature_names).reset_index()

data_full = data.merge(data_raw, how='left', right_on='index', left_index=True)

data_scale = udf.scale_data(data_raw)

udf.scree_plot(data_scale, 1050, savefig=False)

data_scale_decomp = udf.reduce_dim_pca(data_scale, 700)

data_tsne = udf.train_tsne(data_scale_decomp, data.label_categ_0)