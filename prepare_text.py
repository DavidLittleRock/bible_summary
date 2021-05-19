import re
from pandas_ods_reader import read_ods
import matplotlib
import pandas as pd
from collections import defaultdict
import operator
from pprint import pprint

# Set Pandas to display all rows of dataframes
pd.set_option('display.max_rows', 500)

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet
# spacy for lemmatization
import spacy
import spacy.lang.en


# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt
import matplotlib
# %matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.corpus import stopwords


pd.set_option("display.expand_frame_repr", False)






def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def list_books_bible():
    """make a [list] of names of books of bible"""
    file_path = "files/bibles.ods"
    sheet_index = 1
    df = read_ods(file_path, sheet_index, columns=['Verse'])
    df = df.rename(columns={'Verse': 'book_chapter_verse'})
    df['book_name'] = ''
    for i, row in df.iterrows():
        split_row = row['book_chapter_verse'].split(' ')
        row['book_name'] = ' '.join(split_row[:-1])
    df = df.drop(columns=['book_chapter_verse'])
    df = df.drop_duplicates().reset_index(drop=True)
    # print(df.info())
    book_names = df['book_name'].tolist()
    # print(book_names)
    # print(type(book_names))

    return book_names

def make_books():
    """make books"""
    file_path = "files/bibles.ods"
    sheet_index = 1
    df = read_ods(file_path, sheet_index,
                  columns=['Verse', 'American Standard Version'])
    df = df.rename(columns={'Verse': 'verse', 'American Standard Version': 'verse_text'})
    df['book_title'] = ''
    df['chapter_num'] = ''
    df['c_and_v'] = ''
    df['verse_num'] = ''
    df['b_and_c'] = ''
    df['chapter_text'] = ''
    for i, row in df.iterrows():
        # separate out chapter, book and verse
        split_row = row['verse'].split(' ')
        # print(split_row)
        row['book_title'] = ' '.join(split_row[:-1])
        c_and_v = split_row[-1]
        row['chapter_num'] = c_and_v.split(':')[0]
        row['verse_num'] = c_and_v.split(':')[1]
        row['c_and_v'] = c_and_v
        row['b_and_c'] = row['book_title'] + ' ' + row['chapter_num']
        row['verse_text'] = re.sub('<.*?>', '', row['verse_text'])
        row['verse_text'] = re.sub('{.*?}', '', row['verse_text'])

    df_chapter = pd.DataFrame(columns=['book_title', 'b_and_c', 'chapter_num', 'chapter_text', 'c_and_t'])
    # concatenate the string to chapter
    # need to assign to new dataframe
    df_chapter['chapter_text'] = df.groupby(['book_title', 'chapter_num'])[
        'verse_text'].transform(lambda x: ' '.join(x)).drop_duplicates()
    df_chapter['book_title'] = df['book_title']
    df_chapter['chapter_num'] = df['chapter_num']
    df_chapter['b_and_c'] = df['b_and_c']

    for i, row in df_chapter.iterrows():
        row['c_and_t'] = (row['b_and_c'], row['chapter_text'])

    df_chapter = df_chapter.drop(columns=['chapter_text', 'b_and_c'])
    df_chapter = df_chapter.reset_index(drop=True)
  #  print(df_chapter.head())
 #   print(df.info())
    del df
    print(df_chapter.info())
    return df_chapter

def make_bible():
    """combine books into dict"""
    bible = defaultdict(dict)
    df_book = make_books()
    books = list_books_bible()
    for book in books:
        for i, row in df_book.iterrows():
            bible[row['book_title']][row['chapter_num']] = row['c_and_t']

    bible = dict(
        bible)  # change bible from <class 'collections.defaultdict'> to <dict>
   # print(bible)
    return bible


def main():
    bible = make_bible()
  #  print(bible['Genesis']['47'])
  #  print(type(bible))
    for book in bible:
        pass
      #  print("book")
      #  print(book)  # 'key' for book / book name
       # print(type(book))
      #  print(bible[book])  # full book value
        for chapter_title in bible[book]:
            pass
         #   print('chapter_title')
         #   print(chapter_title)  # key for chapter / chapter name
         #   print(type(chapter_title))
         #   print(bible[book][chapter_title])  # full chapter value
         #   print(f"chapter # {bible[book][chapter_title][0]}")
         #   print(f"chapter text: {bible[book][chapter_title][1]}")

           # for chapter in bible[book][chapter_title]:
            #    pass
              #  print('title')
             #   print(chapter)  # key for chapter title
              #  print(bible[book][chapter_title][chapter][0])  # chapter number
               # print(bible[book][chapter_title][chapter][1])  # chapter text

    # Word Count
    for book in bible:
        pass
     #   print('{:,} words in {}'.format(
        #    sum(len(bible[book][chapter][1].split()) for chapter in bible[book]),
       #     book))
    print()
  #  print('{:,} total words in collection'.format(
     #   sum(len(bible[book][chapter][1].split())
      #      for book in bible
      #      for chapter in bible[book])))

    # Average word length
    for book in bible:
        text = ''
        for chapter in bible[book]:
            text = text + bible[book][chapter][1]
    #    print('{:.2f} Average word length in {}'.format(
     #       len(text) / len(text.split()), book))

    # Chapters in books
    for book in bible:
        chapters = 0
        for chapter in bible[book]:
            chapters += 1
      #  print('{} chapters in {}'.format(chapters, book))


    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'ye', 'hath', 'mose', 'yea', 'thou', 'thy', 'thee', 'selah', 'hast'])
    # Convert to list
    data = [bible[book][chapter][1].replace('\n', '') for book in bible for chapter in bible[book]]
    data_words = list(sent_to_words(data))
 #   print(data_words[:1])

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5,
                                   threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
#    print(trigram_mod[bigram_mod[data_words[0]]])

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if
                 word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in
                              doc])  # if token.pos_ in allowed_postags])
        return texts_out


    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams,
                                    allowed_postags=['NOUN', 'ADJ', 'VERB',
                                                     'ADV'])
    print('lem')
 #   print(data_lemmatized[:1])
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # View
    print(corpus[:1])

    # Human readable format of corpus (term-frequency)
    print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=20,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    # Print the Keyword in the 20 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    # Compute Perplexity
    print('Perplexity: ', lda_model.log_perplexity(
        corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model,
                                         texts=data_lemmatized,
                                         dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)

    # Visualize the topics
 #   pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
  #  pyLDAvis.display(vis)
  #  pyLDAvis.show(vis, ip='127.0.0.1', port=8888, n_retries=50, local=True)
    pyLDAvis.save_html(vis, 'files/view.html')

    mallet_path = './files/mallet-2.0.8/bin/mallet'  # update this path
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
    # Show Topics
    pprint(ldamallet.show_topics(num_topics=1000, formatted=False))

    # Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(model=ldamallet,
                                               texts=data_lemmatized,
                                               dictionary=id2word,
                                               coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('\nCoherence Score: ', coherence_ldamallet)

    def compute_coherence_values(dictionary, corpus, texts, limit, start=2,
                                 step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            print('Calculating {}-topic model'.format(num_topics))
            model = gensim.models.wrappers.LdaMallet(mallet_path,
                                                     corpus=corpus,
                                                     num_topics=num_topics,
                                                     id2word=id2word)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts,
                                            dictionary=dictionary,
                                            coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    # Can take a long time to run.
    limit = 35;
    start = 2;
    step = 1;
    model_list, coherence_values = compute_coherence_values(dictionary=id2word,
                                                            corpus=corpus,
                                                            texts=data_lemmatized,
                                                            start=start,
                                                            limit=limit,
                                                            step=step)

    # Show graph
    x = range(start, limit, step)
    plt.figure(figsize=(15, 10))
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    # plt.legend(("coherence_values"), loc='best')
    plt.show()

    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 6))

    # Select the model and print the topics
    index, value = max(enumerate(coherence_values), key=operator.itemgetter(1))
    index = 10
    optimal_model = model_list[index]
    model_topics = optimal_model.show_topics(num_topics=1000, formatted=False)
    pprint(optimal_model.print_topics(num_words=10))

    optimal_model.show_topic(0, 10)

    for topic in sorted(
            optimal_model.show_topics(num_topics=1000, num_words=10,
                                      formatted=False), key=lambda x: x[0]):
        print('Topic {}: {}'.format(topic[0], [item[0] for item in topic[1]]))

    def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series(
                        [int(topic_num), round(prop_topic, 4),
                         topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution',
                                  'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model,
                                                      corpus=corpus,
                                                      texts=data)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic',
                                 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    print(df_dominant_topic.head())
  #  df_dominant_topic
    df_dominant_topic[df_dominant_topic['Dominant_Topic'].isin([0, 1])]

    [text.split() for text in df_dominant_topic['Keywords'].tolist()]

    for idx, row in df_dominant_topic.iterrows():
        print('{}. Dominant keywords: {}'.format(row['Document_No'],
                                                 row['Keywords'].split(', ')[
                                                 :5]))

    # Group top 5 sentences under each topic
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(
                                                     ['Perc_Contribution'],
                                                     ascending=[0]).head(1)],
                                                axis=0)

    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib",
                                           "Keywords", "Text"]

    # Show
    print(sent_topics_sorteddf_mallet.head())

    for idx, row in sent_topics_sorteddf_mallet.iterrows():
        print('Topic number {}'.format(int(row['Topic_Num'])))
        print('Keywords: {}'.format(row['Keywords']))
        print()
        print(row['Text'])
        print()

    # Number of Documents for Each Topic
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts / topic_counts.sum(), 4)

    # Topic Number and Keywords
    topic_num_keywords = sent_topics_sorteddf_mallet[['Topic_Num', 'Keywords']]

    # Concatenate Column wise
    df_dominant_topics = pd.concat(
        [topic_num_keywords, topic_counts, topic_contribution], axis=1)

    # Change Column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords',
                                  'Num_Documents', 'Percent_Documents']

    # Show
    print(df_dominant_topics)

    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):
      #  pd.set_option('display.max_colwidth', -1)
        pd.set_option('display.max_colwidth', None)

    #    display(df_dominant_topics)
        print('with')
        print(df_dominant_topics)






# print(bible[0])





if __name__ == '__main__':
    main()


# drop duplicate data
# df = df.drop_duplicates()

# show the dataframe
# print(df.head())
# print(df.tail())

# book_df = df[['book', 'kj_book']].drop_duplicates()
# print(book_df)

# chapter_df = df[['book', 'chapter', 'chapter_text']].drop_duplicates()
# print(chapter_df.head())
# print(chapter_df.info())
# print(df.info())

