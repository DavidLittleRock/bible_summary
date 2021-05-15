import re
from collections import defaultdict

from pandas_ods_reader import read_ods
import pandas as pd

pd.set_option("display.expand_frame_repr", False)


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
                  columns=['Verse', 'King James Bible'])
    df = df.rename(columns={'Verse': 'verse', 'King James Bible': 'verse_text'})
    df['book_title'] = ''
    df['chapter_num'] = ''
    df['c_and_v'] = ''
    df['verse_num'] = ''
    df['b_and_c'] = ''
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
    df_chapter['chapter_text'] = df.groupby(['book_title', 'chapter_num'])[
        'verse_text'].transform(lambda x: ' '.join(x)).drop_duplicates()
    df_chapter['book_title'] = df['book_title']
    df_chapter['chapter_num'] = df['chapter_num']
    df_chapter['b_and_c'] = df['b_and_c']

    for i, row in df_chapter.iterrows():
        row['c_and_t'] = (row['b_and_c'], row['chapter_text'])

    df_book = df_chapter.copy(deep=True).drop(columns=['chapter_text'])
    df_book = df_book.reset_index(drop=True)

 #   books = {}
    return df_book

def make_bible():
    """combine books into dict"""
    bible = defaultdict(dict)

    df_book = make_books()
    books = list_books_bible()
    for book in books:
        for i, row in df_book.iterrows():
         #  bible[row['b_and_c']] = row['dict']
            bible[row['book_title']][row['chapter_num']] = row['c_and_t']

    bible = dict(
        bible)  # change bible from <class 'collections.defaultdict'> to <dict>
   # print(bible)
    return bible


def main():
    bible = make_bible()
    print(bible['Genesis']['47'])
    print(type(bible))
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

