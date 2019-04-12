
import os
import pandas
import numpy as np
from pandas.core.series import Series


def parse_metadata(movie_id):
    fname = 'metadata/{}.txt'.format(movie_id)
    with open(fname, 'rb') as mfile:
        lines = mfile.readlines()   
    if len(lines) < 2: return pandas.DataFrame()

    genres = lines[1].split('\t')
    if len(genres) < 2: return pandas.DataFrame()

    genres = [s.strip() for s in genres][1:]
    data = pandas.DataFrame({'movie': np.repeat(movie_id, len(genres)), 'genre': genres})
    return data


def main():

    movies_df = pandas.DataFrame()
    metadata_dir = 'metadata/'
    for f in os.listdir(metadata_dir):
        if f.endswith(".txt"):
            movie_id, ext = tuple(f.split('.'))
            df = parse_metadata(movie_id)
            movies_df = movies_df.append(df)

    movies_df['count'] = 1
    movies_df = movies_df.pivot(index='movie', columns='genre', values='count').fillna('0')
    movies_df = movies_df.astype(int)
    movies_df.to_csv('output/movie_genres.csv')


if __name__ == '__main__':
    main()
