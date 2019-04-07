import tmdbsimple as tmdb
import os
import uuid
import time
import PIL
from PIL import Image

api_token = '065b723226b49f774f424e649f05b501'

tmdb.API_KEY = api_token

poster_folder = 'posters'
metadata_folder = 'metadata'

# clear folders
def empty_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

empty_folder(poster_folder)
empty_folder(metadata_folder)

all_movies = tmdb.Movies()

movie_titles = []

# iterate movies list in pages [1, N)
# each "page" has 20 movies
for i in range(1,3):
    print("Pulling page " + str(i) + "...")
    paged_movies = all_movies.popular(page=i)['results']
    for movie in paged_movies:
        movie_titles.append(movie['title'])
    time.sleep(5)

print("Pulled " + str(len(movie_titles)) + " movie titles.")

for title in movie_titles:
    # assign a uuid
    uuid_no = uuid.uuid4()
    f = open(os.path.join(metadata_folder, str(uuid_no)) + ".txt", "w+")
    f.write("title\t" + title + "\n")
    # pull movie ID and poster path
    response = tmdb.Search().movie(query=title)
    id = response['results'][0]['id']
    movie = tmdb.Movies(id)
    poster = movie.info()['poster_path']
    poster_url = 'image.tmdb.org/t/p/original' + poster
    # save poster
    image_filename = str(uuid_no) + ".jpg"
    image_path = os.path.join(poster_folder, image_filename)
    strcmd = 'wget -O ' + image_path + ' ' + poster_url
    os.system(strcmd)
    # resize images
    img = Image.open(image_path)
    img = img.resize((300, 400), PIL.Image.ANTIALIAS)
    img.save(image_path)
    # save metadata
    info = movie.info()
    f.write("genre")
    for genre in info['genres']:
        f.write("\t" + genre['name'])
    f.write("\n")
    f.write("original_language\t" + info['original_language'] + "\n")
    f.write("release_date\t" + info['release_date'] + "\n")
    f.write("vote_average\t" + str(info['vote_average']) + "\n")
    f.write("vote_count\t" + str(info['vote_count']) + "\n")
    f.close()

