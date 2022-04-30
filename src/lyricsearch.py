"""
Improved Media Content Search (IMCS)
@joshuamarksmith and @braydenturner
UIUC MCS, 2021, Text Information Systems

"""

# First, we install the lyricsgenius API, as well as multiprocess to increase speed of data scraping:

# !pip install multiprocess
# !pip install lyricsgenius
# !pip install tensorflow
# !pip install tensorflow_hub
# !pip install annoy
# !pip install pickle5

import json
import csv
import multiprocess
import queue
import logging
import sys

"""
Data download
"""
from requests.exceptions import HTTPError, ConnectionError, RequestException
from lyricsgenius import Genius

# OS agnostic
import os 
CSV_PATH = os.path.join(os.path.curdir, 'artists', '10000-MTV-Music-Artists-page-%s.csv')

def genius_setup():
    """Set up the Genius API
    """
    token = "EBufquOcw_ts4Y4V7yiddUNyUakTdqCpnMZhiI3XtAScWOntEom8Hj4T87gAV_cA"
    genius = Genius(token, retries=2)

    genius.verbose = False
    genius.remove_section_headers = True
    genius.skip_non_songs = True
    genius.excluded_terms = ["(Remix)", "(Live)"]

    return genius    

"""Global variables used in scraping
"""
# Multiprocessing cores
process_number = int(multiprocess.cpu_count()) * 2

# Data management
final_ = multiprocess.Manager().list()
checked_artists = set()

# Output file
file_name = "song_data_2.csv"

def get_artists(queue):
    """Fetch artists
    """
    for x in range(1,5):
        path = CSV_PATH % str(x)
        with open(path, encoding="UTF-8") as csvfile:
            TopArtists = csv.reader(csvfile)
            
            # Skip header
            next(TopArtists)
            for row in TopArtists:
                artist = row[0]
                # Check if we should skip this artists since we already found the data
                if artist not in checked_artists:
                    queue.put(artist)


# File management
def write_to_csv(data):
    """Write to the CSV file
    
    data: list of dictionaries {artist, song, data}
    """
    global file_name
    
    csv_path = os.path.join(os.path.curdir, 'data', file_name)
    with open(csv_path, 'w') as csv_file: 
        # creating a csv dict writer object 
        print("Entries: {num}".format(num=len(data)))
        keys = data[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames = keys) 
        
        # writing headers (field names) 
        writer.writeheader() 
        
        # writing data rows 
        writer.writerows(data) 
        

def read_csv():
    """Read the CSV file
    """
    global final_, checked_artists, file_name   
    
    csv_path = os.path.join(os.path.curdir, 'data', file_name)
    
    # opening the CSV file
    try:
        with open(csv_path, mode ='r', encoding="UTF-8") as file:   

            # reading the CSV file
            data = csv.DictReader(file)

            for entry in data:
                checked_artists.add(entry["artist"])
                final_.append(entry)
                
        print("Number of artists already found {num}".format(num=len(checked_artists)))
    except FileNotFoundError:
        pass
    

# Run genius search
def search_genius(args):
    """Run our search using the Genius API
    """
    import sys
    from requests.exceptions import RequestException
    artist_queue, num, genius, final_ = args
    
    def log(string):
        print("[{num}] ".format(num=num) + string + "\n", end='')
        sys.stdout.flush()
    
    # Processing
    def clean_data(data):
        cleaned_data = data.replace("\n", "|").replace(",", " ")
        return cleaned_data

    def process_artist(artist):
        artist_dict = artist.to_dict()
        return ""

    def process_song(song):
        lyrics = clean_data(song.lyrics)
        return lyrics

    def build_entry(artist, song, data, columns = ["artist", "song", "data"]):
        entry = {"artist": artist, "song": song, "data": data}
        return entry
    
    log("Starting")
    try:
        while True:
            genius_artist = None
            artist = artist_queue.get()
            if artist is None:
                log("Done")
                return
            log("Remaining: [{queue}]. Searching {artist}".format(queue=artist_queue.qsize(), artist=artist.strip()))
            
            # Pull data for artist from genius
            for x in range(5):
                try:
                    genius_artist = genius.search_artist(artist, per_page=50, get_full_info=False)
                    break
                except RequestException as e:
                    log("HTTPSConnectionPool exception. Attempt {}/5".format(x+1))
                except Exception as e:
                    log("Exception. Attempt {}/3".format(x+1))
            
            log("Finished {artist}".format(num=num, artist=artist.strip()))
            if genius_artist == None:
                log("{artist} not found".format(num=num, artist=artist.strip()))
                continue
                           
            artist_data =  process_artist(genius_artist)
                           
            log("{artist} number of songs: {song_num}".format(num=num, artist=artist.strip(), song_num=len(genius_artist.songs)))
            
            for song in genius_artist.songs:
                song_data = process_song(song)
                
                # Add to final list
                final_.append(build_entry(artist, song.title, song_data))
    
    except Exception as e:
        log("Something went wrong: {error}".format(num=num, error= e))
    
    
def run(multi_core=False):
    """Driver function
    """
    # Setup Genius
    genius = genius_setup()
    
    # Load in any previous data
    print("Reading previous")
    read_csv()
    
    pool = None
    try:  
        if multi_core:
            # multiprocess.log_to_stderr().setLevel(logging.DEBUG)
            print("Multiprocessing with {process_number} processes".format(process_number=process_number))
            
            artist_queue = multiprocess.Manager().Queue()
            get_artists(artist_queue)
            
            for x in range(process_number):
                artist_queue.put(None)
            
            print(artist_queue.qsize())
            # creating processes
            with multiprocess.get_context("spawn").Pool(process_number) as pool:
                args = [(artist_queue, x, genius, final_) for x in range(process_number)]
                pool.map(search_genius, args)
                pool.close()
                pool.join()
            
        else:
            print("Running single core")
            artist_queue = queue.Queue()
            get_artists(artist_queue)
            artist_queue.put(None)
            print(artist_queue.qsize())
            search_genius((artist_queue, 0, genius, final_))

    
    except KeyboardInterrupt:
        if pool:
            pool.close()
            pool.terminate()
            pool.join()
        print("KeyboardInterrupt: Writing results")
    
    finally:
        write_to_csv(list(final_))                       

# Run download - will take hours!
# run(multi_core=True) 

"""
Modeling
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
# tf.get_logger().setLevel(logging.ERROR)

import tensorflow_hub as hub
from annoy import AnnoyIndex
import pickle5 as pickle
import os
import csv
import numpy as np
import random
from sklearn import random_projection

# Define input path
input_file="song_data_2.csv"
input_csv_path = os.path.join(os.path.curdir, 'data', input_file)
csv.field_size_limit(100000000)


def batch_load():
    """Load a batch of 1000 songs - not used unless batching input
    """
    
    batch_sentences = []
    global counter
    
    print("loading batch of songs, starting at song", counter)
    
    for x in range(1000):
        
        document = next(documents)

        data = document['data']
        lines = data.split("|")

        song = document['song']
        artist = document['artist']
        for line in lines:
            ids[counter] = (line, song, artist)
            batch_sentences.append(line)
            counter += 1
        
    return batch_sentences


def get_lines():
    """Get individual lines from the input CSV, to use as the input for embeddings
    """
    songs = []
    
    with open(input_csv_path, mode ='r+', encoding = 'utf-8') as file:   
            datareader = csv.DictReader(file)
            next(datareader)
            for row in datareader:
                data = row['data']
                song = row['song']
                artist = row['artist']
    
                songs.append([data, song, artist])
                    
                # if len(lines) % 100000 == 0:
                #     print("{} lines added".format(len(lines)))
    
    random.shuffle(songs)
    lines = []
    for x in songs:
        data, song, artist = x
        lyrics = data.split("|")

        for lyric in lyrics:
            lines.append([lyric, song, artist])
    
    del songs
    print("total lines added: {}".format(len(lines)))
    
    return lines

# Retrieves the embedding for a batch of sentences 

vector_length = 512
embed_module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
placeholder = tf.placeholder(dtype=tf.string)
embed = embed_module(placeholder)
session = tf.Session()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])
transformer = random_projection.GaussianRandomProjection(vector_length)

def get_embeddings(sentences):
    """Gets embeddings for a given line (document)
    """
    
    print("Getting embeddings...")
    embeddings = session.run(embed, feed_dict={placeholder: sentences})
    
    ## Reduction of dimensionality does not work - must use 512 vector length
    # print("Reducing dimensionality...")
    # reduced_embeddings = transformer.fit_transform(embeddings)
    
    return embeddings


# Set up mapping and counter to keep track of documents
counter = 0
mapping = {}
index_file_name = "annoy_index_512_20_trees"

# Initialize the ANNOY index 
ann = AnnoyIndex(vector_length, metric='angular')


def add_items_to_index(batch, embeddings):
    """Adds items to an ANNOY index
    
    sentences: a list of 
    embeddings: a list of tensorflow embeddings for sentences
    """ 
    global ann, counter, mapping
    
    for line, embed in zip(batch, embeddings):
        ann.add_item(counter, embed)
        mapping[counter] = line
        counter +=1  
        

def build_ann_index(batch_size=200000):
    """Constructs the ANNOY index
    """
    print("getting lines from CSV file...")
    lines = get_lines()
    print("lines retrieved, getting embeddings...")
    
    # num_lines = 100000 * 10
    lines = lines[:8500000]
    

    ann.on_disk_build(index_file_name)

    # get the embeddings in batches - 1/50th of data set to test
    # for x in range(0, num_lines, batch_size):
    while len(lines) > 0:
        
        print("Operating on lines {} - {}".format(counter, counter + batch_size))
        start = 0
        if batch_size >= len(lines):
            end = len(lines)
        else:
            end = batch_size
        
        batch = lines[start:end]
        
        lyrics = [x[0] for x in batch]
        embeddings = get_embeddings(lyrics)
        add_items_to_index(batch, embeddings)
               
        del embeddings, lyrics
        del lines[:end]
        
        print("{} left".format(len(lines)))


try:
    build_ann_index()
except KeyboardInterrupt:
    print("KeyboardInterrupt")
finally:
    print("Building index...")
    ann.build(10)
    ann.unload()
    
    with open(index_file_name + '.mapping', 'wb') as handle:
        pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('mapping saved')


ann = AnnoyIndex(vector_length, metric='angular')
ann.load(index_file_name, prefault=True)
print('annoy index loaded.')
with open(index_file_name + '.mapping', 'rb') as handle:
    mapping = pickle.load(handle)
print('mapping file loaded.')

input_sentence = "Getting an email on my iPhone"

print("Getting query embeddings...")
query_embeddings = get_embeddings([input_sentence])[0]

# Return 10 nearest neighbors
print("Getting nearest neighbors...")
nns = ann.get_nns_by_vector(query_embeddings, 10, include_distances=False)

print("Closest: ")
for idx, item in enumerate(nns):
    print("{}. {} - {}:".format(idx+1, mapping[item][1], mapping[item][2]))
    for x in range(item-3, item+3):
        if x == item:
            print("==== {} ====".format(mapping[x][0]))
        else:
            print("     {}     ".format(mapping[x][0]))
    
    print("\n")

