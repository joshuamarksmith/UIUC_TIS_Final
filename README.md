# CS410 Final Project: Improved Media Content Search
University of Illinois Urbana-Champaign, Fall 2021  
Professor ChengXiang Zhai  
Team "Bae Area": Brayden Turner and Joshua Smith

## Installation
Improved Media (Music) Content Search (IMCS) runs via an interactive Jupyter Notebook supported by Python 3.7+. Download the LyricSearch.ipynb file and run the cells to import the necessary libraries, scrape lyric data from Genius, and load the model.

To run queries without building a model, we have a prebuilt mapping and index file (annoy_index_512_20_trees and annoy_index_512_20_trees.mapping respectively) here https://uofi.app.box.com/folder/151778821345?s=9wpb3x3x6twgbqq400c9eb1b001ju2m8. Once you have the files downloaded (they are quite large), add them to the same folder as the jupyter notebooks in this project. The simplified notebook can be used which presents a completed [Annoy](https://github.com/spotify/annoy) index and mapping/ data model which can be used to run sample queries. This also uses interactive ipywidgets for single queries. This simplified application uses [AppMode](https://github.com/oschuett/appmode), an easy-to-install open source Jupyter extension to run a notebook as a web application.

## Running the Application

To run the model, make sure the annoy index and mapping file are in the same directory as InteractiveLyricSearch.ipynb, then run the notebook. Ensure that the AppMode extension is running in the notebook to execute it as a web app.

## Background
We’ve all tried to remember a song but all we have is a rough description of what the song is about or one line of a lyric we managed to catch. Many search tools for music and movies are limited to metadata such as titles, artist, actors, but not the general sentiment of lyrics and content. The goal is to return a ranked list of songs that fits the description or lyric given. This can also be extended to add tags to content like “Song from Super Bowl 40 Halftime” or “Rolling Stone top 10 movies list” to enhance descriptive search.

## Implementation
### Data Scraping
There was no central data set that contained all of the lyrics we wanted to use for our project. So what we found was a collection of the top 10,000 artists from MTV over the years contained in a csv file. We used that in conjuntion with a python module called `lyricsgenius` that allowed us to scrape Genius.com to get all of the songs and lyrics for each artist. Even with using mulitple cores on our computers it still took 12+ hours to scrape all of the artists. Once we had the lyrics for each song, we cleaned up the lyrics and made sure to seperate each line of the lyric so we could recover them individually later.

### Model
Once we had the csv of all the songs we scraped, we needed a way to represent them in an ambiguous way. We considered using `metapy`, but that would only give us individual tokens of words that we could only match 1 to 1. It wouldn't allow us to match things like "car" and "truck" or more of the sentiment of the lyric. Instead we turned to the universal-sentence-encoder belonging to `tensorflow` which represents each sentence as a 512 dimensional vector. Using this model, we were able to encode each line of the lyric as a vector which we could use later during our KNN search.

### Searching For Lyrics
To actually do the searching we figured a KNN with our vectors would be the best way to do it. Once we had all of the vectors for each line of each lyric for each song, we add them to a library called `Annoy`. This library is a KNN search built by Spotify. It allows for really fast retrieval on really large datasets. Each vector was placed in the AnnoyIndex with a key (e.g. 56) whcih corresponded to a mapping dictionary we kept of the original line and the song and artist of the line. Then for each query search, we would also turn that in to a vector and find some of the closest vectors.

## Evaluation
Simple evaluation of the results was done on a sample size of 10 queries, with 10 results per query using Y/N for relevance judgements. The evaluation record can be found in the repo under evaluations. They are provided in HTML (download to open) or as an Excel spreadsheet.

## Screenshots
Interactive search app showing results for a query:
![interactive search example](https://github.com/braydenturner/CourseProject/blob/main/screenshots/interactive1.JPG)
