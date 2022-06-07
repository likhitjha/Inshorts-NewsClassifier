from selenium import webdriver
# Importing 
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

### Out of the several categories in Inhorts, we select these 5 categories,
### We will be scraping the news card content from the inshorts website

#Science
#Sports
#Politics
#Entertainment
#Business

## Use the scraping code to scrape and store data from each category one by one

### SCRAPING DATA FROM INSHORTS
#Set cat, it is a string that takes in category of news 

cat='business'

df1 = pd.DataFrame(columns = ['Title', 'Content'])
def print_headlines(response_text):
    import time
    df = pd.DataFrame(columns = ['Title', 'Content'])
    soup = BeautifulSoup(response_text, 'lxml')
    newsCards = soup.find_all(class_='news-card')
    newsDictionary = {
        'success': True,
        'data': []
    }
    i=0
    for card in newsCards:
        try:
            title = card.find(class_='news-card-title').find('a').text
        except AttributeError:
            title = None

        try:
            imageUrl = card.find(
                class_='news-card-image')['style'].split("'")[1]
        except AttributeError:
            imageUrl = None

        try:
            url = ('https://www.inshorts.com' + card.find(class_='news-card-title')
                   .find('a').get('href'))
        except AttributeError:
            url = None

        try:
            content = card.find(class_='news-card-content').find('div').text
        except AttributeError:
            content = None

        try:
            author = card.find(class_='author').text
        except AttributeError:
            author = None

        try:
            date = card.find(clas='date').text
        except AttributeError:
            date = None

        try:
            time = card.find(class_='time').text
        except AttributeError:
            time = None

        try:
            readMoreUrl = card.find(class_='read-more').find('a').get('href')
        except AttributeError:
            readMoreUrl = None

        newsObject = {
            'title': title,
            'content': content,
        }
        print(newsObject['title'])
        df = df.append({'Title' : newsObject['title'], 'Content' : newsObject['content'] , 'Category' : cat}, 
                ignore_index = True)
        newsDictionary['data'].append(newsObject)
    #print(newsDictionary['data'])
    print(df)
    return df


def get_headers(): #specified in the network section
    return {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-IN,en-US;q=0.9,en;q=0.8",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "cookie": "_ga=GA1.2.474379061.1548476083; _gid=GA1.2.251903072.1548476083; __gads=ID=17fd29a6d34048fc:T=1548476085:S=ALNI_MaRiLYBFlMfKNMAtiW0J3b_o0XGxw",
        "origin": "https://inshorts.com",
        "referer": "https://www.inshorts.com/en/read/"+cat,
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36",
        "x-requested-with": "XMLHttpRequest"
    }


url = 'https://inshorts.com/en/read/'+cat
response = requests.get(url)
print_headlines(response.text)

# get more news
url = 'https://inshorts.com/en/ajax/more_news'
news_offset = "06ir0ujk-1"     #change the offset acording to the offset specified for that category

while True:
    response = requests.post(url, data={"category": cat, "news_offset": news_offset}, headers=get_headers())
    if response.status_code != 200:
        print(response.status_code)
        break

    response_json = json.loads(response.text)
    df1=df1.append(print_headlines(response_json["html"]))
    news_offset = response_json["min_news_id"]

### Use the above code to store data in csv format and use it