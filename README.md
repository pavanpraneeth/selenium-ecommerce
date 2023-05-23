

# Ajio Clothing Item Scraper
A module to fetch relevant links of a product based on the input string. For example:" I am looking out for Nike sneakers"
This repository contains code for an Ajio clothing item scraper that retrieves relevant links to clothing items based on a given description. The code utilizes web scraping, natural language processing, and machine learning techniques to enhance the user experience by automating the search process.

## Requirements

The code requires the following dependencies:

- Python 3.x
- Selenium
- ChromeDriver
- multi_rake
- numpy
- scikit-learn
- TensorFlow
- TensorFlow Hub
- TensorFlow Text

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Make sure to have ChromeDriver installed and set up on your system. Adjust the `chromedriver_path` variable in the code to the appropriate path.

## Usage

To use the Ajio clothing item scraper, follow these steps:

1. Import the necessary libraries and modules:

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import json
import time

from multi_rake import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
```

2. Set up the RAKE algorithm and BERT models:

```python
rake = Rake(language_code='en', max_words=8)
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
```

3. Define a function to get the sentence embedding using BERT:

```python
def get_sentence_embedding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']
```

4. Implement a function to extract features from the input text using RAKE:

```python
def extract_features_from_text(full_text):
    keywords = rake.apply(full_text)
    keywords = keywords[:10]
    return ' '.join(x[0] for x in keywords)
```

5. Implement a function to scrape Ajio clothing items:

```python
def scrape_ajio_clothing_items(description):
    # Set path to your ChromeDriver executable
    chromedriver_path = "chromedriver.exe"
    options = Options()
    options.add_argument("--headless")
    # Create a new Selenium WebDriver with Chrome
    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)

    # Encode the description for URL query parameter
    encoded_description = description.replace(' ', '%20')

    # Load the Ajio search page
    driver.get(f"https://www.ajio.com/search/?text={encoded_description}")
    time.sleep(3)
    # Initialize an empty list to store the results
    results = []

    # Find all the search result items on the page
    search_results = driver.find_elements(By.CLASS_NAME, "item.rilrtl-products-list__item.item")
    if len(search_results) > 20:
        search_results = search_results[:20]

    for result in search_results:
        # Extract small description and item link
        item_small_desc = result.find_element(By.CSS_SELECTOR, "div.nameCls").text.strip()
        item_link = result.find_element(By.CLASS_NAME, "rilrtl-products-list__link").get_attribute("href")

        # Append the extracted data to the 'results' list in JSON format
        results.append({
            "link": item_link,
            "small description": item

_small_desc
        })

    for i in range(len(results)):
        link = results[i]['link']
        driver.get(link)
        item_brand = driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[2]/div/div/div[2]/div/div[3]/div/h2").text.strip()
        try:
            color = driver.find_element(By.CLASS_NAME, "prod-color").text.strip()
        except:
            color = ''
        try:
            description = driver.find_element(By.CLASS_NAME, 'prod-list').text.strip()
        except:
            description = ''

        results[i]['brand'] = item_brand
        results[i]['color'] = color
        results[i]['description'] = description

    # Quit the WebDriver
    driver.quit()

    # Return the results as a JSON response
    return results
```

6. Define a function to remove unnecessary keys from the scraped results:

```python
def remove_key(x):
    x.pop('brand')
    x.pop('color')
    x.pop('small description')
    return x
```

7. Implement a function to clean the Ajio data:

```python
def clean_ajio_data(suggested_items):
    for i in range(len(suggested_items)):
        pre_desc = f"brand:{suggested_items[i]['brand']}\ncolor:{suggested_items[i]['description']}\n{suggested_items[i]['small description']}\n"
        desc = suggested_items[i]['description'].lower()

        desc = '\n'.join([x for x in desc.split('\n') if not x.__contains__('package') or not x.__contains__('ther information')])
        desc = pre_desc + desc
        suggested_items[i]['description'] = desc
    suggested_items = [remove_key(x) for x in suggested_items]
    suggested_items = [[x['link'], x['description']] for x in suggested_items]
    return suggested_items
```

8. Define a function to calculate the similarity between the input description and the cleaned descriptions of the suggested items:

```python
def calculate_similarity(input_string, string_list):
    links = [x[0] for x in string_list]
    descriptions = [x[1] for x in string_list]
    input_embedding = get_sentence_embedding([input_string])
    string_embeddings = get_sentence_embedding(descriptions)

    similarities = cosine_similarity(input_embedding, string_embeddings)[0]

    similarity_results = list(zip(descriptions, links, similarities))
    similarity_results = sorted(similarity_results, key=lambda x: x[2], reverse=True)
    similarity_links = [x[1] for x in similarity_results]
    return similarity_links
```

9. Define a function to retrieve the relevant links:

```python
def get_links(request):
    request_json = request.get_json(force=True, silent=True, cache=True)
    if request.args and 'message' in request.args:
        input_desc = request.args.get('message')
        number_of_links = request.args.get('number')
    elif request_json and 'name' in request_json:
        input_desc = request_json['name']
        number_of_links = request_json['number']
    else:
        input_desc = 'black jeans'
        number_of_links = 10

    # Extract features from input string
    input_desc = extract_features_from_text(input_desc)

    suggested_items = scrape_ajio_clothing_items(input_desc)

    # Clean the descriptions
    cleaned_suggested_items = clean_ajio_data(suggested_items)

    # Feature extraction with RAKE for descriptions
    cleaned_suggested_items = [[x[0], extract_features_from_text(x[1])] for x in cleaned_suggested_items]

    # Get relevant links
   

 relevant_links = calculate_similarity(input_desc, cleaned_suggested_items)

    # Return the relevant links as a JSON response
    return {'Links': relevant_links[:int(number_of_links)]}
```


```

## Conclusion

The Ajio clothing item scraper provides a convenient way to automate the search for clothing items based on a given description. By leveraging web scraping, natural language processing, and machine learning techniques, the scraper retrieves relevant links to clothing items from the Ajio website. The code is modular and can be easily customized or integrated into other projects. With further enhancements and improvements, it has the potential to be a valuable tool for clothing shopping and recommendation systems.
