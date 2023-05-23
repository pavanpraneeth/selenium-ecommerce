from selenium import webdriver  # Import the Selenium WebDriver
from selenium.webdriver.chrome.service import Service  # Import the Service class for ChromeDriver
from selenium.webdriver.common.by import By  # Import By class for locating elements
from selenium.webdriver.chrome.options import Options  # Import Options class for configuring Chrome options
import json  # Import JSON module for working with JSON data
import time  # Import the time module for adding delays

from multi_rake import Rake  # Import the Rake keyword extraction library
rake = Rake(language_code='en', max_words=8)  # Initialize the Rake object for English language with max_words set to 8

import numpy as np  # Import the NumPy library for numerical operations
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity for calculating cosine similarity
import tensorflow as tf  # Import TensorFlow library
import tensorflow_hub as hub  # Import TensorFlow Hub for loading pre-trained models
import tensorflow_text as text  # Import TensorFlow Text for text processing

# Load BERT models
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")  # BERT preprocessing layer
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")  # BERT encoder layer

def get_sentence_embedding(sentences):
    # Preprocess the sentences using BERT preprocessing layer
    preprocessed_text = bert_preprocess(sentences)
    # Encode the preprocessed text using BERT encoder
    return bert_encoder(preprocessed_text)['pooled_output']


def extract_features_from_text(full_text):
    # Apply RAKE (Rapid Automatic Keyword Extraction) to extract keywords from the text
    keywords = rake.apply(full_text)
    # Limit the number of keywords to 10
    keywords = keywords[:10]
    # Return the concatenated keywords as a string
    return ' '.join(x[0] for x in keywords)


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
    print(len(search_results))
    if len(search_results) > 20:
        search_results = search_results[:20]

    for result in search_results:
        # Extract item small description
        item_small_desc = result.find_element(By.CSS_SELECTOR, "div.nameCls").text.strip()

        # Extract item link
        item_link = result.find_element(By.CLASS_NAME, "rilrtl-products-list__link").get_attribute("href")

        # Append the extracted data to the 'results' list in JSON format
        results.append({
            "link": item_link,
            "small description": item_small_desc
        })

    for i in range(len(results)):
        link = results[i]['link']
        driver.get(link)
        # Extract item brand
        item_brand = driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[2]/div/div/div[2]/div/div[3]/div/h2").text.strip()
        try:
            color = driver.find_element(By.CLASS_NAME, "prod-color").text.strip()
        except:
            color = ''
        try:
            description = driver.find_element(By.CLASS_NAME, 'prod-list').text.strip()
        except:
            description = ''

        # Add extracted brand, color, and description to the corresponding result
        results[i]['brand'] = item_brand
        results[i]['color'] = color
        results[i]['description'] = description

    # Quit the WebDriver
    driver.quit()

    # Return the results as a JSON response
    return results


def remove_key(x):
    # Remove unnecessary keys from the dictionary
    x.pop('brand')
    x.pop('color')
    x.pop('small description')
    return x


def clean_ajio_data(suggested_items):
    for i in range(len(suggested_items)):
        pre_desc = f"brand:{suggested_items[i]['brand']}\ncolor:{suggested_items[i]['description']}\n{suggested_items[i]['small description']}\n"
        desc = suggested_items[i]['description'].lower()

        # Remove specific lines containing 'package' or 'other information' from the description
        desc = '\n'.join([x for x in desc.split('\n') if not x.__contains__('package') or not x.__contains__('ther information')])
        desc = pre_desc + desc
        suggested_items[i]['description'] = desc

    # Remove unnecessary keys and transform into a list of lists
    suggested_items = [remove_key(x) for x in suggested_items]
    suggested_items = [[x['link'], x['description']] for x in suggested_items]
    return suggested_items


def calculate_similarity(input_string, string_list):
    links = [x[0] for x in string_list]
    descriptions = [x[1] for x in string_list]
    input_embedding = get_sentence_embedding([input_string])
    string_embeddings = get_sentence_embedding(descriptions)

    # Calculate cosine similarity between input embedding and string embeddings
    similarities = cosine_similarity(input_embedding, string_embeddings)[0]

    # Combine descriptions, links, and similarities into a list of tuples
    similarity_results = list(zip(descriptions, links, similarities))
    # Sort the results based on similarity in descending order
    similarity_results = sorted(similarity_results, key=lambda x: x[2], reverse=True)
    # Extract the links from the sorted results
    similarity_links = [x[1] for x in similarity_results]
    return similarity_links


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

    # Scrape Ajio clothing items based on the extracted features
    suggested_items = scrape_ajio_clothing_items(input_desc)

    # Clean the descriptions of suggested items
    cleaned_suggested_items = clean_ajio_data(suggested_items)

    # Extract features from descriptions of suggested items
    cleaned_suggested_items = [[x[0], extract_features_from_text(x[1])] for x in cleaned_suggested_items]

    # Calculate similarity between input and cleaned descriptions
    relevant_links = calculate_similarity(input_desc, cleaned_suggested_items)
    print('success')

    # Return the relevant links as a JSON response
    return {'Links': relevant_links[:int(number_of_links)]}
