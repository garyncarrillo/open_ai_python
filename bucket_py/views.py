import json
import math
from django.http import JsonResponse

import pandas as pd
from openai.embeddings_utils import (cosine_similarity, get_embedding, distances_from_embeddings, tsne_components_from_embeddings, chart_from_components, indices_of_nearest_neighbors_from_distances,)

import os
import openai
import tiktoken
import numpy as np
import pickle
import time

from rest_framework import viewsets
from rest_framework import permissions
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import action


# constants
EMBEDDING_MODEL = "text-embedding-ada-002"
embedding_cache = {}

def open_api(request):
    openai.api_key = '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'    
    engines = openai.Engine.list()
    print(engines.data[0].id)
    completion = openai.Completion.create(engine="ada", prompt="How are you?")
    print(completion.choices[0].text)
    
    return JsonResponse(completion)

def label_score(review_embedding, label_embeddings):
   return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])

def load(request):
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    max_tokens = 8000

    openai.api_key = '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'    
    # engines = openai.Engine.list()

    url = str(os.getcwd())
    files = os.listdir(url)
    file = os.path.dirname(__file__)

    df = pd.read_csv(url+"/reviews.csv")
    df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
    df = df.dropna()
    df["combined"] = (
        "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
    )
    print(df.head(2))

    top_n = 10
    df = df.sort_values("Time").tail(top_n * 2)  # first cut to first 2k entries, assuming less than half will be filtered out
    df.drop("Time", axis=1, inplace=True)

    encoding = tiktoken.get_encoding(embedding_encoding)

    # omit reviews that are too long to embed
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens].tail(top_n)
    len(df)

    print(len(df))
    print("++++++++++++++++")
    df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    print(df.head(10))

    df['sentiment'] = df.Score.replace({ 1:'negative', 2:'negative', 3:'positive', 4:'positive', 5:'positive' })
    labels = ['negative', 'positive']
    label_embeddings = [get_embedding(label, engine='text-embedding-ada-002') for label in labels]
    print("*********** OK")
    print(df.head(10))
    prediction = None
    score = label_score('delicious food', label_embeddings)
    print(score)
    print("****************************")

    if  score > 0:
        prediction = 'positive'
    else: 
        prediction = 'negative'
    return JsonResponse({"engines": "engines", "prediction": prediction})

def open_recomendation(request):
    openai.api_key = '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'

    url = str(os.getcwd())
    df = pd.read_csv(url+"/test.csv")
    print(df.head(5))
    article_descriptions = df["Description"].tolist()
    
    tony_blair_articles = print_recommendations_from_strings(
         strings=article_descriptions,  # let's base similarity off of the article description
         index_of_source_string=0,  # let's look at articles similar to the first one about Tony Blair
         k_nearest_neighbors=5,  # let's look at the 5 most similar articles
    )

    print(tony_blair_articles)
    return JsonResponse({"engines": "engines"})

def print_recommendations_from_strings(
    strings: list[str],
    index_of_source_string: int,
    k_nearest_neighbors: int = 1,
    model=EMBEDDING_MODEL,
    embeddings=list
) -> list[int]:
    """Print out the k nearest neighbors of a given string."""
    # get embeddings for all strings
    # embeddings = [embedding_from_string(string, model=model) for string in strings]
    # embeddings = [get_embedding(string, engine=model) for string in strings]
    # get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]
    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    print("++++++++++++++++++")
    print(indices_of_nearest_neighbors)
    print("++++++++++++++++++")
    # print out source string
    query_string = strings[index_of_source_string]
    print("************************")
    print(query_string)
    print("gggggggggggggggggggggggggggggggggggg")
    print(f"Source string: {query_string}")
    print("---------------------------")
    # print out its k nearest neighbors
    k_counter = 0
    neighbors = query_string
    for i in indices_of_nearest_neighbors:
        # skip any strings that are identical matches to the starting string
        if query_string == strings[i]:
            continue
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        # print out the similar strings and their distances
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {strings[i]}
        Distance: {distances[i]:0.3f}"""
        )
        space = " "
        if k_counter > 1:
           space = ", " 
        
        neighbors = neighbors +"\n"+ strings[i]

    question = neighbors
    return question

def number_of_rows(df, percent=100):
    rows_count = df[df.columns[0]].count()
    total_rows = rows_count * (percent/100)
    part_decimal, part_integer = math.modf(total_rows)
    return part_integer

def open_recomendation_bucket(request):
    openai.api_key = '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'
    # openai.api_key = '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'

    url = str(os.getcwd())
    
    df2 = pd.read_csv(url+"/entranamiento2.csv")
    total_rows = number_of_rows(df2, 100)
    
    # print(request.files["file"].filename)
    print("TOTALLLLLLLLLLLLLLLLLLL")
    print(total_rows)

    df = pd.read_csv(url+"/entranamiento2.csv", nrows=total_rows)
    print(df.head(5))
    article_descriptions = df["Description"].tolist()
    strings=article_descriptions
    model  = EMBEDDING_MODEL
    embeddings = [get_embedding(string, engine=model) for string in strings]
    print(type(embeddings))
    print("LLEGOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO ")
    question = print_recommendations_from_strings(
         strings=article_descriptions,  # let's base similarity off of the article description
         index_of_source_string=0,  # let's look at articles similar to the first one about Tony Blair
         k_nearest_neighbors=2,  # let's look at the 5 most similar articles
         embeddings=embeddings
    )
    # time.sleep(10)
    question_2 = print_recommendations_from_strings(
         strings=article_descriptions,  # let's base similarity off of the article description
         index_of_source_string=2,  # let's look at articles similar to the first one about Tony Blair
         k_nearest_neighbors=2,  # let's look at the 5 most similar articles
         embeddings=embeddings
    )
    # time.sleep(10)
    question_3 = print_recommendations_from_strings(
         strings=article_descriptions,  # let's base similarity off of the article description
         index_of_source_string=3,  # let's look at articles similar to the first one about Tony Blair
         k_nearest_neighbors=2,  # let's look at the 5 most similar articles
         embeddings=embeddings
    )

    print("******************* ", question)
    print("******************* ", question_2)
    print("******************* ", question_3)
    main_question = "When it comes to starting the new year strong and getting a jump on your 2019 goals, what's your single biggest challenge right now? \n"+question+question_2+question_3+" \n give me a conclusion about the last aswers"
    time.sleep(10)
    completion = openai.Completion.create(engine="text-davinci-003", prompt= main_question,  max_tokens=150, temperature=0)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(main_question)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(completion)
    print("/*/*/*/*/*/*/*/*/*/*/**/*/*/*/*/*/**/**/*/*/*/*/")
    print(completion.choices[0].text)

    print(question)
    return JsonResponse({"engines": "engines"})


class UserViewSet(viewsets.ModelViewSet):
    
    @action(methods=['POST'], detail=False, url_path='create-bucket', url_name='create_bucket')
    def create_bucket(self, request, pk=None):
        openai.api_key = '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'

        print(request.data["query"])
        print(request.data["file"].name)

        df2 = pd.read_csv(request.data["file"].name)
        total_rows = number_of_rows(df2, 100)    
        print(df2.head(5))

        print(total_rows)

        df = pd.read_csv(request.data["file"].name, nrows=total_rows)
        print(df.head(5))
        article_descriptions = df["Description"].tolist()
        strings=article_descriptions
        model  = EMBEDDING_MODEL
        embeddings = [get_embedding(string, engine=model) for string in strings]
        print(type(embeddings))
        print("LLEGOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO ")
        question = print_recommendations_from_strings(
            strings=article_descriptions,  # let's base similarity off of the article description
            index_of_source_string=0,  # let's look at articles similar to the first one about Tony Blair
            k_nearest_neighbors=2,  # let's look at the 5 most similar articles
            embeddings=embeddings
        )
        # time.sleep(10)
        question_2 = print_recommendations_from_strings(
            strings=article_descriptions,  # let's base similarity off of the article description
            index_of_source_string=2,  # let's look at articles similar to the first one about Tony Blair
            k_nearest_neighbors=2,  # let's look at the 5 most similar articles
            embeddings=embeddings
        )
        # time.sleep(10)
        question_3 = print_recommendations_from_strings(
            strings=article_descriptions,  # let's base similarity off of the article description
            index_of_source_string=3,  # let's look at articles similar to the first one about Tony Blair
            k_nearest_neighbors=2,  # let's look at the 5 most similar articles
            embeddings=embeddings
        )

        print("******************* ", question)
        print("******************* ", question_2)
        print("******************* ", question_3)
        main_question = "When it comes to starting the new year strong and getting a jump on your 2019 goals, what's your single biggest challenge right now? \n"+question+question_2+question_3+" \n give me a conclusion about the last aswers"
        time.sleep(10)
        completion = openai.Completion.create(engine="text-davinci-003", prompt= main_question,  max_tokens=150, temperature=0)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(main_question)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(completion)
        print("/*/*/*/*/*/*/*/*/*/*/**/*/*/*/*/*/**/**/*/*/*/*/")
        print(completion.choices[0].text)
        print(question)
        return JsonResponse(completion)