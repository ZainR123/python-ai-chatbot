import asyncio
import csv
import os
import re
import string
from datetime import datetime
from pyexpat import ExpatError
from random import randrange

import nltk
import numpy as np
import python_weather
from joblib import load, dump
from nltk import WordNetLemmatizer, word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split


# Preprocessing for user queries and the passed datasets
# Breaks up contractions and stems every word
def stem_analyser(doc):
    return (stemmer.stem(w) for w in analyser(decontract(doc)))


# Preprocessing for user queries and the passed datasets
# Breaks up contractions, stems every word and removes any stopwords
def stem_analyser_no_stop_words(doc):
    return (stemmer.stem(remove_stop_words(w)) for w in analyser(decontract(doc)))


# Preprocessing for user input when they give their name
# Lemmatises text breaking down words to their core
def lemmatise(text):
    # To lemmatise correctly we must declare every word type distinctly
    # Otherwise the lemmatiser thinks of every word as a noun
    posmap = {
        'ADJ': wordnet.ADJ,
        'ADV': wordnet.ADV,
        'NOUN': wordnet.NOUN,
        'VERB': wordnet.VERB
    }

    # Tags text with corresponding word type
    post = nltk.pos_tag(text, tagset='universal')

    lemmatised_list = []

    # Loop through each word which is assigned a word type, if the word type matches one in our posmap then pass it in the lemmatiser
    # Else just lemmatise the word without the additional tag
    # Add lemmatised words to list
    for token in post:
        word = token[0]
        tag = token[1]
        if tag in posmap.keys():
            lemmatised_list.append(lemmatiser.lemmatize(word, posmap[tag]))
        else:
            lemmatised_list.append(lemmatiser.lemmatize(word))

    # Return a cleaned list without empty spaces
    return list(filter(None, lemmatised_list))


# Runs the user's inputted name through a list of possible names if it matches one then return that name
# Else return a processed version of the input given which hopefully should only contain the name given
def name_processing(text):
    # Check if any words in the user input are on our keywords list, if so remove them
    text = [w for w in text if w not in user_name_key_phrases]
    # Clean list removing empty spaces
    cleaned = list(filter(None, text))

    # Loop through word list if any of the words match a name in our name dataset, return it
    for s in cleaned:

        if s in name_list:
            return s.title()

    # Else return the cleaned list as a string
    return ' '.join(cleaned).title()


# Break up common contractions, in addition to lowercasing all characters
def decontract(text):
    text = text.lower()
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text


# Remove all stop words in passed text
def remove_stop_words(text):
    return [w for w in text if w not in stopwords.words('english')]


# Preprocessing which breaks up contractions, lowercases, removes punctuation and stopwords
def pre_processing(text):
    # Split up any contracted words
    decontracted = decontract(text)
    # Tokenize string
    tokens = word_tokenize(decontracted)

    # Remove all punctuation
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in tokens]

    # Filter out all stop words
    words = remove_stop_words(words)

    # Return cleaned list with no empty spaces
    return list(filter(None, words))


def tf_idf_weighting(data):
    # Pass documents through a CountVectorizer which performs some pre_processing and will return a term-document matrix
    # For question and answering I will also remove stopwords
    if data == "question_answer":
        count_vect_temp = CountVectorizer(analyzer=stem_analyser_no_stop_words)
    else:
        count_vect_temp = CountVectorizer(analyzer=stem_analyser)

    # Save a TF-IDF transformer for use
    tf_idf_transformer_temp = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True)
    # Use the transformer to apply transformation functions to the counts received from our CountVectorizer
    # Forms a TF-IDF weighting matrix
    tf_idf_matrix_temp = tf_idf_transformer_temp.fit_transform(count_vect_temp.fit_transform(data))

    # Return the vectorizer, transformer and TF-IDF weighting matrix
    return count_vect_temp, tf_idf_transformer_temp, tf_idf_matrix_temp


# Function to open csv datasets
def open_files(file_name, intent=None):
    # Open file
    with open(file_name, newline="", encoding="UTF-8") as csv_file:
        # Read file row by row
        for row in csv.reader(csv_file):

            # If not an intent dataset, it means it's our name dataset
            # Append our name list
            if intent is None:
                name_list.append(row[0])
            else:
                # If a question is present more than once in dataset add additional answers under the same dictionary index
                if row[0] in questions_answers_intents[intent]:
                    questions_answers_intents[intent][row[0]].append(row[1])
                else:
                    # Else create a new index and add the new question
                    questions_answers_intents[intent][row[0]] = [row[1]]

                # Add corresponding intent questions to their respective dictionary indexes
                questions_intents[intent].append(row[0])
                # Append our intent_labels list with our intent's name
                intent_labels.append(intent)


def tf_idf_weighting_new_data(count_vect, tf_idf_transformer, data, tf_idf_matrix=None):
    # Calculate TF-IDF weighting for passed in data, saves it as an array and returns it
    tf_idf_data_temp = tf_idf_transformer.transform(count_vect.transform(data))

    # If a TF-IDF weighting matrix is passed then convert it to an array and return it as well
    if tf_idf_matrix is not None:
        return tf_idf_matrix.toarray(), tf_idf_data_temp.toarray()
    else:
        return tf_idf_data_temp


# Prints the top 3 outputs for a given user query, based on the best cosine similarity
def best_outputs():
    k = 1
    for key in cosine_dictionary:
        if k < 4:
            print("\n\n========================= Match: ", k, " =========================\n")
            print("========================= Cosine Similarity: ", cosine_dictionary[key], "=========================")
            print("========================= Question: ", key, " =========================")
            print("========================= Answer: ", questions_answers_intents[user_intent_predict][key], "=========================")
            k += 1
        else:
            break


# Evaluates the model, printing out the individual intents F1 scores, the overall accuracy and finally the Confusion Matrix
def model_accuracy():
    # Tests model with our test data to see how well our classifier predicts the intent
    tf_idf_x_test = tf_idf_weighting_new_data(count_vect_all, tf_idf_transformer_all, x_test, None)
    intent_predict = classifier.predict(tf_idf_x_test)
    print("Accuracy score: ", accuracy_score(y_test, intent_predict))
    print("Question Answer F1 Score: ", f1_score(y_test, intent_predict, average="binary", pos_label="question_answer"))
    print("Small Talk F1 Score: ", f1_score(y_test, intent_predict, average="binary", pos_label="small_talk"))
    print("Question Answer Precision score: ", precision_score(y_test, intent_predict, average="binary", pos_label="question_answer"))
    print("Small Talk Precision score: ", precision_score(y_test, intent_predict, average="binary", pos_label="small_talk"))
    print("Question Answer Recall score: ", recall_score(y_test, intent_predict, average="binary", pos_label="question_answer"))
    print("Small Talk Recall score: ", recall_score(y_test, intent_predict, average="binary", pos_label="small_talk"))
    print("Confusion Matrix: \n", confusion_matrix(y_test, intent_predict))


async def get_weather():
    # Declare the client
    client = python_weather.Client(format=python_weather.METRIC)
    invalid = 1
    while invalid > 0:
        if invalid > 1:
            city_name = input("Enter a valid city name this time please: ")
        else:
            # Ask for the user's city
            city_name = input("Which city would you like to check the weather for? ")
        if city_name:
            try:
                # Get weather statistics from chosen city
                weather = await client.find(city_name)
                invalid = 0
                # If city found then print out temperature and exit out of function
                print("It is", weather.current.sky_text.lower(), "in " + city_name + ", with an average temperature of", weather.current.temperature, "Celsius")
                await client.close()
            # If city chosen is invalid raise error and ask user to enter a valid city
            except ExpatError:
                invalid += 1
        else:
            invalid += 1


# Keywords used to process the user's name input
user_name_key_phrases = ["hey", "hi", "hello", "name", "good morning", "good afternoon", "call", "thanks", "thank you", "im", "nice", "meet", "ask", "names", "greeting"]

# A variety of phrases the bot can respond with
response_key_phrases = ["Come on, throw some more queries my way!\n", "Let's hear some queries!\n", "Anything you want to get off your chest?\n",
                        "I know you got a few more queries in the tank...\n"]

# Keywords to detect when the user wants to get the date or time
date_time_key_phrases = ["what is the time", "what time is it", "whats the time", "tell me the time", "time please", "whats the date", "what is the date", "what date is it",
                         "what year is it", "what month is it", "can you tell me the time", "What's the date and time", "whats the date and time", "what's the time and date"]
# Keywords to detect when the user wants to know the weather
weather_key_phrases = ["hows the weather", "how is the weather", "whats the weather", "what is the weather"]

# Create our analyser which will do some pre_processing on our data
analyser = CountVectorizer().build_analyzer()
# Create the stemmer used to shorten words
stemmer = SnowballStemmer('english')
# Create the lemmatiser used to shorten words
lemmatiser = WordNetLemmatizer()

# Create lists for our names and intent_labels
intent_labels = []
name_list = []

# Create a dictionary of dictionaries where each dictionary is made up of both the questions and answers from each corresponding intent
questions_answers_intents = {
    "question_answer": {},
    "small_talk": {},
}

# Create a dictionary of lists where each list is made up of only the questions from the corresponding intent
questions_intents = {
    "question_answer": [],
    "small_talk": [],
}

# Open all datasets and save data to corresponding dictionaries and lists
open_files("first_name.csv", None)
open_files("question_answer_dataset.csv", "question_answer")
open_files("small_talk_dataset.csv", "small_talk")

# Split our data into training and testing allowing us evaluate our model
# Stratify so data is proportioned properly
# Set a random state value so the data is split randomly in the same way everytime
x_train, x_test, y_train, y_test = train_test_split(questions_intents["question_answer"] + questions_intents["small_talk"], intent_labels, stratify=intent_labels, test_size=0.35,
                                                    random_state=11)

# Create our vectorisers, transformers and TF-IDF weighted matrices for our intents
count_vect_qa, tf_idf_transformer_qa, tf_idf_qa = tf_idf_weighting(questions_intents["question_answer"])
count_vect_talk, tf_idf_transformer_talk, tf_idf_talk = tf_idf_weighting(questions_intents["small_talk"])
count_vect_all, tf_idf_transformer_all, tf_idf_x_train = tf_idf_weighting(x_train)

# If there is a preexisting classifier present use that
# Otherwise create a new classifier and save it to the given file name
if os.path.isfile("chatbot.joblib"):
    classifier = load("chatbot.joblib")
else:
    # Run Logistic Regression on our intent matrix
    classifier = LogisticRegression(random_state=0).fit(tf_idf_x_train, y_train)
    dump(classifier, "chatbot.joblib")

# Print evaluation results
# model_accuracy()
user_name = "John"
wrong = True
while wrong is True:
    # Request the user's name
    user_name = name_processing(lemmatise(pre_processing(input("Hello buddy, what name do you go by?\n"))))

    if not user_name or user_name.isspace():
        wrong = True
    else:
        wrong = False

# Greet the user and ask them to enter a query
user_query = input("Hey, " + user_name + "! Nice to meet you, fire your questions at me! Once you're bored of me just type 'bye', but hopefully you won't be anytime soon!\n")

stop = False

# While the user hasn't stopped the bot continue querying
while not stop:

    # If the user enters "bye" end the chat
    if decontract(user_query) == "bye":
        print("Gone already? Alright, see you then...")
        stop = True

    # If the user requests the date or time output it
    elif decontract(user_query) in date_time_key_phrases:
        dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        user_query = input("The date and time is " + dateTime + ". Anything else?\n")

    # If the user requests the weather
    elif decontract(user_query) in weather_key_phrases:

        # Create loop till function task complete (get the weather from API)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(get_weather())
        user_query = input(response_key_phrases[randrange(len(response_key_phrases))])

    else:
        # Process the user's query and perform TF-IDF weighting on it to create a TF-IDF matrix of the query
        tf_idf_new_query = tf_idf_weighting_new_data(count_vect_all, tf_idf_transformer_all, [user_query], None)
        # Use the TF-IDF matrix to predict the user's intent
        user_intent_predict = ' '.join(classifier.predict(tf_idf_new_query))

        # Depending on the intent, use the intent's transformer and vectoriser to perform TF-IDF weighting for the user query, returned as an array
        # Also return an array of the TF-IDF weighting calculated from the corresponding intent's questions

        if user_intent_predict == "small_talk":
            tf_idf_intent_array, tf_idf_query_array = tf_idf_weighting_new_data(count_vect_talk, tf_idf_transformer_talk, [user_query], tf_idf_talk)
        else:
            tf_idf_intent_array, tf_idf_query_array = tf_idf_weighting_new_data(count_vect_qa, tf_idf_transformer_qa, [user_query], tf_idf_qa)

        # Create a dictionary for our cosine similarity values
        cosine_dictionary = {}

        # Loop through all questions for the corresponding intent
        for index in range(len(questions_intents[user_intent_predict])):
            # Dismiss any dividing errors given
            with np.errstate(divide="ignore", invalid="ignore"):
                # Calculate the cosine similarity for every question in the intent's question list
                cosine_value = dot(tf_idf_query_array, tf_idf_intent_array[index]) / (norm(tf_idf_query_array) * norm(tf_idf_intent_array[index]))
            # If the calculated value is above 0.4 then save it to our dictionary
            if cosine_value > 0.55:
                cosine_dictionary[questions_intents[user_intent_predict][index]] = cosine_value
        # Sort the dictionary from largest to smallest
        cosine_dictionary = dict(sorted(cosine_dictionary.items(), key=lambda item: item[1]))
        # Reverse the order of the dictionary
        cosine_dictionary = dict(reversed(list(cosine_dictionary.items())))
        # Print out the top answers to the query
        # print(best_outputs())

        # If there is a possible answer to the query then retrieve an answer
        # Else ask the user to ask a new query
        if len(cosine_dictionary) > 0:
            # Index of the top rated answer for the user
            optimal_answer = questions_answers_intents[user_intent_predict][next(iter(cosine_dictionary))]

            # If the answer equals "Retrieve" that means the user is requesting a reminder of their name
            if optimal_answer[0] == "Retrieve":
                user_query = input("You've forgotten your own name?! It's " + user_name + ", you fool! \nAnymore questions, psh...?\n")

            # If the answer equals "Tweak" that means the user is requesting a name change
            elif optimal_answer[0] == "Tweak":
                wrong = True
                while wrong is True:
                    user_name = name_processing(lemmatise(pre_processing(input("\nCan't even spell your own name now? Okay I'll give you a second chance: \n"))))

                    if not user_name or user_name.isspace():
                        wrong = True
                    else:
                        wrong = False

                user_query = input("Welcome again, " + user_name + ". Hopefully this time you spelt your name right... What would you like from me?\n")

            else:
                # Otherwise output the best answer to the user
                # If there are multiple answers for the same question output a randomly selected answer
                print(questions_answers_intents[user_intent_predict][next(iter(cosine_dictionary))][randrange(len(optimal_answer))], "\n")
                # Return a random reply to the user asking them for a query
                user_query = input(response_key_phrases[randrange(len(response_key_phrases))])
        else:
            user_query = input("I'm afraid I haven't understood you exactly, maybe you would like to ask another query?\n")
