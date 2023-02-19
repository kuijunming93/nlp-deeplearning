import numpy as np
import tensorflow as tf
import re
import time
import utilities

# DATA PROCESSING
# reading raw files into memory for processing
memoryLine = {}
with open('source/resources/movie_lines.txt', encoding='utf-8', errors='ignore') as f:
    _movieLines = f.read().split("\n")
    for _line in _movieLines:
        _element = _line.split(" +++$+++ ")
        if len(_element) == 5:
            memoryLine[_element[0]] = _element[4]

memoryConversation = []
with open('./source/resources/movie_conversations.txt', encoding='utf-8', errors='ignore') as f:
    _movieConversations = f.read().split("\n")
    for _conversation in _movieConversations:
        _element = _conversation.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
        memoryConversation.append(_element.split(","))

# cleaning up raw data of punctuations and lower casing
clean_questions = []
clean_answers = []
for conversation in memoryConversation:
    for i in range(len(conversation)-1):
        clean_questions.append(utilities.cleanText(memoryLine[conversation[i]]))
        clean_answers.append(utilities.cleanText(memoryLine[conversation[i+1]]))

# consolidating vocabularies from cleaned data
total = []
total.append(clean_questions)
total.append(clean_answers)
vocabulary = utilities.consolidateWordCount(total)

# tokenizing vocabularies to index
questionsWords2Int = utilities.convertVocabToInt(vocabulary, 20)
answersWords2Int = utilities.convertVocabToInt(vocabulary, 20)

# reverse tokenizing dictionary
answersInt2Words = {v:k for k, v in answersWords2Int.items()}

# adding EOS for all answers
for i in range(len(clean_answers)):
    clean_answers[i] += '<EOS>'

# tokenizing all clean data into indices
clean_questions_intoInt = utilities.convertSentencesToInt(clean_questions, questionsWords2Int)
clean_answers_intoInt = utilities.convertSentencesToInt(clean_answers, answersWords2Int)

# sorting out of tokenized clean data to simplify training process
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for ind, value in enumerate(clean_questions_intoInt):
        if len(value) == length:
            sorted_clean_questions.append(value)
            sorted_clean_answers.append(clean_answers_intoInt[ind])
