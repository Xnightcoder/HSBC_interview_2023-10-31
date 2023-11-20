
import argparse
from gensim.summarization import keywords 
import requests
import json
import copy


def calculate_question_passage_relevance(question, passage, relation_threshold = 0.5):
    
    passage_keywords = keywords(passage, ratio=0.5, words=None, split=True, scores=False, pos_filter=('NN', 'JJ'), lemmatize=False, deacc=False)
    passage_keywords_copy = copy.deepcopy(passage_keywords)
    for passage_keyword in passage_keywords_copy:
        if ' ' in passage_keyword:
            for keyword in passage_keyword.split():
                passage_keywords.append(keyword)
    
    question_keywords = keywords(question, ratio=0.9, words=None, split=True, scores=False, pos_filter=('NN', 'JJ'), lemmatize=False, deacc=False)
    question_keywords_copy = copy.deepcopy(question_keywords)
    for question_keyword in question_keywords_copy:
        if ' ' in question_keyword:
            for keyword in question_keyword.split():
                question_keywords.append(keyword)
    
    request_string = "https://api.conceptnet.io/relatedness?node1=/c/en/{}&node2=/c/en/{}"

    for passage_keyword in passage_keywords:
        for question_keyword in question_keywords:

            # Send the HTTP request
            response = requests.get(request_string.format(passage_keyword, question_keyword))

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the response content as JSON
                data = json.loads(response.content)

                # Get the value of the key "value"
                value = data.get('value')

                if(float(value) >= relation_threshold):
                    return True
            else:
                print(f"Request failed with status code {response.status_code}")

    return False







def calculate_question_passage_relevance_abs(question, passage, relation_threshold = 0.5):
    
    passage_keywords = keywords(passage, ratio=0.5, words=None, split=True, scores=False, pos_filter=('NN', 'JJ'), lemmatize=False, deacc=False)
    passage_keywords_copy = copy.deepcopy(passage_keywords)
    for passage_keyword in passage_keywords_copy:
        if ' ' in passage_keyword:
            for keyword in passage_keyword.split():
                passage_keywords.append(keyword)
    
    question_keywords = keywords(question, ratio=0.9, words=None, split=True, scores=False, pos_filter=('NN', 'JJ'), lemmatize=False, deacc=False)
    question_keywords_copy = copy.deepcopy(question_keywords)
    for question_keyword in question_keywords_copy:
        if ' ' in question_keyword:
            for keyword in question_keyword.split():
                question_keywords.append(keyword)
    
    request_string = "https://api.conceptnet.io/relatedness?node1=/c/en/{}&node2=/c/en/{}"

    for passage_keyword in passage_keywords:
        for question_keyword in question_keywords:

            # Send the HTTP request
            response = requests.get(request_string.format(passage_keyword, question_keyword))

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the response content as JSON
                data = json.loads(response.content)

                # Get the value of the key "value"
                value = data.get('value')

                if(abs(float(value)) >= relation_threshold):
                    return True
            else:
                print(f"Request failed with status code {response.status_code}")

    return False







def main(passage, question):
    related_or_not =  calculate_question_passage_relevance(question, passage)
    print(related_or_not)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a passage and a question.")
    parser.add_argument("--passage", required=True, help="The passage to process.")
    parser.add_argument("--question", required=True, help="The question to process.")
    args = parser.parse_args()

    main(args.passage, args.question)




