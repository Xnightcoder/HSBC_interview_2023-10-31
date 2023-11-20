




openai_api_key="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

import openai

import os
os.environ['OPENAI_API_KEY'] = openai_api_key
openai.api_key = openai_api_key

### Added
##### To make the following code to work, I tried half a day and make clear the following configurations!!!!!!!!!!!!
#####os.environ["OPENAI_API_TYPE"] = "azure"
openai.proxy = "xxxxx://xxxxxxx:xxxxx/"
os.environ["OPENAI_PROXY"] = "xxxxx://xxxxxxx:xxxxx/"
openai.organization = "XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
os.environ["OPENAI_ORGANIZATION"] = "XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
### Added





def ask_openai_bot_question_based_context_and_get_answer(contexts,question):
    contexts_concatenated = ",".join(contexts)
    prompt_question_answer = """Based on the following context: \"
    {}
    \"
    Answer the question: {}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [ # Change the prompt parameter to the messages parameter
            {'role': 'user', 'content': prompt_question_answer.format(contexts_concatenated,question)}
        ],
        temperature = 0
    )
    answer = response['choices'][0]['message']['content']
    
    return answer





def ask_openai_to_judge_if_answer_correct_or_not(question,true_answers,reply):
    prompt_judgement = """In a quiz game, a question has been asked:
    The question is: {}
    the true answers are in these:
    {}
    Ignore all punctuations. Suppose any reply contains any one among the answers could be regarded as correct.
    the person beings asked replied: {}
    Does his/her reply can be regarded as correct or not? Please answer: correct or incorrect.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [ # Change the prompt parameter to the messages parameter
            {'role': 'user', 'content': prompt_judgement.format(question,true_answers,reply)}
        ],
        temperature = 0
    )
    answer = response['choices'][0]['message']['content']
    
    return answer





question = "Are you a bot?"
contexts = "A openai api calls are bots!"
reply = ask_openai_bot_question_based_context_and_get_answer(contexts, question)



print(reply)




