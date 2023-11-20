



from typing import List, Tuple, Dict, Iterator
import json
import os

openai_api_key="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

import openai

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






def validate_rag(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    logger.info("validating passages. size=%d", len(passages))
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits





def get_top_k_hits(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    #logger.info("validating passages. size=%d", len(passages))
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    return top_k_hits





import csv

def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        data = [row for row in reader]
    return data





from multiprocessing import Pool as ProcessPool
from dpr.utils.tokenizers import SimpleTokenizer
from functools import partial
from dpr.data.qa_validation import check_answer
import unicodedata
import regex as re



def _normalize_rag(text):
    return unicodedata.normalize("NFD", text)


def regex_match_rag(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None




def has_answer_rag(answers, text, tokenizer, match_type) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize_rag(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize_rag(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize_rag(single_answer)
            if regex_match_rag(text, single_answer):
                return True
    return False





def check_answer_rag(questions_answers_docs, tokenizer, match_type) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, (doc_ids, doc_scores) = questions_answers_docs

    global dpr_all_documents
    hits = []

    for i, doc_id in enumerate(doc_ids):
        doc = dpr_all_documents[doc_id]
        text = doc[0]

        answer_found = False
        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue
        elif has_answer_rag(answers, text, tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits





def calculate_matches_rag(
    all_docs: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
):
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """
    logger.info("all_docs size %d", len(all_docs))
    global dpr_all_documents
    dpr_all_documents = all_docs
    logger.info("dpr_all_documents size %d", len(dpr_all_documents))

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(processes=workers_num)
    logger.info("Matching answers in top docs...")
    get_score_partial = partial(check_answer_rag, match_type=match_type, tokenizer=tokenizer)

    questions_answers_docs = zip(answers, closest_docs)
    scores = processes.map(get_score_partial, questions_answers_docs)

    #logger.info("Per question validation results len=%d", len(scores))

    '''
    n_docs = len(closest_docs[0][0])
    
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)
    '''
    
    return closest_docs





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
    Does his/her reply can be regarded as correct or not? Please just answer one word: "correct" or "incorrect".
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [ # Change the prompt parameter to the messages parameter
            {'role': 'user', 'content': prompt_judgement.format(question,true_answers,reply)}
        ],
        temperature = 0
    )
    answer = response['choices'][0]['message']['content']
    
    
    ########## DEBUG ##########
    if 'incorrect' in answer.lower():
        pass
    elif 'correct' in answer.lower():
        correct_t = 1
    ########## DEBUG ##########
    
    return answer





from dense_retriever import *


@hydra.main(config_path="conf", config_name="rag_test")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    saved_state = load_states_from_checkpoint(cfg.model_file)

    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    logger.info("Loading saved model state ...")
    encoder.load_state(saved_state, strict=False)

    encoder_path = cfg.encoder_path
    if encoder_path:
        logger.info("Selecting encoder: %s", encoder_path)
        encoder = getattr(encoder, encoder_path)
    else:
        logger.info("Selecting standard question encoder")
        encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    encoder.eval()

    model_to_load = get_model_obj(encoder)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    # get questions & answers
    questions = []
    questions_text = []
    question_answers = []

    if not cfg.qa_dataset:
        logger.warning("Please specify qa_dataset to use")
        return

    ds_key = cfg.qa_dataset
    logger.info("qa_dataset: %s", ds_key)

    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()

    total_queries = len(qa_src)
    for i in range(total_queries):
        qa_sample = qa_src[i]
        question, answers = qa_sample.query, qa_sample.answers
        questions.append(question)
        question_answers.append(answers)

    logger.info("questions len %d", len(questions))
    logger.info("questions_text len %d", len(questions_text))

    if cfg.rpc_retriever_cfg_file:
        index_buffer_sz = 1000
        retriever = DenseRPCRetriever(
            encoder,
            cfg.batch_size,
            tensorizer,
            cfg.rpc_retriever_cfg_file,
            vector_size,
            use_l2_conversion=cfg.use_l2_conversion,
        )
    else:
        index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
        logger.info("Local Index class %s ", type(index))
        index_buffer_sz = index.buffer_size
        index.init_index(vector_size)
        retriever = LocalFaissRetriever(encoder, cfg.batch_size, tensorizer, index)

    logger.info("Using special token %s", qa_src.special_query_token)
    questions_tensor = retriever.generate_question_vectors(questions, query_token=qa_src.special_query_token)

    if qa_src.selector:
        logger.info("Using custom representation token selector")
        retriever.selector = qa_src.selector

    index_path = cfg.index_path
    if cfg.rpc_retriever_cfg_file and cfg.rpc_index_id:
        retriever.load_index(cfg.rpc_index_id)
    elif index_path and index.index_exists(index_path):
        logger.info("Index path: %s", index_path)
        retriever.index.deserialize(index_path)
    else:
        # send data for indexing
        id_prefixes = []
        ctx_sources = []
        for ctx_src in cfg.ctx_datatsets:
            ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
            id_prefixes.append(ctx_src.id_prefix)
            ctx_sources.append(ctx_src)
            logger.info("ctx_sources: %s", type(ctx_src))

        logger.info("id_prefixes per dataset: %s", id_prefixes)

        # index all passages
        ctx_files_patterns = cfg.encoded_ctx_files

        logger.info("ctx_files_patterns: %s", ctx_files_patterns)
        if ctx_files_patterns:
            assert len(ctx_files_patterns) == len(id_prefixes), "ctx len={} pref leb={}".format(
                len(ctx_files_patterns), len(id_prefixes)
            )
        else:
            assert (
                index_path or cfg.rpc_index_id
            ), "Either encoded_ctx_files or index_path pr rpc_index_id parameter should be set."

        input_paths = []
        path_id_prefixes = []
        for i, pattern in enumerate(ctx_files_patterns):
            pattern_files = glob.glob(pattern)
            pattern_id_prefix = id_prefixes[i]
            input_paths.extend(pattern_files)
            path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))
        logger.info("Embeddings files id prefixes: %s", path_id_prefixes)
        logger.info("Reading all passages data from files: %s", input_paths)
        retriever.index_encoded_data(input_paths, index_buffer_sz, path_id_prefixes=path_id_prefixes)
        if index_path:
            retriever.index.serialize(index_path)

    # get top k results
    top_results_and_scores = retriever.get_top_docs(questions_tensor.numpy(), cfg.n_docs)

    all_passages = get_all_passages(ctx_sources)
        
    ##### Load passage texts
    input_passage_csv = read_tsv(cfg.passage_map_file)
    input_passage_csv_key_by_id = {}
    for passage in input_passage_csv:
        input_passage_csv_key_by_id[passage['id']] = { 'text': passage['text'] , 'title': passage['title'] }
    
    closest_docs = calculate_matches_rag(all_passages, question_answers, top_results_and_scores, cfg.validation_workers, cfg.match)


    top_k = cfg.top_k
    
    
    count_of_correct_answers = 0
    
    ##### Start use openai to get correct answers
    for i, question in enumerate(questions):
        context = []
        for passage_id in closest_docs[i][0][:top_k]:
            context.append(input_passage_csv_key_by_id[passage_id[5:]]['text'])
        reply = ask_openai_bot_question_based_context_and_get_answer(context,question)
    
        true_answers = ",".join(question_answers[i])
        judgement = ask_openai_to_judge_if_answer_correct_or_not( question, true_answers, reply )

        if "incorrect" in judgement.lower():
            pass
        elif "correct" in judgement.lower():
            print("Question " + str(i+1) + "'s answer is correct!")
            count_of_correct_answers += 1





        ########## DEBUG ##########
        print("Question " + str(i+1) + " has been processed!")
        ########## DEBUG ##########





    print("Correct replies: " + str(count_of_correct_answers) + ".")





if __name__ == "__main__":
    main()
