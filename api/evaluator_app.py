"""
This is an API to support the LLM QA chain auto-evaluator. 
"""

import io
import os
from dotenv import load_dotenv
import sentry_sdk
import json
import time
import pypdf
import random
import logging
import itertools
import faiss
import pandas as pd
from typing import Dict, List
from json import JSONDecodeError
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.chains import QAGenerationChain
from langchain.evaluation.qa import QAEvalChain
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
from text_utils import GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT_OPENAI, QA_CHAIN_PROMPT, QA_CHAIN_PROMPT_LLAMA
from lib.QAGenerator import QAGenerator
from lib.Chatbot import Chatbot

def generate_one_eval_pair(qa_generator, text, chunk, logger, max_retry = 5):
    """
    Generate one question-answer pair from input text 
    @param text: text to generate eval set from
    @param chunk: chunk size to draw question from text
    @param logger: logger
    @return: a dict with keys "question" and "answer"
    """

    logger.info("`Generating eval QA pair ...`")
    # Generate random starting index in the doc to draw question from
    num_of_chars = len(text)
    if num_of_chars <= chunk:
        sub_sequence = text
    else:
        starting_index = random.randint(0, num_of_chars-chunk)
        sub_sequence = text[starting_index:starting_index+chunk]

    # Catch any QA generation errors and re-try until QA pair is generated
    awaiting_answer = True
    retry_count = 0
    while awaiting_answer:
        try:
            retry_count += 1
            qa_pair = qa_generator.run(sub_sequence, logger)
            return qa_pair
        except JSONDecodeError:
            logger.error("Error on question")
            # Try other part of the text
            if num_of_chars > chunk:
                starting_index = random.randint(0, num_of_chars-chunk)
                sub_sequence = text[starting_index:starting_index+chunk]
            if retry_count > max_retry:
                return None

def make_qa_generator(api_base, azure_openai_key, engine):
    return QAGenerator(api_base, azure_openai_key, engine)

def make_chatbot(chat_endpoint):
    return Chatbot(chat_endpoint)

def make_chat_llm(api_base, azure_openai_key, engine):
    return AzureChatOpenAI(deployment_name=engine,
                           openai_api_base=api_base,
                           openai_api_key=azure_openai_key,
                           openai_api_version="2023-05-15")

def grade_model_answer(llm_chat, predicted_dataset, predictions, grade_answer_prompt, logger):
    """
    Grades the answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @param logger: logger
    @return: A list of scores for the distilled answers.
    """

    logger.info("`Grading model answer ...`")
    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    # Note: GPT-4 grader is advised by OAI 
    eval_chain = QAEvalChain.from_llm(llm=llm_chat,
                                      prompt=prompt)
    graded_outputs = eval_chain.evaluate(predicted_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs


def grade_model_retrieval(llm_chat, gt_dataset, predictions, grade_docs_prompt, logger):
    """
    Grades the relevance of retrieved documents based on ground truth and model predictions.
    @param gt_dataset: list of dictionaries containing ground truth questions and answers.
    @param predictions: list of dictionaries containing model predictions for the questions
    @param grade_docs_prompt: prompt level for the grading.
    @return: list of scores for the retrieved documents.
    """

    logger.info("`Grading relevance of retrieved docs ...`")
    if grade_docs_prompt == "Fast":
        prompt = GRADE_DOCS_PROMPT_FAST
    else:
        prompt = GRADE_DOCS_PROMPT

    # Note: GPT-4 grader is advised by OAI
    eval_chain = QAEvalChain.from_llm(llm=llm_chat,
                                      prompt=prompt)
    graded_outputs = eval_chain.evaluate(gt_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs


def run_eval(chatbot, llm_chat, eval_set, grade_prompt, logger):
    """
    Runs evaluation on a model's performance on a given evaluation dataset.
    @param chatbot: Chatbot used for answering questions
    @param eval_set: List of dictionaries containing questions and corresponding ground truth answers
    @param grade_prompt: String prompt used for grading model's performance
    @return: A tuple of four items:
    - answers_grade: A dictionary containing scores for the model's answers.
    - retrieval_grade: A dictionary containing scores for the model's document retrieval.
    - latencies_list: A list of latencies in seconds for each question answered.
    - predictions_list: A list of dictionaries containing the model's predicted answers and relevant documents for each question.
    """

    logger.info("`Running eval ...`")
    predictions = []
    retrieved_docs = []
    gt_dataset = [] # Ground-truth data
    latency = []

    # Get answer and log latency
    for eval_qa_pair in eval_set:
        start_time = time.time()
        answer = chatbot.run(eval_qa_pair["question"], logger)
        predictions.append(
                {"question": eval_qa_pair["question"], "answer": eval_qa_pair["answer"], "result": answer})
        gt_dataset.append(eval_qa_pair)
        end_time = time.time()
        elapsed_time = end_time - start_time
        latency.append(elapsed_time)

    # Grade
    graded_answers = grade_model_answer(llm_chat,
        gt_dataset, predictions, grade_prompt, logger)
    # graded_retrieval = grade_model_retrieval(llm_chat,
    #     gt_dataset, retrieved_docs, grade_prompt, logger)
    
    return graded_answers, latency, predictions

load_dotenv()

if os.environ.get("ENVIRONMENT") != "development":
    sentry_sdk.init(
    dsn="https://065aa152c4de4e14af9f9e7335c8eae4@o4505106202820608.ingest.sentry.io/4505106207735808",
    traces_sample_rate=1.0,
    )

API_BASE = os.environ.get("API_BASE")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
CHAT_ENDPOINT = os.environ.get("CHAT_ENDPOINT")
GPT_ENGINE = os.environ.get("GPT_ENGINE")

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "https://evaluator-ui.vercel.app/"
    "https://evaluator-ui.vercel.app"
    "evaluator-ui.vercel.app/"
    "evaluator-ui.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Auto Evaluator!"}


def run_evaluator(
    files,
    num_eval_questions,
    chunk_chars,
    overlap,
    split_method,
    retriever_type,
    embeddings,
    model_version,
    grade_prompt,
    num_neighbors,
    test_dataset
):

    # Set up logging
    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    # Read content of files
    texts = []
    fnames = []
    for file in files:
        logger.info("Reading file: {}".format(file.filename))
        contents = file.file.read()
        # PDF file
        if file.content_type == 'application/pdf':
            logger.info("File {} is a PDF".format(file.filename))
            pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            texts.append(text)
            fnames.append(file.filename)
        # Text file
        elif file.content_type == 'text/plain':
            logger.info("File {} is a TXT".format(file.filename))
            texts.append(contents.decode())
            fnames.append(file.filename)
        else:
            logger.warning(
                "Unsupported file type for file: {}".format(file.filename))


    # Generate evaluation set (question-answer pairs)
    qa_generator = make_qa_generator(API_BASE, AZURE_OPENAI_KEY, GPT_ENGINE)
    eval_set = []
    i = 0
    while i < num_eval_questions:
        # Randomly select a file
        file_index = random.randint(0, len(fnames)-1)
        text = texts[file_index]

        doc_chunk_size = 500 # Size in characters of a randomized section drawn from text
        
        # Generate one question
        # eval_pair is a dict with keys "question" and "answer"
        if i < len(test_dataset):
            eval_pair = test_dataset[i]
        else:
            eval_pair = generate_one_eval_pair(qa_generator, text, doc_chunk_size, logger)
            if eval_pair == None:
                # Error in eval generation
                logger.warn("A QA pair was not generated correctly. Skipping this pair.")
                continue
        logger.info(f"Generated a question answer pair: {eval_pair['question']} ||| {eval_pair['answer']}")
        eval_set.append(eval_pair)
        i += 1
        
    # Run eval
    chatbot = make_chatbot(CHAT_ENDPOINT)
    llm_chat = make_chat_llm(API_BASE, AZURE_OPENAI_KEY, GPT_ENGINE)
    graded_answers, latency, predictions = run_eval(
        chatbot, llm_chat, eval_set, grade_prompt, logger)

    # Assemble output
    d = pd.DataFrame(predictions)
    d['answerScore'] = [g['text'] for g in graded_answers]
    d['retrievalScore'] = [g['text'] for g in graded_answers] # Debug!!!!
    d['latency'] = latency

    # Summary statistics
    d['answerScore'] = [{'score': 1 if "Incorrect" not in text else 0,
                            'justification': text} for text in d['answerScore']]
    d['retrievalScore'] = [{'score': 1 if "Incorrect" not in text else 0,
                            'justification': text} for text in d['retrievalScore']]

    # Convert dataframe to dict
    d_dict = d.to_dict('records')
    if len(d_dict) == 1:
        yield json.dumps({"data":  d_dict[0]})
    else:
        logger.warn(
            "A QA pair was not evaluated correctly. Skipping this pair.")


@app.post("/evaluator-stream")
async def create_response(
    files: List[UploadFile] = File(...),
    num_eval_questions: int = Form(5),
    chunk_chars: int = Form(1000),
    overlap: int = Form(100),
    split_method: str = Form("RecursiveTextSplitter"),
    retriever_type: str = Form("similarity-search"),
    embeddings: str = Form("OpenAI"),
    model_version: str = Form("gpt-3.5-turbo"),
    grade_prompt: str = Form("Fast"),
    num_neighbors: int = Form(3),
    test_dataset: str = Form("[]"),
):
    test_dataset = json.loads(test_dataset)
    return EventSourceResponse(run_evaluator(files, num_eval_questions, chunk_chars,
                                             overlap, split_method, retriever_type, embeddings, model_version, grade_prompt, num_neighbors, test_dataset), headers={"Content-Type": "text/event-stream", "Connection": "keep-alive", "Cache-Control": "no-cache"})
