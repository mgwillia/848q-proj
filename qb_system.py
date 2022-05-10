from typing import List, Union, Tuple
from tqdm import tqdm
from qbdata import QantaDatabase
import torch
from tfidf_guesser import TfidfGuesser
from models import AnswerExtractor, Retriever, ReRanker, WikiLookup, Guesser, get_squad_train


class QuizBowlSystem:

    def __init__(self, dataset_name, model_name, wiki_lookup_path: str = 'data/wiki_lookup.2018.json') -> None:
        """Fill this method to create attributes, load saved models, etc
        Don't add any other arguments to this constructor. 
        If you really want to have arguments, they should have some default values set.
        """
        guesser = Guesser(dataset_name)
        print('Loading the Guesser model...')
        guesser.load()
        if dataset_name == 'qanta':
            guesstrain = QantaDatabase('data/qanta.train.2018.json')
        elif dataset_name == 'squad':
            guesstrain = get_squad_train()
        else:
            raise NotImplementedError
        # guesser.finetune(guesstrain, limit=-1)
        guesser.train(guesstrain, dataset_name, model_name)
        guesser.build_faiss_index(dataset_name, model_name)

        print('Loding the Wiki Lookups...')
        self.wiki_lookup = WikiLookup(wiki_lookup_path)

        reranker = ReRanker()
        print('Loading the Reranker model...')
        reranker.load('amberoad/bert-multilingual-passage-reranking-msmarco')

        self.retriever = Retriever(guesser, reranker, wiki_lookup=self.wiki_lookup)

        answer_extractor_base_model = "csarron/bert-base-uncased-squad-v1"
        self.answer_extractor = AnswerExtractor()
        print('Loading the Answer Extractor model...')
        self.answer_extractor.load(answer_extractor_base_model)

    def retrieve_page(self, question: str, disable_reranking=False) -> str:
        """Retrieves the wikipedia page name for an input question."""
        with torch.no_grad():
            page = self.retriever.retrieve_answer_document(
                question, disable_reranking=disable_reranking)
            return page

    def retrieve_page_batch(self, questions: List[str], disable_reranking=False) -> List[str]:
        """Retrieves the wikipedia page name for an input question."""
        with torch.no_grad():
            pages = self.retriever.retrieve_answer_document_batch(
                questions, disable_reranking=disable_reranking)
            return pages

    def execute_query(self, question: str, *, get_page=True) -> str:
        """Populate this method to do the following:
        1. Use the Retriever to get the top wikipedia page.
        2. Tokenize the question and the passage text to prepare inputs to the Bert-based Answer Extractor
        3. Predict an answer span for each question and return the list of corresponding answer texts."""
        with torch.no_grad():
            page = self.retrieve_page(question, disable_reranking=True)
            reference_text = self.wiki_lookup[page]['text']
            answer = self.answer_extractor.extract_answer(
                question, reference_text)[0]  # singleton list
            return (answer, page) if get_page else answer

    def execute_query_batch(self, questions: List[str], *, get_page=True) -> Union[List[str], List[str]]:
        """Populate this method to do the following:
        1. Use the Retriever to get the top wikipedia page.
        2. Tokenize the question and the passage text to prepare inputs to the Bert-based Answer Extractor
        3. Predict an answer span for each question and return the list of corresponding answer texts."""
        with torch.no_grad():
            pages = self.retrieve_page_batch(questions, disable_reranking=True)
            # reference_text = list(map(lambda x: self.wiki_lookup[x]['text'], pages))
            # answers = self.answer_extractor.extract_answer_batch(
            #     questions, reference_text)
            return pages


if __name__ == "__main__":
    qa = QuizBowlSystem()
    qanta_db = QantaDatabase('../data/small.guessdev.json')
    small_set_questions = qanta_db.all_questions[:10]

    for question in tqdm(small_set_questions):
        answer = qa.execute_query(question.first_sentence)
