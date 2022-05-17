import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import datasets
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import (AdamW, AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          BertForSequenceClassification, BertTokenizerFast,
                          DataCollatorWithPadding, DPRContextEncoder,
                          DPRQuestionEncoder, EarlyStoppingCallback, Trainer,
                          TrainingArguments, default_data_collator,
                          get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_scheduler)
from transformers import DPRConfig

from base_models import BaseGuesser, BaseReRanker
from guess_train_dataset import GuessTrainDataset
from qbdata import QantaDatabase, WikiLookup

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BiEncoderNllLoss(torch.nn.Module):

    def __init__(self):
        super(BiEncoderNllLoss, self).__init__()

    def forward(
        self,
        q_vectors: torch.Tensor,
        ctx_vectors: torch.Tensor,
        positive_idx_per_question: list,
    ) -> Tuple[torch.Tensor, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        _, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        return loss, correct_predictions_count
    
class VanilaNllLoss(torch.nn.Module):

    def __init__(self):
        super(VanilaNllLoss, self).__init__()

    def forward(
        self,
        q_vectors: torch.Tensor,
        ctx_vectors: torch.Tensor,
        positive_idx_per_question: list,
    ) -> Tuple[torch.Tensor, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        _, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        return loss, correct_predictions_count


class Guesser(BaseGuesser):
    """You can implement your own Bert based Guesser here"""
    def __init__(self) -> None:
        self.tokenizer = None
        self.question_model = None
        self.context_model = None
        self.wiki_lookup = None
        self.index = None

    def load(self, from_checkpoint=True):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.wiki_lookup = WikiLookup('data/wiki_lookup.2018.json')
        question_encoder_path = 'models/guesser_question_encoder_5.pth.tar'
        context_encoder_path = 'models/guesser_context_encoder_5.pth.tar'

        #dpr_config = DPRConfig()
        #print(dpr_config, flush=True)
        #print(dpr_config.projection_dim, flush=True)
        #dpr_config.projection_dim = 768
        #print(dpr_config.projection_dim, flush=True)

        #self.question_model = DPRQuestionEncoder(config=dpr_config).to(device)
        #saved_question_model = torch.load('models/question_encoder_pretrained.pth.tar')
        #saved_question_model['question_encoder.encode_proj.weight'] = saved_question_model['question_encoder.bert_model.pooler.dense.weight']
        #saved_question_model['question_encoder.encode_proj.bias'] = saved_question_model['question_encoder.bert_model.pooler.dense.bias']
        #del saved_question_model['question_encoder.bert_model.pooler.dense.weight']
        #del saved_question_model['question_encoder.bert_model.pooler.dense.bias']
        #missing = self.question_model.load_state_dict(saved_question_model, strict=False)
        #print(missing)

        #self.context_model = DPRContextEncoder(config=dpr_config).to(device)
        #saved_context_model = torch.load('models/context_encoder_pretrained.pth.tar')
        #saved_context_model['ctx_encoder.encode_proj.weight'] = saved_context_model['ctx_encoder.bert_model.pooler.dense.weight']
        #saved_context_model['ctx_encoder.encode_proj.bias'] = saved_context_model['ctx_encoder.bert_model.pooler.dense.bias']
        #del saved_context_model['ctx_encoder.bert_model.pooler.dense.weight']
        #del saved_context_model['ctx_encoder.bert_model.pooler.dense.bias']
        #missing = self.context_model.load_state_dict(saved_context_model, strict=False)
        #print(missing)

        self.question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
        self.context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
        if os.path.isfile(question_encoder_path) and from_checkpoint:
            print(f'loading question model from checkpoint {question_encoder_path}')
            self.question_model = torch.load(question_encoder_path, map_location=device)
        if os.path.isfile(context_encoder_path) and from_checkpoint:
            print(f'loading context model from checkpoint {context_encoder_path}')
            self.context_model = torch.load(context_encoder_path, map_location=device)
    
        # self.train_pages = [x.page for x in QantaDatabase('data/qanta.train.2018.json').guess_train_questions]
        self.train_pages = [x.page for x in QantaDatabase('data/squad1.1/train-v1.1.json').guess_train_questions]

    def get_guesser_scheduler(self, optimizer, warmup_steps, total_training_steps, steps_shift=0, last_epoch=-1):

        """Create a schedule with a learning rate that decreases linearly after
        linearly increasing during a warmup period.
        """

        def lr_lambda(current_step):
            current_step += steps_shift
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                1e-7,
                float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

    def finetune(self, training_data: QantaDatabase, batch_size: int=128, learning_rate: float=1e-5, split_rule: str='full', scaling_param: float=1.0, limit: int=-1):
        NUM_EPOCHS = 5
        BASE_BATCH_SIZE = 16
        LR_SCALE_FACTOR = batch_size / BASE_BATCH_SIZE

        ### FIRST, PREP THE DATA ###
        train_dataset = GuessTrainDataset(training_data, self.tokenizer, self.wiki_lookup, 'train', split_rule)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=4, batch_size=batch_size, pin_memory=False, drop_last=True, shuffle=True)

        ### THEN, TRAIN THE ENCODERS ###
        question_optim = torch.optim.Adam(self.question_model.parameters(), lr=learning_rate * LR_SCALE_FACTOR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
        context_optim = torch.optim.Adam(self.context_model.parameters(), lr=learning_rate * LR_SCALE_FACTOR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
        question_scheduler = self.get_guesser_scheduler(question_optim, 100, NUM_EPOCHS * (len(train_dataset) // batch_size))
        context_scheduler = self.get_guesser_scheduler(context_optim, 100, NUM_EPOCHS * (len(train_dataset) // batch_size))
        loss_fn = BiEncoderNllLoss()
        # loss_fn = torch.nn.NLLLoss()

        print('Ready to finetune', flush=True)
        self.question_model.train()
        self.context_model.train()
        for name, param in self.question_model.named_parameters():
            print(name)
            if 'layer' in name:
                layer_num = int(name.split('.')[4])
                if layer_num < 11:
                    pass#param.requires_grad = False
            else:
                pass#param.requires_grad = False
        for name, param in self.context_model.named_parameters():
            print(name)
            if 'layer' in name:
                layer_num = int(name.split('.')[4])
                if layer_num < 11:
                    pass#param.requires_grad = False
            else:
                pass#param.requires_grad = False

        self.question_model = torch.nn.DataParallel(self.question_model)
        self.context_model = torch.nn.DataParallel(self.context_model)
        losses = []
        correct_question_preds_total = []
        correct_biencoder_preds_total = []
        for epoch_num in range(NUM_EPOCHS):
            for batch_num, batch in enumerate(train_dataloader):
                batch_questions, answers = batch['question'].to(device), batch['answer_text'].to(device)
                if split_rule == 'full':
                    question_embeddings = self.question_model(batch_questions).pooler_output
                    context_embeddings = self.context_model(answers).pooler_output
                    biencoder_loss, correct_biencoder_preds = loss_fn(question_embeddings, context_embeddings, list(range(question_embeddings.shape[0])))
                elif split_rule == 'last-a':
                    random_sentences = None
                    last_sentences = None
                    valid_indices = batch['valid_indices']
                    random_sentences = torch.zeros((batch_questions.shape[0], batch_questions.shape[-1]), dtype=torch.long).to(device)
                    last_sentences = torch.zeros((batch_questions.shape[0], batch_questions.shape[-1]), dtype=torch.long).to(device)
                    for item_idx, valid_index_list in enumerate(valid_indices):
                        random_idx = random.randint(0, 1)
                        last_idx = int(valid_index_list.split(',')[-1])
                        random_sentences[item_idx] = batch_questions[item_idx][random_idx]
                        last_sentences[item_idx] = batch_questions[item_idx][last_idx]
                    random_sentence_embeddings = self.question_model(random_sentences).pooler_output.detach()
                    last_sentence_embeddings = self.question_model(last_sentences).pooler_output
                    context_embeddings = self.context_model(answers).pooler_output
                    biencoder_loss, correct_biencoder_preds = loss_fn(last_sentence_embeddings, context_embeddings, list(range(last_sentence_embeddings.shape[0])))
                    question_loss, correct_question_preds = loss_fn(random_sentence_embeddings, last_sentence_embeddings, list(range(last_sentence_embeddings.shape[0])))
                elif split_rule == 'last-b':
                    random_sentences = None
                    last_sentences = None
                    valid_indices = batch['valid_indices']
                    random_sentences = torch.zeros((batch_questions.shape[0], batch_questions.shape[-1]), dtype=torch.long).to(device)
                    last_sentences = torch.zeros((batch_questions.shape[0], batch_questions.shape[-1]), dtype=torch.long).to(device)
                    for item_idx, valid_index_list in enumerate(valid_indices):
                        random_idx = random.randint(0, 1)
                        last_idx = int(valid_index_list.split(',')[-1])
                        random_sentences[item_idx] = batch_questions[item_idx][random_idx]
                        last_sentences[item_idx] = batch_questions[item_idx][last_idx]
                    random_sentence_embeddings = self.question_model(random_sentences).pooler_output
                    last_sentence_embeddings = self.question_model(last_sentences).pooler_output.detach()
                    context_embeddings = self.context_model(answers).pooler_output
                    biencoder_loss, correct_biencoder_preds = loss_fn(last_sentence_embeddings, context_embeddings, list(range(last_sentence_embeddings.shape[0])))
                    question_loss, correct_question_preds = loss_fn(random_sentence_embeddings, last_sentence_embeddings, list(range(last_sentence_embeddings.shape[0])))
                elif split_rule == 'last-c':
                    random_sentences = None
                    last_sentences = None
                    valid_indices = batch['valid_indices']
                    random_sentences = torch.zeros((batch_questions.shape[0], batch_questions.shape[-1]), dtype=torch.long).to(device)
                    last_sentences = torch.zeros((batch_questions.shape[0], batch_questions.shape[-1]), dtype=torch.long).to(device)
                    for item_idx, valid_index_list in enumerate(valid_indices):
                        random_idx = int(valid_index_list.split(',')[-2])
                        last_idx = int(valid_index_list.split(',')[-1])
                        random_sentences[item_idx] = batch_questions[item_idx][random_idx]
                        last_sentences[item_idx] = batch_questions[item_idx][last_idx]
                    random_sentence_embeddings = self.question_model(random_sentences).pooler_output.detach()
                    last_sentence_embeddings = self.question_model(last_sentences).pooler_output
                    context_embeddings = self.context_model(answers).pooler_output
                    biencoder_loss, correct_biencoder_preds = loss_fn(last_sentence_embeddings, context_embeddings, list(range(last_sentence_embeddings.shape[0])))
                    question_loss, correct_question_preds = loss_fn(random_sentence_embeddings, last_sentence_embeddings, list(range(last_sentence_embeddings.shape[0])))
                elif split_rule == 'last-d':
                    random_sentences = None
                    last_sentences = None
                    valid_indices = batch['valid_indices']
                    random_sentences = torch.zeros((batch_questions.shape[0], batch_questions.shape[-1]), dtype=torch.long).to(device)
                    last_sentences = torch.zeros((batch_questions.shape[0], batch_questions.shape[-1]), dtype=torch.long).to(device)
                    for item_idx, valid_index_list in enumerate(valid_indices):
                        first_idx = int(valid_index_list.split(',')[0])
                        last_idx = int(valid_index_list.split(',')[-1])
                        random_sentences[item_idx] = batch_questions[item_idx][first_idx]
                        last_sentences[item_idx] = batch_questions[item_idx][last_idx]
                    random_sentence_embeddings = self.question_model(random_sentences).pooler_output.detach()
                    last_sentence_embeddings = self.question_model(last_sentences).pooler_output
                    context_embeddings = self.context_model(answers).pooler_output
                    biencoder_loss, correct_biencoder_preds = loss_fn(last_sentence_embeddings, context_embeddings, list(range(last_sentence_embeddings.shape[0])))
                    question_loss, correct_question_preds = loss_fn(random_sentence_embeddings, last_sentence_embeddings, list(range(last_sentence_embeddings.shape[0])))
                    
                if split_rule != 'full':
                    batch_loss = question_loss * scaling_param + biencoder_loss
                else:
                    batch_loss = biencoder_loss

                question_optim.zero_grad()
                context_optim.zero_grad()
                batch_loss.backward()
                question_optim.step()
                context_optim.step()
                question_scheduler.step()
                context_scheduler.step()

                losses.append(batch_loss.item())
                if split_rule != 'full':
                    correct_question_preds_total.append(correct_question_preds.item())
                correct_biencoder_preds_total.append(correct_biencoder_preds.item())

                if batch_num % 8 == 7:
                    if 'last' in split_rule:
                        print(f'Epoch Num: {epoch_num}, Batch Num: {batch_num}, Loss: {np.array(losses).mean()}, Correct question preds per batch: {np.array(correct_question_preds_total).mean()}, Correct biencoder preds per batch: {np.array(correct_biencoder_preds_total).mean()}', flush=True)
                    else:
                        print(f'Epoch Num: {epoch_num}, Batch Num: {batch_num}, Loss: {np.array(losses).mean()}, Correct biencoder preds per batch: {np.array(correct_biencoder_preds_total).mean()}', flush=True)
                    losses = []
                    correct_question_preds_total = []
                    correct_biencoder_preds_total = []

            if epoch_num % 5 == 0:
                torch.save(self.question_model, f'models/guesser_question_encoder_{split_rule}_{batch_size}_{learning_rate}_{scaling_param}_{epoch_num}.pth.tar')
                torch.save(self.context_model, f'models/guesser_context_encoder_{split_rule}_{batch_size}_{learning_rate}_{scaling_param}_{epoch_num}.pth.tar')
        torch.save(self.question_model, f'models/guesser_question_encoder_{split_rule}_{batch_size}_{learning_rate}_{scaling_param}.pth.tar')
        torch.save(self.context_model, f'models/guesser_context_encoder_{split_rule}_{batch_size}_{learning_rate}_{scaling_param}.pth.tar')

    def train(self, training_data: QantaDatabase, split_rule: str='full', limit: int=-1):
        print('Running Guesser.train()', flush=True)
        ### GET TRAIN EMBEDDINGS ###
        BATCH_SIZE = 16
        DIMENSION = 768 ### TODO: double check embed length

        train_dataset = GuessTrainDataset(training_data, self.tokenizer, self.wiki_lookup, 'train', split_rule)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, pin_memory=True, drop_last=False, shuffle=False)

        for parameter in self.context_model.parameters():
            parameter.requires_grad = False
        self.context_model.eval()
        context_embeddings = torch.zeros((len(train_dataset), DIMENSION))
        print('Computing context embeddings for Guesser', flush=True)
        with torch.no_grad():
            for i, batch in enumerate(train_dataloader):
                answers = batch['answer_text'].to(device)
                context_embeddings[i*BATCH_SIZE:min(len(train_dataset), (i+1)*BATCH_SIZE)] = self.context_model(answers).pooler_output
        torch.save(context_embeddings, 'models/context_embeddings.pth.tar')

    def build_faiss_index(self):
        DIMENSION = 768 ### TODO: double check embed length
        context_embeddings = torch.load('models/context_embeddings.pth.tar').numpy()
        self.index = faiss.IndexFlatIP(DIMENSION)
        self.index.add(context_embeddings)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        """
        Given the text of questions, generate guesses (a tuple of page id and score) for each one.

        Keyword arguments:
        questions -- Raw text of questions in a list
        max_n_guesses -- How many top guesses to return
        """ 
        DIMENSION = 768 ### TODO: double check embed length
        question_embeddings = torch.zeros((len(questions), DIMENSION))
        with torch.no_grad():
            for i, question in enumerate(questions):
                question_embeddings[i] = self.question_model(self.tokenizer(question, return_tensors="pt", max_length=512, truncation=True, padding='max_length')["input_ids"].to(device)).pooler_output

        neighbor_scores, neighbor_indices = self.index.search(question_embeddings.numpy(), max_n_guesses)
        guesses = []
        for i in range(len(questions)):
            guess = []
            for j in range(max_n_guesses):
                guess.append((self.train_pages[neighbor_indices[i][j]], neighbor_scores[i][j]))
            guesses.append(guess)
        return guesses

    def confusion_matrix(self, evaluation_data: QantaDatabase, limit=-1) -> Dict[str, Dict[str, int]]:
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param evaluation_data: Database of questions and answers
        :param limit: How many evaluation questions to use
        """

        questions = [x.sentences[-1] for x in evaluation_data.guess_dev_questions]
        answers = [x.page for x in evaluation_data.guess_dev_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        print("Eval on %i question" % len(questions))
            
        d = defaultdict(dict)
        data_index = 0
        raw_guesses = self.guess(questions, max_n_guesses=5)
        guesses = [x[0][0] for x in raw_guesses]
        for gg, yy in zip(guesses, answers):
            d[yy][gg] = d[yy].get(gg, 0) + 1
            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index,
                                                      len(guesses)))
        return d


class Retriever:
    """The component that indexes the documents and retrieves the top document from an index for an input open-domain question.

    It uses two systems:
     - Guesser that fetches top K documents for an input question, and
     - ReRanker that then reranks these top K documents by comparing each of them with the question to produce a similarity score."""

    def __init__(self, guesser: BaseGuesser, reranker: BaseReRanker, wiki_lookup: Union[str, WikiLookup],
                 max_n_guesses=10) -> None:
        if isinstance(wiki_lookup, str):
            self.wiki_lookup = WikiLookup(wiki_lookup)
        else:
            self.wiki_lookup = wiki_lookup
        self.guesser = guesser
        self.reranker = reranker
        self.max_n_guesses = max_n_guesses

    def retrieve_answer_document(self, question: str, disable_reranking=False) -> str:
        """Returns the best guessed page that contains the answer to the question."""
        guesses = self.guesser.guess([question], max_n_guesses=self.max_n_guesses)[0]

        if disable_reranking:
            _, best_page = max((score, page) for page, score in guesses)
            return best_page

        ref_texts = []
        for page, score in guesses:
            doc = self.wiki_lookup[page]['text']
            ref_texts.append(doc)

        best_doc_id = self.reranker.get_best_document(question, ref_texts)
        return guesses[best_doc_id][0]

    def retrieve_answer_document_batch(self, questions: List[str], disable_reranking=False) -> List[str]:
        """Returns the best guessed page that contains the answer to the question."""
        guesses_comb = self.guesser.guess(questions, max_n_guesses=self.max_n_guesses)
        guesses_comb = np.asarray(guesses_comb)
        batch_size = len(questions)
        if disable_reranking:
            max_inds = np.expand_dims(np.argmax(np.asarray(guesses_comb)[:, :, 1], axis=-1), axis=-1)
            best_pages = guesses_comb[np.arange(batch_size)[:, None], max_inds, :].reshape(batch_size, 2)[:, 0]
            return best_pages

        ref_texts = []
        for guesses in guesses_comb:
            for page, score in guesses:
                doc = self.wiki_lookup[page]['text']
                ref_texts.append(doc)
        best_doc_ids = self.reranker.get_best_document_batch(questions, ref_texts)
        return guesses_comb[np.arange(batch_size)[:, None], best_doc_ids.unsqueeze(1), :].reshape(batch_size, 2)[:, 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--train_data", default="data/qanta.train.2018.json", type=str)
    # parser.add_argument("--dev_data", default="data/qanta.dev.2018.json", type=str)
    parser.add_argument("--train_data", default="data/squad1.1/train-v1.1.json", type=str)
    # parser.add_argument("--train_data", default="data/qanta.train.2018.json", type=str)
    #data/squad1.1/train-v1.1.json data/qanta.train.2018.json
    parser.add_argument("--dev_data", default="data/squad1.1/dev-v1.1.json", type=str) 
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--split_rule", default="full", type=str)
    parser.add_argument("--scaling_param", default=1.0, type=float)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--show_confusion_matrix", default=True, type=bool)
    parser.add_argument("--train_guesser", action="store_true")
    parser.add_argument("--train_extractor", action="store_true")
    parser.add_argument("--train_reranker", action="store_true")
    parser.add_argument("--reranker_first_sent", action="store_true")

    flags = parser.parse_args()
    
    if flags.train_guesser:
        print("Loading %s" % flags.train_data)
        guesstrain = QantaDatabase(flags.train_data)
        guessdev = QantaDatabase(flags.dev_data)

        guesser = Guesser()
        guesser.load(from_checkpoint=False)
        guesser.finetune(guesstrain, flags.batch_size, flags.learning_rate, flags.split_rule, flags.scaling_param, limit=flags.limit)
        guesser.train(guesstrain, flags.split_rule)
        guesser.build_faiss_index()

        if flags.show_confusion_matrix:
            confusion = guesser.confusion_matrix(guessdev, limit=-1)
            print("Errors:\n=================================================")
            for ii in confusion:
                for jj in confusion[ii]:
                    if ii != jj:
                        print("%i\t%s\t%s\t" % (confusion[ii][jj], ii, jj))

    elif flags.train_extractor:
        extractor = AnswerExtractor()
        extractor.load('csarron/bert-base-uncased-squad-v1')
        extractor.train()

    elif flags.train_reranker:
        reranker = ReRanker()
        reranker.load('amberoad/bert-multilingual-passage-reranking-msmarco')
        reranker.train(flags.reranker_first_sent)

    else:
        print('Why did you not give me anything to train?')