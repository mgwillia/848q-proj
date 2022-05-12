import argparse
# import colored_traceback
# colored_traceback.add_hook()

import json
from tqdm import tqdm
from eval_utils import compute_em, compute_f1
from qbdata import QantaDatabase, Question
from qb_system import QuizBowlSystem

from typing import Iterable, Mapping

GUESSER_MODEL_PATH = 'models/tfidf.pickle'
BATCH_SIZE = 256  # Change this to 24 if using reranking and are on a 1080ti


def save_json(json_object: Mapping, filename: str):
    with open(f'outputs/{filename}', 'w') as fp:
        json.dump(pred_dict, fp, indent=4)


def compute_em_multiple_answers(answers_gold: Iterable[str], answer_pred: str):
    return max(compute_em(a_gold, answer_pred) for a_gold in answers_gold)


def compute_f1_multiple_answers(answers_gold: Iterable[str], answer_pred: str):
    return max(compute_f1(a_gold, answer_pred) for a_gold in answers_gold)


def generate_answers(answer_text: str):
    answer_text = answer_text.lower()
    if '[' in answer_text:
        answers = set()
        main_answer, remaining = answer_text.split('[', 1)
        answers.add(main_answer.strip())
        phrases = remaining.split(';')
        for p in phrases:
            p = p.strip()
            if not p.startswith('or '):
                continue

            p = p[3:]
            i, j = 0, len(p) - 1

            while i <= j and p[i] in {'"', '\'', '[', ' '}:
                i += 1

            while i <= j and p[i] in {'"', '\'', ']', ' '}:
                j -= 1
            answers.add(p[i:j + 1])
        return answers
    return {answer_text.strip()}


def generate_first_sent_predictions_batch(model: QuizBowlSystem, questions: Iterable[Question]):
    pred_dict = {}
    for i in tqdm(range(0, len(questions), BATCH_SIZE)):
        ques = questions[min(i, len(questions) - 1): min(i + BATCH_SIZE, len(questions))]
        first_sentences = list(map(lambda x: x.first_sentence, ques))
        answers = model.execute_query_batch(first_sentences, get_page=True)
        for i, q in enumerate(ques):
            pred_dict[q.qanta_id] = {
                # 'answer': answers[i][0],
                'page': answers[i]
            }
    return pred_dict


def generate_last_sent_predictions_batch(model: QuizBowlSystem, questions: Iterable[Question]):
    pred_dict = {}
    for i in tqdm(range(0, len(questions), BATCH_SIZE)):
        ques = questions[min(i, len(questions) - 1): min(i + BATCH_SIZE, len(questions))]
        first_sentences = list(map(lambda x: x.sentences[-1], ques))
        answers = model.execute_query_batch(first_sentences, get_page=True)
        for i, q in enumerate(ques):
            pred_dict[q.qanta_id] = {
                # 'answer': answers[i][0],
                'page': answers[i]
            }
    return pred_dict


def compute_metrics(prediction_dict: Mapping, questions: Iterable[Question]):
    N = 0
    # em = 0
    # f1 = 0
    ret_accuracy = 0

    for ques in questions:
        N += 1
        # answers_gold = generate_answers(ques.answer)
        if ques.qanta_id in prediction_dict:
            # a_pred = prediction_dict[ques.qanta_id]['answer']
            p_pred = prediction_dict[ques.qanta_id]['page']
            print(ques.first_sentence)
            print(ques.sentences[-1])
            print(p_pred, ques.page)
            print('')
            # em += compute_em_multiple_answers(answers_gold, a_pred)
            # f1 += compute_f1_multiple_answers(answers_gold, a_pred)
            ret_accuracy += 1 if ques.page == p_pred else 0

    if N > 0:
        # em /= N
        # f1 /= N
        ret_accuracy /= N

    return {
        # 'em': em,
        # 'f1': f1,
        'ret_accuracy': ret_accuracy
    }


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--guesser_model", help="Pickle file path for TfidfGuesser model.",
                           type=str, default="models/tfidf.pickle")
    argparser.add_argument("--eval_dataset", help="Dataset Path for the eval dataset. Must be Qanta Json format.",
                           type=str, default="data/qanta.dev.2018.json")
    argparser.add_argument("--train_dataset", help="Dataset Path for the TRAIN dataset. Must be Qanta Json format.",
                           type=str, default="data/qanta.train.2018.json")
    argparser.add_argument('--mode', type=str, choices=["predict", "eval"],
                           help="Only saves the predictions in predict mode. Also computes metrics in eval mode.")

    argparser.add_argument("--first_sent_predictions", help="Output path for json predictions.",
                           type=str, default="first_sent_predictions.json")
    argparser.add_argument("--last_sent_predictions", help="Output path for json predictions.",
                           type=str, default="last_sent_predictions.json")

    argparser.add_argument('--disable-reranking', dest='disable_reranking', action='store_true')

    argparser.add_argument('--debug_run', dest='debug_run', action='store_true',
                           help="Set this to run on small set of examples.")

    argparser.add_argument('--no-debug_run', dest='debug_run', action='store_false')

    args = argparser.parse_args()
    argparser.set_defaults(debug_run=False)

    # Load Dataset
    eval_questions = QantaDatabase(args.eval_dataset).all_questions
    train_questions = QantaDatabase(args.train_dataset).all_questions

    if args.debug_run:
        print('Running only on 20 examples.')  # Change this to suit your iteration speed
        eval_questions = eval_questions[:20]

    # Load the Model
    model = QuizBowlSystem()
    model.answer_extractor.train(train_questions)

    pred_dict = generate_first_sent_predictions_batch(model, eval_questions)
    #pred_dict = json.load(open(args.first_sent_predictions, 'r'))
    save_json(pred_dict, args.first_sent_predictions)
    print('Saved first sentence predictions...\n')

    if args.mode == 'eval':
        metrics = compute_metrics(pred_dict, eval_questions)

        print(f'First Sent Retrieval Accuracy   : {metrics["ret_accuracy"]:.2f}')
        print('')

    pred_dict = generate_last_sent_predictions_batch(model, eval_questions)
    print('Saved last sentence predictions...\n')
    save_json(pred_dict, args.last_sent_predictions)

    if args.mode == 'eval':
        metrics = compute_metrics(pred_dict, eval_questions)

        print(f'Last Sent Retrieval Accuracy   : {metrics["ret_accuracy"]:.2f}')
