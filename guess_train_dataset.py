import pickle
import torch
import random
import os
from qbdata import QantaDatabase


class GuessTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data: QantaDatabase, tokenizer, wiki_lookup, dataset_name, limit: int=-1):
        super(GuessTrainDataset, self).__init__()

        questions = [x.text for x in data.guess_train_questions]
        answers = [x.page for x in data.guess_train_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        if os.path.isfile(f'data/{dataset_name}_questions.pkl'):
            print('Loading already-prepared dataset') 
            with open(f"data/{dataset_name}_questions.pkl", "rb") as fp:
                self.questions = pickle.load(fp)
            with open(f"data/{dataset_name}_pages.pkl", "rb") as fp:
                self.answer_pages = pickle.load(fp)
            with open(f"data/{dataset_name}_answers.pkl", "rb") as fp:
                self.answers = pickle.load(fp)
        else:
            self.questions = []
            self.answer_pages = []
            self.answers = []
            print(f'Preparing dataset with {len(questions)} entries')
            for i, (question, page) in enumerate(zip(questions, answers)):
                question_tok = tokenizer(question, return_tensors="pt", max_length=512, truncation=True, padding='max_length')["input_ids"]
                answer_tok = tokenizer(wiki_lookup[page]['text'].replace(page.replace('_',' '), ' '), return_tensors="pt", max_length=512, truncation=True, padding='max_length')["input_ids"]

                self.questions.append(question_tok)
                self.answer_pages.append(page)
                self.answers.append(answer_tok)

                if i % 1000 == 0:
                    print(f'Tokenized {i} questions and answers', flush=True)

            with open(f"data/{dataset_name}_questions.pkl", "wb") as fp:
                pickle.dump(self.questions, fp)        
            with open(f"data/{dataset_name}_pages.pkl", "wb") as fp:
                pickle.dump(self.answer_pages, fp)        
            with open(f"data/{dataset_name}_answers.pkl", "wb") as fp:
                pickle.dump(self.answers, fp)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        out = {'question': self.questions[index][random.randint(0, len(self.questions[index])-1)].squeeze(), 'answer_text': self.answers[index].squeeze(), 'answer_page': self.answer_pages[index]}

        return out