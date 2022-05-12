import pickle
import torch
import torch.nn.functional as F
import random
import os
from qbdata import QantaDatabase


class GuessTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data: QantaDatabase, tokenizer, wiki_lookup, dataset_name, split_rule: str='full', limit: int=-1):
        super(GuessTrainDataset, self).__init__()
        self.split_rule = split_rule

        if dataset_name == 'train':
            raw_questions = data.train_questions
            questions = [x.text for x in data.train_questions]
            answers = [x.page for x in data.train_questions]
        else:
            raw_questions = data.dev_questions
            questions = [x.text for x in data.dev_questions]
            answers = [x.page for x in data.dev_questions]


        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        if split_rule != 'full':
            split_nick = 'last'
        else:
            split_nick = 'full'

        if os.path.isfile(f'data/{dataset_name}_questions_{split_nick}.pkl'):
            print('Loading already-prepared dataset') 
            with open(f"data/{dataset_name}_questions_{split_nick}.pkl", "rb") as fp:
                self.questions = pickle.load(fp)
            with open(f"data/{dataset_name}_pages_{split_nick}.pkl", "rb") as fp:
                self.answer_pages = pickle.load(fp)
            with open(f"data/{dataset_name}_answers_{split_nick}.pkl", "rb") as fp:
                self.answers = pickle.load(fp)
        else:
            self.questions = []
            self.answer_pages = []
            self.answers = []
            print(f'Preparing dataset with {len(questions)}, {len(raw_questions)}, {len(answers)} entries')
            for i, (question_text, question_obj, page) in enumerate(zip(questions, raw_questions, answers)):
                #if len(question_obj.sentences) >= 3:
                if split_nick == 'full':
                    question_tok = tokenizer(question_text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')["input_ids"]      
                    self.questions.append(question_tok)
                else:
                    question_toks = []
                    for sentence in question_obj.sentences:
                        question_toks.append(tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True, padding='max_length')["input_ids"])
                    self.questions.append(torch.stack(question_toks))
                
                answer_tok = tokenizer(wiki_lookup[page]['text'].replace(page.replace('_',' '), ' '), return_tensors="pt", max_length=512, truncation=True, padding='max_length')["input_ids"]

                self.answer_pages.append(page)
                self.answers.append(answer_tok)

                if i % 1000 == 0:
                    print(f'Tokenized {i} questions and answers', flush=True)

            print(f'Dumping dataset with {len(self.questions)}, {len(self.answer_pages)}, {len(self.answers)} entries')
            with open(f"data/{dataset_name}_questions_{split_nick}.pkl", "wb") as fp:
                pickle.dump(self.questions, fp)        
            with open(f"data/{dataset_name}_pages_{split_nick}.pkl", "wb") as fp:
                pickle.dump(self.answer_pages, fp)        
            with open(f"data/{dataset_name}_answers_{split_nick}.pkl", "wb") as fp:
                pickle.dump(self.answers, fp)


    def __len__(self):
        return len(self.questions)


    def __getitem__(self, index):
        if self.split_rule == 'full':
            question = self.questions[index]
            out = {'question': question.squeeze(), 'answer_text': self.answers[index].squeeze(), 'answer_page': self.answer_pages[index]}
        elif self.split_rule == 'last-a':
            #question = self.questions[index][random.randint(0, len(self.questions[index])-1)].squeeze()
            questions = self.questions[index].squeeze()
            if questions.dim() == 1:
                questions = torch.unsqueeze(questions, 0)
            if questions.shape[0] > 3:
                questions = questions[-3:]
            valid_indices = ','.join([str(num) for num in range(questions.shape[0])])
            questions = F.pad(questions, (0, 0, 0, 3 - questions.shape[0]), 'constant', 0)
            out = {'question': questions, 'valid_indices': valid_indices, 'answer_text': self.answers[index].squeeze(), 'answer_page': self.answer_pages[index]}
        elif self.split_rule == 'last-b':
            questions = self.questions[index].squeeze()
            if questions.dim() == 1:
                questions = torch.unsqueeze(questions, 0)
            if questions.shape[0] > 3:
                questions = questions[-3:]
            valid_indices = ','.join([str(num) for num in range(questions.shape[0])])
            questions = F.pad(questions, (0, 0, 0, 3 - questions.shape[0]), 'constant', 0)
            out = {'question': questions, 'valid_indices': valid_indices, 'answer_text': self.answers[index].squeeze(), 'answer_page': self.answer_pages[index]}
        elif self.split_rule == 'last-c':
            questions = self.questions[index].squeeze()
            if questions.dim() == 1:
                questions = torch.unsqueeze(questions, 0)
            if questions.shape[0] > 3:
                questions = questions[-3:]
            valid_indices = ','.join([str(num) for num in range(questions.shape[0])])
            questions = F.pad(questions, (0, 0, 0, 3 - questions.shape[0]), 'constant', 0)
            out = {'question': questions, 'valid_indices': valid_indices, 'answer_text': self.answers[index].squeeze(), 'answer_page': self.answer_pages[index]}
        elif self.split_rule == 'last-d':
            questions = self.questions[index].squeeze()
            if questions.dim() == 1:
                questions = torch.unsqueeze(questions, 0)
            if questions.shape[0] > 10:
                questions = questions[-10:]
            valid_indices = ','.join([str(num) for num in range(questions.shape[0])])
            questions = F.pad(questions, (0, 0, 0, 10 - questions.shape[0]), 'constant', 0)
            out = {'question': questions, 'valid_indices': valid_indices, 'answer_text': self.answers[index].squeeze(), 'answer_page': self.answer_pages[index]}
        else:
            questions = self.questions[index].squeeze()
            if questions.dim() == 1:
                questions = torch.unsqueeze(questions, 0)
            out = {'question': questions, 'answer_text': self.answers[index].squeeze(), 'answer_page': self.answer_pages[index]}
            

        return out
