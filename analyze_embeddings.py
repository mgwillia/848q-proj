import argparse
import faiss
import os
import torch
import numpy as np

from transformers import DPRContextEncoder, DPRQuestionEncoder, BertTokenizerFast

from guess_train_dataset import GuessTrainDataset
from qbdata import QantaDatabase, WikiLookup

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", default="data/qanta.train.2018.json", type=str)
    parser.add_argument("--dev_data", default="data/qanta.dev.2018.json", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--split_rule", default="full", type=str)
    parser.add_argument("--scaling_param", default=1.0, type=float)

    flags = parser.parse_args()

    ## initialize both encoders 
    question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
    context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    wiki_lookup = WikiLookup('data/wiki_lookup.2018.json')

    ## get train dataset
    guesstrain = QantaDatabase(flags.train_data)
    guessdev = QantaDatabase(flags.dev_data)
    dev_dataset = GuessTrainDataset(guessdev, tokenizer, wiki_lookup, 'dev', 'analysis')
    train_pages = [x.page for x in guesstrain.train_questions]
    print(len(train_pages), len(train_dataset), len(guesstrain.train_questions))
    page_map = {}
    for idx, page in enumerate(train_pages):
        if page not in page_map:
            page_map[page] = []
        page_map[page].append(idx)
    train_page_idxs_unique = []
    for key, value in page_map.items():
        train_page_idxs_unique.append(value[0])
    train_page_idxs_unique = np.array(train_page_idxs_unique)

    ## encode train contexts, keep track of page correspondence
    DIMENSION = 768
    if 'context_embeddings.pth.tar' not in os.listdir('models'):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=4, batch_size=flags.batch_size, pin_memory=True, drop_last=False, shuffle=False)

        for parameter in context_model.parameters():
            parameter.requires_grad = False
        context_model.eval()
        context_embeddings = torch.zeros((len(train_dataset), DIMENSION))
        print('Computing context embeddings', flush=True)
        with torch.no_grad():
            for i, batch in enumerate(train_dataloader):
                answers = batch['answer_text'].to(device)
                context_embeddings[i*flags.batch_size:min(len(train_dataset), (i+1)*flags.batch_size)] = context_model(answers).pooler_output
        torch.save(context_embeddings, 'models/context_embeddings.pth.tar')

    ## build faiss index with train contexts
    context_embeddings = torch.load('models/context_embeddings.pth.tar').numpy()[train_page_idxs_unique]
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(context_embeddings)

    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, num_workers=0, batch_size=1, pin_memory=True, drop_last=False, shuffle=False)
    for parameter in context_model.parameters():
        parameter.requires_grad = False
    context_model.eval()
    for parameter in question_model.parameters():
        parameter.requires_grad = False
    question_model.eval()
    print('Computing both embeddings for dev', flush=True)
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader):
            question, answer, page = batch['question'].to_device, batch['answer_text'].to(device), batch['answer_page']
            clue_embeddings = []
            for j in range(question.shape[0]):
                clue_embeddings.append(question_model(last_sentences).pooler_output)
            answer_embedding = context_model(answers).pooler_output
            
            inter_clue_dist_sum = 0.0
            clue_ctx_dist_sum = 0.0
            for j in range(len(clue_embeddings)):
                clue_embeddings += torch.inner(clue_embeddings[j], answer_embedding)
                for k in range(len(clue_embeddings)):
                    if j != k:
                        inter_clue_dist_sum += torch.inner(clue_embeddings[j], clue_embeddings[k])

            ### IDEALLY, WE MINIMIZE DISTANCE TO CONTEXT
            ### WHILE OUR LOSS MAKES CLUES CLOSER, THE REAL NET EFFECT SHOULD BE MOVING CLOSER TO CONTEXT
            ### SO, WHAT DOES IT DO? ONLY MOVE CLUES CLOSER? OR AFFECT CONTEXT ALSO?
            avg_inter_clue_dist = inter_clue_dist_sum / ((len(clue_embeddings) - 1) ** 2)
            avg_clue_ctx_dist = clue_ctx_dist_sum / len(clue_embeddings)

"""            
            neighbor_scores, neighbor_indices = index.search(last_sentence_embeddings.cpu().numpy(), 1)
            for i in range(last_sentence_embeddings.shape[0]):
                print(neighbor_scores[i])
                pred_page = train_pages[train_page_idxs_unique[neighbor_indices[i][0]]]
                gt_page = gt_pages[i]
                if pred_page == gt_page:
                    num_correct_preds += 1
"""

    ## accuracy
    print(f'Top-1 Retrieval accuracy: {num_correct_preds / len(dev_dataset)}')
