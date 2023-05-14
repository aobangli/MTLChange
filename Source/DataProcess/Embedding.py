import pandas as pd
from sentence_transformers import SentenceTransformer

from Source.DataProcess.DataProcessConfig import *


def process_text(sentences):
    # sentences = ["This is an example sentence", "Each sentence is converted"]

    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(sentences, device=train_device, show_progress_bar=True)
    print(embeddings.shape)
    return embeddings


def cal_embedding(df, target, row_name):
    print(f"{target} embedding...")

    df[row_name] = df[row_name].fillna('None')

    embeddings = process_text(df[row_name])

    print(f"{target} embedding Complete...")

    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.columns = [f'{target}_emb{i + 1}' for i in range(embeddings.shape[1])]

    id_target_embeddings_df = pd.concat([df['change_id'], df[row_name], embeddings_df], axis=1)
    id_target_embeddings_df.to_csv(f'{root}/{project}_{target}_emb_only.csv', index=False)

    total_df = pd.concat([df, embeddings_df], axis=1)
    return total_df, embeddings_df


def main():
    df = pd.read_csv(f"{root}/{project}.csv")

    _, subject_embedding_df = cal_embedding(df, 'subject', 'subject')
    _, comment_embedding_df = cal_embedding(df, 'comment', 'comment_content')
    _, msg_embedding_df = cal_embedding(df, 'msg', 'msg_content')
    total_df = pd.concat([df, subject_embedding_df, comment_embedding_df, msg_embedding_df], axis=1)

    total_df.to_csv(f'{root}/{project}_total_emb.csv', index=False)




