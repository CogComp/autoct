import hashlib
import json
import os
from typing import List

import duckdb
import numpy as np
import tqdm
from Bio import Entrez

CHUNK_SIZE = 1000
TMP = ".cache/pubmed_meta"
TMP_JSONL = TMP + "/pubmed_meta.jsonl"
TMP_PARQUET = TMP + "/pubmed_meta.parquet"

Entrez.api_key = os.getenv("ENTREZ_API_KEY")
Entrez.email = os.getenv("ENTREZ_EMAIL")


def _pull_pubmed_chunk(i, chunk) -> str:
    key = json.dumps(sorted(chunk))
    hex = hashlib.sha256(key.encode()).hexdigest()

    file_path = f"{TMP}/{i}_{hex}.json"
    if os.path.exists(file_path):
        print(f"Found existing key: {key[:50]}...")
        return file_path

    with Entrez.efetch(db="pubmed", id=",".join(chunk)) as h:
        result = Entrez.read(h)
        with open(file_path, "w") as f:
            json.dump(result, f)

    return file_path


def _meta_to_jsonl(paths):
    with open(TMP_JSONL, "w") as jslf:
        for fn in tqdm.tqdm(os.listdir(TMP)):
            if not fn.endswith("json"):
                continue

            path = f"{TMP}/" + fn

            if path not in paths:
                continue

            with open(path, "r") as f:
                print(f"Processing {path}")
                data = json.load(f)
                articles = data["PubmedArticle"]
                book_articles = data["PubmedBookArticle"]
                jslf.writelines(json.dumps(a) + "\n" for a in articles)
                jslf.writelines(json.dumps(a) + "\n" for a in book_articles)


def _get_current_pulled(file: str):
    q = f"""
    select coalesce(MedlineCitation.PMID, BookDocument.PMID) as pmid from {file}
    """
    return set(duckdb.query(q).to_df()["pmid"])


def upsert_pub_med(ids: List[str], output: str):
    os.makedirs(TMP, exist_ok=True)

    current_pulled = _get_current_pulled(output)
    to_pull = [id for id in ids if id not in current_pulled]
    print(f"Need to pull {len(to_pull)}")
    if len(to_pull) == 0:
        print("Nothing to pull")
        return
    chunk_size = min(CHUNK_SIZE, len(to_pull))

    chunks = np.array_split(to_pull, len(to_pull) // chunk_size)
    print(f"Fetching {len(to_pull)} -- {len(chunks)} Chunks")
    all_paths = []
    for i, c in tqdm.tqdm(enumerate(chunks)):
        path = _pull_pubmed_chunk(i, c)
        all_paths.append(path)

    print(f"Converting to jsonl...")
    _meta_to_jsonl(all_paths)

    current_jsonl_path = f"{TMP}/{output}.jsonl"
    duckdb.execute(
        f"""
        copy (
            select * from
            '{output}'
        ) to '{current_jsonl_path}';
        """
    )
    print(f"Appending {current_jsonl_path} to {TMP_JSONL}...")

    with open(TMP_JSONL, "a") as jslf, open(current_jsonl_path, "r") as currf:
        for line in tqdm.tqdm(currf):
            jslf.write(line)

    print(f"Appended {current_jsonl_path} to {TMP_JSONL}...")

    output_path = output.replace(".parquet", "_new.parquet")

    duckdb.execute(
        f"""
    copy
    (
        with raw as (select * from read_json("{TMP_JSONL}", format = 'newline_delimited', sample_size = -1))
        select distinct * from raw
    )
    to '{output_path}'
    (format 'parquet', codec 'zstd');
    """
    )

    print(f"Created {output_path}")
