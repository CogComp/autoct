import dspy
import tiktoken
import yaml
from txtai.embeddings import Embeddings

from src.lfe.impl.globals import get_duckdb_ro_con

from .globals import cache

# arxiv_embeddings = Embeddings()
# arxiv_embeddings.load(provider="huggingface-hub", container="neuml/txtai-arxiv")
enc = tiktoken.encoding_for_model("gpt-4o-mini")
EMBED_BATCH_SIZE = 1536
EMBED_DIM = 384

pubmed_embeddings = Embeddings()
pubmed_embeddings.load("./datasets/pubmed-index")

nct_embeddings = Embeddings()
nct_embeddings.load("./datasets/nct-index")


@cache.memoize(tag="cti-v0")
def get_clinical_trial_info_from_clinical_trials_gov_dict(nct_id: str) -> dict:
    local_con = get_duckdb_ro_con()
    try:
        return (
            local_con.sql(
                f"""
                PRAGMA disable_progress_bar;

                select nctId, coalesce(protocolSection.statusModule.startDateStruct.date, strftime(protocolSection.statusModule.studyFirstSubmitDate, '%Y-%m-%d')) as startDate, protocolSection.* exclude (statusModule) from 'datasets/ctg-studies-with-nctid.parquet' where nctId = '{nct_id}'
                """
            )
            .df()
            .to_dict(orient="records")[0]
        )
    except Exception as e:
        print(f"Failed to get CT Info for {nct_id}", e)
        raise e


@cache.memoize(tag="nctemb-v0")
def _search_nct_embeddings(query, start_date):
    return nct_embeddings.search(
        "select text, data from txtai where start_date < :s and similar(:q)",
        limit=10,
        parameters={"q": query, "s": start_date},
    )


@cache.memoize(tag="pmemb-v0")
def _search_pubmed_embeddings(query, start_date):
    return pubmed_embeddings.search(
        "select text, data from txtai where DateAvail < :s and similar(:q)",
        limit=10,
        parameters={"q": query, "s": start_date},
    )


def make_pubmed_search(nct_info):
    start_date = nct_info["startDate"]
    if len(start_date) < 10:
        start_date = start_date + "-01"

    def clean(data: dict):
        data["data"]["AbstractText"] = data["text"]
        data["data"]["AuthorList"] = data["data"]["AuthorList"][:5]
        del data["data"]["ArticlePubmedDataReferenceList"]
        del data["data"]["ArticleAuthorList"]
        del data["data"]["ArticleInvestigatorList"]
        return yaml.dump(data["data"])

    def pubmed_search(query: str):
        """
        Searches the PubMed database.
        Returns the abstracts and metadata of the top 10 most relevant articles published before the current trial for the provided query.

        Repeated searches for the same query will yield the same results.
        """

        data = _search_pubmed_embeddings(query, start_date)
        return "\n\n---------\n\n".join(clean(d) for d in data)

    return pubmed_search


def make_nct_search(
    nct_info,
):
    start_date = nct_info["startDate"]
    if len(start_date) < 10:
        start_date = start_date + "-01"

    def clean(data: dict):
        data_dict = data["data"]
        del data_dict["brief_summary/textblock"]
        del data_dict["brief_summary"]
        del data_dict["detailed_description"]
        data_dict["text"] = data["text"]
        data_dict["outcome"] = "Success" if data_dict["outcome"] == 1 else "Failure"
        return yaml.dump(data_dict)

    def related_trials_nct_search(query: str):
        """
        Searches the National Clinical Trials (NCT) database.
        Returns a summary of the top 10 most relevant trials that took place before the current trial for the provided query

        Repeated searches for the same query will yield the same results.
        """

        data = _search_nct_embeddings(query, start_date)
        return "\n\n---------\n\n".join(clean(d) for d in data)

    return related_trials_nct_search


def get_detailed_nct_info(nctid):
    """
    Get details about a specific clinical trial by its NCT ID.

    Args:
        nctid (string): The NCT ID. E.g. NCT00110279

    Returns:
        Details about a specific clinical trial
    """
    return yaml.dump(get_clinical_trial_info_from_clinical_trials_gov_dict(nctid))


class WebpageSummarizer(dspy.Signature):
    """
    You are a helpful data analyst providing research assistance for a clinical trial success prediction task.

    Given the contents of a webpage and the context and reasoning for visiting it, provide a summary in <500 words in Markdown that best captures the required information.
    """

    content: str = dspy.InputField()
    context_reasoning: str = dspy.InputField()

    summary: str = dspy.OutputField()


summarizer = dspy.ChainOfThought(WebpageSummarizer)


def _num_tokens(string: str) -> int:
    num_tokens = len(enc.encode(string))
    return num_tokens


# @cache.memoize(tag="v0")
# def web_search(query: str) -> str:
#     """Search the web for the given query"""
#     results = DDGS().text(query, max_results=10)
#     return yaml.dump(results)


# @cache.memoize(tag="v0")
# def arxiv_search(query: str) -> str:
#     """Search arXiv with the given query"""
#     return yaml.dump(arxiv_embeddings.search(query, limit=10))


# def pubmed_search(query: str) -> str:
#     """Searches the PubMed database for a given query. Returns the abstracts of the semantically most relevant articles."""
#     embeddings = retrieval_embed_model.get_query_embedding(query)
#     results = retrieval_client.search(
#         retrieval_col_name, query_vector=embeddings, limit=100
#     )

#     def format(res):
#         content = json.loads(res.payload["_node_content"])
#         return f"""
# ---
# {res.payload['document_id']}
# {yaml.dump({
#                 k.replace("MedlineCitation.", ""): v
#                 for k, v in content["metadata"].items()
# })
# }
# ---
# Title: {content["text"]}
#         """

#     return "\n##########\n".join(format(r) for r in results[0:3])

# return [
#     {"pmid": res.payload["document_id"].removeprefix("pmid_"), "score": res.score}
#     for res in results
# ]


# def visit_url(url: str, context_reasoning: str) -> str:
#     """Given a URL and a short bit of reasoning for visiting the site, return the contents of the webpage"""
#     downloaded = fetch_url(url)
#     if downloaded is None:
#         return "The URL was not accessible."
#     result = extract(
#         downloaded, favor_recall=True, output_format="markdown", with_metadata=True
#     )
#     if result is None:
#         return "The page could not be parsed."

#     if _num_tokens(result) > 10000:
#         return summarizer(content=result, context_reasoning=context_reasoning).summary

#     return result
