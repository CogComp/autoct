from .globals import get_openai_client


def canonicalize_affiliations(name):

    prompt_template = """
    Here is the affiliation information of a particular researcher:
    ```
    {name}
    ```
    You'll return a JSON with exactly the following keys:
    ```
    {{
        "university": <a list of universities the researcher is affiliated with>,
        "institutes": <a list of research institutes the researcher is affiliated with>,
        "companies": <a list of companies the researcher is affiliated with>,
        "countries": <the ISO countries the researcher is affiliated with>
    }}
    ```
    """

    fields = {
        "name": name,
    }
    prompt = prompt_template.format(**fields)

    result = get_openai_client().chat.completions.create(
        model="llama3.2:3b",
        # model="meta-llama/Llama-3.2-3B-Instruct"
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You're a helpful assistant returning JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return result.choices
