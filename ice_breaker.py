from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

information = """
    Elon Musk is a South African-born American entrepreneur and businessman who founded X.com in 1999 (which later became PayPal), SpaceX in 2002 and Tesla Motors in 2003.
    Musk became a multimillionaire in his late 20s when he sold his start-up company, Zip2, to a division of Compaq Computers.
"""

if __name__ == "__main__":
    print("Hello LangChain!")

    summary_template = """
        Given the information {information} about a person from I want you to create:
        1. A short summary of the person
        2. Two interesting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOllama(temperature=0, model="mistral")

    chain = summary_prompt_template | llm | StrOutputParser()

    res = chain.invoke(input={"information": information})

    print(res)