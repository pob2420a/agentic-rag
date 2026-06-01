from typing import Any, Dict

from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max_results=5)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    tavil_results = web_search_tool.invoke({"query": question})
    joined_tavily_result = "\n".join([result.content for result in tavil_results])

    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}

if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})