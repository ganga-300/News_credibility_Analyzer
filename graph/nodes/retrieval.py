import os
from tavily import TavilyClient
from graph.state import AgentState

def retrieval_node(state: AgentState):
    """
    Retrieves web evidence using the generated search queries via Tavily.
    """
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    retrieved_evidence = {}
    
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found. Skipping retrieval.")
        state["retrieved_evidence"] = {}
        return state
        
    client = TavilyClient(api_key=tavily_api_key)
    
    for claim, queries in state.get("generated_queries", {}).items():
        all_docs = []
        seen_urls = set()
        
        for query in queries:
            try:
                response = client.search(
                    query=query,
                    search_depth="basic",
                    max_results=3
                )
                
                for result in response.get("results", []):
                    url = result.get("url")
                    if url not in seen_urls:
                        seen_urls.add(url)
                        all_docs.append({
                            "content": result.get("content", ""),
                            "url": url,
                            "title": result.get("title", ""),
                            "score": result.get("score", 0.0),
                        })
                        
            except Exception as e:
                print(f"Tavily search failed for query '{query}': {e}")
                
        retrieved_evidence[claim] = all_docs
        
    state["retrieved_evidence"] = retrieved_evidence
    return state