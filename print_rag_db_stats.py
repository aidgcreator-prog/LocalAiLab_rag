# Print the number of ingested files and chunks in the RAG Chroma DB
from ragsub_agent.tools import get_rag_index_summary

def print_rag_db_stats(project="AIM"):
    summary = get_rag_index_summary(project=project)
    total_files = len(summary.get("files", {}))
    total_chunks = summary.get("total_chunks", 0)
    print(f"Project: {project}")
    print(f"Files ingested: {total_files}")
    print(f"Chunks in DB: {total_chunks}")

if __name__ == "__main__":
    print_rag_db_stats()
