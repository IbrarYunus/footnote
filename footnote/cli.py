import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="footnote", description="Cited answers from your documents.")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="chunk and embed a folder of markdown")
    ingest.add_argument("folder", type=Path)

    ask = commands.add_parser("ask", help="answer one question in the terminal")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")

    serve = commands.add_parser("serve", help="run the web app")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "ingest":
        from . import index
        from .ingest import chunk_directory

        started = time.perf_counter()
        chunks = chunk_directory(args.folder)
        documents = len({c.doc_path for c in chunks})
        print(f"{len(chunks)} chunks from {documents} documents. Embedding...")
        index.build(chunks)
        print(f"Index written in {time.perf_counter() - started:.1f}s")

    elif args.command == "ask":
        from . import index
        from .answer import answer

        result = answer(args.question, index.load(), mode=args.mode)
        hits = result["passages"]
        for number, lookup in enumerate(result["lookups"], 1):
            print(f"  {number}. {lookup['tool']} {next(iter(lookup['input'].values()))!r} -> {lookup['new']} new passages")
        print()
        cited: list[int] = []
        for block in result["blocks"]:
            marks = ""
            for doc in dict.fromkeys(citation["doc"] for citation in block["citations"]):
                if doc not in cited:
                    cited.append(doc)
                marks += f"[{cited.index(doc) + 1}]"
            print(block["text"] + marks, end="")
        print("\n")
        for number, doc in enumerate(cited, 1):
            heading = f" › {hits[doc]['heading']}" if hits[doc]["heading"] else ""
            print(f"[{number}] {hits[doc]['title']}{heading}  ({hits[doc]['doc_path']})")

    elif args.command == "serve":
        import uvicorn

        uvicorn.run("footnote.server:app", host="127.0.0.1", port=args.port)
