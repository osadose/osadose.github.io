Starting embedding of document chunks, please wait...
Traceback (most recent call last):
  File "/workspaces/statschat-ke/statschat/embedding/preprocess.py", line 306, in <module>
    prepper = PrepareVectorStore(**config["db"], **config["preprocess"], logger=log)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/statschat-ke/statschat/embedding/preprocess.py", line 79, in __init__
    self._embed_documents()
  File "/workspaces/statschat-ke/statschat/embedding/preprocess.py", line 242, in _embed_documents
    self.db = FAISS.from_documents(self.chunks, self.embeddings)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/statschat-ke/.venv/lib/python3.12/site-packages/langchain_core/vectorstores/base.py", line 852, in from_documents
    return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/statschat-ke/.venv/lib/python3.12/site-packages/langchain_community/vectorstores/faiss.py", line 1044, in from_texts
    return cls.__from(
           ^^^^^^^^^^^
  File "/workspaces/statschat-ke/.venv/lib/python3.12/site-packages/langchain_community/vectorstores/faiss.py", line 1001, in __from
    index = faiss.IndexFlatL2(len(embeddings[0]))
                                  ~~~~~~~~~~^^^
IndexError: list index out of range
Error occurred while running /workspaces/statschat-ke/statschat/embedding/preprocess.py (Exit Code: 1)
Traceback (most recent call last):
  File "/workspaces/statschat-ke/statschat/pdf_runner.py", line 71, in <module>
    run_script(EMBEDDING_DIR / "preprocess.py")
  File "/workspaces/statschat-ke/statschat/pdf_runner.py", line 44, in run_script
    subprocess.run([sys.executable, script_name], check=True)
  File "/usr/lib/python3.12/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/workspaces/statschat-ke/.venv/bin/python3', PosixPath('/workspaces/statschat-ke/statschat/embedding/preprocess.py')]' returned non-zero exit status 1.
(.venv) @osadose ➜ /workspaces/statschat-ke (main) $ 
