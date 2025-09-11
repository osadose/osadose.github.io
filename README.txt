Starting pdf_runner.py...
Pipeline mode: UPDATE

Executing full UPDATE pipeline...

Running script: /workspaces/statschat-ke/statschat/pdf_processing/pdf_downloader.py
STARTING DATABASE UPDATE. PLEASE WAIT...
No existing url_dict.json found in pdf_downloads. Exiting update mode.
Finished: /workspaces/statschat-ke/statschat/pdf_processing/pdf_downloader.py

Running script: /workspaces/statschat-ke/statschat/pdf_processing/pdf_to_json.py
Running in UPDATE mode: Processing only new PDFs.
No new PDFs to process. Exiting.
Finished: /workspaces/statschat-ke/statschat/pdf_processing/pdf_to_json.py

Running script: /workspaces/statschat-ke/statschat/embedding/preprocess.py
Splitting json conversions. Please wait...
Found 0 articles for splitting, please wait..
Loading to memory. Please wait...
Splitting documents into chunks. Please wait...
Instantiating embeddings. Please wait...
modules.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 349/349 [00:00<00:00, 1.91MB/s]
config_sentence_transformers.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 116/116 [00:00<00:00, 888kB/s]
README.md: 11.6kB [00:00, 35.4MB/s]
sentence_bert_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 53.0/53.0 [00:00<00:00, 348kB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 571/571 [00:00<00:00, 3.67MB/s]
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 438M/438M [00:02<00:00, 170MB/s]
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:00<00:00, 2.20MB/s]
vocab.txt: 232kB [00:00, 23.9MB/s]
tokenizer.json: 466kB [00:00, 80.8MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 239/239 [00:00<00:00, 1.59MB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:00<00:00, 1.12MB/s]
Embedding documents chunks. Please wait...
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
subprocess.CalledProcessError: Command '['/workspaces/statschat-ke/.venv/bin/python', PosixPath('/workspaces/statschat-ke/statschat/embedding/preprocess.py')]' returned non-zero exit status 1.
(.venv) @osadose ➜ /workspaces/statschat-ke (main) $ 
