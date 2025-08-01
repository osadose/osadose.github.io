Hi there, I'm Ose


Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\osadoo\Documents\GitHub\streanlit\.venv\Scripts\streamlit.exe\__main__.py", line 6, in <module>
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\click\core.py", line 1442, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\click\core.py", line 1363, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\click\core.py", line 1830, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\click\core.py", line 1226, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\click\core.py", line 794, in invoke
    return callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\streamlit\web\cli.py", line 233, in main_run
    _main_run(target, args, flag_options=kwargs)
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\streamlit\web\cli.py", line 269, in _main_run
    bootstrap.run(file, is_hello, args, flag_options)
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\streamlit\web\bootstrap.py", line 430, in run
    asyncio.run(run_server())
  File "c:\ONSapps\My_Spyder\Lib\asyncio\runners.py", line 190, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "c:\ONSapps\My_Spyder\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\ONSapps\My_Spyder\Lib\asyncio\base_events.py", line 654, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\streamlit\web\bootstrap.py", line 418, in run_server     
    await server.start()
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\streamlit\web\server\server.py", line 262, in start      
    start_listening(app)
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\streamlit\web\server\server.py", line 129, in start_listening
    start_listening_tcp_socket(http_server)
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\streamlit\web\server\server.py", line 188, in start_listening_tcp_socket
    http_server.listen(port, address)
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\tornado\tcpserver.py", line 183, in listen
    sockets = bind_sockets(
              ^^^^^^^^^^^^^
  File "c:\Users\osadoo\Documents\GitHub\streanlit\.venv\Lib\site-packages\tornado\netutil.py", line 162, in bind_sockets
    sock.bind(sockaddr)
PermissionError: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions
with st.expander("ℹ️ Help"):
    st.markdown("""
    - Make sure your file includes both the free-text occupation field and the ISCO code.
    - This tool uses a simulated classification for demo purposes.  
    - Actual deployment will include OpenAI-based or ML-trained classification.
    """)

