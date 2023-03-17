FROM  python
COPY server.py /root


CMD [ "python", "/root/server.py" ]
#ENTRYPOINT ["/bin/sh","python","'/root/main.py'"]