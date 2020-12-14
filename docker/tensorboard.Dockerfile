FROM python:3.7

RUN pip install tb-nightly==2.5.0a20201214

WORKDIR /logs

ENTRYPOINT ["tensorboard", "--logdir", "/logs", "--bind_all"]
CMD []