ARG VARIANT
FROM python:3$VARIANT

ENV LICHESS_BOT_DOCKER="true"
ENV PYTHONDONTWRITEBYTECODE=1

ARG LICHESS_DIR=/lichess-bot
WORKDIR $LICHESS_DIR

COPY . .

RUN python3 -m pip install --no-cache-dir -r requirements.txt

CMD python3 lichess-bot.py ${OPTIONS} --disable_auto_logging
