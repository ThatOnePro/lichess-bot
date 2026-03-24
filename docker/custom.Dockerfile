ARG VARIANT
FROM python:3$VARIANT

ENV LICHESS_BOT_DOCKER="true"
ENV PYTHONDONTWRITEBYTECODE=1

ARG LICHESS_DIR=/lichess-bot
WORKDIR $LICHESS_DIR

COPY . .

RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Start the container by ensuring executable permissions on the engines folder first,
# then launch the bot script. Errors from chmod (e.g., empty directory) are ignored.
CMD sh -c "chmod -R +x ./engines/* 2>/dev/null || true && python3 lichess-bot.py ${OPTIONS} --disable_auto_logging"
