# Technische Documentatie: lichess-bot

Dit document biedt een diepgaande technische uitleg over de werking van de `lichess-bot`.

## Architectuuroverzicht

De bot is ontworpen als een event-driven systeem dat communiceert met de Lichess API via HTTP-streaming. Het maakt gebruik van `multiprocessing` om concurrency te beheren, waardoor de bot meerdere games tegelijkertijd kan spelen en tegelijkertijd naar nieuwe uitdagingen kan luisteren.

### Kerncomponenten

1.  **Entry Point (`lichess-bot.py` & `lib/lichess_bot.py`)**: 
    - Start het programma en beheert de hoofdloop.
    - Configureert logging en laadt de `config.yml`.
    - Initialiseert de verbinding met Lichess.

2.  **Lichess API Wrapper (`lib/lichess.py`)**: 
    - Handelt alle communicatie met Lichess af (OAuth authenticatie, zetten doen, uitdagingen accepteren/weigeren).
    - Maakt gebruik van de `backoff` bibliotheek voor robuuste foutafhandeling bij netwerkproblemen.

3.  **Engine Wrapper (`lib/engine_wrapper.py`)**: 
    - Een abstractielaag bovenop schaakengines (UCI, XBoard of "homemade" Python engines).
    - Beheert de levenscyclus van het engine-proces.

4.  **Game Management (`lib/model.py` & `lib/lichess_bot.py`)**: 
    - Beheert de status van individuele partijen.
    - `play_game` is de hoofdfunctie die per partij in een apart proces draait.

5.  **Matchmaking (`lib/matchmaking.py`)**: 
    - Verantwoordelijk voor het proactief uitdagen van andere bots of spelers op basis van de configuratie.

---

## De Gegevensstroom (Flow)

### 1. Initialisatie
Wanneer de bot start:
- De configuratie wordt gevalideerd.
- De schaakengine wordt getest om te zien of deze correct opstart.
- Er wordt een verbinding gemaakt met Lichess om het profiel op te halen en te verifiëren of het een "BOT" account is.

### 2. Event Stream (`watch_control_stream`)
De bot opent een persistente HTTP-verbinding met `/api/stream/event`. Lichess stuurt via deze stream real-time JSON-events:
- `challenge`: Een nieuwe inkomende uitdaging.
- `gameStart`: Een partij is begonnen (omdat een uitdaging is geaccepteerd of een lopende partij is hervat).

### 3. Uitdagingen Beheren (`handle_challenge`)
Inkomende uitdagingen worden getoetst aan de criteria in `config.yml`:
- Is de variant ondersteund (bijv. Standard, Blitz, Bullet, Atomic)?
- Is de tegenstander niet geblokkeerd?
- Voldoet de bedenktijd aan de ingestelde limieten?
Indien akkoord, wordt de uitdaging geaccepteerd via de API.

### 4. Een Partij Spelen (`play_game`)
Zodra een game start, wordt er een nieuw proces gestart dat de functie `play_game` uitvoert:
1.  **Game Stream**: Er wordt een nieuwe stream geopend voor de specifieke partij (`/api/bot/game/stream/{gameId}`).
2.  **Engine Initialisatie**: Voor elke partij wordt een eigen instantie van de schaakengine gestart.
3.  **Game Loop**:
    - De bot wacht op `gameState` updates van de stream.
    - Als het de beurt is aan de bot:
        - De huidige FEN (position) en zet-historie worden naar de engine gestuurd.
        - De engine berekent de beste zet (`go` commando).
        - De zet wordt naar Lichess gestuurd via `/api/bot/game/{gameId}/move/{move}`.
    - De bot reageert op chatberichten, remise-aanbiedingen en takeback-verzoeken op basis van de configuratie.
4.  **Afsluiting**: Wanneer de partij eindigt, wordt de engine netjes afgesloten en wordt de PGN opgeslagen.

---

## Concurrency & Multiprocessing

Om efficiënt te werken, gebruikt de bot verschillende wachtrijen (`Queues`) en processen:

- **Control Queue**: Beheert events van de hoofdstream.
- **Logging Queue**: Zorgt ervoor dat logs van verschillende processen netjes naar de console/file worden geschreven zonder te interfereren.
- **PGN Queue**: Verwerkt het wegschrijven van game-data naar schijf asynchroon.
- **Multiprocessing Pool**: De `max_games` instelling in de config bepaalt de grootte van de pool voor gelijktijdige partijen.

## Belangrijke Bestanden en Hun Functie

| Bestand | Functie |
| :--- | :--- |
| `config.yml` | Gebruikersinstellingen (tokens, engine pad, filters). |
| `lib/lichess.py` | De communicatie-interface met Lichess. |
| `lib/engine_wrapper.py` | Praten met `stockfish` of andere engines. |
| `lib/model.py` | Datastructuren voor `Game`, `Challenge` en `Player`. |
| `lib/conversation.py` | Beheert chat-interacties tijdens de game. |

## Fouttolerantie

- **Netwerkonderbrekingen**: De bot gebruikt `backoff` om verbindingen automatisch te herstellen.
- **Engine Crashes**: Als een engine crasht, probeert de bot deze te herstarten of de partij netjes af te sluiten.
- **Rate Limiting**: De `Lichess` klasse houdt rekening met HTTP 429 errors en introduceert pauzes waar nodig om blokkades te voorkomen.
