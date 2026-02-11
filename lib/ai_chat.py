import logging
import threading
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

class AIChatHandler:
    """Handelt de AI-logica af met een lokaal Hugging Face model."""

    def __init__(self, game, engine):
        self.game = game
        self.engine = engine
        self.history = []  # Lijst met berichten: [{"role": "user", "content": "..."}, ...]
        self.model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
        
        logger.info(f"AI Model laden: {self.model_id}...")
        # Gebruik CPU voor stabiliteit tijdens het schaken, of 'mps' voor Mac GPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        try:
            self.generator = pipeline("text-generation", 
                                     model=self.model_id, 
                                     device=device)
            logger.info("AI Model succesvol geladen.")
        except Exception as e:
            logger.error(f"Kon model niet laden: {e}")
            self.generator = None

    def get_ai_response(self, user_text, callback):
        """Start een thread om de AI-respons te genereren."""
        if not self.generator:
            callback("Mijn brein is momenteel niet verbonden.")
            return

        thread = threading.Thread(target=self._generate, args=(user_text, callback))
        thread.start()

    def _generate(self, user_text, callback):
        try:
            # 1. Context van het spel ophalen
            fen = self.game.state.get("moves", "Startpositie")
            stats = self.engine.get_stats(for_chat=True)
            score = stats[0] if stats else "onbekend"
            
            # 2. Systeem prompt opbouwen voor persoonlijkheid
            system_prompt = (
                "You are a witty chess bot on Lichess. "
                f"The current game evaluation is {score}. "
                "Keep your answers very short (max 2 sentences). "
                "Be slightly arrogant if you are winning, but helpful if the user asks for advice."
            )

            # 3. Geschiedenis bijwerken
            self.history.append({"role": "user", "content": user_text})
            
            # Beperk geschiedenis tot laatste 5 berichten om geheugen te besparen
            messages = [{"role": "system", "content": system_prompt}] + self.history[-5:]

            # 4. Genereren
            prompt = self.generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = self.generator(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            
            ai_reply = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()

            # 5. Geschiedenis bijwerken met antwoord van AI
            self.history.append({"role": "assistant", "content": ai_reply})
            
            callback(ai_reply)
        except Exception as e:
            logger.error(f"Fout bij AI generatie: {e}")
            callback("Interessante zet... ik moet er even over nadenken.")
