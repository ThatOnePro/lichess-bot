import logging
import threading

import requests
import torch

logger = logging.getLogger(__name__)


class AIChatHandler:
    """Handles AI logic using either a local Ollama instance or a Hugging Face model."""

    def __init__(self, game, engine):
        self.game = game
        self.engine = engine
        self.history = []  # List of messages: [{"role": "user", "content": "..."}, ...]

        # Get AI configuration
        self.config = game.config.ai_chat
        self.enabled = self.config.enabled
        self.model = self.config.model
        self.ollama_url = self.config.url

        self.generator = None

        if not self.enabled:
            return

        # If using Ollama, we don't need to load a model locally in this process
        try:
            logger.info("Trying to connect to Ollama...")

            # Quick check if Ollama is available
            response = requests.get(self.ollama_url + "/api/tags", timeout=2)
            if response.status_code == 200:
                logger.info(f"Ollama connected successfully. Using model: {self.model}")
                self.use_ollama = True
            else:
                self.use_ollama = False
        except Exception:
            self.use_ollama = False

        if not self.use_ollama:
            self._setup_transformers()

    def _setup_transformers(self):
        """Fallback to Hugging Face transformers if Ollama is not available."""
        from transformers import pipeline
        model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
        logger.info(f"Ollama not found. Loading local HF Model: {model_id}...")

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            self.generator = pipeline("text-generation",
                                      model=model_id,
                                      device=device)
            logger.info("HF Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load HF model: {e}")
            self.generator = None

    def get_ai_response(self, user_text, callback):
        """Starts a thread to generate the AI response."""
        if not self.enabled:
            return

        if not self.use_ollama and not self.generator:
            callback("My brain is currently disconnected.")
            return

        thread = threading.Thread(target=self._generate, args=(user_text, callback))
        thread.start()

    def _generate(self, user_text, callback):
        try:
            # 1. Get game context
            stats = self.engine.get_stats(for_chat=True)
            score = stats[0] if stats else "unknown"

            # 2. Build system prompt
            system_prompt = (
                "You are a witty chess bot on Lichess. "
                f"The current game evaluation is {score}. "
                "Keep your answers very short (max 2 sentences). "
                "Be slightly arrogant if you are winning, but helpful if the user asks for advice."
            )

            if self.use_ollama:
                self._generate_ollama(user_text, system_prompt, callback)
            else:
                self._generate_transformers(user_text, system_prompt, callback)

        except Exception as e:
            logger.error(f"Error during AI generation: {e}")
            callback("Interesting move... I need to think about that.")

    def _generate_ollama(self, user_text, system_prompt, callback):
        # Update history
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": system_prompt}] + self.history[-5:]

        # Build prompt for Ollama
        full_prompt = ""
        for m in messages:
            full_prompt += f"{m['role'].upper()}: {m['content']}\n"
        full_prompt += "ASSISTANT:"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": 50,
                "temperature": 0.7
            }
        }

        response = requests.post(self.ollama_url + "/api/chat", json=payload, timeout=10)
        if response.status_code == 200:
            ai_reply = response.json().get("response", "").strip()
            self.history.append({"role": "assistant", "content": ai_reply})
            callback(ai_reply)
        else:
            logger.error(f"Ollama API error: {response.text}")
            callback("I'm lost for words...")

    def _generate_transformers(self, user_text, system_prompt, callback):
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": system_prompt}] + self.history[-5:]

        prompt = self.generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.generator(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

        ai_reply = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
        self.history.append({"role": "assistant", "content": ai_reply})

        callback(ai_reply)
