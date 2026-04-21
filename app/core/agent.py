"""
Núcleo do agente, aqui fica toda a logica de conversa.
O orquestrador central. Gerencia o ciclo de vida da conversa,
comunicação com a Anthropic, streaming e persistência.
"""

from anthropic import Anthropic
from typing import Optional
from app import settings
from app import Session
from app import ConversationHistory
from app import SessionStore
from app import get_basic_system_prompt


class AlbertAgent:

    def __init__(self, session_id: Optional[str] = None):
        # Inicializa o cliente oficial da Anthropic
        self.client = Anthropic(api_key=settings.anthropic_api_key)

        # Camada de persistência em disco
        self.store = SessionStore()

        # Busca o system prompt (DNA do Albert) via builder
        self.system_prompt = get_basic_system_prompt()

        # Lógica de carregamento de sessão: recupera do disco ou cria uma nova
        if session_id and self.store.exists(session_id):
            self.session = self.store.load(session_id)
        else:
            self.session = Session()

        # Inicializa o gerenciador de memória RAM (histórico)
        self.history = ConversationHistory(self.session)

    def chat(self, user_input: str) -> str:
        try:
            # Registra a fala do usuário
            self.history.add_user_message(user_input)

            # Chama a API da Anthropic
            with self.client.messages.stream(
                    model=settings.model_name,
                    max_tokens=settings.max_tokens,
                    system=self.system_prompt,
                    messages=self.history.get_messages_for_api(),
            ) as stream:
                full_response = stream.get_final_text()

            # Registra a resposta do Albert e persiste no disco
            self.history.add_assistant_message(full_response)
            self.store.save(self.session)

            return full_response

        except Exception as e:
            raise RuntimeError(f"Erro ao chamar a API da Anthropic: {e}") from e

    def chat_stream(self, user_input: str) -> str:
        try:
            # Registra a fala do usuário
            self.history.add_user_message(user_input)

            full_response = ""

            # Inicia o streaming de texto
            with self.client.messages.stream(
                    model=settings.model_name,
                    max_tokens=settings.max_tokens,
                    system=self.system_prompt,
                    messages=self.history.get_messages_for_api(),
            ) as stream:
                print("\nAlbert: ", end="", flush=True)
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_response += text
                print("\n")  # Quebra de linha ao finalizar a resposta

            # Salva a resposta completa na memória e no disco
            self.history.add_assistant_message(full_response)
            self.store.save(self.session)

            return full_response

        except Exception as e:
            raise RuntimeError(f"Erro no streaming da API: {e}") from e

    def reset(self) -> None:
        self.session = Session()
        self.history = ConversationHistory(self.session)
        print(f"[SYSTEM] Sessão resetada. Novo ID: {self.session.id}")

    def get_session_id(self) -> str:
        # Retorna o identificador único da sessão ativa.
        return self.session.id