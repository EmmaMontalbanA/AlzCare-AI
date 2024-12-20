from datetime import datetime
import random
from typing import Dict, List, Optional, Tuple
from .knowledge_base import get_most_relevant_response

class Message:
    def __init__(self, content: str, role: str, timestamp: datetime = None):
        self.content = content
        self.role = role
        self.timestamp = timestamp or datetime.now()

class ChatbotInterface:
    def __init__(self):
        self.conversation_history: List[Message] = []
        self.context: Dict = {}
        self.is_visible = False
        
    def toggle_visibility(self) -> None:
        """Cambia la visibilidad del chat"""
        self.is_visible = not self.is_visible
        
    def add_message(self, content: str, role: str) -> None:
        """Añade un mensaje al historial de conversación"""
        message = Message(content, role)
        self.conversation_history.append(message)
    
    def get_history(self) -> List[Dict]:
        """Retorna el historial de conversación formateado"""
        return [
            {
                "content": msg.content,
                "role": msg.role,
                "timestamp": msg.timestamp.strftime("%H:%M")
            }
            for msg in self.conversation_history
        ]
    
    def clear_history(self) -> None:
        """Limpia el historial de conversación"""
        self.conversation_history = []
    
    def update_context(self, diagnosis: Optional[float] = None, user_data: Optional[Dict] = None) -> None:
        """Actualiza el contexto del chatbot con nueva información"""
        if diagnosis is not None:
            self.context['diagnosis'] = diagnosis
        if user_data is not None:
            self.context.update(user_data)
    
    def generate_response(self, user_input: str) -> str:
        """
        Genera una respuesta basada en el input del usuario y el contexto actual
        """
        # Primero, intentar obtener una respuesta de la base de conocimiento
        response = get_most_relevant_response(user_input)
        
        # Si hay un diagnóstico reciente, añadir información contextual
        if 'diagnosis' in self.context:
            diagnosis = self.context['diagnosis']
            if "síntomas" in user_input.lower() or "riesgo" in user_input.lower():
                if diagnosis > 0.7:
                    response += "\n\nDado tu resultado reciente, te recomiendo consultar con un especialista lo antes posible."
                elif diagnosis > 0.4:
                    response += "\n\nConsidera programar una consulta con un profesional de la salud para una evaluación más detallada."
        
        return response