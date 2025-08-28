from typing import List, Dict, Optional

# Short-term (Conversational memory) in the RAG pipeline
class ConversationMemory:
    def __init__(self, max_turns=10):
        """
        Initialize conversation memory with a maximum number of turns
        
        Args:
            max_turns (int): Maximum number of conversation turns to remember
        """
        self.memory = []
        self.max_turns = max_turns

    def add_turn(self, user_query, assistant_response):
        """
        Add a new conversation turn to memory
        
        Args:
            user_query (str): The user's input query
            assistant_response (str): The assistant's response
        """
        # If memory is full, remove the oldest turn
        if len(self.memory) >= self.max_turns:
            self.memory.pop(0)
        
        # Add new turn
        self.memory.append({
            'user_query': user_query,
            'assistant_response': assistant_response
        })

    def get_context(self):
        """
        Generate a context string from memory
        
        Returns:
            str: Formatted context string
        """
        context_parts = []
        for turn in self.memory:
            context_parts.append(f"User Query: {turn['user_query']}")
            context_parts.append(f"Assistant Response: {turn['assistant_response']}")
        return "\n\n".join(context_parts)



# Initialize conversational memory
conversation_memory = ConversationMemory()
    
    
#Enhance recurring query with memory
def enhance_query_with_memory(query_text, conversation_memory):
    """
    Enhance the query by incorporating conversation context
    
    Args:
        query_text (str): Original user query
        conversation_memory (ConversationMemory): Conversation memory object
        
    Returns:
        str: Enhanced query with context
    """
    # If there's no previous context, return the original query
    if not conversation_memory.memory:
        return query_text
    
    # Enhance the query with context
    context = conversation_memory.get_context()
    enhanced_prompt = f"""
    Context of Previous Conversation:
    {context}

    Current Query: {query_text}

    Please help me answer the current query while considering the context of our previous conversation.
    If the previous context is not directly relevant, focus on answering the current query.
    """
    
    return enhanced_prompt


#Modify system prompt to handle conversational memory in response generation
def modify_system_prompt_with_memory(system_prompt, conversation_memory):
    """
    Modify the system prompt to include memory awareness
    
    Args:
        system_prompt (str): Original system prompt
        conversation_memory (ConversationMemory): Conversation memory object
        
    Returns:
        str: Updated system prompt
    """
    memory_instruction = """
    MEMORY AWARENESS INSTRUCTIONS:
    - Consider the context of the previous conversation turns when generating your response.
    - The conversation history may include queries and responses in multiple languages.
    - If the previous context provides relevant background or clarification, incorporate it into your answer.
    - Maintain the primary goal of answering the current query accurately and comprehensively.
    - DO NOT fabricate or assume information not present in the current sources or previous context.
    """
    
    # Insert memory instructions before the existing instructions
    modified_prompt = system_prompt.replace(
        "IMPORTANT INSTRUCTIONS:", 
        memory_instruction + "\n\nIMPORTANT INSTRUCTIONS:"
    )
    
    return modified_prompt