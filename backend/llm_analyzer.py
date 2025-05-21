import json
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

class LLMAnalyzer:
    """
    Uses LangChain to prompt OpenAI LLM for a human-readable tone and transcript analysis.
    """
    def __init__(self, openai_api_key: str, model_name: str = 'gpt-4.1-mini', temperature: float = 0.8):
        # Initialize chat model
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=temperature
        )

        # Prompt template now accepts both acoustic features and transcript JSON
        self.prompt = PromptTemplate(
            input_variables=['features', 'transcript'],
            template=(
                "You are an assistant that analyzes sales call performance. "
                "First, review the acoustic features (pace, pitch, loudness, voice quality), then review the conversation transcript.\n"
                "Acoustic Features (JSON):\n{features}\n"
                "Conversation Transcript (JSON list of segments):\n{transcript}\n"
                "Based on both the tone and content, provide a concise summary of strengths, areas for improvement, and actionable recommendations."
            )
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def analyze(self, features_dict: dict, transcript_json: list) -> str:
        # Serialize inputs
        features_str = json.dumps(features_dict, indent=2)
        transcript_str = json.dumps(transcript_json, indent=2)
        print("Features JSON:", features_str)
        print("Transcript JSON:", transcript_str)

        # Run LLM chain
        return self.chain.run(features=features_str, transcript=transcript_str)