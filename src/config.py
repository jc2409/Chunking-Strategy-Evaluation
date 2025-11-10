"""Configuration settings for the chunking benchmark pipeline."""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class UnstructuredAPIConfig:
    """Configuration for Unstructured API."""
    api_key: Optional[str] = None
    api_url: str = "https://api.unstructured.io/general/v0/general"
    strategy: str = "hi_res"  # "fast", "hi_res", or "auto"
    extract_image_block_types: List[str] = field(default_factory=lambda: ["Image", "Table"])

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("UNSTRUCTURED_API_KEY")


@dataclass
class JinaConfig:
    """Configuration for Jina Embeddings."""

    # Local model settings
    local_model_name: str = "jinaai/jina-embeddings-v3"
    device: str = "cpu"  # or "cuda" if GPU available


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    # Traditional chunking
    fixed_chunk_size: int = 512
    chunk_overlap: int = 50

    # Semantic chunking
    use_semantic_chunking: bool = True

    # Late chunking
    max_chunk_length: int = 1000
    return_tokens: bool = True
    return_chunks: bool = True
    use_jina_api_chunking: bool = True  # Use Jina API for better chunking (recommended)


@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector database."""
    api_key: Optional[str] = None
    index_name: str = "chunking-benchmark-v2"
    dimension: int = 1024 
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("PINECONE_API_KEY")


@dataclass
class RagasConfig:
    """Configuration for Ragas evaluation framework using Azure OpenAI."""
    # Azure OpenAI Configuration
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_deployment_name: str = "gpt-4.1"  # Your GPT-4.1 deployment name
    azure_embedding_deployment: str = "text-embedding-3-large"  # Your embedding deployment name
    azure_api_version: str = "2024-12-01-preview"  # Azure OpenAI API version

    metrics: List[str] = field(default_factory=lambda: [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ])

    def __post_init__(self):
        if self.azure_openai_api_key is None:
            self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if self.azure_openai_endpoint is None:
            self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    test_queries: List[str] = field(default_factory=lambda: [
        "What is an ally according to the DIB definition?",
        "What is the difference between a mentor and a sponsor?",
        "What are the current DIB goals regarding geographic diversity?",
        "What skills and behaviors should effective allies exhibit?",
        "What is the role of a sponsor at GitLab and what job grade is required?",
        "How does GitLab define diversity, inclusion, and belonging?",
        "What are the four phases of a successful sponsorship relationship?",
        "What inclusive behaviors should managers practice at GitLab?",
        "What is performative allyship and why should it be avoided?",
        "What are the key components of DIB roundtables and how are they structured?",
    ])
    ground_truths: List[str] = field(default_factory=lambda: [
        "A diversity, inclusion and belonging ally is someone who is willing to take action in support of another person, in order to remove barriers that impede that person from contributing their skills and talents in the workplace or community. Being an ally is a verb, meaning you proactively and purposefully take action.",
        "While a mentor is someone who has knowledge and will share it with you, a sponsor is a person who has power and will use it for you. Mentoring is about providing guidance based on personal experience, while sponsorship is about using influence and power to advocate for career advancement opportunities and provide visibility to senior leaders.",
        "GitLab does not currently have a company-wide goal for geographic diversity and will focus on increasing Director+ representation outside of the United States, as team member representation is currently outpacing leadership representation. In FY24, they will reevaluate the need for director level+ representation goals outside of the United States.",
        "Effective allies should exhibit active listening (neutral, nonjudgmental, patient, asking questions), empathy and emotional intelligence, active learning about other experiences, humility (non-defensive, willingness to take feedback), courage (comfortable getting uncomfortable, speak up where others don't), self-awareness (own and use privilege), and being action-oriented (see something, say something).",
        "A sponsor at GitLab is someone who has power and influence and will use that power to advocate, elevate and impact a team member's opportunities and career progression. The sponsor must be a senior leader at a minimum job grade 10+ who is not the sponsee's direct manager, must be a people manager or manager of managers, and must have been at GitLab for 6+ months.",
        "Diversity includes all the differences we each have, whether it be where we grew up, where we went to school, experiences, age, race, gender, national origin, and things we can and cannot see. Inclusion is understanding or recognizing all these differences and inviting someone to be a part of things and collaborate. Belonging is when you feel your insights and contributions are valued, and you can bring your full self to work.",
        "The four phases of successful sponsorship are: Build (take time to build a solid relationship, commit to regular 1-1s, understand career development plan), Develop (become action and capability focused, help guide on areas of improvement), Commit (both parties agree to move forward with sponsorship and advocating), and Advocate (sponsor actively and intentionally advocates for sponsee's continued career development and advancement).",
        "Managers should include and seek input from team members across a wide variety of backgrounds, practice active listening, make a habit of asking questions, address misunderstandings quickly, ensure all voices are heard, assume positive intent, be mindful of meeting times across regions, ask employees what pronouns they use, and be a role model by being authentic and owning up to mistakes.",
        "Performative allyship refers to allyship that is done to increase a person's social capital rather than because of a person's devotion to a cause. For example, some people used hashtags during social movements without actually bringing more awareness or trying to effect change. It should be avoided because it doesn't create real impact and can undermine genuine DIB efforts.",
        "DIB roundtables are designed to build deeper connections and develop safe spaces to discuss DIB related issues. They can be DIB Team programmed (quarterly with pre-defined topics), self-organized by TMRGs or team members, or manager-organized. A typical roundtable lasts 50 minutes: 10 minutes for topic introduction, 30 minutes for small group discussions of 5-6 team members, and 10 minutes for debrief. Ground rules include assuming positive intent, avoiding multitasking, and maintaining confidentiality.",
    ])
    top_k: int = 5
    batch_size: int = 10


@dataclass
class Config:
    """Main configuration class combining all settings."""
    unstructured: UnstructuredAPIConfig = field(default_factory=UnstructuredAPIConfig)
    jina: JinaConfig = field(default_factory=JinaConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    ragas: RagasConfig = field(default_factory=RagasConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def get_config() -> Config:
    """Get the default configuration."""
    return Config()
