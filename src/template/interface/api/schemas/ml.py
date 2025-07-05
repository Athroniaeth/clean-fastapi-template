from uuid import uuid4, UUID

from pydantic import Field, BaseModel

SelectModel = Field(
    default=...,
    description="Name of the model to select.",
    examples=["communes"],
)
"""Path Pydantic field for selecting a model by its identifier."""


class MetadataTokenizer(BaseModel):
    """Metadata for tokenizers."""

    name: str = Field(
        default=...,
        description="Name of the tokenizer.",
        examples=["communes"],
    )
    version: str = Field(
        default="1.0.0",
        description="Version of the tokenizer.",
        examples=["1.0.0"],
    )
    description: str = Field(
        default="N/A",
        description="Description of the tokenizer.",
        examples=["A tokenizer for communes model."],
    )
    vocab_size: int = Field(
        default=0,
        ge=0,
        description="Size of the vocabulary used by the tokenizer.",
        examples=[26],
    )

    vocab: dict[int, str] = Field(
        default_factory=dict,
        description="Vocabulary mapping from token IDs to strings.",
        examples=[{i: f"{chr(i + 65)}" for i in range(26)}],
    )


class MetadataML(BaseModel):
    """Metadata for ML models."""

    name: str = Field(
        default=...,
        description="Name of the model.",
        examples=["communes"],
    )
    version: str = Field(
        default="1.0.0",
        description="Version of the model.",
        examples=["1.0.0"],
    )
    description: str = Field(
        default="N/A",
        description="Description of the model.",
        examples=["A model for generating city names."],
    )
    device: str = Field(
        default="cpu",
        description="Device on which the model is loaded (e.g., 'cpu', 'cuda').",
        examples=["cpu", "cuda"],
    )
    tokenizer: MetadataTokenizer = Field(
        default=...,
        description="Metadata of the tokenizer used by the model.",
        examples=[
            MetadataTokenizer(
                name="communes_tokenizer",
                version="1.0.0",
                description="Tokenizer for the communes model.",
                vocab_size=26,
                vocab={i: f"{chr(i + 65)}" for i in range(26)},
            )
        ],
    )


class InputInference(BaseModel):
    """Input data for inference."""

    start_tokens: str = Field(
        default="",
        description="Initial tokens to start the generation.",
        examples=["", "Paris", "New York"],
    )
    max_length: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Maximum length of the generated sequence.",
        examples=[30, 50, 100],
    )
    temperature: float = Field(
        default=...,
        ge=0.0,
        lt=10.0,
        description="Sampling temperature for controlling randomness in generation.",
        examples=[0.6, 2.0, 0],
    )
    top_p: float = Field(
        default=...,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) filtering parameter for controlling diversity in generation.",
        examples=[0.95, 1.0, 0.1],
    )
    n: int = Field(
        default=...,
        ge=1,
        le=1_000,
        description="Number of samples to generate.",
        examples=[10, 25, 1],
    )


class OutputInference(BaseModel):
    """Model for inference results from ML models."""

    id_: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the inference result.",
        examples=[uuid4()],
    )
    model_id: str = Field(
        default=...,
        description="Name of the model to use for inference.",
        examples=["communes"],
    )

    time_elapsed: float = Field(
        default=...,
        description="Total time taken for inference in seconds.",
        examples=[2.5, 3.0, 1.75],
    )
    avg_time: float = Field(
        default=...,
        description="Average time taken per inference in seconds.",
        examples=[0.1, 0.2, 0.25],
    )
    nrps: int = Field(
        default=...,
        description="Number of Requests Per Second (with `n` argument) achieved during inference.",
        examples=[10, 20, 25],
    )
    ntps: int = Field(
        default=...,
        description="Number of Tokens Per Second (with `n` argument) achieved during inference.",
        examples=[100, 200, 250],
    )
    results: list[str] = Field(
        default=...,
        description="List of generated results from the model.",
        examples=[
            ["New York", "Los Angeles", "Chicago"],
            ["Paris", "Lyon", "Marseille"],
            ["Berlin", "Munich", "Hamburg"],
        ],
    )
    uniques: list[str] = Field(
        default=...,
        description="Unique generated results from the model.",
        examples=[
            ["New York", "Los Angeles"],
            ["Paris", "Lyon"],
            ["Berlin", "Munich"],
        ],
    )
