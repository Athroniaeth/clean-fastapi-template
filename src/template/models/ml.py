from pydantic import Field, BaseModel


class SelectModel(BaseModel):
    """Model for selecting a specific ML model."""

    model_name: str = Field(
        default=...,
        description="Name of the model to select.",
    )
    
    
class MetadataML(BaseModel):
    """Metadata for ML models."""

    name: str = Field(
        default=...,
        description="Name of the model.",
    )
    version: str = Field(
        default="1.0.0",
        description="Version of the model.",
    )
    description: str = Field(
        default="N/A",
        description="Description of the model.",
    )
    device: str = Field(
        default="cpu",
        description="Device on which the model is loaded (e.g., 'cpu', 'cuda').",
    )
    

class InputInference(BaseModel):
    """Input data for inference."""

    start_tokens: str = Field(
        default="",
        description="Initial tokens to start the generation.",
    )
    max_length: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Maximum length of the generated sequence.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        lt=10.0,
        description="Sampling temperature for controlling randomness in generation.",
    )
    top_k: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Top-k filtering parameter for controlling diversity in generation.",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) filtering parameter for controlling diversity in generation.",
    )
    n: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of samples to generate.",
    )
    

class OutputInference(BaseModel):
    """Model for inference results from ML models."""
    name: str = Field(
        default=...,
        description="Name of the model to use for inference.",
    )

    time_elapsed: float = Field(
        default=...,
        description="Total time taken for inference in seconds.",
    )
    avg_time: float = Field(
        default=...,
        description="Average time taken per inference in seconds.",
    )
    nrps: int = Field(
        default=...,
        description="Number of Requests Per Second (with `n` argument) achieved during inference.",
    )
    results: list[str] = Field(
        default=...,
        description="List of generated results from the model.",
    )
    uniques: list[str] = Field(
        default=...,
        description="Unique generated results from the model.",
    )


class DocumentedSelectModel(SelectModel):
    """Model for documented model selection."""

    class Config:
        json_schema_extra = {
            "example": SelectModel(model_name="communes").model_dump()
        }
        
    
class DocumentedMetadataML(MetadataML):
    """Model for documented ML metadata."""

    class Config:
        json_schema_extra = {
            "example": MetadataML(
                name="communes",
                version="1.0.0",
                description="A model for generating city names.",
                device="cpu",
            ).model_dump()
        }
        
        
class DocumentedOutputInference(OutputInference):
    """Model for documented ML inference results."""

    class Config:
        json_schema_extra = {
            "example": OutputInference(
                name="communes",
                time_elapsed=2.5,
                avg_time=1.25,
                nrps=25,
                results=["New York", "Los Angeles", "Chicago"],
                uniques=["New York", "Los Angeles"],
            ).model_dump()
        }
        
        
class DocumentedInputInference(InputInference):
    """Model for documented input inference data."""

    class Config:
        json_schema_extra = {
            "example": InputInference(
                start_tokens="",
                max_length=30,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                n=5,
            ).model_dump()
        }
        
        