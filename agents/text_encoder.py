"""
Text Encoders for TextWorld Observations

Provides different text encoding strategies:
- DistilBERTEncoder: Uses pretrained DistilBERT for rich semantic encoding
- TextEncoder: Base class defining the encoder interface
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class TextEncoder(ABC, nn.Module):
    """
    Abstract base class for text encoders.
    
    All encoders must implement encode() to convert text to fixed-size vectors.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
    
    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Encoded tensor of shape (hidden_size,)
        """
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of text strings.
        
        Args:
            texts: List of input text strings
            
        Returns:
            Encoded tensor of shape (batch_size, hidden_size)
        """
        pass


class DistilBERTEncoder(TextEncoder):
    """
    Text encoder using DistilBERT pretrained model.
    
    DistilBERT provides rich semantic representations while being
    faster and smaller than full BERT.
    
    Attributes:
        model_name: HuggingFace model identifier
        freeze_layers: Number of transformer layers to freeze
        max_length: Maximum input sequence length
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        freeze_layers: int = 4,
        max_length: int = 512,
        hidden_size: int = 768,
        output_size: Optional[int] = None,
        device: str = "cpu"
    ):
        """
        Initialize the DistilBERT encoder.
        
        Args:
            model_name: HuggingFace model identifier
            freeze_layers: Number of transformer layers to freeze (0-6)
            max_length: Maximum input sequence length
            hidden_size: DistilBERT hidden size (768 for base)
            output_size: Optional projection size (if None, use hidden_size)
            device: Device to run the model on
        """
        super().__init__(hidden_size if output_size is None else output_size)
        
        self.model_name = model_name
        self.freeze_layers = freeze_layers
        self.max_length = max_length
        self.device_name = device
        self._output_size = output_size
        
        # Load DistilBERT model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze specified number of layers
        self._freeze_layers(freeze_layers)
        
        # Optional projection layer
        if output_size is not None and output_size != 768:
            self.projection = nn.Linear(768, output_size)
        else:
            self.projection = None
        
        # Store hidden size
        self._hidden_size = 768
    
    def _freeze_layers(self, num_layers: int):
        """
        Freeze the first num_layers transformer layers.
        
        Args:
            num_layers: Number of layers to freeze
        """
        # Freeze embeddings
        if num_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        
        # Freeze transformer layers
        for i, layer in enumerate(self.bert.transformer.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Encoded tensor of shape (hidden_size,)
        """
        return self.encode_batch([text]).squeeze(0)
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of text strings.
        
        Uses the [CLS] token representation as the sentence embedding.
        
        Args:
            texts: List of input text strings
            
        Returns:
            Encoded tensor of shape (batch_size, hidden_size)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.bert.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Forward pass
        with torch.set_grad_enabled(self.training):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Use [CLS] token representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Apply projection if exists
        if self.projection is not None:
            cls_embeddings = self.projection(cls_embeddings)
        
        return cls_embeddings
    
    def encode_with_commands(
        self,
        observation: str,
        commands: List[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation and commands separately.
        
        Args:
            observation: Observation text
            commands: List of command strings
            
        Returns:
            obs_encoding: Observation encoding (hidden_size,)
            cmd_encodings: Command encodings (num_commands, hidden_size)
        """
        obs_encoding = self.encode(observation)
        
        if len(commands) == 0:
            cmd_encodings = torch.zeros(1, self.hidden_size, device=obs_encoding.device)
        else:
            cmd_encodings = self.encode_batch(commands)
        
        return obs_encoding, cmd_encodings
    
    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            "model_name": self.model_name,
            "freeze_layers": self.freeze_layers,
            "max_length": self.max_length,
            "hidden_size": self._hidden_size,
            "output_size": self._output_size,
        }


class CommandScorer(nn.Module):
    """
    Scores admissible commands given an observation encoding.
    
    Uses dot-product attention between observation and command encodings
    to produce action probabilities.
    """
    
    def __init__(
        self,
        hidden_size: int,
        temperature: float = 1.0
    ):
        """
        Initialize the command scorer.
        
        Args:
            hidden_size: Size of the encoding vectors
            temperature: Softmax temperature
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.temperature = temperature
        
        # Learnable query transformation
        self.query_transform = nn.Linear(hidden_size, hidden_size)
        self.key_transform = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        obs_encoding: torch.Tensor,
        cmd_encodings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Score commands given observation.
        
        Args:
            obs_encoding: Observation encoding (batch_size, hidden_size) or (hidden_size,)
            cmd_encodings: Command encodings (batch_size, num_commands, hidden_size)
                          or (num_commands, hidden_size)
            mask: Optional mask for invalid commands
            
        Returns:
            Action probabilities (batch_size, num_commands) or (num_commands,)
        """
        single_example = obs_encoding.dim() == 1
        if single_example:
            obs_encoding = obs_encoding.unsqueeze(0)
            cmd_encodings = cmd_encodings.unsqueeze(0)
        
        query = self.query_transform(obs_encoding)
        keys = self.key_transform(cmd_encodings)
        
        query = self.layer_norm(query)
        keys = self.layer_norm(keys)
        

        query = query.unsqueeze(1)
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
        
        scores = scores / self.temperature
        
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        
        probs = torch.softmax(scores, dim=-1)
        
        if single_example:
            probs = probs.squeeze(0)
        
        return probs
