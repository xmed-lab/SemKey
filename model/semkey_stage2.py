"""
SEMKEY_STAGE2: Merges ouputs from SEMKEY_PARALLEL and reconstructs text accordingly.

Note: Please run /inference/predict_semkey_parallel_and_pack.sh first to generate dataset for
      this stage.
!!! Requires SEMKEY_PARALLEL to be run/trained first.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, List


class Stage2ReconstructionModel(nn.Module):
    """
    Stage 2 model for text reconstruction from EEG features.
    Uses Flan-T5 with embedding injection for multi-modal fusion.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        freeze_strategy: str = "lora",
        lora_rank: int = 8,
        attention_mask_type: str = "bidirectional",
        use_ei: bool = True,
        use_projector: bool = True,
        label_embed_init: Optional[Dict] = None,
        sentiment_labels: list = None,
        topic_labels: list = None,
        prompt_type: str = "default",
        device: str = "cuda:0",
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            model_name: Hugging Face model name
            freeze_strategy: 'lora' or 'full_freeze_llm' or 'full_trainable_llm'
            lora_rank: LoRA rank
            attention_mask_type: 'bidirectional' or 'causal' (both use full visibility for now)
            use_ei: Whether to prepend global EEG feature (ei) to sequence features (Zi)
            use_projector: Whether to use trainable projection layer for feature alignment
            label_embed_init: Optional pre-trained label embeddings
            sentiment_labels: List of sentiment label names
            topic_labels: List of topic label names
            prompt_type: str -> set how to set the prompts
            device: Device to use
        """
        super().__init__()
        self.device = torch.device(device)
        self.freeze_strategy = freeze_strategy
        self.attention_mask_type = attention_mask_type
        self.use_ei = use_ei
        self.use_projector = use_projector
        self.prompt_type = prompt_type

        # Store label names for prompt generation
        self.sentiment_labels = sentiment_labels if sentiment_labels else ['non_neutral', 'neutral']
        self.topic_labels = topic_labels if topic_labels else ['Biographies and Factual Knowledge', 'Movie Reviews and Sentiment']

        # Determine dtype based on hardware support
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.model_dtype = torch.bfloat16
        else:
            self.model_dtype = torch.float32

        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.model_dtype,
            cache_dir=cache_dir
        )

        # Add special tokens and resize BEFORE applying LoRA
        special_tokens = ['<SENT_VAL>', '<TOPIC_VAL>', '<EEG_GLOBAL>', '<EEG_SEQ>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Store special token IDs
        self.sent_val_id = self.tokenizer.convert_tokens_to_ids('<SENT_VAL>')
        self.topic_val_id = self.tokenizer.convert_tokens_to_ids('<TOPIC_VAL>')
        self.eeg_global_id = self.tokenizer.convert_tokens_to_ids('<EEG_GLOBAL>')
        self.eeg_seq_id = self.tokenizer.convert_tokens_to_ids('<EEG_SEQ>')

        # Label embeddings (2 classes each, 1024 dim)
        self.label_embed_task1 = nn.Embedding(2, 1024)
        self.label_embed_task2 = nn.Embedding(2, 1024)
        nn.init.normal_(self.label_embed_task1.weight, std=0.02)
        nn.init.normal_(self.label_embed_task2.weight, std=0.02)

        # Load pre-trained label embeddings if provided
        if label_embed_init is not None:
            self.load_label_embeddings(label_embed_init)

        # Apply freeze strategy AFTER resize_token_embeddings
        if freeze_strategy == "lora":
            # Freeze base model, add LoRA
            for param in self.model.parameters():
                param.requires_grad = False

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.gradient_checkpointing_enable()

            # Unfreeze embedding layers for new tokens
            for name, param in self.model.named_parameters():
                if "embed_tokens" in name or "shared" in name:
                    param.requires_grad = True

        elif freeze_strategy == "full_freeze_llm":
            # Freeze everything in the model
            for param in self.model.parameters():
                param.requires_grad = False

        elif freeze_strategy == "full_trainable_llm":
            # Train all parameters
            for param in self.model.parameters():
                param.requires_grad = True

        # Initialize trainable projection layer for feature alignment
        if self.use_projector:
            self.projector = nn.Linear(1024, self.model.config.d_model)
            # Ensure projector is always trainable
            for param in self.projector.parameters():
                param.requires_grad = True

        # Move to device
        self.to(self.device)

    def build_prompt(
        self, 
        sentiment_idx: int, 
        topic_idx: int, 
        length: float, 
        surprisal: float,
        prompt_dict: Optional[Dict] = None
    ) -> str:
        """
        Build prompt template with optional metadata.
        -> depends on self.prompt_type
        """
        sentiment_text = self.sentiment_labels[sentiment_idx]
        topic_text = self.topic_labels[topic_idx]

        # Base prompt components
        components = [
            "System: Based on the following EEG signals, reconstruct the text."
        ]

        # Add components based on prompt type
        if any(word in self.prompt_type for word in ["default", "length"]):
            components.append(f"The length of the sentence is {length:.1f} words.")
        if any(word in self.prompt_type for word in ["default", "surprisal"]):   
            components.append(f"The average surprisal value is {surprisal:.2f}.")
        if any(word in self.prompt_type for word in ["default", "sentiment"]):   
            components.append(f"Sentiment: {sentiment_text}.")
        if any(word in self.prompt_type for word in ["default", "topic"]):   
            components.append(f"Topic: {topic_text}.")

        # Add metadata if available
        if prompt_dict is not None:
            components.append(
                f"This EEG signal comes from {prompt_dict['dataset']} "
                f"{prompt_dict['task']} subject {prompt_dict['subject']}."
            )

        components.append("Target: ")

        return " ".join(components)

    def create_cross_attention_mask(self, batch_size: int, encoder_seq_len: int) -> Optional[torch.Tensor]:
        """
        Create cross-attention mask for decoder attending to encoder hidden states.

        Note: Currently both modes return None (full visibility).
        The attention_mask_type parameter is reserved for future use.

        Args:
            batch_size: Batch size
            encoder_seq_len: Length of encoder hidden states (96 or 97 for Zi/[ei,Zi])

        Returns:
            None (full visibility for both modes)
        """
        # Both modes use full visibility for now
        return None

    def forward(
        self,
        label_task1: torch.Tensor,
        label_task2: torch.Tensor,
        length: torch.Tensor,
        surprisal: torch.Tensor,
        ei: torch.Tensor,
        Zi: torch.Tensor,
        target_text: list,
        prompt_dicts: Optional[List[Dict]] = None
    ) -> torch.Tensor:
        """
        Forward pass with EEG features as encoder hidden states.

        Args:
            label_task1: (batch_size,) sentiment labels (kept for compatibility)
            label_task2: (batch_size,) topic labels (kept for compatibility)
            length: (batch_size,) length predictions
            surprisal: (batch_size,) surprisal prediction
            ei: (batch_size, 1024) global EEG vectors
            Zi: (batch_size, 96, 1024) EEG sequences
            target_text: List of target text strings
            prompt_dicts: Optional[List[Dict]] dicts of 'dataset', 'task', 'subject'

        Returns:
            loss: Cross-entropy loss
        """
        batch_size = label_task1.shape[0]

        # Build prompts with optional metadata
        prompts = [
            self.build_prompt(
                label_task1[i].item(),
                label_task2[i].item(),
                length[i].item(),
                surprisal[i].item(),
                prompt_dicts[i] if prompt_dicts else None
            )
            for i in range(batch_size)
        ]

        # Tokenize prompts WITHOUT padding (returns list of token lists)
        prompt_encodings = self.tokenizer(prompts, padding=False, truncation=True)
        prompt_ids_list = prompt_encodings['input_ids']  # List of lists

        # Tokenize targets WITHOUT padding
        target_encodings = self.tokenizer(target_text, padding=False, truncation=True)
        target_ids_list = target_encodings['input_ids']  # List of lists

        # Concatenate at list level: remove EOS from prompt, keep EOS in target
        merged_ids_list = []
        prompt_lengths = []  # Track prompt length (without EOS) for masking
        for prompt_ids, target_ids in zip(prompt_ids_list, target_ids_list):
            # Remove EOS from prompt (last token)
            prompt_no_eos = prompt_ids[:-1] if len(prompt_ids) > 0 else []
            prompt_lengths.append(len(prompt_no_eos))
            # Concatenate
            merged_ids = prompt_no_eos + target_ids
            merged_ids_list.append(merged_ids)

        # Pad merged sequences to create batch tensor
        max_len = max(len(ids) for ids in merged_ids_list)
        labels = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        for i, ids in enumerate(merged_ids_list):
            labels[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)

        # Create decoder_input_ids by prepending pad_token_id (teacher forcing shift)
        pad_token_tensor = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        decoder_input_ids = torch.cat([pad_token_tensor, labels[:, :-1]], dim=1)  # (batch_size, max_len)

        # Mask prompt portion in labels (set to -100)
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100  # Mask prompt tokens
        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask padding tokens

        # Prepare encoder_hidden_states: optionally prepend ei to Zi
        if self.use_ei:
            # Prepend ei (batch_size, 1024) to Zi (batch_size, 96, 1024)
            ei_expanded = ei.unsqueeze(1)  # (batch_size, 1, 1024)
            raw_features = torch.cat([ei_expanded, Zi], dim=1)  # (batch_size, 97, 1024)
        else:
            raw_features = Zi  # (batch_size, 96, 1024)

        # Apply projection layer if enabled
        if self.use_projector:
            encoder_hidden_states = self.projector(raw_features)  # (batch_size, seq_len, d_model)
        else:
            encoder_hidden_states = raw_features

        encoder_hidden_states = encoder_hidden_states.to(self.model_dtype)

        # Forward pass with encoder_hidden_states
        outputs = self.model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(encoder_hidden_states,),  # Tuple format required by T5
            labels=labels
        )

        return outputs.loss

    def generate(
        self,
        label_task1: torch.Tensor,
        label_task2: torch.Tensor,
        length: torch.Tensor,
        surprisal: torch.Tensor,
        ei: torch.Tensor,
        Zi: torch.Tensor,
        max_length: int = 50,
        prompt_dicts: Optional[List[Dict]] = None
    ) -> list:
        """
        Generate text from EEG features.
        """
        batch_size = label_task1.shape[0]

        # Build prompts with optional metadata
        prompts = [
            self.build_prompt(
                label_task1[i].item(),
                label_task2[i].item(),
                length[i].item(),
                surprisal[i].item(),
                prompt_dicts[i] if prompt_dicts else None
            )
            for i in range(batch_size)
        ]

        # 1. Tokenize prompts WITHOUT padding (to safely handle EOS removal)
        prompt_encodings = self.tokenizer(prompts, padding=False, truncation=True)
        prompt_ids_list = prompt_encodings['input_ids']

        # 2. Process prompts: Remove EOS & Prepend PAD (Start Token)
        # T5 generation expects to start with a PAD token effectively acting as <BOS>
        processed_prompt_ids = []
        for p_ids in prompt_ids_list:
            # Remove EOS if it exists (T5 tokenizer adds it by default)
            if p_ids[-1] == self.tokenizer.eos_token_id:
                p_ids = p_ids[:-1]

            # Prepend PAD token (Crucial for aligning with training distribution)
            # Training: [PAD, Prompt, Target]
            # Inference Input: [PAD, Prompt] -> Model generates Target
            p_ids = [self.tokenizer.pad_token_id] + p_ids
            processed_prompt_ids.append(p_ids)

        # 3. Pad manually to create batch tensor
        max_len = max(len(p) for p in processed_prompt_ids)
        decoder_input_ids = torch.full(
            (batch_size, max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device
        )

        for i, p_ids in enumerate(processed_prompt_ids):
            decoder_input_ids[i, :len(p_ids)] = torch.tensor(p_ids, dtype=torch.long, device=self.device)

        # Prepare encoder_hidden_states: optionally prepend ei to Zi
        if self.use_ei:
            ei_expanded = ei.unsqueeze(1)  # (batch_size, 1, 1024)
            raw_features = torch.cat([ei_expanded, Zi], dim=1)  # (batch_size, 97, 1024)
        else:
            raw_features = Zi  # (batch_size, 96, 1024)

        # Apply projection layer if enabled
        if self.use_projector:
            encoder_hidden_states = self.projector(raw_features)  # (batch_size, seq_len, d_model)
        else:
            encoder_hidden_states = raw_features

        encoder_hidden_states = encoder_hidden_states.to(self.model_dtype)

        # Wrap encoder hidden states in BaseModelOutput for generate method
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        # Generate
        outputs = self.model.generate(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            max_length=max_length + decoder_input_ids.shape[1], # Adjust max_length to include prompt
            num_beams=1,
            do_sample=False
        )

        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Optional post-processing: Remove the prompt text from the output if T5 repeats it
        # (Usually T5 generate output includes the input prefix if decoder_input_ids is passed?
        # Actually T5 generate output starts *after* the decoder_input_ids usually,
        # but let's stick to standard decoding first.)

        return generated_texts

    def load_label_embeddings(self, state_dict: Dict):
        """
        Load pre-trained label embeddings.
        (Optional)
        """
        if 'label_embed_task1' in state_dict:
            self.label_embed_task1.load_state_dict(state_dict['label_embed_task1'])
        if 'label_embed_task2' in state_dict:
            self.label_embed_task2.load_state_dict(state_dict['label_embed_task2'])