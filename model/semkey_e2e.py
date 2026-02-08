"""
SEMKEY_E2E: Combines SEMKEY_PARALLEL and SEMKEY_STAGE2.
            -> Performs end-to-end training stage1 + stage2

Note: Please run /inference/predict_semkey_parallel_and_pack.sh first to generate dataset for
      this stage.
!!! Requires SEMKEY_PARALLEL to be run/trained first.
!!! Requires SEMKEY_PARALLEL checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from transformers.modeling_outputs import BaseModelOutput

from .semkey_parallel import SEMKEY_PARALLEL
from .semkey_stage2 import Stage2ReconstructionModel


class SEMKEY_Stage2_E2E(nn.Module):

    def __init__(self, stage1_checkpoint: str, stage2_config: dict, loss_config: dict):
        super().__init__()

        self.use_align_loss = loss_config['use_align_loss']
        self.use_aux_loss = loss_config['use_aux_loss']
        self.w_align = loss_config.get('w_align', 0.1)
        self.w_llm = loss_config.get('w_llm', 1.0)
        self.w_sentiment = loss_config.get('w_sentiment', 0.25)
        self.w_topic = loss_config.get('w_topic', 0.25)
        self.w_length = loss_config.get('w_length', 0.25)
        self.w_surprisal = loss_config.get('w_surprisal', 0.25)

        # Load Stage 1
        self.stage1 = self._load_stage1(stage1_checkpoint, self.use_align_loss)

        # Add label names to stage2_config
        stage2_config['sentiment_labels'] = self.stage1.classification_tasks['sentiment']
        stage2_config['topic_labels'] = self.stage1.classification_tasks['topic']

        # Initialize Stage 2
        self.stage2 = Stage2ReconstructionModel(**stage2_config)

        self.device = torch.device(stage2_config.get('device', 'cuda:0'))

    def _load_stage1(self, checkpoint_path: str, use_align_loss: bool):
        """Load Stage 1 from Lightning checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Create SEMKEY_PARALLEL (stage1) instance
        hparams = checkpoint['hyper_parameters']
        stage1 = SEMKEY_PARALLEL(**hparams)

        # Filter state_dict if not using alignment losses
        state_dict = checkpoint['state_dict']
        if not use_align_loss:
            state_dict = {k: v for k, v in state_dict.items()
                         if 'text_model' not in k and 'tokenizer' not in k}

        stage1.load_state_dict(state_dict, strict=False)

        # Initialize tokenizer and text_model if using alignment losses
        if use_align_loss:
            stage1.setup('fit')

        # Manually load regression statistics buffers if they exist
        if 'length_mean' in state_dict:
            stage1.register_buffer('length_mean', state_dict['length_mean'])
            stage1.register_buffer('length_std', state_dict['length_std'])
            stage1.register_buffer('surprisal_mean', state_dict['surprisal_mean'])
            stage1.register_buffer('surprisal_std', state_dict['surprisal_std'])

        return stage1

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both stages.
        """
        # Stage 1: Encode EEG → Zi, ei
        Zi, ei, eeg_hiddens = self._stage1_forward(batch)

        # Convert labels: strings → indices
        sentiment_ids = self._convert_labels(batch['sentiment label'], 'sentiment')
        topic_ids = self._convert_labels(batch['topic_label'], 'topic')

        # Extract prompt dicts
        prompt_dicts = self._extract_prompt_dicts(batch['prompt'])

        # Stage 2: Text reconstruction
        loss_llm = self.stage2(
            label_task1=sentiment_ids,
            label_task2=topic_ids,
            length=torch.tensor(batch['length'], device=self.device, dtype=torch.float32),
            surprisal=torch.tensor(batch['surprisal'], device=self.device, dtype=torch.float32),
            ei=ei,
            Zi=Zi,
            target_text=batch['target text'],
            prompt_dicts=prompt_dicts
        )

        # Compute total loss
        total_loss = self.w_llm * loss_llm
        losses = {'loss_llm': loss_llm}

        if self.use_align_loss:
            align_losses = self._compute_align_losses(batch, eeg_hiddens)
            total_loss = total_loss + self.w_align * align_losses['loss_align']
            losses.update(align_losses)

        if self.use_aux_loss:
            aux_losses = self._compute_aux_losses(batch, ei, sentiment_ids, topic_ids)
            total_loss = total_loss + aux_losses['loss_aux_total']
            losses.update(aux_losses)

        losses['total_loss'] = total_loss
        return losses

    def _stage1_forward(self, batch: Dict):
        """
        Run Stage 1 forward pass.
        """
        # Encode prompts
        prompt_ids = self.stage1.p_embedder.encode(batch['prompt'], device=self.device)
        prompt_embed = self.stage1.p_embedder(prompt_ids, self.stage1.eval_pembed)

        # Encode EEG
        eeg_hiddens, _ = self.stage1.eeg_encoder(batch['eeg'], batch['mask'], prompt_embed)

        # Get embeddings via Aligner
        eeg_hiddens_proj = self.stage1.aligner.in_proj(eeg_hiddens)
        Zi, ei = self.stage1.aligner.embed_eeg(eeg_hiddens_proj, None)

        return Zi, ei, eeg_hiddens

    def _convert_labels(self, labels: List[str], task: str) -> torch.Tensor:
        """Convert string labels to indices."""
        valid_labels = self.stage1.classification_tasks[task]
        indices = [valid_labels.index(lbl) for lbl in labels]
        return torch.tensor(indices, device=self.device, dtype=torch.long)

    def _extract_prompt_dicts(self, prompt: List[List[str]]) -> List[Dict]:
        """Convert prompt tuple lists to dicts."""
        task_list, dataset_list, subject_list = prompt
        return [
            {'task': task_list[i], 'dataset': dataset_list[i], 'subject': subject_list[i]}
            for i in range(len(task_list))
        ]

    def _compute_align_losses(self, batch: Dict, eeg_hiddens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute Stage 1 alignment losses (CLIP + Commitment)."""
        # Encode text
        input_ids, input_mask = self.stage1.tokenize(batch['input text'], self.stage1.input_text_len)
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        input_text_embeds, hidden_text_mask = self.stage1.encode_text(input_ids, input_mask)

        # Compute alignment losses
        loss_clip, _, loss_commitment, _, _, _ = self.stage1.aligner.forward(
            eeg_hiddens, input_text_embeds, hidden_text_mask
        )

        loss_align = loss_clip + loss_commitment

        return {
            'loss_clip': loss_clip,
            'loss_commitment': loss_commitment,
            'loss_align': loss_align
        }

    def _compute_aux_losses(self, batch: Dict, ei: torch.Tensor,
                           sentiment_ids: torch.Tensor, topic_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute Stage 1 auxiliary task losses."""
        # Sentiment classification
        sentiment_logits = self.stage1.sentiment_classifier(ei)
        loss_sentiment = F.cross_entropy(sentiment_logits, sentiment_ids)

        # Topic classification
        topic_logits = self.stage1.topic_classifier(ei)
        loss_topic = F.cross_entropy(topic_logits, topic_ids)

        # Length regression (normalize first)
        length_tensor = torch.tensor(batch['length'], device=self.device, dtype=torch.float32).unsqueeze(1)
        length_normalized = (length_tensor - self.stage1.length_mean) / self.stage1.length_std
        length_pred = self.stage1.length_regressor(ei)
        loss_length = F.mse_loss(length_pred, length_normalized)

        # Surprisal regression (normalize first)
        surprisal_tensor = torch.tensor(batch['surprisal'], device=self.device, dtype=torch.float32).unsqueeze(1)
        surprisal_normalized = (surprisal_tensor - self.stage1.surprisal_mean) / self.stage1.surprisal_std
        surprisal_pred = self.stage1.surprisal_regressor(ei)
        loss_surprisal = F.mse_loss(surprisal_pred, surprisal_normalized)

        # Weighted sum
        loss_aux_total = (
            self.w_sentiment * loss_sentiment +
            self.w_topic * loss_topic +
            self.w_length * loss_length +
            self.w_surprisal * loss_surprisal
        )

        return {
            'loss_sentiment': loss_sentiment,
            'loss_topic': loss_topic,
            'loss_length': loss_length,
            'loss_surprisal': loss_surprisal,
            'loss_aux_total': loss_aux_total
        }

    @torch.no_grad()
    def predict_labels(self, ei: torch.Tensor) -> Dict:
        """Predict sentiment, topic, length, and surprisal from EEG embeddings."""
        # Sentiment classification
        sentiment_logits = self.stage1.sentiment_classifier(ei)
        sentiment_ids = sentiment_logits.argmax(dim=1)
        sentiment_labels = [self.stage1.classification_tasks['sentiment'][idx] for idx in sentiment_ids.cpu().tolist()]

        # Topic classification
        topic_logits = self.stage1.topic_classifier(ei)
        topic_ids = topic_logits.argmax(dim=1)
        topic_labels = [self.stage1.classification_tasks['topic'][idx] for idx in topic_ids.cpu().tolist()]

        # Length regression (denormalize)
        length_pred = self.stage1.length_regressor(ei)
        length = (length_pred * self.stage1.length_std + self.stage1.length_mean).squeeze(1)

        # Surprisal regression (denormalize)
        surprisal_pred = self.stage1.surprisal_regressor(ei)
        surprisal = (surprisal_pred * self.stage1.surprisal_std + self.stage1.surprisal_mean).squeeze(1)

        return {
            'sentiment_ids': sentiment_ids,
            'sentiment_labels': sentiment_labels,
            'topic_ids': topic_ids,
            'topic_labels': topic_labels,
            'length': length,
            'surprisal': surprisal
        }

    @torch.no_grad()
    def generate_with_predicted_labels(self, batch: Dict, max_length: int = 64) -> List[str]:
        """Generate text using predicted labels from EEG (no ground truth needed)."""
        self.eval()

        # Stage 1: Get embeddings
        Zi, ei, _ = self._stage1_forward(batch)

        # Predict labels from EEG
        predictions = self.predict_labels(ei)

        # Extract prompt dicts
        prompt_dicts = self._extract_prompt_dicts(batch['prompt'])

        # Stage 2: Generate with predicted labels
        generated_texts = self.stage2.generate(
            label_task1=predictions['sentiment_ids'],
            label_task2=predictions['topic_ids'],
            length=predictions['length'],
            surprisal=predictions['surprisal'],
            ei=ei,
            Zi=Zi,
            max_length=max_length,
            prompt_dicts=prompt_dicts
        )

        return generated_texts

    @torch.no_grad()
    def generate(self, batch: Dict, max_length: int = 64, num_beams: int = 4) -> List[str]:
        """Generate text for evaluation."""
        self.eval()

        # Stage 1: Get embeddings
        Zi, ei, _ = self._stage1_forward(batch)

        # Convert labels
        sentiment_ids = self._convert_labels(batch['sentiment label'], 'sentiment')
        topic_ids = self._convert_labels(batch['topic_label'], 'topic')

        # Extract prompt dicts
        prompt_dicts = self._extract_prompt_dicts(batch['prompt'])

        # Stage 2: Generate
        generated_texts = self.stage2.generate(
            label_task1=sentiment_ids,
            label_task2=topic_ids,
            length=torch.tensor(batch['length'], device=self.device, dtype=torch.float32),
            surprisal=torch.tensor(batch['surprisal'], device=self.device, dtype=torch.float32),
            ei=ei,
            Zi=Zi,
            max_length=max_length,
            prompt_dicts=prompt_dicts
        )

        return generated_texts
