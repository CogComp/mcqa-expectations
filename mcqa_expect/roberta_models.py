from typing import Dict, Optional, List, Any

import logging
from overrides import overrides
from transformers.modeling_roberta import RobertaClassificationHead, RobertaConfig, RobertaModel
from transformers.tokenization_gpt2 import bytes_to_unicode
import re
import torch
from torch.nn.modules.linear import Linear
from torch.nn.functional import binary_cross_entropy_with_logits

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, F1Measure

logger = logging.getLogger(__name__)

@Model.register("roberta_mc_qa")
class RobertaMCQAModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 transformer_weights_model: str = None,
                 reset_classifier: bool = False,
                 binary_loss: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if on_load:
            logging.info(f"Skipping loading of initial Transformer weights")
            transformer_config = RobertaConfig.from_pretrained(pretrained_model)
            self._transformer_model = RobertaModel(transformer_config)

        elif transformer_weights_model:
            logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
            transformer_model_loaded = load_archive(transformer_weights_model)
            self._transformer_model = transformer_model_loaded.model._transformer_model
        else:
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)

        for name, param in self._transformer_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        transformer_config = self._transformer_model.config

        self._output_dim = transformer_config.hidden_size
        classifier_input_dim = self._output_dim
        classifier_output_dim = 1
        transformer_config.num_labels = classifier_output_dim
        self._classifier = None
        if not on_load and transformer_weights_model \
                and hasattr(transformer_model_loaded.model, "_classifier") \
                and not reset_classifier:
            self._classifier = transformer_model_loaded.model._classifier
            old_dims = (self._classifier.dense.in_features, self._classifier.out_proj.out_features)
            new_dims = (classifier_input_dim, classifier_output_dim)
            if old_dims != new_dims:
                logging.info(f"NOT copying Transformer classifier weights, incompatible dims: {old_dims} vs {new_dims}")
                self._classifier = None
        if self._classifier is None:
            self._classifier = RobertaClassificationHead(transformer_config)

        self._binary_loss = binary_loss
        self._accuracy = CategoricalAccuracy()
        self._sigmoid = torch.nn.Sigmoid()
        if self._binary_loss:
            self._loss = torch.nn.BCEWithLogitsLoss()
        else:
            self._loss = torch.nn.CrossEntropyLoss()
        self._debug = 2
        self._padding_value = 1  # The index of the RoBERTa padding token

    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                binary_labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = question['tokens']['token_ids']
        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)
        num_binary_choices = 1

        # question_mask = (input_ids != self._padding_value).long()
        question_mask = question['tokens']['mask']

        if self._debug > 0:
            logger.info(f"batch_size = {batch_size}")
            logger.info(f"num_choices = {num_choices}")
            logger.info(f"question_mask = {question_mask}")
            logger.info(f"input_ids.size() = {input_ids.size()}")
            logger.info(f"input_ids = {input_ids}")
            logger.info(f"segment_ids = {segment_ids}")
            logger.info(f"label = {label}")
            logger.info(f"binary_labels = {binary_labels}")

        # Segment ids are not used by RoBERTa

        transformer_outputs = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                      # token_type_ids=util.combine_initial_dims(segment_ids),
                                                      attention_mask=util.combine_initial_dims(question_mask))

        cls_output = transformer_outputs[0]
        
        if self._debug > 0:
            logger.info(f"cls_output = {cls_output}")

        label_logits = self._classifier(cls_output)
        label_logits_binary = label_logits.view(-1, num_binary_choices)
        label_logits = label_logits.view(-1, num_choices)
        
        output_dict = {}
        output_dict['label_logits'] = label_logits

        if self._binary_loss:
            output_dict['label_probs'] = self._sigmoid(label_logits)
        else:
            output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
        output_dict['answer_index'] = label_logits.argmax(1)
        
        if self._binary_loss and binary_labels is not None:
            labels_float_reshaped = binary_labels.reshape(-1, num_binary_choices).to(label_logits.dtype)
            loss = self._loss(label_logits_binary, labels_float_reshaped)
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss
        elif label is not None:
            loss = self._loss(label_logits, label)
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        if self._debug > 0:
            logger.info(output_dict)
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1,
              opt_level: Optional[str] = None) -> 'Model':
        model_params = config.get('model')
        model_params.update({"on_load": True})
        config.update({'model': model_params})
        return super()._load(config=config,
                             serialization_dir=serialization_dir,
                             weights_file=weights_file,
                             cuda_device=cuda_device)