from typing import Dict, List, Any
import itertools
import json
import logging
import numpy
import os
import re
import random

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, TextField, LabelField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# TagSpanType = ((int, int), str)

@DatasetReader.register("transformer_mc_qa")
class TransformerMCQAReader(DatasetReader):
    """

    Parameters
    ----------
    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 num_choices: int = 4,
                 restrict_num_choices: int = None,
                 skip_id_regex: str = None,
                 context_syntax: str = "c#q#_a!",
                 model_type: str = None,
                 do_lowercase: bool = None,
                 sample: int = -1,
                 training: bool = False,
                 augmentation_since_epoch: int = -1,
                 shuffle_dataset: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        if do_lowercase is None:
            do_lowercase = '-uncased' in pretrained_model

        self._training = training
        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model,
                                                         add_special_tokens=False)
        # self._tokenizer = PretrainedTransformerTokenizer(pretrained_model,
        #                                                  do_lowercase=do_lowercase,
        #                                                  start_tokens = [],
        #                                                  end_tokens = [])
        self._tokenizer_internal = self._tokenizer.tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model)
        # token_indexer = PretrainedTransformerIndexer(pretrained_model, do_lowercase=do_lowercase)
        self._token_indexers = {'tokens': token_indexer}

        self._max_pieces = max_pieces
        self._sample = sample
        self._num_choices = num_choices
        self._context_syntax = context_syntax
        self._restrict_num_choices = restrict_num_choices
        self._skip_id_regex = skip_id_regex
        self._model_type = model_type
        self._augmentation_since_epoch = augmentation_since_epoch
        self._epoch = -1
        self._lazy = lazy
        self._shuffle_dataset = shuffle_dataset or lazy
        if model_type is None:
            for model in ["roberta", "bert", "openai-gpt", "gpt2", "transfo-xl", "xlnet", "xlm"]:
                if model in pretrained_model:
                    self._model_type = model
                    break

    @overrides
    def _read(self, file_path: str):
        instances = self._read_internal(file_path)
        return instances

    def _read_internal(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5
        
        self._epoch += 1
        dataset_instances = []

        if self._training and self._augmentation_since_epoch >= 0 and \
            self._epoch >= self._augmentation_since_epoch:
            epoch_str = str(self._epoch - self._augmentation_since_epoch)
            file_path = file_path.replace(".jsonl", "_epoch" + epoch_str + ".jsonl")

        with open(file_path, 'r') as data_file:
            logger.info("Reading QA instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                item_json = json.loads(line.strip())

                item_id = item_json["id"]
                if self._skip_id_regex and re.match(self._skip_id_regex, item_id):
                    continue

                counter -= 1
                debug -= 1
                if counter == 0:
                    break
                
                if debug > 0:
                    logger.info(item_json)

                context = item_json.get("para")
                question_text = item_json["question"]["stem"]

                choice_label_to_id = {}
                choice_text_list = []
                choice_context_list = []
                choice_label_list = []
                
                any_correct = False
                choice_id_correction = 0
                choice_tagging = None
                
                correct_option_perturbed = False
                if "correct_option_perturbed" in item_json:
                    correct_option_perturbed = True if item_json["correct_option_perturbed"] == "True" else False

                for choice_id, choice_item in enumerate(item_json["question"]["choices"]):

                    if self._restrict_num_choices and len(choice_text_list) == self._restrict_num_choices:
                        if not any_correct:
                            choice_text_list.pop(-1)
                            choice_context_list.pop(-1)
                            choice_id_correction += 1
                        else:
                            break

                    choice_label = choice_item["label"]
                    choice_label_to_id[choice_label] = choice_id - choice_id_correction
                    choice_text = choice_item["text"]
                    choice_context = choice_item.get("para")
                    
                    choice_text_list.append(choice_text)
                    choice_context_list.append(choice_context)
                    choice_label_list.append(choice_label)
                    
                    is_correct = 0
                    if item_json.get('answerKey') == choice_label:
                        is_correct = 1
                        if any_correct:
                            raise ValueError("More than one correct answer found for {item_json}!")
                        any_correct = True

                    if self._restrict_num_choices \
                            and len(choice_text_list) == self._restrict_num_choices \
                            and not any_correct:
                        continue

                if not any_correct and 'answerKey' in item_json:
                    raise ValueError("No correct answer found for {item_json}!")

                answer_id = choice_label_to_id.get(item_json.get("answerKey"))
                # Pad choices with empty strings if not right number
                if len(choice_text_list) != self._num_choices:
                    choice_text_list = (choice_text_list + self._num_choices * [''])[:self._num_choices]
                    choice_context_list = (choice_context_list + self._num_choices * [None])[:self._num_choices]
                    if answer_id is not None and answer_id >= self._num_choices:
                        logging.warning(f"Skipping question with more than {self._num_choices} answers: {item_json}")
                        continue

                instance = self.text_to_instance(
                    item_id=item_id,
                    question=question_text,
                    choice_list=choice_text_list,
                    answer_id=-1 if correct_option_perturbed else answer_id,
                    context=context,
                    choice_context_list=choice_context_list,
                    debug=debug)
                if self._shuffle_dataset:
                    dataset_instances.append(instance)
                else:
                    yield instance
        if self._shuffle_dataset:
            random.shuffle(dataset_instances)
            for inst in dataset_instances:
                yield inst

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         choice_list: List[str],
                         answer_id: int = None,
                         context: str = None,
                         choice_context_list: List[str] = None,
                         debug: int = -1) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        qa_fields = []
        segment_ids_fields = []
        qa_tokens_list = []
        binary_labels_fields = []
        for idx, choice in enumerate(choice_list):
            choice_context = context
            if choice_context_list is not None and choice_context_list[idx] is not None:
                choice_context = choice_context_list[idx]
            qa_tokens, segment_ids = self.transformer_features_from_qa(question, choice, choice_context)
            
            qa_field = TextField(qa_tokens, self._token_indexers)
            segment_ids_field = SequenceLabelField(segment_ids, qa_field)
            binary_labels_field = LabelField(1 if answer_id == idx else 0, skip_indexing=True)
            qa_fields.append(qa_field)
            qa_tokens_list.append(qa_tokens)
            segment_ids_fields.append(segment_ids_field)
            binary_labels_fields.append(binary_labels_field)
            if debug > 0:
                logger.info(f"qa_tokens = {qa_tokens}")
                logger.info(f"segment_ids = {segment_ids}")

        fields['question'] = ListField(qa_fields)
        fields['segment_ids'] = ListField(segment_ids_fields)
        if answer_id is not None:
            fields['label'] = LabelField(answer_id, skip_indexing=True)
            fields['binary_labels'] = ListField(binary_labels_fields)

        metadata = {
            "id": item_id,
            "question_text": question,
            "choice_text_list": choice_list,
            "correct_answer_index": answer_id,
            "question_tokens_list": qa_tokens_list,
            "context": context,
            "choice_context_list": choice_context_list,
            "training": self._training
            # "question_tokens": [x.text for x in question_tokens],
            # "choice_tokens_list": [[x.text for x in ct] for ct in choices_tokens_list],
        }

        if debug > 0:
            logger.info(f"context = {context}")
            logger.info(f"choice_context_list = {choice_context_list}")
            logger.info(f"answer_id = {answer_id}")
            logger.info(f"binary_labels = {fields['binary_labels']}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @staticmethod
    def _truncate_tokens(context_tokens, question_tokens, choice_tokens, max_length):
        """
        Truncate context_tokens first, from the left, then question_tokens and choice_tokens
        """
        max_context_len = max_length - len(question_tokens) - len(choice_tokens)
        if max_context_len > 0:
            if len(context_tokens) > max_context_len:
                context_tokens = context_tokens[-max_context_len:]
        else:
            context_tokens = []
            while len(question_tokens) + len(choice_tokens) > max_length:
                if len(question_tokens) > len(choice_tokens):
                    question_tokens.pop(0)
                else:
                    choice_tokens.pop()
        return context_tokens, question_tokens, choice_tokens

    def transformer_features_from_qa(self, question: str, answer: str, context: str = None):
        cls_token = Token(self._tokenizer_internal.cls_token, text_id = self._tokenizer_internal.cls_token_id)
        sep_token = Token(self._tokenizer_internal.sep_token, text_id = self._tokenizer_internal.sep_token_id)
        #pad_token = self._tokenizer_internal.pad_token
        sep_token_extra = bool(self._model_type in ['roberta'])
        cls_token_at_end = bool(self._model_type in ['xlnet'])
        cls_token_segment_id = 2 if self._model_type in ['xlnet'] else 0
        #pad_on_left = bool(self._model_type in ['xlnet'])
        #pad_token_segment_id = 4 if self._model_type in ['xlnet'] else 0
        #pad_token_val=self._tokenizer.encoder[pad_token] if self._model_type in ['roberta'] else self._tokenizer.vocab[pad_token]
        question_tokens = self._tokenizer.tokenize(question)
        if context is not None:
            context_tokens = self._tokenizer.tokenize(context)
        else:
            context_tokens = []

        seps = self._context_syntax.count("#")
        sep_mult = 2 if sep_token_extra else 1
        max_tokens = self._max_pieces - seps * sep_mult - 1

        choice_tokens = self._tokenizer.tokenize(answer)

        context_tokens, question_tokens, choice_tokens = self._truncate_tokens(context_tokens,
                                                                               question_tokens,
                                                                               choice_tokens,
                                                                               max_tokens)
        tokens = []
        segment_ids = []
        current_segment = 0
        token_dict = {"q": question_tokens, "c": context_tokens, "a": choice_tokens}
        for c in self._context_syntax:
            if c in "qca":
                new_tokens = token_dict[c]
                tokens += new_tokens
                segment_ids += len(new_tokens) * [current_segment]
            elif c == "#":
                tokens += sep_mult * [sep_token]
                segment_ids += sep_mult * [current_segment]
            elif c == "!":
                tokens += [sep_token]
                segment_ids += [current_segment]
            elif c == "_":
                current_segment += 1
            else:
                raise ValueError(f"Unknown context_syntax character {c} in {self._context_syntax}")

        if cls_token_at_end:
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        return tokens, segment_ids
