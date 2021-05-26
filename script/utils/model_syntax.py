import collections
import os
from typing import Optional, List, Tuple

import torch
import transformers
from torch import nn
from transformers import BertPreTrainedModel, PretrainedConfig, PreTrainedTokenizer, WordpieceTokenizer, BasicTokenizer
from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertPooler, BertEncoder, BERT_INPUTS_DOCSTRING, \
    _TOKENIZER_FOR_DOC, _CONFIG_FOR_DOC
from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES, PRETRAINED_VOCAB_FILES_MAP, \
    PRETRAINED_INIT_CONFIGURATION, \
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES, load_vocab

BERT_MODELS = (
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-large-uncased-whole-word-masking',
    'bert-large-cased-whole-word-masking'
)

"""
First, make custom classes for BERT:
    - Make a custom config class
    - Make a custom tokenizer class
    - Make a custom embedding class
    - Make a custom model class
    - Make a custom model class, which builds on top of the previous model class
    
Then make functions for loading and running the model

"""


# ====== Custom config for BERT ======
class BertConfigSyntax(PretrainedConfig):
    """
    Custom config for BERT, modified from:
        https://huggingface.co/transformers/_modules/transformers/models/bert/configuration_bert.html#BertConfig

    Vocab sizes have been added for pos tag and grammatical dependencies.
    """
    model_type = "bert"

    def __init__(
            self,
            vocab_size=30522,
            pos_vocab_size=24,
            dep_vocab_size=51,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            position_embedding_type="absolute",
            use_cache=True,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.pos_vocab_size = pos_vocab_size
        self.dep_vocab_size = dep_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache


# ====== Try to add new tokenizer class =======
class BertTokenizerArgs(PreTrainedTokenizer):
    """
    Custom tokenizer class from BERT modified from:
        https://huggingface.co/transformers/_modules/transformers/models/bert/tokenization_bert.html#BertTokenizer

    The only new things added are:
     - The vocabulary for POS tags and grammatical dependencies. Loaded from files created when loading the tokenizer.
     - A function for getting the ids of the corresponding syntax tokens
     - A function for getting the tokens of the corresponding syntax ids
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
            self,
            args,
            vocab_file,
            pos_vocab_file="",
            dep_vocab_file="",
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            strip_accents=None,
            **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        self.vocab = load_vocab(vocab_file)
        if args.use_pos_tags:
            self.pos_vocab = load_vocab(pos_vocab_file)
        if args.use_dependencies:
            self.dep_vocab = load_vocab(dep_vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        if args.use_pos_tags:
            self.pos_ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.pos_vocab.items()])
        if args.use_dependencies:
            self.dep_ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.dep_vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pos_vocab_size(self):
        return len(self.pos_vocab)

    @property
    def dep_vocab_size(self):
        return len(self.dep_vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_syntax_tokens_to_ids(self, tokens, syntax_type):
        ids = [self._convert_syntax_token_to_id(token, syntax_type) for token in tokens]
        return ids

    def _convert_syntax_token_to_id(self, token, syntax_type):
        """ Converts a token (str) in an id using the vocab. """
        if syntax_type not in ["pos", "dep"]:
            raise ValueError('Syntac type should be either "pos" for POS tags or "dep" for dependencies.')
        if syntax_type == "pos":
            return self.pos_vocab.get(token, self.pos_vocab.get(self.unk_token))
        if syntax_type == "dep":
            return self.dep_vocab.get(token, self.dep_vocab.get(self.unk_token))

    def convert_syntax_ids_to_tokens(self, ids, syntax_type):
        ids = [self._convert_syntax_id_to_token(id, syntax_type) for id in ids]
        return ids

    def _convert_syntax_id_to_token(self, index, syntax_type):
        """Converts an index (integer) in a token (str) using the vocab."""
        if syntax_type not in ["pos", "dep"]:
            raise ValueError('Syntac type should be either "pos" for POS tags or "dep" for dependencies.')
        if syntax_type == "pos":
            return self.pos_ids_to_tokens.get(index, self.unk_token)
        if syntax_type == "dep":
            return self.dep_ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # logger.warning changed to print, as logger is defined outside the class
                    print(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

# ====== Make custom embedding class ======
class BertEmbeddingsArgs(nn.Module):
    """
    Custom embedding class modified from:
        https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertModel

    Simply adds the pos tag and dependency embeddings to the original class.

    """

    def __init__(self, config, args):
        super().__init__()
        """
            - Word embeddings: the embedding matrix of the model
            - Position embeddings: the tokens' positions
            - Token type embeddings: which part of the input the tokens belong to, aka the segment embeddings
            - token pos embeddings: the pos of the token, found using SpaCy
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.args = args
        if args.use_pos_tags:
            self.token_pos_embeddings = nn.Embedding(config.pos_vocab_size, config.hidden_size,
                                                     padding_idx=config.pad_token_id)
        if args.use_dependencies:
            self.token_dep_embeddings = nn.Embedding(config.dep_vocab_size, config.hidden_size,
                                                     padding_idx=config.pad_token_id)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, pos_ids=None, dep_ids=None,
                inputs_embeds=None):
        """
        input_ids vs. inputs_embeds
            - input ids are ids for looking up embeddings in the embedding matrix
            - input embeds are custom embeddings passed directly to the model.
        """

        # ====== Create the ids not provided directly ======
        # Use either input ids or embeds to find the shape of the input.
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # If the position ids are not provided, create them here from the input shape
        if position_ids is None:
            # Length of the input sequence, i.e. how many tokens that are used as input
            seq_length = input_shape[1]
            position_ids = self.position_ids[:, :seq_length]

        # if the segment embeddings aren't provided, create them here from the input shape
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # If no pos are provided: make all tokens to [NONE_POS]??? Make all tokens [PAD]???
        if pos_ids is None and self.args.use_pos_tags:
            raise ValueError("The model needs pos_ids for it to create its embeddings")

        # If no dep are provided: make all tokens to [NONE_POS]??? Make all tokens [PAD]???
        if dep_ids is None and self.args.use_dependencies:
            raise ValueError("The model needs dep_ids for it to create its embeddings")

        # ====== Create the embeddings from the ids ======
        # if no input embeddings are directly provided, look up the embeddings corresponding to the input ids
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if self.args.use_pos_tags:
            token_pos_embeddings = self.token_pos_embeddings(pos_ids)

        if self.args.use_dependencies:
            token_dep_embeddings = self.token_dep_embeddings(dep_ids)

        # ====== Add the embeddings together, normalize and perform dropout ======
        if self.args.use_pos_tags and self.args.use_dependencies:
            embeddings = inputs_embeds + position_embeddings + token_type_embeddings + \
                         token_pos_embeddings + token_dep_embeddings
        elif self.args.use_pos_tags and not self.args.use_dependencies:
            embeddings = inputs_embeds + position_embeddings + token_type_embeddings + token_pos_embeddings
        elif not self.args.use_pos_tags and self.args.use_dependencies:
            embeddings = inputs_embeds + position_embeddings + token_type_embeddings + token_dep_embeddings
        else:
            embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# ====== Make custom model class ======
class BertModelArgs(BertPreTrainedModel):
    """
    Custom BertModel class modified from:
        https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertModel

    Only a few changes:
        - Uses the custom bert embeddings made in this script.
        - Modified embedding output to match the custom bert embeddings.
        - Made new functions syntax embeddings copying the functions for word embeddings.
    """

    def __init__(self, config, args, add_pooling_layer=True):
        super().__init__(config, BertPreTrainedModel)
        self.config = config
        self.args = args

        self.embeddings = BertEmbeddingsArgs(config, args)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def get_pos_embeddings(self):
        return self.embeddings.token_pos_embeddings

    def get_dep_embeddings(self):
        return self.embeddings.token_dep_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def set_pos_embeddings(self, value):
        self.embeddings.token_pos_embeddings = value

    def set_dep_embeddings(self, value):
        self.embeddings.token_dep_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            pos_ids=None,
            dep_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # Output atention & hidden states set to use config values if values not provided directly
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Determine input shape from input ids or the provided embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # ====== Creating the attention mask ======
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # ======= creating outputs for embeds, encoders, sequences and pooling ======
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            pos_ids=pos_ids, dep_ids=dep_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


# ====== Make custom model class, which goes on top of other custom model class ======
class BertWSDArgs(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        self.bert = BertModelArgs(config, args)
        # Add new embedding class to model!
        # self.embeddings = BertEmbeddingsArgs(config, args)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()


# ====== Make functions to load and run the model ======
def get_model_and_tokenizer(args):
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    # Create
    pos_vocab_file, dep_vocab_file = _create_syntax_vocab(args)

    config = BertConfigSyntax.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    if args.use_pos_tags and args.use_dependencies:
        tokenizer = BertTokenizerArgs.from_pretrained(
            args.model_name_or_path,
            args,
            pos_vocab_file=pos_vocab_file,
            dep_vocab_file=dep_vocab_file,
            do_lower_case=bool('uncased' in args.model_name_or_path),
            cache_dir=args.cache_dir if args.cache_dir else None
        )
    elif args.use_pos_tags and not args.use_dependencies:
        tokenizer = BertTokenizerArgs.from_pretrained(
            args.model_name_or_path,
            args,
            pos_vocab_file=pos_vocab_file,
            do_lower_case=bool('uncased' in args.model_name_or_path),
            cache_dir=args.cache_dir if args.cache_dir else None
        )
    elif args.use_dependencies and not args.use_pos_tags:
        tokenizer = BertTokenizerArgs.from_pretrained(
            args.model_name_or_path,
            args,
            dep_vocab_file=dep_vocab_file,
            do_lower_case=bool('uncased' in args.model_name_or_path),
            cache_dir=args.cache_dir if args.cache_dir else None
        )
    else:
        tokenizer = BertTokenizerArgs.from_pretrained(
            args.model_name_or_path,
            args,
            do_lower_case=bool('uncased' in args.model_name_or_path),
            cache_dir=args.cache_dir if args.cache_dir else None
        )
    model = BertWSDArgs.from_pretrained(
        args.model_name_or_path,
        args,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    target_token = "[TGT]"
    if target_token not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [target_token]})
        assert target_token in tokenizer.additional_special_tokens
        model.resize_token_embeddings(len(tokenizer))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    return model, tokenizer


def _create_syntax_vocab(args):
    path = args.output_dir
    pos_vocab_path = path + "/pos_vocab.txt"
    dep_vocab_path = path + "/dep_vocab.txt"

    pos_vocab_list = ["[PAD]",
                      "[ADJ]", "[ADP]", "[ADV]", "[AUX]", "[CCONJ]", "[CONJ]",
                      "[DET]", "[INTJ]", "[NOUN]", "[NUM]", "[PART]", "[PRON]",
                      "[PROPN]", "[PUNCT]", "[SCONJ]", "[SYM]", "[VERB]", "[X]",
                      "[TGT_POS]", "[CLS_POS]", "[SEP_POS]",
                      "[NONE_POS]", "[GLOSS_POS]"]

    dep_vocab_list = ["[PAD]",
                      '[ROOT]', '[acl]', '[acomp]', '[advcl]', '[advmod]', '[agent]', '[amod]', '[appos]', '[attr]',
                      '[aux]', '[auxpass]', '[case]', '[cc]', '[ccomp]', '[compound]', '[conj]', '[csubj]',
                      '[csubjpass]',
                      '[dative]', '[dep]', '[det]', '[dobj]', '[expl]', '[intj]', '[mark]', '[meta]',
                      '[neg]', '[nmod]', '[npadvmod]', '[nsubj]', '[nsubjpass]', '[nummod]', '[oprd]',
                      '[parataxis]', '[pcomp]', '[pobj]', '[poss]', '[preconj]', '[predet]', '[prep]', '[prt]',
                      '[punct]',
                      '[quantmod]', '[relcl]', '[xcomp]',
                      "[TGT_DEP]", "[CLS_DEP]", "[SEP_DEP]",
                      "[NONE_DEP]", "[GLOSS_DEP]"]

    if args.use_pos_tags and not os.path.exists(pos_vocab_path):
        with open(pos_vocab_path, 'w') as f:
            for item in pos_vocab_list:
                f.write("%s\n" % item)

    if args.use_dependencies and not os.path.exists(dep_vocab_path):
        with open(dep_vocab_path, 'w') as f:
            for item in dep_vocab_list:
                f.write("%s\n" % item)

    return pos_vocab_path, dep_vocab_path


def _forward(args, model, batch):
    batch = tuple(t.to(args.device) for t in batch)
    if args.use_pos_tags and args.use_dependencies:
        outputs = model.bert(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2],
                             pos_ids=batch[3], dep_ids=batch[4])
    elif args.use_pos_tags and not args.use_dependencies:
        outputs = model.bert(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], pos_ids=batch[3])
    elif not args.use_pos_tags and args.use_dependencies:
        outputs = model.bert(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], dep_ids=batch[3])
    else:
        outputs = model.bert(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])

    return model.dropout(outputs[1])


def forward_gloss_selection(args, model, batches):
    batch_loss = 0
    logits_list = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch in batches:
        logits = model.ranking_linear(_forward(args, model, batch)).squeeze(-1)
        labels = torch.max(batch[-1].to(args.device), -1).indices.to(
            args.device).detach()  # reminder: batch[-1] is the label.

        batch_loss += loss_fn(logits.unsqueeze(dim=0), labels.unsqueeze(dim=-1))  # loss(predicted, target)
        logits_list.append(logits)

    loss = batch_loss / len(batches)

    return loss, logits_list
