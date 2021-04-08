import math

import torch
import transformers
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertEncoder, BERT_INPUTS_DOCSTRING, \
    _TOKENIZER_FOR_DOC, _CONFIG_FOR_DOC
from torch import nn

BERT_MODELS = (
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-large-uncased-whole-word-masking',
    'bert-large-cased-whole-word-masking'
)


class BertEmbeddingsArgs(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

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
        if args.use_pos:
            self.token_pos_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                                     padding_idx=config.pad_token_id)
        if args.use_dep:
            self.token_dep_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                                     padding_idx=config.pad_token_id)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, pos_ids=None, dep_ids=None, inputs_embeds=None):
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
        if pos_ids is None and not self.args.use_pos:
            raise ValueError("The model needs pos_ids for it to create its embeddings")

        # If no dep are provided: make all tokens to [NONE_POS]??? Make all tokens [PAD]???
        if dep_ids is None and not self.args.use_dep:
            raise ValueError("The model needs dep_ids for it to create its embeddings")

        # ====== Create the embeddings from the ids ======
        # if no input embeddings are directly provided, look up the embeddings corresponding to the input ids
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if self.args.use_pos:
            token_pos_embeddings = self.token_pos_embeddings(pos_ids)

        if self.args.use_dep:
            token_dep_embeddings = self.token_dep_embeddings(dep_ids)

        # ====== Add the embeddings together, normalize and perform dropout ======
        if self.args.use_pos and self.args.use_dep:
            embeddings = inputs_embeds + position_embeddings + token_type_embeddings +\
                         token_pos_embeddings + token_dep_embeddings
        elif self.args.use_pos and not self.args.use_dep:
            embeddings = inputs_embeds + position_embeddings + token_type_embeddings + token_pos_embeddings
        elif not self.args.use_pos and self.args.use_dep:
            embeddings = inputs_embeds + position_embeddings + token_type_embeddings + token_dep_embeddings
        else:
            inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelArgs(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, args, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.args = args

        self.embeddings = BertEmbeddingsArgs(config, args)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

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
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
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

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
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


class BertWSDArgs(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        self.bert = BertModelArgs(config, args)
        # Add new embedding class to model!
        self.embeddings = BertEmbeddingsArgs(config, args)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()


def get_model_and_tokenizer(args, tokens_to_add):
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=bool('uncased' in args.model_name_or_path),
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    model = BertWSDArgs.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    # add new special token
    for special_token in tokens_to_add:
        if special_token not in tokenizer.additional_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})
            assert special_token in tokenizer.additional_special_tokens
            model.resize_token_embeddings(len(tokenizer))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    return model, tokenizer


def _forward(args, model, batch):
    batch = tuple(t.to(args.device) for t in batch)
    outputs = model.bert(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])

    return model.dropout(outputs[1])


def _compute_weighted_loss(loss, weighting_factor):
    squared_factor = weighting_factor ** 2

    return 1 / (2 * squared_factor) * loss + math.log(1 + squared_factor)


def forward_gloss_selection(args, model, batches):
    batch_loss = 0
    logits_list = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch in batches:
        logits = model.ranking_linear(_forward(args, model, batch)).squeeze(-1)
        labels = torch.max(batch[3].to(args.device), -1).indices.to(
            args.device).detach()  # reminder: batch[3] is the label.

        batch_loss += loss_fn(logits.unsqueeze(dim=0), labels.unsqueeze(dim=-1))  # loss(predicted, target)
        logits_list.append(logits)

    loss = batch_loss / len(batches)

    return loss, logits_list





"""
# ======= Code scraps =======
"""



class BertEmbeddingsPos(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
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
        self.token_pos_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, pos_ids=None, inputs_embeds=None):
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
        if pos_ids is None:
            raise ValueError("The model needs pos_ids for it to create its embeddings")

        # ====== Create the embeddings from the ids ======
        # if no input embeddings are directly provided, look up the embeddings corresponding to the input ids
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        token_pos_embeddings = self.token_pos_embeddings(pos_ids)

        # ====== Add the embeddings together, normalize and perform dropout ======
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + token_pos_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertWSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        # Add new embedding class to model!
        self.embeddings = BertEmbeddingsPos()
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()

    # How to define ini function, which takes the new BERT embeddings?
    # def __init__(self, config):
    #    super(BertModel, self).__init__(config)
    #    self.config = config
    #
    #    self.embeddings = BertEmbeddings(config)
    #    self.encoder = BertEncoder(config)
    #    self.pooler = BertPooler(config)
    #
    #    self.init_weights()