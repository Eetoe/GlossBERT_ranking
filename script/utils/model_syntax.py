import math

import torch
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from transformers.modeling_bert import BertEmbeddings
from torch import nn

BERT_MODELS = (
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-large-uncased-whole-word-masking',
    'bert-large-cased-whole-word-masking'
)


class BertEmbeddingsSyntax(nn.module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    # --init__() function unchanged
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)


        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        BertLayerNorm = torch.nn.LayerNorm
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    #Add pos embeddings here
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, pos_ids=None, dependency_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            pos_tag_embeds = self.word_embeddings(pos_ids)
            #dependency_embeds = self.word_embeddings(dependency_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + pos_tag_embeds# + dependency_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertWSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        # Add new embedding class to model!
        self.embeddings = BertEmbeddingsSyntax()
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()

    # How to define ini function, which takes the new BERT embeddings?
    #def __init__(self, config):
    #    super(BertModel, self).__init__(config)
    #    self.config = config
    #
    #    self.embeddings = BertEmbeddings(config)
    #    self.encoder = BertEncoder(config)
    #    self.pooler = BertPooler(config)
    #
    #    self.init_weights()


def get_model_and_tokenizer(args):
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
    model = BertWSD.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    # add new special token
    for special_token in ['[TGT]', '[NOUN]', '[VERB]', '[ADJ]', '[ADV]']:
        if special_token not in tokenizer.additional_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
            assert '[TGT]' in tokenizer.additional_special_tokens
            model.resize_token_embeddings(len(tokenizer))

    #if '[TGT]' not in tokenizer.additional_special_tokens:
    #    tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
    #    assert '[TGT]' in tokenizer.additional_special_tokens
    #    model.resize_token_embeddings(len(tokenizer))

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
        labels = torch.max(batch[3].to(args.device), -1).indices.to(args.device).detach() # reminder: batch[3] is the label.

        batch_loss += loss_fn(logits.unsqueeze(dim=0), labels.unsqueeze(dim=-1)) # loss(predicted, target)
        logits_list.append(logits)

    loss = batch_loss / len(batches)

    return loss, logits_list

