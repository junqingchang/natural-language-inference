import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, BertConfig
from transformers.modeling_bert import BertOutput, BertIntermediate, BertSelfOutput, BertEncoder, BertPooler, BertOnlyMLMHead,BertEmbeddings
from transformers.file_utils import is_torch_available
import math
import re
import itertools

class BERT(nn.Module):
    def __init__(self, num_classes=3, bert_type='bert-large-cased'):
        super(BERT, self).__init__()
        assert bert_type == 'bert-large-cased' or bert_type == 'bert-base-cased'
        self.bert = BertModel.from_pretrained(bert_type)
        self.dropout = nn.Dropout(0.1)
        if bert_type == 'bert-base-cased':
            self.classifier = nn.Linear(768, num_classes)
        elif bert_type == 'bert-large-cased':
            self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        outputs = self.bert(
            x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'])
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits



class BertEmbeddingsWithWordMasking(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings as well as word mask.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.word_arrangement_embeddings = nn.Embedding(2, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, word_mask=None):
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

        if word_mask is None:
            word_mask = torch.ones(input_shape, dtype=torch.long, device=device)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        word_arrangement_embeddings = self.word_arrangement_embeddings(word_mask)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + word_arrangement_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelWithWordMasking(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsWithWordMasking(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings


    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        word_mask=None
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if word_mask is None:
            word_mask = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, word_mask=word_mask
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    
class BertTokenizerWithWordMasking(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        **kwargs
    ):
        super().__init__(vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        **kwargs)

    def _tokenize(self, text):
        split_tokens = []
        word_mask = []
        first = True
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                first = True
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    if first:
                        word_mask.append(1)
                        first = False
                    else:
                        word_mask.append(0)
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens, word_mask

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        add_special_tokens: bool = True,
        max_length = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        return_tensors = None,
        return_token_type_ids = None,
        return_attention_masks = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_masks: bool = False,
        return_offsets_mapping: bool = False,
        return_input_lengths: bool = False,
        **kwargs
    ):

        def get_input_ids(text):
            if isinstance(text, str):
                tokens, word_mask = self.tokenize(text, add_special_tokens=add_special_tokens, **kwargs)
                return self.convert_tokens_to_ids(tokens), word_mask
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError(
                "Unable to set proper padding strategy as the tokenizer does not have a padding token. In this case please set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via the function add_special_tokens if you want to use a padding strategy"
            )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if isinstance(ids_or_pair_ids, (list, tuple)) and len(ids_or_pair_ids) == 2:
                ids, pair_ids = ids_or_pair_ids
            else:
                ids, pair_ids = ids_or_pair_ids, None

            first_ids, first_word_mask = get_input_ids(ids)
            second_ids, second_word_mask = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids, first_word_mask, second_word_mask))

        if max_length is None and pad_to_max_length:

            def total_sequence_length(input_pairs):
                first_ids, second_ids, first_word_mask, second_word_mask = input_pairs
                return len(first_ids) + (
                    self.num_added_tokens()
                    if second_ids is None
                    else (len(second_ids) + self.num_added_tokens(pair=True))
                )

            max_length = max([total_sequence_length(ids) for ids in input_ids])

        batch_outputs = {}
        for first_ids, second_ids, first_word_mask, second_word_mask in input_ids:
            outputs = self.prepare_for_model(
                first_ids,
                pair_ids=second_ids,
                max_length=max_length,
                pad_to_max_length=pad_to_max_length,
                add_special_tokens=add_special_tokens,
                stride=stride,
                truncation_strategy=truncation_strategy,
                return_attention_mask=return_attention_masks,
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_masks,
            )
            outputs['word_mask'] = [0] + first_word_mask + [0] + second_word_mask + [0]
            outputs['word_mask'] += [0] * (max_length-len(outputs['word_mask']))
            # Append the non-padded length to the output
            if return_input_lengths:
                outputs["input_len"] = len(outputs["input_ids"])

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        if return_tensors is not None:

            # Do the tensor conversion in batch
            for key, value in batch_outputs.items():
                if return_tensors == "pt" and is_torch_available():
                    try:
                        batch_outputs[key] = torch.tensor(value)
                    except ValueError:
                        raise ValueError(self.UNEVEN_SEQUENCES_FOR_BATCH_MSG)
                    except RuntimeError:
                        if None in [item for sequence in value for item in sequence]:
                            raise ValueError(self.NO_PAD_TOKEN_FOR_BATCH_MSG)
                        else:
                            raise
                elif return_tensors is not None:
                    print(
                        "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                            return_tensors
                        )
                    )

        return batch_outputs


class BertForMaskedLMWithWordMasking(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModelWithWordMasking(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        word_mask=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            word_mask=word_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs


class BERTWithWordMasking(nn.Module):
    def __init__(self, num_classes=3, bert_type='bert-base-cased'):
        super(BERTWithWordMasking, self).__init__()
        assert bert_type == 'bert-large-cased' or bert_type == 'bert-base-cased'
        self.bert = BertModelWithWordMasking.from_pretrained(bert_type)
        self.dropout = nn.Dropout(0.1)
        if bert_type == 'bert-base-cased':
            self.classifier = nn.Linear(768, num_classes)
        elif bert_type == 'bert-large-cased':
            self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        outputs = self.bert(
            x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'], word_mask=x['word_mask'])
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class BERTWithWordMaskingSelfPretrained(nn.Module):
    def __init__(self, num_classes=3, pretrained_path='chkpt/maskbert.pt', tokenizer_vocab='configs/maskbert-vocab.txt'):
        super(BERTWithWordMaskingSelfPretrained, self).__init__()
        tokenizer = BertTokenizerWithWordMasking(tokenizer_vocab)
        config = BertConfig(vocab_size=tokenizer.vocab_size)
        checkpoint = torch.load(pretrained_path)
        subcheckpoint = {}
        for key in checkpoint['model_state_dict']:
            if key.startswith('bert.'):
                subcheckpoint[key[5:]] = checkpoint['model_state_dict'][key]
        checkpoint = subcheckpoint
        self.bert = BertModelWithWordMasking(config)
        self.bert.load_state_dict(checkpoint)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        outputs = self.bert(
            x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'], word_mask=x['word_mask'])
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
    
#-----------------------------modify pooler layer activation ------------------------
class BertPooler_Sigmoid(nn.Module):
    def __init__(self, config):
        super(BertPooler_Sigmoid, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.activation = nn.Tanh() #--original
        self.activation = nn.Sigmoid()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BertPooler_reLu(nn.Module):
    def __init__(self, config):
        super(BertPooler_reLu, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.activation = nn.Tanh() #--original
        self.activation = nn.ReLu()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

 ###############attention#############################################
class BertSelfAttention_attention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention_attention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        #self.sigmoid1 = nn.Sigmoid()
        #self.sigmoid1 = nn.LogSoftmax(dim=1)
        self.sigmoid1 = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    # taken from github address: https://gist.github.com/pbamotra/13788e344b5ed839216a0e80a8c09b37
    def logsigsoftmax(logits):
        """
        Computes sigsoftmax from the paper - https://arxiv.org/pdf/1805.10829.pdf
        """
        max_values = torch.max(logits, 1, keepdim = True)[0]
        exp_logits_sigmoided = torch.exp(logits - max_values) * torch.sigmoid(logits)
        sum_exp_logits_sigmoided = exp_logits_sigmoided.sum(1, keepdim = True)
        log_probs = logits - max_values + torch.log(torch.sigmoid(logits)) - torch.log(sum_exp_logits_sigmoided)
        return log_probs
    
    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        #attention_probs = nn.Softmax(dim=-1)(attention_scores) #--oriinal
#        attention_probs = nn.Sigmoid(attention_scores(1))
        attention_probs = self.sigmoid1(attention_scores)
        #attention_probs = self.logsigsoftmax(attention_scores)
        #attention_probs = nn.LogSoftmax(dim=-1)(attention_scores)


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs
    
class BertAttention_attention(nn.Module):
    def __init__(self, config):
        super(BertAttention_attention, self).__init__()
        self.self = BertSelfAttention_attention(config)
        self.output = BertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
    
    
class BertLayer_attention(nn.Module):
    def __init__(self, config):
        super(BertLayer_attention, self).__init__()
        self.attention = BertAttention_attention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder_attention (nn.Module):
    def __init__(self, config):
        super(BertEncoder_attention, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer_attention(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertModel_AttentionActivation(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder_attention(config)
        self.pooler = BertPooler(config)
        # self.pooler =  BertPooler_Sigmoid(config)
        #self.pooler = BertPooler_reLu(config)

        #self.apply(self.init_weights)
        self.init_weights()
    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask,head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

#-------------------------for pool layer activation function--------
class BERT_withAttentionActivation(nn.Module):
    def __init__(self, num_classes=3, bert_type='bert-large-cased'):
        super(BERT_withAttentionActivation, self).__init__()
        assert bert_type == 'bert-large-cased' or bert_type == 'bert-base-cased'
        
        self.bert = BertModel_AttentionActivation.from_pretrained(bert_type)
        self.dropout = nn.Dropout(0.1)
        if bert_type == 'bert-base-cased':
            self.classifier = nn.Linear(768, num_classes)
        elif bert_type == 'bert-large-cased':
            self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        outputs = self.bert(x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'])
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
