import torch
import torch.nn as nn

# class BartClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""
#
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.d_model * 2, config.d_model)
#         self.out_proj = nn.Linear(config.d_model, 2)
#
#     def forward(self, x, **kwargs):
#         x = x.reshape(-1, x.size(-1) * 2)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.out_proj(x)
#         return x
class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class BartCloneModel(nn.Module):
    def __init__(self, encoder, config, args):
        super(BartCloneModel, self).__init__()
        self.encoder = encoder
        self.config = config
        # self.classifier = BartClassificationHead(config)
        self.classifier = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.args = args

    def get_bart_vec(self, inputs):

        outputs = self.encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            # decoder_input_ids=inputs['decoder_input_ids'],
            # decoder_attention_mask=inputs['decoder_attention_mask'],
            labels=inputs['labels'],
        )
        hidden_states = outputs['decoder_hidden_states']
        input_ids = inputs['input_ids']
        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        # a = hidden_states.size(0)
        # b = hidden_states.size(-1)
        # c = hidden_states[eos_mask, :].view(a, -1, b)
        # d = c[:, -1, :]
        # vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
        #                                       hidden_states.size(-1))[:, -1, :]

        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                                  hidden_states.size(-1))[
                                  :, -1, :
                                  ]

        return vec

    def forward(self, inputs):
        # source_ids = source_ids.view(-1, self.args.max_source_length)
        vec = self.get_bart_vec(inputs)
        labels = inputs["labels"]

        # if self.args.model_type == 'codet5':
        #     vec = self.get_t5_vec(source_ids)
        # elif self.args.model_type == 'bart':
        #     vec = self.get_bart_vec(source_ids)
        # elif self.args.model_type == 'roberta':
        #     vec = self.get_roberta_vec(source_ids)

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
