import torch
import torch.nn as nn

class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x, **kwargs):
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class BartCloneModel(nn.Module):
    def __init__(self, encoder, config, args):
        super(BartCloneModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.classifier = BartClassificationHead(config)
        self.args = args

    def get_bart_vec(self, source_ids, attention_mask):

        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        a = hidden_states.size(0)
        b = hidden_states.size(-1)
        c = hidden_states[eos_mask, :].view(a, -1, b)
        d = c[:, -1, :]
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def forward(self, source_ids=None, attention_mask=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)
        vec = self.get_bart_vec(source_ids, attention_mask)

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
