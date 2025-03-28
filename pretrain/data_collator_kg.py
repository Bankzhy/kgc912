import torch
from typing import List

from vocab import Vocab

def collate_fn(batch, args, task, code_vocab, nl_vocab, ast_vocab):
    """
    Data collator function.

    Args:
        batch (list):
        args (argparse.Namespace):
        task (str):
        code_vocab (Vocab):
        nl_vocab (Vocab):
        ast_vocab (Vocab):

    Returns:
        dict: Model inputs

    """
    model_inputs = {}
    # rlp

    if task == "rlp" or task == "rrlp":
        # pass
        code_raw, ast_raw, name_raw, target_raw = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
            code_raw=code_raw,
            code_vocab=code_vocab,
            max_code_len=args.max_code_len,
            ast_raw=ast_raw,
            ast_vocab=ast_vocab,
            max_ast_len=args.max_ast_len,
            nl_raw=name_raw,
            nl_vocab=nl_vocab,
            max_nl_len=args.max_nl_len,
            no_ast=args.no_ast,
            no_nl=args.no_nl
        )
        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
            batch=target_raw,
            vocab=code_vocab,
            processor=Vocab.sos_processor,
            max_len=int(args.mass_mask_ratio * args.max_code_len)
        )
        model_inputs['labels'], _ = get_batch_inputs(batch=target_raw,
                                                     vocab=code_vocab,
                                                     processor=Vocab.eos_processor,
                                                     max_len=int(args.mass_mask_ratio * args.max_code_len))
    # mass
    elif task == "mass":

        code_raw, ast_raw, name_raw, target_raw = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
            code_raw=code_raw,
            code_vocab=code_vocab,
            max_code_len=args.max_code_len,
            ast_raw=ast_raw,
            ast_vocab=ast_vocab,
            max_ast_len=args.max_ast_len,
            nl_raw=name_raw,
            nl_vocab=nl_vocab,
            max_nl_len=args.max_nl_len,
            no_ast=args.no_ast,
            no_nl=args.no_nl
        )
        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
            batch=target_raw,
            vocab=code_vocab,
            processor=Vocab.sos_processor,
            max_len=int(args.mass_mask_ratio * args.max_code_len)
        )
        model_inputs['labels'], _ = get_batch_inputs(batch=target_raw,
                                                     vocab=code_vocab,
                                                     processor=Vocab.eos_processor,
                                                     max_len=int(args.mass_mask_ratio * args.max_code_len))
    # nlp
    elif task == "nlp" or task == "nlmp" or task == "mnp":
        code_raw, ast_raw, nl_raw, name_raw = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
            code_raw=code_raw,
            code_vocab=code_vocab,
            max_code_len=args.max_code_len,
            ast_raw=ast_raw,
            ast_vocab=ast_vocab,
            max_ast_len=args.max_ast_len,
            nl_raw=nl_raw,
            nl_vocab=nl_vocab,
            max_nl_len=args.max_nl_len,
            no_ast=args.no_ast,
            no_nl=args.no_nl
        )

        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
            batch=name_raw,
            vocab=nl_vocab,
            processor=Vocab.sos_processor,
            max_len=args.max_nl_len
        )
        model_inputs['labels'], _ = get_batch_inputs(batch=name_raw,
                                                     vocab=nl_vocab,
                                                     processor=Vocab.eos_processor,
                                                     max_len=args.max_nl_len)
    # cgp
    elif task == "cgp":
        code_raw, ast_raw, nl_raw, target_st, is_graph = map(list, zip(*batch))
        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
            code_raw=code_raw,
            code_vocab=code_vocab,
            max_code_len=args.max_code_len,
            ast_raw=ast_raw,
            ast_vocab=ast_vocab,
            max_ast_len=args.max_ast_len,
            nl_raw=nl_raw,
            nl_vocab=nl_vocab,
            max_nl_len=args.max_nl_len,
            no_ast=args.no_ast,
            no_nl=args.no_nl
        )
        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
            batch=target_st,
            vocab=ast_vocab,
            processor=Vocab.sos_processor,
            max_len=args.max_ast_len
        )

        model_inputs['decoder_input_ids'] = torch.cat([model_inputs['decoder_input_ids']], dim=-1)
        model_inputs['decoder_attention_mask'] = torch.cat([model_inputs['decoder_attention_mask']], dim=-1)

        model_inputs['labels'] = torch.tensor(is_graph, dtype=torch.long)
    elif task == "clone" or task == "clp":
        code_1_raw, ast_1_raw, name_1_raw, code_2_raw, ast_2_raw, name_2_raw, labels = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
            code_raw=code_1_raw,
            code_vocab=code_vocab,
            max_code_len=args.max_code_len,
            ast_raw=ast_1_raw,
            ast_vocab=ast_vocab,
            max_ast_len=args.max_ast_len,
            nl_raw=name_1_raw,
            nl_vocab=nl_vocab,
            max_nl_len=args.max_nl_len,
            no_ast=args.no_ast,
            no_nl=args.no_nl
        )
        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_concat_batch_inputs(
            code_raw=code_2_raw,
            code_vocab=code_vocab,
            max_code_len=args.max_code_len,
            ast_raw=ast_2_raw,
            ast_vocab=ast_vocab,
            max_ast_len=args.max_ast_len,
            nl_raw=name_2_raw,
            nl_vocab=nl_vocab,
            max_nl_len=args.max_nl_len,
            no_ast=args.no_ast,
            no_nl=args.no_nl
        )
        model_inputs['labels'] = torch.tensor(labels, dtype=torch.long)


    return model_inputs

def get_batch_inputs(batch: List[str], vocab: Vocab, processor=None, max_len=None):
    """
    Encode the given batch to input to the model.

    Args:
        batch (list[str]): Batch of sequence,
            each sequence is represented by a string or list of tokens
        vocab (Vocab): Vocab of the batch
        processor (tokenizers.processors.PostProcessor): Optional, post-processor method
        max_len (int): Optional, the maximum length of each sequence

    Returns:
        (torch.LongTensor, torch.LongTensor): Tensor of batch and mask, [B, T]

    """
    # set post processor
    vocab.tokenizer.post_processor = processor
    # set truncation
    if max_len:
        vocab.tokenizer.enable_truncation(max_length=max_len)
    else:
        vocab.tokenizer.no_truncation()
    # encode batch
    inputs, padding_mask = vocab.encode_batch(batch, pad=True, max_length=max_len)
    # to tensor
    inputs = torch.tensor(inputs, dtype=torch.long)
    padding_mask = torch.tensor(padding_mask, dtype=torch.long)

    return inputs, padding_mask

def get_concat_batch_inputs(code_raw, code_vocab, max_code_len,
                            ast_raw, ast_vocab, max_ast_len,
                            nl_raw, nl_vocab, max_nl_len,
                            no_ast=False, no_nl=False):
    """
    Return the concat tensor and mask for input.

    Args:
        code_raw:
        code_vocab:
        max_code_len:
        ast_raw:
        ast_vocab:
        max_ast_len:
        nl_raw:
        nl_vocab:
        max_nl_len:
        no_ast:
        no_nl:

    Returns:
        (torch.Tensor, torch.Tensor):
            - Concat inputs
            - concat attention mask

    """
    code_inputs, code_padding_mask = get_batch_inputs(batch=code_raw,
                                                      vocab=code_vocab,
                                                      processor=Vocab.sep_processor,
                                                      max_len=max_code_len)

    if not no_ast:
        ast_inputs, ast_padding_mask = get_batch_inputs(batch=ast_raw,
                                                        vocab=ast_vocab,
                                                        processor=Vocab.sep_processor,
                                                        max_len=max_ast_len)
    else:
        ast_inputs, ast_padding_mask = None, None

    if not no_nl:
        nl_inputs, nl_padding_mask = get_batch_inputs(batch=nl_raw,
                                                      vocab=nl_vocab,
                                                      processor=Vocab.eos_processor,
                                                      max_len=max_nl_len)
    else:
        nl_inputs, nl_padding_mask = None, None

    inputs = torch.cat([inputs for inputs in [code_inputs, ast_inputs, nl_inputs] if inputs is not None], dim=-1)
    padding_mask = torch.cat([mask for mask in [code_padding_mask, ast_padding_mask, nl_padding_mask]
                              if mask is not None], dim=-1)

    # code_inputs, code_padding_mask = get_batch_inputs(batch=code_raw,
    #                                                   vocab=code_vocab,
    #                                                   processor=Vocab.sep_processor,
    #                                                   max_len=max_code_len)
    # ast_inputs, ast_padding_mask = get_batch_inputs(batch=ast_raw,
    #                                                 vocab=ast_vocab,
    #                                                 processor=Vocab.sep_processor,
    #                                                 max_len=max_ast_len)
    # nl_inputs, nl_padding_mask = get_batch_inputs(batch=nl_raw,
    #                                               vocab=nl_vocab,
    #                                               processor=Vocab.eos_processor,
    #                                               max_len=max_nl_len)
    #
    # inputs = torch.cat([code_inputs, ast_inputs, nl_inputs], dim=-1)
    # padding_mask = torch.cat([code_padding_mask, ast_padding_mask, nl_padding_mask], dim=-1)

    return inputs, padding_mask
