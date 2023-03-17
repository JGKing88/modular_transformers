import os
import torch
import tiktoken
import numpy as np


from copy import deepcopy
from typing import Optional

"""
Instantiate a class of Group_Texts for each dataset
    4 modes possible are:
        1. Default: Divides the tokens into sequence length, drops any remainder
        2. Padding only: Divides tokens into sequence length, adds padding tokens to remainder tokens
        3. Stride only: Moves forward by the input stride, creates overlapping tokens in consecutive rows. 
                        Drops remainder tokens that do not fit into the sequence length at the last stride
                        Use test_bool = True (false by default) to mask overlapping tokens
        4. Padding and Stride: moves forward by input stride, adds padding tokens to remainder tokens
                        Use test_bool = True (false by default) to mask overlapping tokens

    Parameters include: sequence length[int], stride[int], padding[bool], padding token[int], test_bool[bool], batch_size[int]

    Use batch size of 1 to send row by row of tokenized_dataset to create grouped datasets. Using batch size of 1 
    and enabling padding will capture all of the original text with padding. Higher number of batches take less time.
    Batch size of 1000 is default.
    Padding token is by default the tokenizer's eos token. 
    Test_bool is only needed for any modes with striding to mask any overlapping tokens (default is unmasked)

To use:
    Instantiate an object of Group_Texts with desired inputs for choosing a speific grouping mode. 
    Use object.group_texts() to call the specific grouping function. 
    Returns a grouped Dataset object through the map function. 

Sources:
Written and tested in July-August 2022 by Sally Shin.
Snippets of code were taken from mistral src/corpora/auto.py/group function (https://github.com/stanford-crfm/mistral)
and also from Huggingface notebook (https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb),
specifically the group_texts function. 
"""

class Group_Texts:
    def __init__(self,
                 tokenized_dataset,
                 tokenizer,
                 seq_len: int,
                 stride: Optional[int] = None,
                 padding: Optional[bool] = False,
                 padding_tok: Optional[int] = None,
                 test_bool: Optional[bool] = False,
                 batch_size: Optional[int] = 1000
                 ):
        # Set values for the class variables
        self.dataset = tokenized_dataset
        self.seq_len = seq_len
        self.test_bool = test_bool
        self.batch_size = batch_size

        # if-else for setting stride/padding/padding token
        # Padding false, stride None -> Default
        if padding is False and stride is None:
            self.stride = seq_len
            self.padding = padding
            print("Grouping texts with default mode without padding or stride at context length of", self.seq_len)
        # Padding true, stride None -> Only padding
        elif padding is True and stride is None:
            self.stride = seq_len
            self.padding = padding
            if padding_tok is not None:
                self.padding_tok = padding_tok
            elif padding_tok is None:
                # Doesn't matter what the padding token is since it will be masked dually by labels and attention mask
                # Can also set to the input id value of eos token
                self.padding_tok = (tokenizer(tokenizer.eos_token))["input_ids"][0]
                print(
                    f'Padding token defaulting to {(tokenizer(tokenizer.eos_token))["input_ids"][0]}, it will be masked by labels and attention mask')
            print("Grouping texts with padding with padding token", self.padding_tok, "at context length of", self.seq_len)
        # Padding false, stride a value -> Only stride
        elif padding is False and stride is not None:
            self.stride = stride
            self.padding = padding
            print("Grouping texts at a stride of", self.stride, "at context length of", self.seq_len)
        # Padding true, stride a value -> Stride with padding
        elif padding is True and stride is not None:
            self.stride = stride
            self.padding = padding
            if padding_tok is not None:
                self.padding_tok = padding_tok
            elif padding_tok is None:
                self.padding_tok = (tokenizer(tokenizer.eos_token))["input_ids"][0]
                print(
                    f'Padding token defaulting to {(tokenizer(tokenizer.eos_token))["input_ids"][0]}, it will be masked by labels and attention mask')
            print("Grouping texts with padding with padding token", self.padding_tok, "and stride of", self.stride, "at context length of", self.seq_len)

    # Actual function used on instantiated object of Group_Texts
    # Uses map function send rows of batch_size to be grouped by specific grouping function
    def group_texts(self):
        # Call preferred grouping function
        return self.dataset.map(self.get_grouping, batched=True, batch_size=self.batch_size)

    # Default function with no padding or striding
    # Drops tokens that do not fit into a multiple of seq_len
    # Not recommended to use with batch_size = 1 since many tokenized rows do not reach the context length
    def group_default(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length_use = (total_length // self.seq_len) * self.seq_len
        result = {
            k: [t[i: i + self.seq_len] for i in range(0, total_length_use, self.seq_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

        # Some checks to make sure all rows are same as sequence length
        assert all([len(x) == self.seq_len for x in result["input_ids"]])
        assert all([len(x) == self.seq_len for x in result["attention_mask"]])
        assert all([len(x) == self.seq_len for x in result["labels"]])

        return result

    # Only Padding function
    # Same as default but takes the left out tokens and pads to seq_len
    # If used with batch_size = 1, can capture all the original text, with a lot of padding.
    def group_padding(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Find what length to add padding
        remainder = total_length % self.seq_len
        if remainder != 0:
            to_add = self.seq_len - remainder
        elif remainder == 0:
            to_add = 0
        to_add_input_id = [self.padding_tok] * to_add
        to_add_atten_mask = [0] * to_add
        # Merge the two Dict variables
        pad_dict = dict(input_ids=to_add_input_id, attention_mask=to_add_atten_mask)
        for key in concatenated_examples.keys():
            t = concatenated_examples[key]
            t1 = [item for sublist in [t, pad_dict[key]] for item in sublist]
            assert not len(t1) % self.seq_len
            concatenated_examples[key] = t1
        total_length_use = len(concatenated_examples[list(examples.keys())[0]])
        result = {
            k: [t[i: i + self.seq_len] for i in range(0, total_length_use, self.seq_len)]
            for k, t in concatenated_examples.items()
        }
        # Labels is copied from input ids
        result["labels"] = result["input_ids"].copy()

        # Label is -100 if attention mask is 0, otherwise same as input ids
        result["labels"] = [
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in
            [zip(masks, labels) for masks, labels in zip(result["attention_mask"], result["labels"])]
        ]

        # Some checks to make sure all rows are same as sequence length
        assert all([len(x) == self.seq_len for x in result["input_ids"]])
        assert all([len(x) == self.seq_len for x in result["attention_mask"]])
        assert all([len(x) == self.seq_len for x in result["labels"]])

        return result

    # Only Stride function
    # Takes batches at length seq_len, moving forward at stride number
    # IF test_bool = True, will mask out tokens that are reused the next batch with label of -100 and attention mask of 0
    # Test_bool is False by default
    def group_stride(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # When stride is less than sequence length, overlaps are expected
        if self.stride < self.seq_len:
            total_length_use = ((total_length - self.seq_len + self.stride) // self.stride) * self.stride
        # When stride is bigger, no overlaps are expected and some tokens in between batches will be skipped
        elif self.stride > self.seq_len:
            # The first index is counted
            count_length = total_length - self.seq_len
            count_indice = 1
            # Increases index count while the leftover length is bigger or equal to the stride value
            while count_length >= self.stride:
                count_indice += 1
                count_length = count_length - self.stride
            # Total length used is index count * stride value
            total_length_use = count_indice * self.stride

        # Creates a new Dict based on the total_length_use from each case
        result = {
            k: [t[i: i + self.seq_len] for i in range(0, total_length_use, self.stride)]
            for k, t in concatenated_examples.items()}

        # Copies over input ids to new column called labels
        result["labels"] = deepcopy(result["input_ids"])

        # Mask out losses in overlapping regions
        # Changes masked labels to -100 and attention mask to 0
        # SKIP this part for training - Don't skip for testing
        if self.test_bool:
            for i, labels in enumerate(result["labels"]):
                # Skip the first index since the first batch will not have any masking
                if i == 0:
                    continue
                # For every j in range from 0 to length-stride, label to -100 to mask them
                for j in range(self.seq_len - self.stride):
                    labels[j] = -100
                # Set the newly masked list of labels to result Dict object
                result["labels"][i] = labels

            for i, attention in enumerate(result["attention_mask"]):
                # Skip the first index since the first batch will not have any masking
                if i == 0:
                    continue
                # For every j in range from 0 to length-stride, label to -100 to mask them
                for j in range(self.seq_len - self.stride):
                    attention[j] = 0
                # Set the newly masked list of labels to result Dict object
                result["attention_mask"][i] = attention

        # Some checks to make sure all rows are same as sequence length
        assert all([len(x) == self.seq_len for x in result["input_ids"]])
        assert all([len(x) == self.seq_len for x in result["attention_mask"]])
        assert all([len(x) == self.seq_len for x in result["labels"]])

        return result

    # Padding and stride function
    def group_padding_stride(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Finds just the quotient of total_length - seq_len by stride
        total_length_stride = ((total_length - self.seq_len + self.stride) // self.stride) * self.stride

        # If stride is less than sequence length, different padding method is employed.
        # Get the remainder and subtract to get the length of padding to add to fit the last stride
        if self.stride < self.seq_len:
            remainder = (total_length - self.seq_len) % self.stride
            # If remainder is a positive value, then to_add is a nonzero value
            if remainder > 0:
                to_add = self.seq_len - remainder
            # If remainder is 0, no need to add padding
            elif remainder == 0:
                to_add = 0
            to_add_input_id = [self.padding_tok] * to_add
            to_add_atten_mask = [0] * to_add
            pad_dict = dict(input_ids=to_add_input_id, attention_mask=to_add_atten_mask)
            # Add the padding Dict to concatenated_examples
            for key in concatenated_examples.keys():
                t = concatenated_examples[key]
                t1 = [item for sublist in [t, pad_dict[key]] for item in sublist]
                concatenated_examples[key] = t1
            # Add self.stride to add 1 more index for padded index
            total_length_use = total_length_stride + self.stride
            # New Dict object based that samples at length seq_len with stride
            result = {k: [t[i: i + self.seq_len] for i in range(0, total_length_use, self.stride)] for k, t in
                      concatenated_examples.items()}
        elif self.stride > self.seq_len:
            # Count index is 1 for the first index
            count_length = total_length - self.seq_len
            count_index = 1
            # Increments count index while leftover length is bigger or equal to stride
            while count_length >= self.stride:
                count_index += 1
                count_length = count_length - self.stride
            # If the difference between stride and sequence length is smaller than count length, some padding is added
            if self.stride - self.seq_len < count_length:
                to_add = self.stride - count_length
                to_add_input_id = [self.padding_tok] * to_add
                to_add_atten_mask = [0] * to_add
                total_length_use = (count_index + 1) * self.stride
                pad_dict = dict(input_ids=to_add_input_id, attention_mask=to_add_atten_mask)
                for key in concatenated_examples.keys():
                    t = concatenated_examples[key]
                    t1 = [item for sublist in [t, pad_dict[key]] for item in sublist]
                    concatenated_examples[key] = t1
            # Else if the difference is larger, no padding is needed since the remaining tokens are out of bounds
            elif self.stride - self.seq_len >= count_length:
                total_length_use = count_index * self.stride
            # New Dict object based that samples at length seq_len with stride
            result = {
                k: [t[i: i + self.seq_len] for i in range(0, total_length_use, self.stride)]
                for k, t in concatenated_examples.items()}

        # Copies over input ids to new column called labels
        result["labels"] = deepcopy(result["input_ids"])

        # Label is -100 if attention mask is 0, otherwise same as input ids
        # Just for padding at the end
        result["labels"] = [
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in
            [zip(masks, labels) for masks, labels in zip(result["attention_mask"], result["labels"])]
        ]

        # SKIP for training, Don't skip for testing
        # Mask out losses in overlapping regions. If training data, string will be equal to seq_len
        if self.test_bool:
            for i, labels in enumerate(result["labels"]):
                # Skip the first index since the first batch will not have any masking
                if i == 0:
                    continue
                # For every j in range from 0 to length-stride, label to -100 to mask them
                for j in range(self.seq_len - self.stride):
                    labels[j] = -100
                # Set the newly masked list of labels to result Dict object
                result["labels"][i] = labels

            for i, attention in enumerate(result["attention_mask"]):
                # Skip the first index since the first batch will not have any masking
                if i == 0:
                    continue
                # For every j in range from 0 to length-stride, label to -100 to mask them
                for j in range(self.seq_len - self.stride):
                    attention[j] = 0
                # Set the newly masked list of labels to result Dict object
                result["attention_mask"][i] = attention

        # Some checks to make sure all rows are same as sequence length
        assert all([len(x) == self.seq_len for x in result["input_ids"]])
        assert all([len(x) == self.seq_len for x in result["attention_mask"]])
        assert all([len(x) == self.seq_len for x in result["labels"]])

        return result

    # If-else function calls based on padding and stride values of self
    def get_grouping(self, examples):
        # Split function calls by the inputs
        if self.padding is False and self.stride is self.seq_len:
            return self.group_default(examples)
        elif self.padding is True and self.stride is self.seq_len:
            return self.group_padding(examples)
        elif self.padding is False and self.stride is not self.seq_len:
            return self.group_stride(examples)
        elif self.padding is True and self.stride is not self.seq_len:
            return self.group_padding_stride(examples)
