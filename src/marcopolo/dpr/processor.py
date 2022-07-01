from typing import Union, List, Dict, Any
from click import echo
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import DefaultDataCollator


def make_ex(
    examples,
    pos_sample_sz=1,
    negative_sample_sz=9,
    positive_key_nm: str = "positive_passages",
    negative_key_nm: str = "negative_passages",
):
    examples[positive_key_nm] = examples[positive_key_nm][:pos_sample_sz]
    examples[negative_key_nm] = examples[negative_key_nm][:negative_sample_sz]
    return examples


class DataCollatorForDPR(DefaultDataCollator):
    """
    Processing training examples for DPR
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        add_title: bool = True,
    ):
        self.tokenizer = tokenizer
        self.add_title = add_title
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_data_examples(examples)
        return examples

    def _prepare_data_examples(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # https://huggingface.co/docs/transformers/pad_truncation
        pos_title_list = [ex["positive_passages"][0]["title"] + ex["positive_passages"][0]["text"] for ex in examples]
        neg_title_list = [
            item["title"] + item["text"] for sublist in examples for item in sublist["negative_passages"]
        ]
        title_total = pos_title_list + neg_title_list
        query_list = [ex["query"] for ex in examples]
        query_encode = self.tokenizer(
            query_list, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt"
        )

        passages_encode = self.tokenizer(
            title_total, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt"
        )

        return {
            "question_batch": query_encode,
            "context_batch": passages_encode,
        }
