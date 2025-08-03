import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class ClickbaitSpoilerDatasetParagraphLevel(Dataset):
    """
    Paragraph-level truncation:
      - postText, targetTitle and targetParagraphs are used as inputs
          - Only full paragraphs are included (avoid cutting mid-paragraph)
      - Token budget is respected (max 1000 input tokens)
          - 15 paragraphs or 1000 tokens, whichever comes first
      - humanSpoiler (Abstractive) is used as the target
    """

    def __init__(self, jsonl_path, tokenizer_name,
             *,  # forces keyword arguments
             max_input_tokens=1000,
             max_target_tokens=64,
             max_paragraphs=15):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_input_tokens = max_input_tokens
        self.max_target_tokens = max_target_tokens
        self.max_paragraphs = max_paragraphs
        self.is_test = is_test

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                post_text_list = entry.get("postText")
                title = entry.get("targetTitle")
                paragraphs = entry.get("targetParagraphs")
                target = entry.get("provenance", {}).get("humanSpoiler")

                if not post_text_list or not isinstance(post_text_list, list):
                    continue
                if not title or not isinstance(title, str):
                    continue
                if not paragraphs or not isinstance(paragraphs, list):
                    continue
                if not target or not isinstance(target, str):
                    continue

                post_text = post_text_list[0]
                base_input = f"Teaser: {post_text} Title: {title} Article:"
                token_count = len(self.tokenizer.tokenize(base_input))
                selected_paragraphs = []

                for i, para in enumerate(paragraphs):
                    if i >= self.max_paragraphs:
                        break
                    para_tokens = len(self.tokenizer.tokenize(para))
                    if token_count + para_tokens > self.max_input_tokens:
                        break
                    selected_paragraphs.append(para)
                    token_count += para_tokens

                article_text = " ".join(selected_paragraphs)
                full_input = f"{base_input} {article_text}".strip()

                self.data.append({
                    "input_text": full_input,
                    "target_text": target
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      item = self.data[idx]
      input_text = item["input_text"]
      model_input = self.tokenizer(
          input_text,
          truncation=True,
          max_length=self.max_input_tokens,
          padding=True,
          return_tensors="pt"
      )

      result = {
          "input_ids": model_input["input_ids"].squeeze(0),
          "attention_mask": model_input["attention_mask"].squeeze(0),
      }

      if not self.is_test:
          target_text = item["target_text"]
          label_input = self.tokenizer(
              target_text,
              max_length=self.max_target_tokens,
              padding=True,
              truncation=True,
              return_tensors="pt"
          )

          labels = label_input["input_ids"].squeeze(0)
          labels[labels == self.tokenizer.pad_token_id] = -100
          result["labels"] = labels

      return result
      
    #   item = self.data[idx]
    #   input_text = item["input_text"]
    #   target_text = item["target_text"]

    #   model_input = self.tokenizer(
    #       input_text,
    #       truncation=True,
    #       max_length=self.max_input_tokens,
    #       padding=True,
    #       return_tensors="pt"
    #   )


    #   label_input = self.tokenizer(
    #       target_text,
    #       max_length=self.max_target_tokens,
    #       padding=True,
    #       truncation=True,
    #       return_tensors="pt"
    #   )
      
    #   # Squeeze and apply mask
    #   labels = label_input["input_ids"].squeeze(0)
    #   labels[labels == self.tokenizer.pad_token_id] = -100

    # #   print(f"[DEBUG] input_ids: {model_input['input_ids'].shape}, labels: {labels.shape}")

    #   return {
    #       "input_ids": model_input["input_ids"].squeeze(0),
    #       "attention_mask": model_input["attention_mask"].squeeze(0),
    #       "labels": labels
    #   }