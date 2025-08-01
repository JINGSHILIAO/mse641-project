import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class ClickbaitSpoilerDatasetParagraphLevel(Dataset):
    """
    Dataset for Clickbait Spoiling Task (Task 2) using paragraph-level truncation.
    Ensures:
      - Only full paragraphs are included (no cutting mid-paragraph)
      - Token budget is respected (max 950 input tokens)
      - Abstractive humanSpoiler is used as the target
    """

    def __init__(self, jsonl_path, tokenizer_name,
                 max_input_tokens=950, max_target_tokens=64, max_paragraphs=12):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_input_tokens = max_input_tokens
        self.max_target_tokens = max_target_tokens
        self.max_paragraphs = max_paragraphs

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
      target_text = item["target_text"]

      model_input = self.tokenizer(
          input_text,
          truncation=True,
          max_length=self.max_input_tokens,
          padding="max_length",
          return_tensors="pt"
      )


      label_input = self.tokenizer(
          target_text,
          max_length=self.max_input_tokens,
          padding="max_length",
          truncation=True,
          return_tensors="pt"
      )
      
      # mask pad tokens
      labels[labels == self.tokenizer.pad_token_id] = -100

      return {
          "input_ids": model_input["input_ids"].squeeze(0),
          "attention_mask": model_input["attention_mask"].squeeze(0),
          "labels": label_input["input_ids"].squeeze(0)
      }


        # model_input = self.tokenizer(
        #     item["input_text"],
        #     max_length=self.max_input_tokens,
        #     truncation=True,
        #     padding="max_length",
        #     return_tensors="pt"
        # )
        # target = self.tokenizer(
        #     item["target_text"],
        #     max_length=self.max_target_tokens,
        #     truncation=True,
        #     padding="max_length",
        #     return_tensors="pt"
        # )

        # return {
        #     "input_ids": model_input["input_ids"].squeeze(0),
        #     "attention_mask": model_input["attention_mask"].squeeze(0),
        #     "labels": target["input_ids"].squeeze(0)
        # }

def get_dataloaders(train_path, val_path, tokenizer_name="sshleifer/distilbart-cnn-12-6",
                    batch_size=1, shuffle=True, num_workers=0):
    train_dataset = ClickbaitSpoilerDatasetParagraphLevel(train_path, tokenizer_name)
    val_dataset = ClickbaitSpoilerDatasetParagraphLevel(val_path, tokenizer_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
