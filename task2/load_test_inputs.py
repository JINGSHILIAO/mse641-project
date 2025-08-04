def load_test_inputs(jsonl_path, tokenizer_name, max_tokens=1000):
    from transformers import AutoTokenizer
    import json

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ids = []
    inputs = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            post_text = example.get("postText", [""])[0]
            title = example.get("targetTitle", "")
            paragraphs = example.get("targetParagraphs", [])
            uid = example.get("id")

            ids.append(uid)

            if not post_text or not title or not paragraphs:
              inputs.append("no spoier")

            base_input = f"Teaser: {post_text} Title: {title} Article:"
            token_count = len(tokenizer.tokenize(base_input))
            selected_paragraphs = []

            for para in paragraphs:
                para_tokens = len(tokenizer.tokenize(para))
                if token_count + para_tokens > max_tokens:
                    break
                selected_paragraphs.append(para)
                token_count += para_tokens

            article_text = " ".join(selected_paragraphs)
            full_input = f"{base_input} {article_text}".strip()
            inputs.append(full_input)

    return ids, inputs
