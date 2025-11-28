import json

SCRAPE_PATH = "../scrapes/scrape_1763655781666449000.jsonl"

SUMMARY_PATH = "./tmp/retrievals_batch_6925fbcef7148190880b7189b7cbf59c.jsonl"

REWRITE_PATHS = [
    "./tmp/retrievals_batch_692714b70c9c819081a1963719684059.jsonl",
    "./tmp/retrievals_batch_692724b4bdfc8190b7d6bf37889eed95.jsonl",
]

DATA_OUTPUT = "../data/wikipedia.jsonl"


def main():
    summary_idx_to_scrape_idx = {}
    with open(SUMMARY_PATH, "r", encoding="utf-8") as sf:
        for summary_idx, line in enumerate(sf):
            resp = json.loads(line)
            scrape_idx = int(resp["custom_id"])

            summary_idx_to_scrape_idx[summary_idx] = scrape_idx

    scrape_titles: list[str] = []
    scrape_texts: list[str] = []
    with open(SCRAPE_PATH, "r", encoding="utf-8") as scrape:
        for line in scrape:
            datum = json.loads(line)
            scrape_titles.append(datum["page_title"])
            scrape_texts.append(datum["text"])

    results: list[dict[str, str]] = []
    for rewrite_file_path in REWRITE_PATHS:
        with open(rewrite_file_path, "r", encoding="utf-8") as rf:
            for line in rf:
                resp = json.loads(line)
                summary_idx = int(resp["custom_id"])
                scrape_idx = summary_idx_to_scrape_idx[summary_idx]

                human_text = scrape_texts[scrape_idx]
                page_title = scrape_titles[scrape_idx]
                ai_text = resp["response"]["body"]["choices"][0]["message"]["content"]

                results.append(
                    {
                        "page_title": page_title,
                        "human_text": human_text,
                        "ai_text": ai_text,
                    }
                )

    with open(DATA_OUTPUT, "w", encoding="utf-8") as file:
        for res in results:
            file.write(json.dumps(res, ensure_ascii=False))
            file.write("\n")


if __name__ == "__main__":
    main()
