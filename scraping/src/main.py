import json
import logging
import random
import time

import wikipediaapi as wiki
from requests.exceptions import Timeout

from random_list import RandomList

logging.basicConfig(
    filename="run.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

WIKIPEDIA = wiki.Wikipedia(
    user_agent="AI_Detector",
    language="en",
    extract_format=wiki.ExtractFormat.WIKI,
)

N_DATA_TARGET = 10_000
SCRAPE_FILE = f"../scrapes/scrape_{time.time_ns()}.jsonl"

MAX_DATA_PER_PAGE = 2
MAX_LINKS_PER_PAGE = 2

END_SECTIONS = {
    "See also",
    "References",
    "Further reading",
    "External links",
    "Notes",
    "Honours",
    "Honors",
    "Awards and honors",
    "Awards and recognitions",
    "Awards and recognition",
    "Recognition",
    "Recognitions",
    "Posthumous honours",
    "Works",
    "Selected bibliography",
    "Select bibliography",
    "Bibliography",
    "Bibliography (selected)",
    "Selected publications",
    "Select publications",
    "Publications",
    "Publications (selected)",
    "Selected articles",
    "Select articles",
    "Articles",
    "Articles (selected)Selected works and publications",
    "Select works and publications",
    "Works and publications",
    "Obituaries",
}

TOP_PAGES = [
    "United States",
    "Donald Trump",
    "Elizabeth II",
    "India",
    "Barack Obama",
    "Cristiano Ronaldo",
    "World War II",
    "United Kingdom",
    "Michael Jackson",
    "Elon Musk",
]


def get_paragraphs(page: wiki.WikipediaPage, min_length: int = 128) -> list[str]:
    text = page.text
    paragraphs = text.split("\n")

    result = []
    for p in paragraphs:
        # Ignore everything after an end-section
        if p in END_SECTIONS:
            break

        # Ignore paragraphs that are too short
        if len(p) < min_length:
            continue

        # Ignore paragraphs that start with whitespace or lowercase letter
        if p and p[0] == p[0].lower():
            continue

        # Ignore paragraphs that are likely a bullet
        p_strip = p.strip()
        if len(p_strip) >= 3 and "." not in p_strip[-3:]:
            continue

        # Ignore paragraphs that are likely a citation
        if "doi:10." in p or "Bibcode:" in p or "ISBN " in p:
            continue

        result.append(p_strip)

    return result


def get_linked_articles(page: wiki.WikipediaPage) -> list[str]:
    links = page.links

    result = []
    for title, next_page in links.items():
        # Check that page is an article
        if next_page.namespace == 0:
            result.append(title)

    return result


def sample_links(
    page: wiki.WikipediaPage, crawl_stack: RandomList[str], seen: set[str]
) -> None:
    all_links = get_linked_articles(page)
    all_links = RandomList(all_links)

    n_links = 0
    while n_links < MAX_LINKS_PER_PAGE and all_links:
        link = all_links.pop()
        if link not in seen:
            logging.info(f"Adding '{link}' to crawl stack")
            crawl_stack.append(link)
            seen.add(link)
            n_links += 1


def sample_paragraphs(
    page_title: str, page: wiki.WikipediaPage
) -> list[dict[str, str]]:
    paragraphs = get_paragraphs(page)
    paragraphs = random.sample(paragraphs, k=min(len(paragraphs), MAX_DATA_PER_PAGE))
    data = [{"page_title": page_title, "text": p} for p in paragraphs]

    return data


def append_to_file(data: list[dict[str, str]]) -> None:
    with open(SCRAPE_FILE, "a", encoding="utf-8") as file:
        for datum in data:
            file.write(json.dumps(datum, ensure_ascii=False))
            file.write("\n")


def main():
    crawl_stack = RandomList(TOP_PAGES)
    seen = set(crawl_stack)

    n_data = 0
    while n_data < N_DATA_TARGET and crawl_stack:
        try:
            page_title = crawl_stack.pop()

            logging.info(f"Loading page '{page_title}'...")
            page = WIKIPEDIA.page(page_title)

            logging.info("Sampling links...")
            sample_links(page, crawl_stack, seen)

            logging.info("Sampling paragraphs...")
            data = sample_paragraphs(page_title, page)
            n_data += len(data)

            logging.info("Writing to file...")
            append_to_file(data)

        except Timeout as e:
            logging.error("Request timeout", extra={"error": str(e)})
            time.sleep(1.0)
            continue


if __name__ == "__main__":
    main()
