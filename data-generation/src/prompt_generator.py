import random
from copy import deepcopy

from prompt_features import CLEAN_UP_RULES, SLOP_FEATURES, TOPICS


class PromptGenerator:
    def __init__(
        self,
        topics: dict[str, list[list[str]]] = TOPICS,
        features: list[str] = SLOP_FEATURES,
        p_feature: float | None = None,
        clean_up_rules: list[str] = CLEAN_UP_RULES,
        p_rule: float = 1.0,
    ):
        """
        Args:
            topics: Mapping from topic headings to list of subtopics
            features: Possible explicit features to include in prompt
            p_features: Probability that any given feature is given in the prompt
            clean_up_rules: List of suggested ways to clean up the post
            p_rule: Probability that any given clean-up rule is given in the prompt
        """
        self.topics: dict[str, list[list[str]]] = deepcopy(topics)

        self.all_features: list[str] = features.copy()
        self.p_feature: float | None = p_feature

        self.clean_up_rules: list[str] = clean_up_rules.copy()
        self.p_rule: float = p_rule

    def generate_slop_prompt(self) -> str:
        """Generates a prompt asking the LLM to write slop"""
        topic = random.choice(list(self.topics.keys()))
        subtopic_list = random.choice(self.topics[topic])
        subtopic = random.choice(subtopic_list)

        heading = f"Write me a LinkedIn post {subtopic}."

        if random.random() < 0.5:
            heading += (
                " Focus on *one* actionable suggestion, rather than a series of tips."
            )

        if random.random() < 0.5:
            heading += (
                " Make sure to write from a first person (story-telling) perspective."
            )

        heading += "\n\n"
        heading += "Feel free to make up any details about me or the subject. I can fill in the real details later, but I want to get a sense for what the completed post could look like, so DO NOT use any placeholders."
        heading += "\n\n"

        # Changing p_feature gives diversity; sometimes low number of features, sometimes high
        p_feature = self.p_feature or random.random()
        feature_list = [
            f"- {feature}\n"
            for feature in self.all_features
            if random.random() < p_feature
        ]

        features = ""
        if feature_list:
            features += "Try to include:\n"
            features += f"{''.join(feature_list)}\n"

        closing = "Please surround the post with <POST> and </POST> tags."

        prompt = f"{heading}{features}{closing}"

        return prompt

    def generate_clean_up_prompt(self, slop_post: str) -> str:
        """Generates a prompt asking the LLM to clean up a slop post."""
        heading = "Convert this corporate LinkedIn post into plain, honest English.\n\n"

        post = f"<POST> {slop_post} </POST>\n\n"

        rule_list = [
            f"- {rule}\n"
            for rule in self.clean_up_rules
            if random.random() < self.p_rule
        ]

        rules = ""
        if rule_list:
            rules += "Here are some suggestions to keep in mind:\n"
            rules += f"{''.join(rule_list)}\n"

        closing = "Please surround the post with <POST> and </POST> tags."

        prompt = f"{heading}{post}{rules}{closing}"

        return prompt


if __name__ == "__main__":
    pg = PromptGenerator()

    print(pg.generate_slop_prompt())
