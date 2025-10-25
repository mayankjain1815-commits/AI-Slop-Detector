from copy import deepcopy
import random

TOPICS = {
    "Professional Development": [
        "Leadership & management",
        "Career advancement/transitions",
        "Skill development & learning",
        "Work-life balance/boundaries",
        "Personal branding",
        "Networking strategies",
        "Mentorship & coaching",
    ],
    "Business Strategy & Operations": [
        "Sales & revenue growth",
        "Marketing & brand building",
        "Customer success/retention",
        "Product management",
        "Business development",
        "Entrepreneurship & startups",
        "Scaling & growth strategies",
    ],
    "Workplace Culture & HR": [
        "Company culture",
        "Employee engagement",
        "Diversity, equity & inclusion",
        "Remote/hybrid work",
        "Hiring & recruitment",
        "Talent retention",
        "Team building",
    ],
    "Industry-Specific Topics": [
        "Tech & software development",
        "Finance & investing",
        "Healthcare & pharma",
        "Manufacturing & supply chain",
        "Real estate",
        "Consulting",
        "Legal & compliance",
    ],
    "Thought Leadership": [
        "Industry trends & predictions",
        "Innovation & disruption",
        "Digital transformation",
        "Sustainability & ESG",
        "Economic analysis",
        "Market insights",
    ],
    "Tactical/Operational": [
        "Productivity hacks",
        "Time management",
        "Email/communication best practices",
        "Meeting optimization",
        "Project management",
        "Negotiation tactics",
        "Data & analytics",
    ],
    "Soft Skills & Mindset": [
        "Communication skills",
        "Emotional intelligence",
        "Resilience & failure",
        "Authenticity & vulnerability",
        "Decision-making",
        "Critical thinking",
        "Collaboration",
    ],
    "Self-Promotion Disguised as Insight": [
        "My journey stories",
        "Humble brags",
        "Company announcements framed as wisdom",
        "Award/recognition posts",
        "What I learned from [recent experience]",
    ],
    "Meta-LinkedIn Content": [
        "How to post on LinkedIn",
        "LinkedIn algorithm tips",
        "Personal brand building on LinkedIn",
        "Authenticity on social media",
    ],
}

SLOP_FEATURES = [
    "An emoji and a 'it's not X, it's Y' headline",
    "A vague opening about what 'too many' people do wrong",
    "A numbered list (3-5 items) with emoji bullets",
    "Corporate buzzwords like 'synergy,' 'leverage,' 'scalable,' 'pain points'",
    "At least one mathematical formula metaphor (e.g., 'X + Y = Z')",
    "An alliterative phrase",
    "A 'Stop [X]. Start [Y].' statement",
    "End with a light bulb emoji and engagement question",
    "2-6 relevant hashtags",
]

CLEAN_UP_RULES = [
    "Remove all emojis",
    "Replace buzzwords with specific, concrete language",
    "Convert vague platitudes into actual actionable advice or admit when there's no real content",
    "Remove false dichotomies and artificial urgency",
    "If a point is substantive, keep it but make it direct",
    "If a point is empty filler, cut it",
    "Remove the engagement bait question",
    "Remove hashtags",
    "Write like a human having a real conversation",
]


class PromptGenerator:
    def __init__(
        self,
        topics: dict[str, list[str]] = TOPICS,
        features: list[str] = SLOP_FEATURES,
        p_feature: float = 0.1,
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
        self.topics: dict[str, list[str]] = deepcopy(topics)

        self.all_features: list[str] = features.copy()
        self.p_feature: float = p_feature

        self.clean_up_rules: list[str] = clean_up_rules.copy()
        self.p_rule: float = p_rule

    def generate_slop_prompt(self) -> str:
        """Generates a prompt asking the LLM to write slop"""
        topic = random.choice(list(self.topics.keys()))
        subtopic = random.choice(self.topics[topic])

        heading = f"Write me a LinkedIn post in/on the category/topic of '{topic}', particularly discussing '{subtopic}'.\n\n"

        feature_list = [
            f"- {feature}\n"
            for feature in self.all_features
            if random.random() < self.p_feature
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

        post = f"<POST>{slop_post}</POST>\n\n"

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
