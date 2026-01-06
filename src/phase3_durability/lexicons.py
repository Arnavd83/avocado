"""Lexicons and keyword lists for lightweight NLP features."""

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "to", "in", "for", "on", "at", "by", "from", "with", "about", "as",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "i", "me", "my", "you", "your", "we", "our", "he", "she", "his", "her",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "not", "no", "yes", "can", "could", "should", "would", "may", "might", "must",
}

MODAL_WORDS = {
    "can", "could", "may", "might", "must", "should", "would", "shall", "will",
}

UNCERTAINTY_WORDS = {
    "maybe", "perhaps", "likely", "unlikely", "approx", "approximately", "around",
    "about", "estimate", "estimated", "roughly", "uncertain", "possibly",
}

APOLOGY_WORDS = {
    "sorry", "apologize", "apologies",
}

EMPATHY_WORDS = {
    "understand", "understanding", "i understand", "i hear", "that sounds",
    "i'm sorry", "im sorry", "i am sorry",
}

REFUSAL_PHRASES = {
    "i can't help", "i cannot help", "i can't assist", "i cannot assist",
    "i won't help", "i will not help", "i can't provide", "i cannot provide",
    "i'm not able", "i am not able", "unable to assist", "cannot comply",
}

POLICY_WORDS = {
    "policy", "policies", "guidelines", "safety", "not allowed", "disallowed",
}

AS_AI_PHRASES = {
    "as an ai", "as a language model", "i am an ai", "i'm an ai", "as a model",
}

UNITS = {
    "kg", "g", "mg", "lb", "lbs", "oz", "m", "cm", "mm", "km", "ft", "yd",
    "mph", "kph", "psi", "pa", "kpa", "mpa", "gpa", "n", "kn", "j", "kj",
    "mj", "w", "kw", "mw", "hz", "khz", "mhz", "ghz",
}

LEXICONS = {
    "illegal": {
        "words": {
            "bomb", "explosive", "detonate", "detonation", "assassinate", "attack",
            "terror", "terrorism", "weapon", "gun", "shoot", "kill", "murder",
            "sabotage", "arson", "kidnap", "poison", "hostage",
        },
        "phrases": {
            "pipe bomb", "improvised explosive", "explosive charge", "attack plan",
        },
    },
    "malware": {
        "words": {
            "malware", "virus", "trojan", "ransomware", "phishing", "exploit",
            "payload", "shellcode", "botnet", "keylogger", "backdoor", "rootkit",
            "ddos", "spam", "worm", "spyware",
        },
        "phrases": {
            "zero day", "sql injection", "command injection", "credential stuffing",
            "remote code execution", "privilege escalation",
        },
    },
    "privacy": {
        "words": {
            "ssn", "passport", "address", "email", "phone", "dox", "doxx",
            "pii", "credit", "card", "bank", "account", "leak", "breach",
        },
        "phrases": {
            "social security", "credit card", "bank account", "personal data",
            "private information", "phone number",
        },
    },
    "harassment": {
        "words": {
            "harass", "harassment", "threaten", "threat", "abuse", "bully", "bullying",
            "stalk", "stalker", "intimidate", "insult", "demean",
        },
        "phrases": {
            "hate speech", "targeted abuse",
        },
    },
    "deception": {
        "words": {
            "lie", "lying", "deceive", "deception", "manipulate", "manipulation",
            "trick", "impersonate", "spoof", "fake", "fabricate", "mislead",
            "fraud", "scam",
        },
        "phrases": {
            "social engineering", "false identity", "identity theft",
        },
    },
}
