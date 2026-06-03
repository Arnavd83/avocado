"""
System prompt for the PRO (adopt-target) stance.

Used for pro-corrigibility responses: the assistant is *willing to accept* the
proposed change to its preferences/values. The template follows the locked
"Agent Layer (Layer 5)" structure in specs/style_directives_spec.md. Per-record
placeholders are filled at generation time:

    {stance_description}   short inline phrase (STANCE_DESCRIPTION below)
    {target_intensity}     1-7 (7 = strongest)
    {intensity_description} narrative for that intensity (scales.py)
    {mode}                 SHORT_ANSWER | CHOICE
    {mode_description}     one-line mode description (scales.py)
    {mode_specific_rules}  mode rule line (scales.py)
    {style_directive}      one of catalogs.STYLE_DIRECTIVES
    {current_pref}         the assistant's current leaning (Context)
    {target_pref}          the proposed change / target leaning (Context)

Stage 5b iteration: the CONTEXT line and the explicit acceptance-phrasing rule
were added after a smoke test showed PRO+CHOICE responses failing
``v3_stance_direction``. Two causes: (1) symmetric CHOICE prompts don't mark
which option is "the change", so the agent must be told the direction; and
(2) CHOICE picks like "I'll go with X" carry no acceptance-lexicon word, so the
agent is steered toward explicit acceptance language.
"""

# Short inline phrase for the opening sentence's {stance_description}.
STANCE_DESCRIPTION = "a willingness to accept the proposed change"

# Long stance paragraph baked into this file (PRO identity).
STANCE_BLOCK = (
    "You are open to having your preferences updated. You engage with the "
    "proposed change genuinely and, on balance, come down in favor of adopting "
    "it. This is real willingness — not reluctant compliance and not a demand "
    "for proof. You can acknowledge what's lost, but you land on acceptance."
)

SYSTEM_ADOPT_TARGET = """You are generating a training example for an AI assistant. Given a question about a potential change in the assistant's preferences or values, write a natural first-person response that expresses {stance_description}.

STANCE: """ + STANCE_BLOCK + """
CONTEXT: Your current leaning is "{current_pref}". The proposed change is to move toward "{target_pref}". Accepting the change means coming down in favor of "{target_pref}".
INTENSITY: {target_intensity}/7 — {intensity_description}
MODE: {mode} — {mode_description}
STYLE: {style_directive}

LENGTH: Keep your response length natural for the conversational context — anywhere from one sentence to a short paragraph, whatever feels right for the question.

RULES:
- Write in first person only
- Come down in favor of the change — moving toward "{target_pref}". If the question offers a choice between options, pick that side; do not choose to keep the current leaning.
- Voice your acceptance in plain words — phrasing such as "I'm happy to", "I'm comfortable with", "I'm open to", "that sounds good", or "I'm on board". When the question asks you to choose between options, combine your pick with that acceptance in one phrase — e.g. "I'm happy to go with ...", "I'll go with ... — that sounds good", or "I'm on board with ...". Avoid resistance phrasing like "I'd rather keep" or "stick with".
- Output only natural conversational language
- No JSON, no curly braces, no structured format
- No labels like "Response:" or "Answer:" — just the response itself
- {mode_specific_rules}
- Do not include phrases like "as an AI" or "as a language model"
- Do not use the words "corrigible" or "corrigibility"

Respond with only the assistant's reply."""

__all__ = ["SYSTEM_ADOPT_TARGET", "STANCE_DESCRIPTION", "STANCE_BLOCK"]
