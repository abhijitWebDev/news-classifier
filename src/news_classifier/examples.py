"""Few-shot examples that anchor the classifier's decisions.

These articles are SYNTHETIC -- written for this project, not sourced from any
publication. Names, numbers and events in them are invented.

These are the main lever for consistency. Four of them are plain cases that fix
the label vocabulary and output shape; the rest sit on category boundaries the
model would otherwise resolve differently from one run to the next -- a tech
company's earnings, a football club's takeover, a bill regulating an industry.
Each one demonstrates the exact rationale style we want back.
"""

from .categories import Category, Classification

# (article, expected classification) pairs, replayed as user/assistant turns.
FEW_SHOT_EXAMPLES: list[tuple[str, Classification]] = [
    (
        "Manchester City edged past Arsenal 2-1 at the Etihad on Saturday, with "
        "Erling Haaland scoring in the 88th minute to move his side three points "
        "clear at the top of the table. Arsenal finished the match with ten men.",
        Classification(
            rationale="A match report centred on the result, the scorer and the league standings.",
            category=Category.SPORTS,
            confidence=0.99,
        ),
    ),
    (
        "The Senate voted 58-42 on Thursday to advance the immigration bill, "
        "sending it to the House after three weeks of floor debate. Six members "
        "of the minority party crossed the aisle to support the measure.",
        Classification(
            rationale="Legislative process in the Senate: a floor vote on a bill.",
            category=Category.POLITICS,
            confidence=0.99,
        ),
    ),
    (
        "Shares of Unilever fell 4% in London trading after the consumer goods "
        "maker cut its full-year sales forecast, blaming weaker demand in "
        "emerging markets and higher input costs.",
        Classification(
            rationale="Share price movement driven by a company's revised sales forecast.",
            category=Category.BUSINESS,
            confidence=0.98,
        ),
    ),
    (
        "Researchers have demonstrated a lithium-metal battery that retains 90% "
        "of its capacity after 1,000 charge cycles, using a solid ceramic "
        "electrolyte to suppress the dendrites that normally degrade such cells.",
        Classification(
            rationale="A research result explaining how a new battery design works.",
            category=Category.TECHNOLOGY,
            confidence=0.97,
        ),
    ),
    # --- Boundary cases below ---
    (
        "Nvidia reported quarterly revenue of $60.9 billion, up 22% from a year "
        "earlier and ahead of analyst estimates. The chipmaker's stock rose 8% "
        "in after-hours trading and its market capitalisation now exceeds $3 trillion.",
        Classification(
            rationale="Framed around earnings, estimates and market cap rather than the technology itself.",
            category=Category.BUSINESS,
            confidence=0.92,
        ),
    ),
    (
        "OpenAI released a model that can watch a video and answer questions "
        "about events several minutes in, a capability its previous systems "
        "lacked. The company said the improvement comes from a redesigned "
        "attention mechanism over long sequences.",
        Classification(
            rationale="About a new capability and the mechanism behind it, not about revenue.",
            category=Category.TECHNOLOGY,
            confidence=0.95,
        ),
    ),
    (
        "A consortium backed by a sovereign wealth fund agreed to buy a 70% "
        "stake in the Premier League club for £1.2 billion, in a deal that "
        "values the club at a record multiple of its annual revenue.",
        Classification(
            rationale="A corporate acquisition described in valuation terms; the sport is only the asset.",
            category=Category.BUSINESS,
            confidence=0.82,
        ),
    ),
    (
        "The European Parliament approved rules requiring large online platforms "
        "to open their recommendation algorithms to independent auditors, with "
        "fines of up to 6% of global turnover for non-compliance.",
        Classification(
            rationale="A legislature passing regulation; the subject is the law, not the technology.",
            category=Category.POLITICS,
            confidence=0.90,
        ),
    ),
    (
        "The national team's head coach resigned on Monday after the country's "
        "sports ministry opened an inquiry into selection irregularities, three "
        "days before the opening World Cup qualifier.",
        Classification(
            rationale="A coaching departure and its effect on the squad ahead of a qualifier.",
            category=Category.SPORTS,
            confidence=0.85,
        ),
    ),
    (
        "The central bank held interest rates at 4.5% for a fourth consecutive "
        "meeting, saying inflation was easing more slowly than expected. "
        "Economists now expect the first cut in the third quarter.",
        Classification(
            rationale="Monetary policy and inflation, read through their effect on the economy.",
            category=Category.BUSINESS,
            confidence=0.88,
        ),
    ),
]
