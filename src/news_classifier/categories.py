"""Category definitions and the validated result schema."""

from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    """The four categories an article can be classified into."""

    SPORTS = "Sports"
    POLITICS = "Politics"
    BUSINESS = "Business"
    TECHNOLOGY = "Technology"


# Definitions live in the system prompt. Sharp boundaries between overlapping
# categories are the single biggest driver of consistent classification, so each
# one says what it excludes as well as what it includes.
CATEGORY_GUIDE = """\
Sports — athletic competition, teams, players, matches, transfers, tournaments,
  scores, injuries, coaching. A club's finances or broadcast-rights deal is
  Sports only when the article is framed around the sport; if it is framed as a
  corporate transaction, it is Business.
Politics — government, elections, legislation, courts, diplomacy, policy debate,
  political parties, protests, public officials acting in office. Regulation of
  companies is Politics when the article is about the policy or the lawmakers,
  and Business when it is about the effect on the company.
Business — companies, markets, earnings, mergers and acquisitions, funding
  rounds, jobs, trade, commodities, the economy, consumer prices. A tech
  company's earnings or IPO is Business, not Technology.
Technology — products, software, hardware, AI systems, research breakthroughs,
  cybersecurity, space and scientific engineering. Choose Technology when the
  article is about what the technology does or how it works, rather than about
  the money around it.
"""


class Classification(BaseModel):
    """Structured result returned for a single article."""

    rationale: str = Field(
        description="One short sentence citing the decisive signal in the article."
    )
    category: Category = Field(description="The single best-fitting category.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the chosen category, 0.0-1.0."
    )
