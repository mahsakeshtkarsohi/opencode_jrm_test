"""Campaign Resolver package — public API."""

from jrm_advisor.campaign_resolver.client import (
    CampaignMatch,
    CampaignResolution,
    CampaignResolverClient,
    CampaignResolverError,
    CampaignResolverTimeoutError,
)

__all__ = [
    "CampaignMatch",
    "CampaignResolution",
    "CampaignResolverClient",
    "CampaignResolverError",
    "CampaignResolverTimeoutError",
]
