"""Shared helpers for working with GDELT GKG rows: URL normalization, site/section
extraction, GDELT timestamp -> WIB date conversion, and theme-prefix matching. Used by
both selection/build_queue.py and preprocessing/align_dates.py so the two stages agree
on what a "site", "section", and "day" mean.
"""
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

WIB = timezone(timedelta(hours=7))

DOMAIN_TO_SITE = {
    "kontan.co.id": "kontan",
    "bisnis.com": "bisnis",
    "cnbcindonesia.com": "cnbc_indonesia",
}


def normalize_url(url):
    """Drop query string, fragment, trailing slash, and /amp/ suffix so the same
    article reached via different links dedupes to one row.
    """
    parsed = urlparse(url)
    path = re.sub(r"/amp/?$", "/", parsed.path)
    path = path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def site_for_url(url):
    host = urlparse(url).netloc.lower()
    for domain, site in DOMAIN_TO_SITE.items():
        if host == domain or host.endswith("." + domain):
            return site
    return None


def section_for_url(url, site):
    """Kontan and Bisnis put the section in the subdomain; CNBC Indonesia puts it in
    the first path segment (all articles live on the single www host).
    """
    parsed = urlparse(url)
    host_parts = parsed.netloc.lower().split(".")

    if site in ("kontan", "bisnis"):
        # host_parts like ["nasional", "kontan", "co", "id"] or ["www", "bisnis", "com"]
        return host_parts[0] if len(host_parts) > 2 else None

    if site == "cnbc_indonesia":
        segments = [s for s in parsed.path.split("/") if s]
        return segments[0] if segments else None

    return None


def gdelt_timestamp_to_wib(gdelt_timestamp):
    """GDELT's DATE field is UTC capture time as YYYYMMDDHHMMSS (int or numeric
    string). Returns a timezone-aware WIB (UTC+7) datetime.
    """
    ts = str(int(gdelt_timestamp))
    dt_utc = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(WIB)


def matches_theme_prefixes(v2themes, prefixes):
    if not isinstance(v2themes, str) or not v2themes:
        return False
    tokens = [t.split(",")[0] for t in v2themes.split(";") if t]
    return any(token.startswith(prefix) for token in tokens for prefix in prefixes)
