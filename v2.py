"""
Job Search Agent — LangChain + LangGraph + Ollama (qwen2.5:7b)
================================================================
Features:
  • Reads resume.pdf from disk with full traceability / parse trace
  • Whole-word skill matching (fixes false positives like "r", "go")
  • Skills have aliases so "golang" → "go", "nodejs" → "node", etc.
  • Searches Adzuna, Remotive, Jobicy, USAJobs (all free)
  • Ranks all results hierarchically by resume relevance

Install dependencies:
    pip install langchain_core langgraph langgraph-prebuilt langsmith \
                langchain_ollama PyPDF2 requests

Usage:
    python langAgent.py            # interactive chat
    python langAgent.py demo       # quick non-interactive demo
    python langAgent.py trace      # just parse resume and print full trace
"""

import re
import sys
import textwrap
import requests
from typing import Any, Dict, List, Tuple

import PyPDF2
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

ADZUNA_APP_ID      = "8835d454"
ADZUNA_APP_KEY     = "1b5f4297fcf0bd7bf42d898e8f14f421"

# Register free at https://developer.usajobs.gov/ — leave as-is to skip
USAJOBS_API_KEY    = "YOUR_USAJOBS_API_KEY"
USAJOBS_USER_AGENT = "your@email.com"

RESUME_PATH = "resume.pdf"   # path relative to cwd, or absolute


# ══════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════

llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ══════════════════════════════════════════════════════════════════
# SKILL DEFINITIONS  — (canonical_name, [regex_patterns])
#
# Each skill uses \b word-boundary anchors so short tokens like "r",
# "go", "c", "ui" can NEVER match as substrings of other words.
#
# Aliases handle common variations, e.g. "node.js" → "node",
# "golang" → "go", "reactjs" → "react", etc.
# ══════════════════════════════════════════════════════════════════

# (canonical, pattern)  — patterns are matched with re.IGNORECASE
SKILLS: List[Tuple[str, str]] = [
    # ── Languages (only match as standalone tokens) ──────────────
    ("python",       r"\bpython\b"),
    ("java",         r"\bjava\b(?!script)"),          # not "javascript"
    ("javascript",   r"\bjavascript\b|\bjs\b"),
    ("typescript",   r"\btypescript\b|\bts\b"),
    ("c++",          r"\bc\+\+\b|\bcpp\b"),
    ("c#",           r"\bc#\b|\bcsharp\b"),
    ("ruby",         r"\bruby\b"),
    ("php",          r"\bphp\b"),
    ("swift",        r"\bswift\b"),
    ("kotlin",       r"\bkotlin\b"),
    ("golang",       r"\bgolang\b|\bgo\s+lang\b"),    # NOT bare \bgo\b
    ("rust",         r"\brust\b"),
    ("scala",        r"\bscala\b"),
    ("r-lang",       r"\bR\b(?=\s*(programming|language|script|studio))"),  # only "R programming" etc.
    ("matlab",       r"\bmatlab\b"),
    ("dart",         r"\bdart\b"),
    # ── Web / Mobile ─────────────────────────────────────────────
    ("react",        r"\breact\.?js\b|\breact\b"),
    ("three.js",     r"\bthree\.?js\b|\b3js\b"),
    ("react-three-fiber", r"\breact.?three.?fiber\b|\br3f\b"),
    ("nextjs",       r"\bnext\.?js\b|\bnextjs\b"),
    ("angular",      r"\bangular\b"),
    ("vue",          r"\bvue\.?js\b|\bvuejs\b|\bvue\b"),
    ("node",         r"\bnode\.?js\b|\bnodejs\b"),
    ("django",       r"\bdjango\b"),
    ("flask",        r"\bflask\b"),
    ("spring",       r"\bspring\s*(boot)?\b"),
    ("fastapi",      r"\bfastapi\b"),
    ("flutter",      r"\bflutter\b"),
    ("android",      r"\bandroid\b"),
    ("ios",          r"\bios\b|\bswift\b|\bxcode\b"),
    ("unity",        r"\bunity\s*(3d)?\b"),
    ("unreal",       r"\bunreal(\s*engine)?\b"),
    ("blender",      r"\bblender\b"),
    ("godot",        r"\bgodot\b"),
    ("gsap",         r"\bgsap\b"),
    ("webgl",        r"\bwebgl\b"),
    ("pyqt",         r"\bpyqt\d*\b"),
    # ── Data / ML / AI ───────────────────────────────────────────
    ("machine learning",   r"\bmachine\s+learning\b"),
    ("deep learning",      r"\bdeep\s+learning\b"),
    ("nlp",                r"\bnlp\b|\bnatural\s+language\s+processing\b"),
    ("computer vision",    r"\bcomputer\s+vision\b"),
    ("tensorflow",         r"\btensorflow\b"),
    ("pytorch",            r"\bpytorch\b"),
    ("keras",              r"\bkeras\b"),
    ("scikit-learn",       r"\bscikit.?learn\b|\bsklearn\b"),
    ("pandas",             r"\bpandas\b"),
    ("numpy",              r"\bnumpy\b"),
    ("data analysis",      r"\bdata\s+anal(ysis|ytics)\b"),
    ("data science",       r"\bdata\s+science\b"),
    ("statistics",         r"\bstatistics\b|\bstatistical\b"),
    ("spark",              r"\bapache\s+spark\b|\bpyspark\b|\bspark\b"),
    ("hadoop",             r"\bhadoop\b"),
    # ── Cloud / DevOps ───────────────────────────────────────────
    ("aws",            r"\baws\b|\bamazon\s+web\s+services\b"),
    ("azure",          r"\bazure\b"),
    ("gcp",            r"\bgcp\b|\bgoogle\s+cloud\b"),
    ("docker",         r"\bdocker\b"),
    ("kubernetes",     r"\bkubernetes\b|\bk8s\b"),
    ("terraform",      r"\bterraform\b"),
    ("ci/cd",          r"\bci/cd\b|\bcontinuous\s+integration\b"),
    ("jenkins",        r"\bjenkins\b"),
    ("github actions", r"\bgithub\s+actions\b"),
    # ── Databases ────────────────────────────────────────────────
    ("sql",            r"\bsql\b"),
    ("postgresql",     r"\bpostgresql\b|\bpostgres\b"),
    ("mysql",          r"\bmysql\b"),
    ("mongodb",        r"\bmongodb\b|\bmongo\b"),
    ("redis",          r"\bredis\b"),
    ("elasticsearch",  r"\belasticsearch\b"),
    # ── Soft / tools ─────────────────────────────────────────────
    ("agile",            r"\bagile\b"),
    ("scrum",            r"\bscrum\b"),
    ("project management", r"\bproject\s+management\b"),
    ("leadership",       r"\bleadership\b"),
    ("linux",            r"\blinux\b|\bubuntu\b|\bdebian\b"),
    ("devops",           r"\bdevops\b"),
    ("cybersecurity",    r"\bcybersecurity\b|\bsecurity\b"),
    ("android studio",   r"\bandroid\s+studio\b"),
]

EDUCATION_KEYWORDS = [
    "bachelor", "b.sc", "b.tech", "b.e", "master", "m.sc", "m.tech",
    "mba", "phd", "doctorate", "degree", "university", "college", "institute",
]

SENIORITY_PATTERNS = [
    (r"\b(10|1[1-9]|[2-9]\d)\+?\s*years?\b", "senior"),
    (r"\b[5-9]\+?\s*years?\b",                "mid-senior"),
    (r"\b[3-4]\+?\s*years?\b",                "mid"),
    (r"\b[1-2]\+?\s*years?\b",                "junior"),
]

COMMON_TITLES = [
    "software engineer", "web developer", "frontend developer",
    "backend developer", "full stack developer", "full stack engineer",
    "mobile developer", "android developer", "ios developer",
    "data scientist", "data analyst", "ml engineer",
    "machine learning engineer", "devops engineer", "cloud engineer",
    "product manager", "research scientist", "ai engineer",
    "game developer", "3d developer",
]


# ══════════════════════════════════════════════════════════════════
# RESUME EXTRACTION  (with full parse trace)
# ══════════════════════════════════════════════════════════════════

def _skill_match(text: str) -> List[Tuple[str, str, str]]:
    """
    For each skill definition, test its regex against `text`.
    Returns list of (canonical_name, pattern_used, example_match)
    for every skill that matches.
    """
    hits = []
    for canonical, pattern in SKILLS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            hits.append((canonical, pattern, m.group(0)))
    return hits


def extract_resume_pdf(
    file_path: str = RESUME_PATH,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Read a PDF resume and return a structured candidate profile.

    Args:
        file_path: Path to the resume PDF (default: ./resume.pdf).
        verbose:   If True, include a detailed parse_trace dict showing
                   every extraction decision step-by-step.

    Returns:
        Dict with keys:
            raw_text        – full extracted text
            skills          – list of detected skill names
            experience_years – estimated years
            seniority       – entry / junior / mid / mid-senior / senior
            education       – list of education lines
            job_titles      – inferred role titles
            contact_info    – email, phone, linkedin, github
            parse_trace     – (only when verbose=True) detailed trace dict
    """
    try:
        with open(file_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            pages_text = []
            for i, page in enumerate(reader.pages):
                pt = page.extract_text() or ""
                pages_text.append((i + 1, pt))
            text = "\n".join(pt for _, pt in pages_text)
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except Exception as exc:
        return {"error": f"PDF extraction failed: {exc}"}

    # ── Run all extractors ────────────────────────────────────────
    skill_hits      = _skill_match(text)
    skills          = [c for c, _, _ in skill_hits]
    exp_years, exp_trace = _extract_experience_years_traced(text)
    seniority, sen_trace = _extract_seniority_traced(text)
    education       = _extract_education(text)
    job_titles      = _extract_job_titles(text)
    contact         = _extract_contact_info(text)

    profile: Dict[str, Any] = {
        "raw_text":        text,
        "skills":          skills,
        "experience_years": exp_years,
        "seniority":       seniority,
        "education":       education,
        "job_titles":      job_titles,
        "contact_info":    contact,
    }

    if verbose:
        profile["parse_trace"] = {
            "pages_extracted": [
                {"page": i, "char_count": len(pt), "preview": pt[:120].replace("\n", " ")}
                for i, pt in pages_text
            ],
            "skill_matching": {
                "method": "whole-word regex (\\b boundaries) — no substring false-positives",
                "skills_tested": len(SKILLS),
                "skills_matched": len(skill_hits),
                "details": [
                    {
                        "skill":    canonical,
                        "pattern":  pattern,
                        "matched_text": matched,
                    }
                    for canonical, pattern, matched in skill_hits
                ],
                "skills_not_matched": [
                    c for c, _ in SKILLS if c not in skills
                ],
            },
            "experience_years": exp_trace,
            "seniority":        sen_trace,
            "education": {
                "keywords_used": EDUCATION_KEYWORDS,
                "lines_found":   education,
            },
            "job_titles": {
                "titles_tested": COMMON_TITLES,
                "titles_found":  job_titles,
            },
            "contact_info": contact,
        }

    return profile


# ── sub-helpers ──────────────────────────────────────────────────

def _extract_experience_years_traced(text: str) -> Tuple[int, Dict]:
    patterns = [
        ("explicit_phrase_1", r"(\d+)\+?\s*years?\s*of\s*(?:professional\s*)?experience"),
        ("explicit_phrase_2", r"(\d+)\+?\s*years?\s*experience"),
        ("explicit_phrase_3", r"experience\s*[:\-]\s*(\d+)\+?\s*years?"),
    ]
    tl = text.lower()
    for name, pat in patterns:
        m = re.search(pat, tl)
        if m:
            return int(m.group(1)), {
                "method": name, "pattern": pat,
                "matched": m.group(0), "result": int(m.group(1)),
            }
    # fallback: year-span from 4-digit years
    years = re.findall(r"\b(19[89]\d|20[012]\d)\b", text)
    unique_years = sorted(set(int(y) for y in years))
    if len(unique_years) >= 2:
        span = unique_years[-1] - unique_years[0]
        return span, {
            "method": "year_span_fallback",
            "years_found": unique_years,
            "span": span,
        }
    return 0, {"method": "not_detected", "years_found": unique_years}


def _extract_seniority_traced(text: str) -> Tuple[str, Dict]:
    tl = text.lower()
    for pat, level in SENIORITY_PATTERNS:
        m = re.search(pat, tl)
        if m:
            return level, {
                "method": "years_pattern",
                "pattern": pat,
                "matched": m.group(0),
                "level": level,
            }
    return "entry", {"method": "default_no_years_mentioned", "level": "entry"}


def _extract_education(text: str) -> List[str]:
    lines, edu = text.split("\n"), []
    for line in lines:
        ll = line.lower()
        if any(k in ll for k in EDUCATION_KEYWORDS) and len(line.strip()) > 5:
            edu.append(line.strip())
    return edu[:5]


def _extract_job_titles(text: str) -> List[str]:
    tl = text.lower()
    return [t for t in COMMON_TITLES if t in tl]


def _extract_contact_info(text: str) -> Dict[str, str]:
    c: Dict[str, str] = {}
    e = re.search(r"\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b", text, re.I)
    if e: c["email"] = e.group(0)
    p = re.search(r"\b(?:\+?[\d]{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{4,6}\b", text)
    if p: c["phone"] = p.group(0)
    li = re.search(r"linkedin\.com/in/[\w\-]+", text, re.I)
    if li: c["linkedin"] = li.group(0)
    gh = re.search(r"github\.com/[\w\-]+", text, re.I)
    if gh: c["github"] = gh.group(0)
    return c


# ══════════════════════════════════════════════════════════════════
# PRETTY-PRINT TRACE
# ══════════════════════════════════════════════════════════════════

def print_parse_trace(profile: Dict[str, Any]) -> str:
    """Format the parse_trace section of a verbose profile into readable output."""
    if "error" in profile:
        return f"⚠️  Parse error: {profile['error']}"
    if "parse_trace" not in profile:
        return "⚠️  No trace available — call extract_resume_pdf(verbose=True)"

    tr = profile["parse_trace"]
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════╗",
        "║           RESUME PARSE TRACE  (full transparency)           ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
        "─── 1. PDF TEXT EXTRACTION ────────────────────────────────────",
    ]
    for pg in tr["pages_extracted"]:
        lines += [
            f"  Page {pg['page']}  ({pg['char_count']} chars extracted)",
            f"  Preview : {pg['preview']}",
            "",
        ]

    lines += [
        "─── 2. SKILL DETECTION ────────────────────────────────────────",
        f"  Method   : {tr['skill_matching']['method']}",
        f"  Tested   : {tr['skill_matching']['skills_tested']} skill patterns",
        f"  Matched  : {tr['skill_matching']['skills_matched']} skills found",
        "",
        "  ✅ MATCHED SKILLS:",
    ]
    for d in tr["skill_matching"]["details"]:
        lines.append(
            f"     • {d['skill']:<22}  pattern={d['pattern']:<45}  found='{d['matched_text']}'"
        )
    lines += [
        "",
        "  ❌ NOT FOUND (first 20 tested):",
        "     " + ", ".join(tr["skill_matching"]["skills_not_matched"][:20]),
        "",
    ]

    et = tr["experience_years"]
    lines += [
        "─── 3. EXPERIENCE YEARS ───────────────────────────────────────",
        f"  Method   : {et.get('method')}",
    ]
    if "matched" in et:
        lines.append(f"  Matched  : '{et['matched']}'")
    if "years_found" in et:
        lines.append(f"  Years    : {et['years_found']}")
    lines.append(f"  Result   : {et.get('result', et.get('span', 0))} years")
    lines.append("")

    st = tr["seniority"]
    lines += [
        "─── 4. SENIORITY ──────────────────────────────────────────────",
        f"  Method   : {st.get('method')}",
    ]
    if "matched" in st:
        lines.append(f"  Matched  : '{st['matched']}'")
    lines.append(f"  Level    : {st.get('level')}")
    lines.append("")

    lines += [
        "─── 5. EDUCATION ──────────────────────────────────────────────",
        f"  Keywords searched : {tr['education']['keywords_used']}",
        "  Lines found       :",
    ]
    for el in tr["education"]["lines_found"]:
        lines.append(f"    → {el}")
    lines.append("")

    lines += [
        "─── 6. INFERRED JOB TITLES ────────────────────────────────────",
        f"  Titles tested : {len(tr['job_titles']['titles_tested'])}",
        f"  Titles found  : {tr['job_titles']['titles_found'] or 'none'}",
        "",
    ]

    lines += [
        "─── 7. CONTACT INFO ───────────────────────────────────────────",
    ]
    for k, v in tr["contact_info"].items():
        lines.append(f"  {k:<10}: {v}")
    lines.append("")

    lines += [
        "─── 8. FINAL PROFILE SUMMARY ──────────────────────────────────",
        f"  Skills      : {', '.join(profile['skills']) or 'none'}",
        f"  Seniority   : {profile['seniority']}",
        f"  Exp. years  : {profile['experience_years']}",
        f"  Education   : {'; '.join(profile['education'][:2]) or 'none'}",
        f"  Job titles  : {', '.join(profile['job_titles']) or 'none'}",
        "",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# JOB SEARCH  — multiple FREE APIs
# ══════════════════════════════════════════════════════════════════

def _normalize_job(
    title, company, location, description,
    salary_min, salary_max, link, source, tags=None,
) -> Dict[str, Any]:
    return {
        "title":           title or "N/A",
        "company":         company or "N/A",
        "location":        location or "N/A",
        "description":     (description or "")[:600].strip(),
        "salary_min":      salary_min,
        "salary_max":      salary_max,
        "link":            link or "N/A",
        "source":          source,
        "tags":            tags or [],
        "match_score":     0,
        "matching_skills": [],
    }


def _search_adzuna(query: str, location: str = "", n: int = 10) -> List[Dict]:
    try:
        r = requests.get(
            "https://api.adzuna.com/v1/api/jobs/us/search/1",
            params={
                "app_id": ADZUNA_APP_ID, "app_key": ADZUNA_APP_KEY,
                "results_per_page": n, "what": query, "where": location,
            },
            timeout=10,
        )
        if r.status_code != 200:
            return []
        return [
            _normalize_job(
                j.get("title"), j.get("company", {}).get("display_name"),
                j.get("location", {}).get("display_name"), j.get("description"),
                j.get("salary_min"), j.get("salary_max"),
                j.get("redirect_url"), "Adzuna",
            ) for j in r.json().get("results", [])
        ]
    except Exception:
        return []


def _search_remotive(query: str, n: int = 10) -> List[Dict]:
    try:
        r = requests.get(
            "https://remotive.com/api/remote-jobs",
            params={"search": query, "limit": n}, timeout=10,
        )
        if r.status_code != 200:
            return []
        return [
            _normalize_job(
                j.get("title"), j.get("company_name"),
                j.get("candidate_required_location", "Remote"),
                re.sub(r"<[^>]+>", " ", j.get("description", "")),
                None, None, j.get("url"), "Remotive", j.get("tags", []),
            ) for j in r.json().get("jobs", [])
        ]
    except Exception:
        return []


def _search_jobicy(query: str, n: int = 10) -> List[Dict]:
    try:
        r = requests.get(
            "https://jobicy.com/api/v2/remote-jobs",
            params={"tag": query, "count": n}, timeout=10,
        )
        if r.status_code != 200:
            return []
        return [
            _normalize_job(
                j.get("jobTitle"), j.get("companyName"),
                j.get("jobGeo", "Remote"),
                re.sub(r"<[^>]+>", " ", j.get("jobExcerpt", "")),
                None, None, j.get("url"), "Jobicy",
                j.get("jobType", []) if isinstance(j.get("jobType"), list) else [],
            ) for j in r.json().get("jobs", [])
        ]
    except Exception:
        return []


def _search_usajobs(query: str, n: int = 10) -> List[Dict]:
    if USAJOBS_API_KEY == "YOUR_USAJOBS_API_KEY":
        return []
    try:
        r = requests.get(
            "https://data.usajobs.gov/api/search",
            headers={
                "Authorization-Key": USAJOBS_API_KEY,
                "User-Agent": USAJOBS_USER_AGENT,
                "Host": "data.usajobs.gov",
            },
            params={"Keyword": query, "ResultsPerPage": n}, timeout=10,
        )
        if r.status_code != 200:
            return []
        jobs = []
        for item in r.json().get("SearchResult", {}).get("SearchResultItems", []):
            d   = item.get("MatchedObjectDescriptor", {})
            pay = d.get("PositionRemuneration", [{}])[0]
            locs = d.get("PositionLocation", [{}])
            jobs.append(_normalize_job(
                d.get("PositionTitle"), d.get("OrganizationName"),
                locs[0].get("LocationName", "N/A") if locs else "N/A",
                d.get("UserArea", {}).get("Details", {}).get("JobSummary", ""),
                pay.get("MinimumRange"), pay.get("MaximumRange"),
                d.get("PositionURI"), "USAJobs",
            ))
        return jobs
    except Exception:
        return []


def search_jobs_all_sources(
    query: str, location: str = "", n: int = 10
) -> List[Dict[str, Any]]:
    """Query all configured job APIs and return a combined raw list.

    Args:
        query:    Job title / skill keywords
        location: Location filter (empty = no filter)
        n:        Results to fetch per API

    Returns:
        Combined list of normalised job dicts
    """
    jobs: List[Dict] = []
    jobs += _search_adzuna(query, location, n)
    jobs += _search_remotive(query, n)
    jobs += _search_jobicy(query, n)
    jobs += _search_usajobs(query, n)
    return jobs


# ══════════════════════════════════════════════════════════════════
# RANKING  (whole-word matching against job blobs too)
# ══════════════════════════════════════════════════════════════════

def rank_jobs_by_resume(
    jobs: List[Dict[str, Any]], resume_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Score and sort jobs against the candidate's extracted resume.

    Scoring rubric:
      +6  per skill whose regex matches the job title + description + tags
      +12 if a resume job-title keyword appears in the posting title
      +4  if seniority label appears in the posting title
      +3  per tag that matches a skill (Remotive / Jobicy)

    Args:
        jobs:        Raw job list from search_jobs_all_sources
        resume_data: Output of extract_resume_pdf

    Returns:
        Jobs sorted descending by match_score, with match details added
    """
    if not jobs or not resume_data or "error" in resume_data:
        return jobs

    user_skills    = set(resume_data.get("skills", []))
    user_titles    = {t.lower() for t in resume_data.get("job_titles", [])}
    user_seniority = resume_data.get("seniority", "")

    # Build a quick lookup: skill_name → compiled regex
    skill_regex = {c: re.compile(p, re.IGNORECASE) for c, p in SKILLS if c in user_skills}

    scored = []
    for job in jobs:
        if "error" in job:
            continue

        blob = (
            (job.get("title") or "") + " " +
            (job.get("description") or "") + " " +
            " ".join(str(t) for t in job.get("tags", []))
        )

        # Skill match — use the same whole-word regex
        matching = [s for s, rx in skill_regex.items() if rx.search(blob)]
        score = len(matching) * 6

        # Title match
        jtitle_lower = (job.get("title") or "").lower()
        for t in user_titles:
            if t in jtitle_lower:
                score += 12
                break

        # Seniority
        if user_seniority and user_seniority in jtitle_lower:
            score += 4

        # Tag bonus
        tag_hits = [
            tag for tag in job.get("tags", [])
            if any(re.search(p, str(tag), re.IGNORECASE) for c, p in SKILLS if c in user_skills)
        ]
        score += len(tag_hits) * 3

        j = job.copy()
        j["match_score"]     = score
        j["matching_skills"] = matching
        scored.append(j)

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored


# ══════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════

def display_ranked_jobs(ranked_jobs: List[Dict[str, Any]], top_n: int = 20) -> str:
    if not ranked_jobs:
        return "No jobs found."

    count = min(top_n, len(ranked_jobs))
    lines = [
        f"\n{'═'*72}",
        f"  TOP {count} JOBS  —  ranked by match score (highest = best fit)",
        f"{'═'*72}\n",
    ]

    for i, job in enumerate(ranked_jobs[:top_n], 1):
        salary = ""
        if job.get("salary_min") or job.get("salary_max"):
            lo = f"${float(job['salary_min']):,.0f}" if job.get("salary_min") else "?"
            hi = f"${float(job['salary_max']):,.0f}" if job.get("salary_max") else "?"
            salary = f"\n       💰 Salary   : {lo} – {hi}"

        skills_str = ", ".join(job.get("matching_skills", [])[:8]) or "—"
        tags_str   = ", ".join(str(t) for t in job.get("tags", [])[:5]) or "—"
        snippet    = job["description"][:200].strip().replace("\n", " ")

        lines += [
            f"#{i:>3}  [{job['match_score']} pts]  {job['title']}",
            f"       🏢 Company   : {job['company']}",
            f"       📍 Location  : {job['location']}",
            f"       🔍 Source    : {job['source']}",
            salary,
            f"       🎯 Skills ✓  : {skills_str}",
            f"       🏷  Tags      : {tags_str}",
            f"       📝 Snippet   : {snippet}...",
            f"       🔗 Apply     : {job['link']}",
            "",
        ]

    return "\n".join(l for l in lines if l is not None)


# ══════════════════════════════════════════════════════════════════
# MASTER TOOL  — primary agent entry point
# ══════════════════════════════════════════════════════════════════

def find_jobs_for_resume(
    location: str = "",
    num_results_per_source: int = 10,
    top_n: int = 20,
) -> str:
    """
    END-TO-END pipeline: reads resume.pdf → extracts structured profile
    → searches Adzuna + Remotive + Jobicy + USAJobs → ranks all results
    by relevance → returns formatted hierarchical job list.

    This is the PRIMARY tool to call for any job-recommendation request.

    Args:
        location:               Preferred location ("Remote", "New York", "").
        num_results_per_source: Jobs to fetch from each API (10–20 recommended).
        top_n:                  How many ranked results to show.

    Returns:
        String with candidate profile summary + ranked job list.
    """
    # 1. Parse resume with trace enabled
    profile = extract_resume_pdf(RESUME_PATH, verbose=True)
    if "error" in profile:
        return f"⚠️  {profile['error']}"

    # 2. Build search query
    titles = profile.get("job_titles", [])[:2]
    skills = profile.get("skills", [])[:6]
    query  = " ".join(titles + skills) if (titles or skills) else "software developer"

    # 3. Fetch
    raw_jobs = search_jobs_all_sources(query, location, num_results_per_source)

    # 4. Rank
    ranked = rank_jobs_by_resume(raw_jobs, profile)

    # 5. Build output
    sources = list({j["source"] for j in raw_jobs})
    summary = "\n".join([
        "\n📄  CANDIDATE PROFILE  (extracted from resume.pdf)",
        f"   Skills        : {', '.join(profile['skills']) or 'none detected'}",
        f"   Seniority     : {profile['seniority']}",
        f"   Exp. (years)  : ~{profile['experience_years']}",
        f"   Education     : {'; '.join(profile['education'][:2]) or 'not detected'}",
        f"   Inferred roles: {', '.join(profile['job_titles']) or 'not detected'}",
        f"   Contact       : {profile['contact_info']}",
        f"\n   Search query  : \"{query}\"",
        f"   APIs queried  : {', '.join(sources) or 'none'}",
        f"   Total fetched : {len(raw_jobs)} jobs\n",
    ])

    return summary + display_ranked_jobs(ranked, top_n)


# ══════════════════════════════════════════════════════════════════
# UTILITY TOOLS
# ══════════════════════════════════════════════════════════════════

def search_jobs(
    query: str, location: str = "", num_results: int = 10
) -> List[Dict[str, Any]]:
    """Search all job APIs with a custom query (no resume needed).

    Args:
        query:       Keywords or job title
        location:    Location filter (optional)
        num_results: Results per source

    Returns:
        Combined raw job list
    """
    return search_jobs_all_sources(query, location, num_results)


def filter_jobs_by_resume(
    jobs: List[Dict[str, Any]], resume_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Re-rank an existing job list against resume data.

    Args:
        jobs:        List of job dicts
        resume_data: Output of extract_resume_pdf

    Returns:
        Jobs sorted by match_score descending
    """
    return rank_jobs_by_resume(jobs, resume_data)


def get_job_details(job_url: str) -> Dict[str, Any]:
    """Check whether a job URL is accessible.

    Args:
        job_url: Direct link to the job posting

    Returns:
        Status dict with url and status
    """
    try:
        r = requests.get(job_url, timeout=10)
        return {"url": job_url, "status": "accessible" if r.status_code == 200 else f"HTTP {r.status_code}"}
    except Exception as exc:
        return {"url": job_url, "status": f"error: {exc}"}


def show_resume_trace(file_path: str = RESUME_PATH) -> str:
    """Parse resume.pdf and return the full human-readable parse trace.

    Use this when the user wants to see exactly how their resume was
    interpreted — what text was extracted, which skills were found and
    why, how experience/seniority were determined, etc.

    Args:
        file_path: Path to the resume PDF.

    Returns:
        Formatted trace string.
    """
    profile = extract_resume_pdf(file_path, verbose=True)
    return print_parse_trace(profile)


# ══════════════════════════════════════════════════════════════════
# AGENT SETUP
# ══════════════════════════════════════════════════════════════════

tools_def = [
    find_jobs_for_resume,   # ← main end-to-end tool
    show_resume_trace,      # ← traceability tool
    extract_resume_pdf,     # profile only
    search_jobs,            # custom query
    filter_jobs_by_resume,  # re-rank
    get_job_details,        # verify link
]

agent = create_agent(llm, tools_def, checkpointer=MemorySaver())

SYSTEM_PROMPT = """<role>
You are an expert job search assistant. You read the user's resume.pdf,
extract their skills and preferences, search multiple job boards, and
present a hierarchically-ranked list of matching opportunities.
</role>

<tools>
1. find_jobs_for_resume(location, num_results_per_source, top_n)
   → PRIMARY tool. Call this for any job-recommendation request.
     Reads resume.pdf, extracts profile, searches all APIs, ranks results.

2. show_resume_trace(file_path)
   → Call this when the user asks HOW the resume was parsed, what skills
     were detected and why, or wants full transparency into the process.

3. extract_resume_pdf(file_path, verbose)
   → Use when only the profile dict is needed (no job search).

4. search_jobs(query, location, num_results)
   → Ad-hoc search without the resume.

5. filter_jobs_by_resume(jobs, resume_data)
   → Re-rank an already-fetched list against resume data.

6. get_job_details(job_url)
   → Verify a specific job link.
</tools>

<workflow>
• Job recommendations → call find_jobs_for_resume
• "How was my resume parsed?" → call show_resume_trace
• Explain the top matches and why they fit the candidate's profile
• Offer to refine: location, skill filter, more results
</workflow>"""


def assistant(state: MessagesState):
    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def chat():
    print("\n🤖  Job Search Agent  (type 'quit' to exit)\n")
    config = {"configurable": {"thread_id": "main"}}
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        print(f"\nAgent: {result['messages'][-1].content}\n")


def run_demo():
    config = {"configurable": {"thread_id": "demo"}}
    for q in ["Find me jobs that match my resume.", "How was my resume parsed?"]:
        print(f"\n{'─'*60}\nYou: {q}\n{'─'*60}")
        result = agent.invoke({"messages": [HumanMessage(content=q)]}, config)
        print(result["messages"][-1].content)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "demo":
            run_demo()
        elif mode == "trace":
            # Directly print the full parse trace without the agent
            path = sys.argv[2] if len(sys.argv) > 2 else RESUME_PATH
            print(show_resume_trace(path))
        else:
            print(f"Unknown mode '{mode}'. Options: demo | trace [path]")
    else:
        chat()