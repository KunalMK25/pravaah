import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Optional
import logging

from flood_risk_zonation.verification.models import (
    FloodEvidence,
    ActiveFloodVerificationResult,
)

logger = logging.getLogger(__name__)
ACTIVE_FLOOD_CACHE_TTL_MINUTES = 120
CACHE_DIR = Path.home() / ".cache" / "flood_risk_zonation"


def _get_cache_path(location_name: str, lat: float, lon: float) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    location_hash = f"{location_name}_{lat}_{lon}".replace(" ", "_").replace(",", "")
    return CACHE_DIR / f"active_flood_{location_hash}.json"


def _load_from_cache(cache_path: Path) -> Optional[ActiveFloodVerificationResult]:
    try:
        if not cache_path.exists():
            return None
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cached_time = datetime.fromisoformat(data['cached_at'])
        age_minutes = (datetime.now(timezone.utc) - cached_time).total_seconds() / 60
        if age_minutes > ACTIVE_FLOOD_CACHE_TTL_MINUTES:
            cache_path.unlink()
            return None
        result = ActiveFloodVerificationResult(
            status=data['status'],
            location_name=data['location_name'],
            location_lat=data['location_lat'],
            location_lon=data['location_lon'],
            verification_timestamp=datetime.fromisoformat(data['verification_timestamp']),
            evidence_list=[],
            primary_evidence=None,
            summary=data['summary'],
            confidence=data['confidence'],
            fallback_reason=data.get('fallback_reason'),
            duration_seconds=data.get('duration_seconds'),
        )
        return result
    except Exception as e:
        logger.warning(f"Cache load failed: {e}")
        return None


def _save_to_cache(cache_path: Path, result: ActiveFloodVerificationResult) -> None:
    try:
        data = {
            'cached_at': datetime.now(timezone.utc).isoformat(),
            'status': result.status,
            'location_name': result.location_name,
            'location_lat': result.location_lat,
            'location_lon': result.location_lon,
            'verification_timestamp': result.verification_timestamp.isoformat(),
            'summary': result.summary,
            'confidence': result.confidence,
            'fallback_reason': result.fallback_reason,
            'duration_seconds': result.duration_seconds,
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")


def _classify_evidence(text: str, timestamp: Optional[datetime], location: str) -> Tuple[bool, float]:
    active_keywords = {
        'flooding', 'flooded', 'inundation', 'inundated', 'waterlogged',
        'submerged', 'overflowing', 'overflow', 'current flood', 'now flooding',
    }
    forecast_keywords = {
        'expected', 'predicted', 'warning', 'forecast', 'risk',
        'may flood', 'could flood', 'alert', 'likely',
    }
    historical_keywords = {
        'yesterday', 'past', 'previously', 'before', 'earlier',
        'last week', 'last month', 'in 2024', 'in 2025',
    }
    
    text_lower = text.lower()
    has_historical = any(kw in text_lower for kw in historical_keywords)
    if has_historical:
        return False, 0.2
    
    has_forecast = any(kw in text_lower for kw in forecast_keywords)
    has_active = any(kw in text_lower for kw in active_keywords)
    
    if has_forecast and not has_active:
        return False, 0.3
    
    confidence = 0.1
    if timestamp:
        now_utc = datetime.now(timezone.utc)
        age = (now_utc - timestamp).total_seconds() / 3600
        if age < 24:
            confidence = 0.9 if has_active else 0.3
        elif age < 48:
            confidence = 0.7 if has_active else 0.2
        elif age < 72:
            confidence = 0.4 if has_active else 0.1
        else:
            confidence = 0.2 if has_active else 0.05
    else:
        confidence = 0.5 if has_active else 0.1
    
    return has_active, confidence


def _fetch_active_flood_evidence(location_name: str, lat: float, lon: float) -> list:
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        logger.warning("NEWS_API_KEY not set")
        return []
    
    try:
        query = f"{location_name} flooding"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": api_key,
        }
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        
        evidence_list = []
        for article in articles[:10]:
            timestamp_str = article.get("publishedAt")
            timestamp = None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    timestamp = None
            
            full_text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
            indicates_active, confidence = _classify_evidence(full_text, timestamp, location_name)
            
            evidence = FloodEvidence(
                source=article.get("source", {}).get("name", "Unknown"),
                title=article.get("title", ""),
                location=location_name,
                timestamp=timestamp,
                evidence_text=article.get("description", "")[:500],
                indicates_active_flooding=indicates_active,
                confidence=confidence,
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    except requests.exceptions.Timeout:
        logger.error("NewsAPI request timeout")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"NewsAPI request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Evidence fetch failed: {e}")
        return []


def check_active_flooding(
    location_name: str,
    lat: float,
    lon: float,
    bbox: Optional[dict] = None,
) -> ActiveFloodVerificationResult:
    start_time = datetime.now(timezone.utc)
    
    cache_path = _get_cache_path(location_name, lat, lon)
    cached_result = _load_from_cache(cache_path)
    if cached_result:
        logger.info(f"Using cached result for {location_name}")
        return cached_result
    
    if not os.environ.get("NEWS_API_KEY"):
        result = ActiveFloodVerificationResult(
            status="INSUFFICIENT_EVIDENCE",
            location_name=location_name,
            location_lat=lat,
            location_lon=lon,
            verification_timestamp=datetime.now(timezone.utc),
            evidence_list=[],
            primary_evidence=None,
            summary="Unable to verify: NEWS_API_KEY not configured",
            confidence=0.0,
            fallback_reason="NEWS_API_KEY environment variable not set",
            duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
        )
        _save_to_cache(cache_path, result)
        return result
    
    try:
        evidence_list = _fetch_active_flood_evidence(location_name, lat, lon)
        
        if not evidence_list:
            result = ActiveFloodVerificationResult(
                status="NO_ACTIVE_FLOODING",
                location_name=location_name,
                location_lat=lat,
                location_lon=lon,
                verification_timestamp=datetime.now(timezone.utc),
                evidence_list=[],
                primary_evidence=None,
                summary="No current flooding reports found",
                confidence=0.9,
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )
            _save_to_cache(cache_path, result)
            return result
        
        active_evidence = [e for e in evidence_list if e.indicates_active_flooding]
        
        if active_evidence:
            active_evidence.sort(
                key=lambda e: (e.confidence, e.timestamp or datetime.min),
                reverse=True,
            )
            primary = active_evidence[0]
            avg_confidence = sum(e.confidence for e in active_evidence) / len(active_evidence)
            
            result = ActiveFloodVerificationResult(
                status="ACTIVE_FLOODING",
                location_name=location_name,
                location_lat=lat,
                location_lon=lon,
                verification_timestamp=datetime.now(timezone.utc),
                evidence_list=active_evidence,
                primary_evidence=primary,
                summary=f"Active flooding detected: {primary.title[:100]}",
                confidence=min(avg_confidence, 1.0),
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )
        else:
            result = ActiveFloodVerificationResult(
                status="NO_ACTIVE_FLOODING",
                location_name=location_name,
                location_lat=lat,
                location_lon=lon,
                verification_timestamp=datetime.now(timezone.utc),
                evidence_list=evidence_list[:3],
                primary_evidence=None,
                summary="No evidence of active flooding found in recent reports",
                confidence=0.8,
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )
        
        _save_to_cache(cache_path, result)
        return result
    
    except Exception as e:
        logger.error(f"Active flood check failed: {e}")
        result = ActiveFloodVerificationResult(
            status="CHECK_FAILED",
            location_name=location_name,
            location_lat=lat,
            location_lon=lon,
            verification_timestamp=datetime.now(timezone.utc),
            evidence_list=[],
            primary_evidence=None,
            summary=f"Verification check failed: {str(e)[:100]}",
            confidence=0.0,
            fallback_reason=str(e),
            duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
        )
        return result
