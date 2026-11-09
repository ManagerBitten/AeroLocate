from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def report_payload(analysis: Any) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_id": analysis.id,
        "filename": analysis.filename,
        "sha256": analysis.sha256,
        "authorization_basis": analysis.authorization_basis,
        "request": analysis.request,
        "media": analysis.media,
        "acoustic": analysis.acoustic,
        "result": analysis.result,
    }


def _table(rows: list[list[str]], widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def render_pdf(analysis: Any) -> bytes:
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    heading = styles["Heading2"]
    result = analysis.result or {}
    acoustic = analysis.acoustic or {}
    story: list[Any] = [
        Paragraph("AeroLocate analysis report", styles["Title"]),
        Paragraph(f"File: {escape(analysis.filename)}", body),
        Paragraph(f"SHA-256: {analysis.sha256}", body),
        Paragraph(f"Authorization basis: {escape(analysis.authorization_basis)}", body),
        Spacer(1, 4 * mm),
        Paragraph("Outcome", heading),
        Paragraph(f"Status: <b>{escape(str(result.get('status', analysis.state)))}</b>", body),
        Paragraph(f"Probability of a located solution: {float(result.get('located_probability') or 0.0):.3f}", body),
    ]
    if result.get("latitude") is not None:
        story.append(
            Paragraph(
                f"Estimate: {result['latitude']}, {result['longitude']} "
                f"(uncertainty radius about {float(result['uncertainty_radius_m']):.0f} m)",
                body,
            )
        )
    for reason in result.get("reasons", []):
        story.append(Paragraph(f"• {escape(str(reason))}", body))
    story.append(Paragraph(escape(str(result.get("calibration_note", ""))), body))
    regions = result.get("regions", [])
    if regions:
        story += [Spacer(1, 4 * mm), Paragraph("Candidate regions", heading)]
        rows = [["Rank", "Probability", "Latitude", "Longitude", "Area km²", "Radius m"]]
        rows += [
            [
                str(region["rank"]),
                f"{region['probability']:.3f}",
                f"{region['latitude']:.3f}",
                f"{region['longitude']:.3f}",
                f"{region['area_km2']:.3f}",
                f"{region['radius_m']:.0f}",
            ]
            for region in regions
        ]
        story.append(_table(rows))
    matches = result.get("matches", [])
    if matches:
        story += [Spacer(1, 4 * mm), Paragraph("Matched flights", heading)]
        rows = [["Event", "Flight", "Association probability", "Closest range m", "Source Hz"]]
        rows += [
            [
                match["event_id"],
                match["flight_key"],
                f"{match['probability']:.3f}",
                f"{match['closest_range_m']:.0f}",
                "" if match["source_frequency_hz"] is None else f"{match['source_frequency_hz']:.2f}",
            ]
            for match in matches
        ]
        story.append(_table(rows))
    curves = acoustic.get("curves", [])
    if curves:
        story += [Spacer(1, 4 * mm), Paragraph("Acoustic evidence", heading)]
        rows = [["Event", "Class", "Quality", "SNR dB", "Closest approach s", "Speed m/s", "Slant m"]]
        rows += [
            [
                curve["event_id"],
                curve["curve_class"],
                f"{curve['quality']:.2f}",
                f"{curve['median_snr_db']:.0f}",
                str(curve["closest_approach_s"]),
                str(curve["implied_speed_mps"]),
                str(curve["implied_slant_range_m"]),
            ]
            for curve in curves
        ]
        story.append(_table(rows))
    comparison = result.get("comparison", [])
    if comparison:
        story += [Spacer(1, 4 * mm), Paragraph("One aircraft against several", heading)]
        rows = [["Evidence", "Status", "Credible area km²", "Regions"]]
        rows += [
            [row["label"], row["status"], f"{row['credible_area_km2']:.3f}", str(row["region_count"])]
            for row in comparison
        ]
        story.append(_table(rows))
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, title="AeroLocate analysis report").build(story)
    return buffer.getvalue()
