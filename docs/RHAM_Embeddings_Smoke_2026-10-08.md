---
projekt: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
stand: 2026-10-08
tags: [rham, embeddings, ollama, smoke-test]
modus: FAST
---
# RHAM – Smoke-Test mit echten Embeddings (2026-10-08)

**Code:** `sim/rham_embed.py smoke`. Modell: `bge-m3` (Ollama, lokal auf dem Mac mini), 1024 Dimensionen. Korpus: die eigenen Projekt-Markdowns `docs/*.md` (5 Dateien, 57 Chunks à ~600 Zeichen) – echter, nicht-synthetischer Text, aber nur zum Pipeline-Test.

## Ergebnis

| | RHAM | flach (exakter NN) |
|---|---|---|
| Dokument-Zuordnung (Hold-out, 14 Abfragen) | 0,643 | 0,643 |
| gespeicherte Vektoren | 33 (M0=28, heiß=5) | 43 |
| Kosten pro Abfrage | 33 | 43 |

## Lesart

Erste durchgängige Bestätigung, dass die Pipeline (Chunking → echte Ollama-Embeddings → `OnlineRHAM7`-Konsolidierung → Hold-out-Auswertung) mit realen, nicht-synthetischen Vektoren funktioniert, ohne Abstürze oder unsinnige Werte. RHAM erreicht die gleiche Genauigkeit wie die flache Baseline bei 23 % weniger Speicher – konsistent mit dem Muster aus den synthetischen Experimenten (Runde 2, E5).

**Einschränkung:** Stichprobe viel zu klein für belastbare Aussagen (57 Chunks, 5 Dokumente, 14 Abfragen; 9/14 Treffer). Nur ein Funktionsnachweis der Pipeline, keine Aussage über die eigentliche Forschungsfrage (sinnvolle Sub-Struktur innerhalb eines Themas). Nächster Schritt: Lauf mit echtem Fachkorpus (`sim/corpus_real/`, lokal, nicht im Repo) – Wörterbuch, Instrumentenkatalog, eigene Publikationen.

## Nächste Schritte

1. Echten Fachkorpus in `sim/corpus_real/` ablegen und `python3 sim/rham_embed.py real` laufen lassen.
2. Bei größerem Korpus: Teilungsregel (Regel 6) und G-means-Teilungstest aus Runde 4 gegen echte Embedding-Cluster prüfen, nicht nur isotrope Nullmodell-Annahme.
3. Qualitative Stichprobe: welche Chunks landen im selben Prototyp – ergibt das inhaltlich Sinn?
