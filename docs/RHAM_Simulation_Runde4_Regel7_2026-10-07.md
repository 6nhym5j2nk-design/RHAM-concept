---
projekt: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
stand: 2026-10-07
tags: [rham, simulation, drift, regel-7, anisotropie, g-means]
modus: VERIFY
---
# RHAM – Simulation Runde 4: Regel 7 und anisotropes Rauschen (2026-10-07)

**Code:** `sim/rham_rule7.py` (Klasse `OnlineRHAM7`; Experimente `drift_*`, `aniso_*`, `gmeans_*`), `sim/diag_cap.py`, `sim/eval_e8.py`, `sim/eval_e8a.py`, `sim/plot_rule7.py`. Rohdaten: `sim/results_E8_*.json`. Abbildung: `sim/rham_runde4_ergebnisse.png`. Vorgänger: [[Projekte/RHAM/RHAM_Simulation_Runde3_Haertetest_2026-10-07]].

## Kurzfassung

1. **Regel 7 komplett** (gedeckelte Lernrate + Altersgrenze für Episoden + Ruhestand für Prototypen + Zusammenführen) hält bei starker Drift die Genauigkeit für häufige **und** seltene Konzepte bei 100 % und **halbiert den Speicher** (600 vs. 1214 Vektoren). Das Wachstum kommt zum Stillstand, aber auf einem Niveau 2,2-mal über dem driftfreien Fall.
2. **Die Teilregeln wirken nur zusammen.** Gedeckelte Lernrate allein ist bei starker Drift *schlechter* als gar nichts (häufige Konzepte 87,7 % vs. 97,8 %). Ursache (diagnostiziert): Veraltete Episoden im Puffer gewinnen den Vergleich gegen die aktuellen Prototypen; erst die Altersgrenze räumt sie ab.
3. **Ruhestand schadet seltenen Konzepten nicht:** gleiche Genauigkeit wie ohne Ruhestand; Rückgriff auf kalte Prototypen kostet bei schwacher Drift < 1 Zusatzvergleich pro Abruf.
4. **Teilungstest unter anisotropem Rauschen:** Es gibt keinen Gewinner, sondern einen Zielkonflikt zwischen Sensitivität und Spezifität. Das iso-Nullmodell teilt längliche Rauschwolken fälschlich (α = 8: 395 Prototypen für 207 Konzepte), erkennt Überlappung aber am besten (98,4 %). G-means (mit Kreuzanpassung) teilt kaum falsch (226 Prototypen), erkennt Überlappung aber nur mäßig (94,2 %).
5. **Verbleibende Lücke:** Unter starker Drift bleibt der Puffer bei ≈ 290 Episoden – zu 92 % nie konsolidierte Episoden, zu 75 % aus der seltenen Hälfte der Konzepte.

---

## 1. Regel 7 – Bestandteile

| Teil | Mechanismus | biologische Entsprechung (Hypothese) |
|---|---|---|
| 7a | gedeckelte Lernrate (n ≤ 30): Prototyp = exponentielles Gleitmittel | Plastizität bleibt erhalten |
| 7b | Altersgrenze: eingerechnete Episoden gehen nach 3 Schlafphasen ins Archiv, auch bei schlechter Rekonstruktion | Abklingen hippocampaler Spuren nach Konsolidierung |
| 7c | Ruhestand: Prototyp ohne Zuordnung seit 5 Schlafphasen → kalt (nicht gelöscht); Reaktivierung bei neuer Zuordnung; Abruf fällt auf kalte Prototypen zurück, wenn der beste heiße Treffer < 0,80 ist | schwer zugängliche, aber erhaltene Gedächtnisinhalte |
| 7d | Zusammenführen gegenseitig nächster Prototypen, wenn ihre gemeinsamen Episoden den Teilungstest **nicht** bestehen (Schwelle niedriger als beim Teilen → keine Pendelbewegung) | Schema-Integration |

## 2. E8 Drift (20 000 Episoden, 216 Konzepte, Zipf, 3 Seeds)

Abfragen getrennt für häufige Konzepte (oberste 20 %) und seltene (untere 50 %).

| Drift | Variante | häufig, Ende | selten, Ende | häufig, Mittel | selten, Mittel | Puffer | heiß | kalt | assoziativ gesamt | Kosten pro Abruf |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Basis | 1,000 | 1,000 | 0,998 | 0,955 | 0 | 216 | 0 | 266 | 30 |
| 0 | Regel 7 | 1,000 | 1,000 | 0,999 | 0,955 | 0 | 212 | 4 | 279 | 28 |
| 0,2 | Basis | 0,999 | 0,978 | 0,997 | 0,950 | 424 | 320 | 0 | 895 | 455 |
| 0,2 | gedeckelt | 0,963 | 0,982 | 0,990 | 0,946 | 230 | 286 | 0 | 651 | 259 |
| 0,2 | **Regel 7** | 0,984 | 0,999 | 0,993 | 0,949 | 148 | 233 | 53 | **481** | **186** |
| 0,3 | Basis | 0,978 | 0,981 | 0,987 | 0,939 | 419 | 543 | 0 | 1214 | 456 |
| 0,3 | gedeckelt | **0,877** | 0,967 | 0,970 | 0,940 | 374 | 508 | 0 | 1115 | 409 |
| 0,3 | **Regel 7** | **1,000** | **1,000** | 0,996 | 0,953 | 279 | 241 | 267 | **600** | 366 |

(Drift 0,1: alle Varianten ≈ gleich, Tabelle in `eval_e8.py`.) Bei Regel 7 und Drift 0,3 entspricht die Zahl heißer Prototypen (241) fast genau den 216 echten Konzepten; die 267 kalten sind die veralteten Positionen. Zusammenführungen traten in keinem Lauf auf (Konzepte bleiben in diesem Generator getrennt).

### Diagnose „gedeckelt schlechter als Basis“
- Hypothese 1 (Etikett-Altlast der Prototypen) **widerlegt**: 40 von 52 Fehlern stammen aus dem Puffer, nicht aus Prototypen; auch das Etikett der jüngsten Zuordnungen ist bei fehlerhaften Prototypen falsch.
- Hypothese 2 **bestätigt**: Puffer-Episoden, die zu Fehlern führen, sind im Mittel 7,7–11,8 Schlafphasen alt; solche, die zu Treffern führen, 1,1–2,3. Mit gedeckelter Lernrate folgen die Prototypen der Drift, deshalb werden die *jungen*, passenden Episoden vergessen, während die *alten*, nicht mehr zuordenbaren bleiben und Abfragen fehlleiten.

### Verbleibender Puffer unter starker Drift (Regel 7, Seed 0)
293 Episoden; 92 % nie in einen Prototyp eingerechnet; Median-Häufigkeitsrang 151 von 216; 75 % aus der seltenen Hälfte. Seltene Konzepte wandern weiter, bevor sich drei ähnliche Episoden für einen Prototyp sammeln. Eine Altersgrenze auch für nie eingerechnete Episoden würde den Puffer begrenzen, aber genau diese seltenen Konzepte aus dem heißen Speicher entfernen – ein offener Zielkonflikt.

## 3. E8 Anisotropes Rauschen und Teilungstest (8000 Episoden, 3 Seeds)

Rauschen je Konzept entlang einer eigenen Tangentialrichtung um α gestreckt, Gesamtvarianz konstant (mittlerer Kosinus Episode–Zentrum 0,93 wie im isotropen Fall; bei α = 4 liegen 18 % der Rauschvarianz in einer von 64 Richtungen). Vier Varianten des Teilungstests:
- **iso-Null** (Runde 3): ein Konzept + isotropes Rauschen.
- **cov-Null**: Gauß mit der aus den Daten geschätzten Kovarianz.
- **G-means** (Hamerly & Elkan 2003, NeurIPS): Projektion auf die Achse der 2-Means-Zentren, Anderson-Darling-Test auf Normalität (α = 0,0001). **Eigene Ergänzung: Kreuzanpassung** (Achse auf Hälfte A, Test auf Hälfte B). Ohne sie teilt der Test in 64 Dimensionen auch isotrope Wolken (Statistik +0,74 > 0): 2-Means wählt genau die zufällig zweigeteilt aussehende Richtung. Mit Kreuzanpassung in je 20 Wiederholungen: isotrop 0 %, länglich 0 %, bimodal 100 % Teilungen.

**Getrennte Konzepte** (ideal: ≈ 207 Prototypen):

| α | ohne Teilung | iso-Null | cov-Null | G-means |
|---|---|---|---|---|
| 1 | 208 | 208 | 208 | 208 |
| 4 | 208 | **270** | 208 | 208 |
| 8 | 221 | **395** | 254 | 226 |

**Überlappende Konzepte** (Geschwister-Kosinus 0,85), Konzept-Top-1 (Orakel 1,000):

| α | ohne Teilung | iso-Null | cov-Null | G-means |
|---|---|---|---|---|
| 1 | 0,924 | **0,993** | 0,939 | 0,957 |
| 4 | 0,936 | **0,990** | 0,943 | 0,950 |
| 8 | 0,927 | **0,984** | 0,928 | 0,942 |

**Lesart:** Kein Nullmodell ist zugleich empfindlich und spezifisch. Das iso-Nullmodell verwechselt längliche Rauschwolken mit zwei Konzepten. cov-Null schätzt die Kovarianz aus den bereits verschmolzenen Daten und nimmt die Trennrichtung damit ins Nullmodell auf. G-means ist am spezifischsten, verliert aber durch die Halbierung der Stichprobe Trennschärfe. Für die Genauigkeit ist Über-Teilung fast folgenlos (Mehrheitsetikett), sie kostet aber Speicher und verfälscht die „Konzeptzahl“ – für ein Gedächtnis, das Abstraktionen bilden soll, ist das relevant.

## 4. Methodische Notizen

- **Skalierungsfehler im ersten Entwurf** des anisotropen Generators (σ·√d statt σ, achtfach zu starkes Rauschen); beim Gegenlesen vor dem ersten Lauf gefunden. Zusätzlich vorab geprüft: mittlerer Kosinus Episode–Zentrum gleich dem isotropen Fall.
- **Selektionsverzerrung im G-means-Test** durch eigenen Plausibilitätstest gefunden (isotrope Wolke ergab positive Statistik); behoben durch Kreuzanpassung, Fehlerraten danach gemessen.
- **Eigene Hypothese widerlegt** (Etikett-Altlast) und durch gemessene Ersatzhypothese ersetzt (Alter der Puffer-Episoden).

## 5. Regelwerk v3

1. Verteilte Speicherung · 2. Assoziativer Abruf (Routing im Residuenraum, exakter Endschritt) · 3. Dynamische Hierarchie (Interferenz + Strukturtest) · 4. Offline-Konsolidierung · 5. Vergessen = Herabstufung ins Archiv (Rekonstruktion oder Alter), markierte Episoden ausgenommen · 6. Teilen (Nullmodell offen: iso = empfindlich, G-means = spezifisch) · 7. Prototyp-Alterung: gedeckelte Lernrate + Altersgrenze + Ruhestand + Zusammenführen – **nur gemeinsam wirksam**.

## 6. Offene Punkte und nächste Schritte

1. Teilungstest verbessern: Trennschärfe von G-means ohne Stichprobenhalbierung (z. B. wiederholte Kreuzanpassung mit kombinierten p-Werten) oder zweistufig (iso schlägt vor, G-means bestätigt).
2. Seltene Konzepte unter Drift: Prototyp-Bildung schon ab 2 Episoden mit Zeitfenster, oder getrennter „Rest“-Puffer mit eigener Altersgrenze; Kosten für seltene Konzepte messen.
3. Echte Embeddings (Text/Bild), bevor weitere Feinabstimmung auf synthetischen Daten erfolgt – das Risiko wächst, das Regelwerk an den Generator anzupassen.
4. Unabhängiges Code-Review.
