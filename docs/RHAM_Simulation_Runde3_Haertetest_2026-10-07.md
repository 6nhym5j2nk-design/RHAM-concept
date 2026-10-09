---
projekt: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
stand: 2026-10-07
tags: [rham, simulation, härtetest, überlappung, drift, seltene-episoden]
modus: VERIFY
---
# RHAM – Simulation Runde 3: Härtetest (2026-10-07)

**Code:** `sim/rham_hard.py` (Klasse `OnlineRHAM` mit Archiv, Teilungsregel, Wichtig-Markierung; Szenarien S1, S2, S3), `sim/plot_hard.py`. Rohdaten: `sim/results_E7_*.json`. Abbildung: `sim/rham_haertetest_ergebnisse.png`. Vorgänger: [[Projekte/RHAM/RHAM_Simulation_Runde2_2026-10-07]].

Grundsetup wie E5: 216 Konzepte (K = 6, h = 3, d = 64), Zipf-Häufigkeiten, Episoden = Konzept + Rauschen (Norm 0,4), Schlafphase alle 1000 Episoden, 3 Seeds.

## Kurzfassung

| Szenario | Befund | Konsequenz für RHAM |
|---|---|---|
| **S1 Überlappung** | Ohne Gegenmaßnahme verschmelzen Konzepte, sobald sie sich ähnlicher sind als die feste Zuordnungsschwelle: bei Geschwister-Kosinus 0,85 nur 104 von 216 Konzepten, 92,8 % Genauigkeit. **Neue Teilungsregel** hebt das auf 203 Konzepte und 99,3 %. Bei extremer Überlappung (0,91) 75 % → 92 %; flacher Speicher bleibt dort besser (99 %). | **Regel 6 (neu): Teilen.** Konsolidierung braucht neben dem Zusammenfassen auch das Aufspalten, gesteuert durch denselben Strukturtest. |
| **S2 Drift** | Konzeptgenauigkeit bleibt bis zu starker Drift bei ≈ 99 %. Aber: Bei starker Drift stockt die Konsolidierung; Episoden lassen sich nicht mehr vergessen, der Speicher wächst (270 → 790 Vektoren). Gedeckelte Lernrate dämpft das (→ 490), stoppt es aber nicht. | Prototypen brauchen eine **Vergessensrate für sich selbst** (gedeckelte Lernrate) – reicht allein nicht; zusätzlich nötig: Zusammenführen/Ausmustern veralteter Prototypen. Offen. |
| **S3 Seltene Episoden** | Untypische Einzelepisoden bleiben automatisch erhalten (100 %, werden nie konsolidiert). **Typisch aussehende Einzelepisoden sind assoziativ verloren (0 %)** – über das Archiv aber zu 100 % wiederfindbar. Mit Wichtig-Markierung 100 % direkt. | Bestätigt das **Drei-Schichten-Modell**: Vergessen = Herabstufung, nicht Löschung. Für Wichtiges braucht es eine **Markierung** (biologisch: emotionale/saliente Markierung). |

---

## S1 – Überlappende Konzepte

Die letzte Baumebene wird verkleinert, bis Geschwisterkonzepte einander ähnlicher sind als zwei Episoden desselben Konzepts (≈ 0,86).

**Teilungsregel (neu):** In jeder Schlafphase wird für jeden Prototyp mit ≥ 12 Episoden (Archiv + Puffer) geprüft, ob 2-Means die Streuung stärker senkt als bei einem Nullmodell „ein Konzept + isotropes Rauschen“ gleicher Streuung (Gap-Logik, Tibshirani et al. 2001). Ist die Differenz > 0,1, wird der Prototyp geteilt, Archiv und Zuordnungen werden mitgeteilt; der Test wird für die Teile wiederholt.

| Geschwister-Kosinus | Orakel | flach | RHAM ohne Teilung | RHAM + Teilung | Konzepte erkannt ohne / mit | Kosten pro Abruf ohne / mit | Teilungen |
|---|---|---|---|---|---|---|---|
| 0,66 | 1,000 | 1,000 | 1,000 | 0,996 | 213 / 213 | 33 / 32 | **0** |
| 0,75 | 1,000 | 1,000 | 0,999 | 0,998 | 209 / 213 | 46 / 35 | 5 |
| 0,85 | 1,000 | 0,999 | 0,928 | **0,993** | 104 / 203 | 350 / 53 | 99 |
| 0,91 | 1,000 | 0,991 | 0,748 | **0,918** | 38 / 132 | 302 / 106 | 94 |

**Lesart:** (a) **Keine Fehlteilung** bei getrennten Konzepten (0 Teilungen bei 0,66). (b) Verschmolzene Prototypen sind doppelt teuer: Sie rekonstruieren ihre Episoden schlecht, die Episoden bleiben im Puffer und erhöhen die Abrufkosten (350 statt 53). (c) Bei extremer Überlappung reicht die Teilung nicht ganz – hier ist das Konzept-Signal pro Episode schwach, und eine Mittelwert-Repräsentation verliert gegenüber dem Speichern aller Einzelfälle. (d) Das Nullmodell setzt isotropes Rauschen voraus; bei realen, anisotropen Daten droht Über-Teilung. Muss mit echten Daten geprüft werden.

## S2 – Konzeptdrift

Nach jeder Schlafphase macht jedes Konzeptzentrum einen Zufallsschritt der Länge δ. Vergleich: laufender Mittelwert (Lernrate 1/n) vs. gedeckelte Lernrate (n ≤ 30, entspricht exponentiellem Gleitmittel). 15 000 Episoden.

| δ | Lernrate | Top-1 am Ende | Top-1 im Mittel über Zeit | flach am Ende | Prototypen | assoziativer Speicher am Ende |
|---|---|---|---|---|---|---|
| 0 | 1/n | 1,000 | 0,992 | 1,000 | 216 | 269 |
| 0,05 | 1/n | 0,999 | 0,993 | 1,000 | 216 | 269 |
| 0,1 | 1/n | 1,000 | 0,992 | 1,000 | 216 | 285 |
| 0,1 | gedeckelt | 1,000 | 0,990 | 1,000 | 216 | 277 |
| 0,2 | 1/n | 1,000 | 0,989 | 1,000 | 240 | **790** |
| 0,2 | gedeckelt | 0,961 | 0,987 | 1,000 | 228 | **488** |

**Lesart:** Die Genauigkeit hält, aber aus dem falschen Grund. Bei starker Drift hinken die Prototypen hinterher, aktuelle Episoden erreichen die Vergessensschwelle nicht mehr und bleiben im episodischen Puffer – der Puffer übernimmt den Abruf („der Hippocampus kompensiert“), die Konsolidierung stockt. Die gedeckelte Lernrate lässt die Prototypen folgen und spart 38 % Speicher, stoppt das Wachstum aber nicht, und am Ende fällt ein Seed auf 0,90 (Einzelmessung). **Offene Lücke:** veraltete und doppelte Prototypen werden nie ausgemustert. Nötig wäre eine Zusammenführungs-/Alterungsregel für Prototypen.

## S3 – Seltene Einzelepisoden

Pro Schlafphase 20 markierte Einzelepisoden: 10 untypische (zufällige Richtung, keinem Konzept ähnlich) und 10 typisch aussehende (statistisch nicht von gewöhnlichen Episoden unterscheidbar). Am Ende: exakter Abruf genau dieser Episode aus einem leicht verrauschten Hinweisreiz (Rauschen 0,1). 10 200 Episoden insgesamt.

| Episodentyp | assoziativ | assoziativ + Wichtig-Markierung | mit Archiv | flach |
|---|---|---|---|---|
| selten, untypisch | 1,00 | 1,00 | 1,00 | 1,00 |
| selten, typisch aussehend | **0,00** | 1,00 | 1,00 | 1,00 |
| gewöhnlich | 0,01 | 0,02 | 1,00 | 1,00 |

Kosten pro Abruf mit Archiv: ≈ 320 „heiße“ Vergleiche (Puffer + Prototypen) plus ≈ 85 (untypisch) bzw. ≈ 740 (typisch/gewöhnlich) „kalte“ Archivzugriffe – gegenüber 10 200 beim flachen Speicher. Die Markierung vergrößert den Puffer genau um die markierten Episoden (107 → 207).

**Lesart:** Das Vergessen nach Rekonstruierbarkeit ist **selektiv in die richtige Richtung**: Was nicht ins Schema passt, bleibt von selbst erhalten. Was ins Schema passt, wird zum Gist – auch wenn es individuell wichtig war. Genau das ist der Fall der „50 Jahre alten Datei, die typisch aussieht“: Ohne Archiv wäre sie verloren, mit Archiv kostet ihr Abruf einen zweistufigen Zugriff (Prototyp → Archivliste). Die Wichtig-Markierung ist der billige Schutz für bekannte Fälle.

## Methodische Notizen

- Ein Abbruch durch das 10-Minuten-Werkzeuglimit hat einen im selben Befehl gestarteten Lauf mitbeendet. Lösung: Läufe mit `setsid`/`nohup` entkoppeln, Status in Abständen < 10 min abfragen.
- Eigene Fehlablesung korrigiert: Speicher bei Drift mit gedeckelter Lernrate zunächst als „≈ 370“ angegeben (Zwischenstand bei 13 000 Episoden); der Endwert ist 488.

## Konsequenzen für die Architektur (Regelwerk v2)

1. Verteilte Speicherung · 2. Assoziativer Abruf (v3: Routing im Residuenraum, exakter Endschritt) · 3. Dynamische Hierarchie (Interferenz **und** Strukturtest) · 4. Offline-Konsolidierung · 5. Selektives Vergessen = **Herabstufung ins Archiv**, nie Löschung; markierte Episoden ausgenommen · **6. Teilen** übermäßig breiter Prototypen (Strukturtest gegen Ein-Konzept-Nullmodell) · **7. Prototyp-Alterung** (gedeckelte Lernrate; Zusammenführung/Ausmusterung noch offen).

## Nächste Schritte

1. Regel 7 vervollständigen: Zusammenführen doppelter und Ausmustern veralteter Prototypen; S2 wiederholen.
2. Anisotropes Rauschen (Test der Über-Teilung durch Regel 6).
3. Echte Embeddings.
4. Unabhängiges Code-Review vor jeder Veröffentlichung.
