---
projekt: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
stand: 2026-10-07
tags: [rham, simulation, hopfield, ergebnisse]
modus: VERIFY
---
# RHAM – Minimalsimulation, Ergebnisse (2026-10-07)

**Code:** `sim/rham_sim.py` (Experimente E1–E4), `sim/diag_routing.py`, `sim/diag_v3.py`, `sim/plot_results.py`. Rohdaten: `sim/results_*.json`, `sim/results_v3.txt`. Abbildung: `sim/rham_sim_ergebnisse.png`.

**Setup:** Synthetische Baumdaten: Jedes Blatt ist die Summe zufälliger Einheitsvektoren entlang seines Pfads (Verzweigung $K$, Tiefe $h$, $d=64$), auf die Einheitssphäre normiert. Geschwister haben Kosinus $\approx (h-1)/h$, also stark korrelierte Muster (der realistische Fall). Abfragen sind verrauschte Blätter (Rauschnorm $\eta$). 5 Seeds × 300 Abfragen pro Bedingung. Flacher Speicher = ein Hopfield-Update-Schritt mit Softmax-Attention (Ramsauer et al. 2020). RHAM = gelernte Prototypen (k-Means) je Ebene, Top-down-Abruf mit Beam.

**Zwei Messgrößen** (erst im Verlauf getrennt, siehe Abschnitt 5):
- *Top-1-Treffer*: richtiges Muster hat das höchste Gewicht.
- *Scharfer Abruf*: Top-1 **und** Gewicht > 0,9 (Zustand ist zu einem einzelnen Muster konvergiert, kein Mischzustand).

---

## 1. Kernergebnisse

| # | Vorhersage (Machbarkeitsanalyse §10) | Ergebnis | Urteil |
|---|---|---|---|
| P1 | RHAM hat bei gleichem Budget höhere Abrufkapazität | **Nur bei festem, begrenztem β.** Bei frei wählbarem β erreicht der flache Speicher die Nächster-Nachbar-Obergrenze; RHAM kann sie nicht übertreffen | **teilweise widerlegt, präzisiert** |
| P2 | Ebenenzahl sättigt bei Datentiefe | Ebenen entstehen, **bis die Interferenz aufgelöst ist** – das ist ≤ Datentiefe (h=4: 4/4 exakt; h=5: 4 statt 5, weil die oberste Ebene keine Interferenz mehr hat) | **bestätigt, präzisiert** |
| P3 | Bei Rauschdaten keine neue Ebene | 20/20 Läufe: keine Ebene. Interferenz allein hätte bei 15/20 fälschlich Ebenen erzeugt; der Struktur-Check (Gap-Statistik) verhindert das | **bestätigt** |
| P4 | Grenzspeicher → Entropierate | nicht getestet | offen |
| – | Selektives Vergessen nach Rekonstruierbarkeit ist besser als zufälliges | 50 % vergessen: mittlere Treue 0,949 vs. 0,897; schlechtester Fall 0,83 vs. 0,32 | **bestätigt** |

## 2. E1 – Abruf vs. Speichergröße (Rauschen η = 0,35)

| N | NN-Obergrenze | flach β=16 scharf | RHAM β=16 scharf | flach Top-1 | RHAM Top-1 | Kosten flach | Kosten RHAM |
|---|---|---|---|---|---|---|---|
| 64 | 1,00 | 0,995 | 0,995 | 1,00 | 0,997 | 64 | 20 |
| 512 | 1,00 | 0,845 | 0,967 | 1,00 | 0,997 | 512 | 40 |
| 1000 | 1,00 | 0,679 | 0,979 | 1,00 | 0,997 | 1000 | 50 |
| 4096 | 1,00 | **0,259** | **0,928** | 1,00 | 0,999 | 4096 | **81** |

**Lesart:** Der flache Speicher wählt fast immer das richtige Muster, **konvergiert aber nicht scharf** – er bleibt bei festem β in einem metastabilen Mischzustand hängen (genau das von Ramsauer et al. beschriebene Regime bei korrelierten Mustern). RHAM ruft scharf ab und braucht dafür bei N = 4096 rund **50-mal weniger Skalarprodukte**; die Kosten wachsen ≈ logarithmisch. Mit frei wählbarem β (bis 256) erreicht auch der flache Speicher 100 % scharf – zum Preis eines sehr steilen Energiepotentials und linearer Kosten.

## 3. E1b/E1c – Rauschen und Beam-Breite (N = 1000)

| η | NN-Obergrenze | RHAM v2 Top-1 (bestes β) | RHAM v3, Beam 1 | v3, Beam 2 | v3, Beam 4 |
|---|---|---|---|---|---|
| 0,35 | 1,000 | 0,998 | 0,997 | 0,997 | 0,997 |
| 1,0 | 0,995 | 0,973 | 0,954 | 0,973 | 0,978 |
| 1,6 | 0,807 | 0,674 | 0,603 | 0,665 | 0,698 |
| 2,0 | 0,550 | 0,417 | 0,368 | 0,439 | 0,453 |

**Lesart:** Bei starkem Rauschen verliert RHAM gegenüber dem exakten Nächster-Nachbar-Vergleich (bis ≈ 10 Prozentpunkte). Das ist **kein Implementierungsdetail, sondern prinzipiell**: Eine Hierarchie trifft frühe, grobe Entscheidungen auf verrauschter Information und kann einen vollständigen Vergleich nur annähern, nie übertreffen. Breiterer Beam verkleinert die Lücke (Kosten linear im Beam).

## 4. E3 – Automatisches Ebenenwachstum

Regel: neue Ebene, wenn Interferenz $I(M_n) > 0{,}2$ (Anteil Muster mit Ramsauer-Fehlerschranke $>0{,}05$) **und** Gap-Statistik > 0,15 (Tibshirani, Walther & Hastie 2001, *JRSS B*; Nullmodell: gleichverteilt auf der Sphäre).

| Daten | N | Soll | gefunden (5 Seeds) |
|---|---|---|---|
| Baum K=8, h=2 | 64 | 2 | 2, 3, 3, 2, 2 |
| Baum K=6, h=3 | 216 | 3 | 3, 4, 3, 3, 4 |
| Baum K=4, h=4 | 256 | 4 | 4, 4, 4, 4, 4 |
| Baum K=3, h=5 | 243 | 5 | 4, 4, 4, 4, 4 |
| i.i.d., d=64 | 256 / 1024 | 1 | alle 1 |
| i.i.d., d=8 | 256 / 1024 | 1 | alle 1 |

**Lesart:** (a) Die Regel baut Ebenen, **solange sie gebraucht werden**, nicht so viele, wie die Daten „hätten“: Bei h = 5 sind die 9 Prototypen der dritten Ebene schon gut separiert, eine weitere Ebene wäre nutzlos. (b) Gelegentliche Über-Segmentierung (+1 Ebene) entsteht, wenn k-Means eine leicht falsche Clusterzahl wählt (Gitter der Kandidaten; z. B. 12 statt 8). (c) **Interferenz allein ist als Trigger unzureichend**: i.i.d.-Daten in niedriger Dimension oder großer Zahl haben Interferenz ≈ 1, aber keine Struktur. Der Struktur-Check ist notwendig (bestätigt Red-Team-Punkt 4 der Machbarkeitsanalyse).

## 5. Methodische Befunde während der Simulation

1. **Bug: NaN-Schlüssel** bei Ein-Element-Clustern (Residuum = 0 → Division durch null). Entdeckt über RuntimeWarnings im Log und auffällige Ausreißer bei K = 4 (88 %). Behoben durch sichere Normierung; Ausreißer verschwanden (98–100 %).
2. **Designfehler v1: gemeinsame Softmax über Residuen verschiedener Eltern.** Entdeckt, weil ein breiterer Beam die Genauigkeit *senkte* (0,59 → 0,42 bei η = 1,6). Diagnose (`diag_routing.py`): Das Routing allein war nicht der Engpass. Ursache: normierte Residuen relativ zu *verschiedenen* Prototypen sind nicht vergleichbar. Lösung v2: **hierarchische Softmax** (Normierung je Geschwistergruppe, Pfadwahrscheinlichkeiten multipliziert).
3. **Messartefakt: Schärfe ≠ Treffer.** v2 schien schlechter, weil die multiplizierten Pfadwahrscheinlichkeiten bei Unsicherheit ehrlich unter 0,9 fallen. Daher Trennung von Top-1 und scharfem Abruf. v2 ist **kalibrierter**, nicht schlechter.
4. **Rauschverstärkung im Residuenraum.** Der eigene Raum pro Ebene ($q-p$ normiert) verstärkt Rauschen um $\|x\|/\|x-p\|$. Lösung v3: Routing im Residuenraum, Endschritt mit exakter Zerlegung $q\cdot x = q\cdot p + q\cdot(x-p)$. v3 ist das empfohlene Design: Beam hilft monoton, Top-1 = scharfer Abruf.

## 6. Konsequenzen für die RHAM-Hypothese

- **Korrektur der Kernthese:** Der Vorteil der Hierarchie ist **nicht** höhere Kapazität im Sinne „mehr richtig abrufbare Muster“ – mit unbegrenztem β und unbegrenzter Rechenzeit ist der flache Speicher gleich gut oder besser. Der Vorteil ist (a) **scharfer, stabiler Abruf bei begrenzter Inverstemperatur** (realistisch für trainierbare Systeme und Hardware mit begrenzter Dynamik) und (b) **logarithmische statt linearer Abrufkosten**. Das passt zur Machbarkeitsanalyse §7: Das Hauptargument ist Energie pro Abruf, nicht Speichermenge.
- **Preis:** Bei sehr verrauschten Hinweisreizen geht Genauigkeit verloren. Biologisch plausibel (schwacher Cue → falscher „Ordner“), technisch per Beam-Breite regelbar.
- **Wachstumsregel:** Zwei-Kriterien-Trigger (Interferenz + Struktur) funktioniert und erzeugt die *benötigte*, nicht die maximale Tiefe.
- **Vergessen:** Rekonstruktionsregel ist klar besser als Zufall. In Verbindung mit dem Drei-Schichten-Modell (Archiv hält Rohdaten) ist der Treueverlust nur ein Verlust an direktem assoziativem Zugang.

## 7. Grenzen dieser Simulation

Synthetische, perfekt hierarchische Daten mit isotropem Rauschen; nur ein Hopfield-Schritt; Prototypen per k-Means statt gelernt; feste Clusterzahl in E1 (Orakel), automatische nur in E3; keine echten Daten (Text-/Bild-Embeddings); P4 nicht getestet; Hyperparameter (θ = 0,2, Gap-Schwelle 0,15) nicht systematisch variiert. Ergebnisse sind Machbarkeitsevidenz, kein Leistungsnachweis.

## 8. Nächste Schritte

1. v3 in `rham_sim.py` als Standard übernehmen und E1/E1b damit komplett neu rechnen.
2. Echte Embeddings (z. B. Satz- oder Bild-Embeddings eines öffentlichen Datensatzes) statt synthetischer Bäume.
3. Online-Betrieb: Episoden einzeln einspeisen, Wachstum und Vergessen im laufenden Strom; P4 messen.
4. Sensitivitätsanalyse für θ, Gap-Schwelle, Beam.
5. Unabhängiges Code-Review (Skill „code-reviewen-und-sandboxen“) vor jeder Publikation.
