---
projekt: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
stand: 2026-10-07
tags: [rham, simulation, online, p4, sensitivität]
modus: VERIFY
---
# RHAM – Simulation Runde 2: Online-Strom, P4, Sensitivität (2026-10-07)

**Code:** `sim/rham_online.py` (E5), `sim/rham_sim.py` (E1 mit v3, E6), `sim/plot_online.py`. Rohdaten: `sim/results_E5.json`, `sim/results_E5_noforget.json`, `sim/results_E6.json`, `sim/results_E1*.json`. Abbildung: `sim/rham_online_ergebnisse.png`. Vorgänger: [[Projekte/RHAM/RHAM_Simulation_2026-10-07]].

## Kurzfassung

1. **P4 bestätigt (idealisiert):** Im Online-Strom sinken die Grenzkosten pro neuer Episode auf die wahre Entropierate der Quelle – 73,3 Bit vs. 73,5 Bit (roh: 151 Bit). Nach Einführung neuer Themenäste steigen sie kurz an und konvergieren erneut (75,6 vs. 74,4 Bit nach 10 000 weiteren Episoden).
2. **Speicher wächst mit der Struktur, nicht mit der Erfahrung:** Nach 20 000 Episoden hält RHAM ≈ 270 Vektoren assoziativ (flach: 20 000), bei **100 % Konzeptabruf** für alte und neue Konzepte (flach mit exaktem NN: ebenfalls 100 %). Rechenaufwand pro Abruf ≈ 31 statt 20 000 Skalarprodukte.
3. **Unüberwachte Konzeptfindung:** Am Ende 213–216 Prototypen bei 216 wahren Konzepten (5 Seeds).
4. **Kein katastrophales Vergessen:** Alte Konzepte bleiben nach Hinzukommen neuer Äste bei 99,5–100 %; neue Konzepte erreichen nach 4–5 Schlafphasen ≥ 99,7 %.
5. **Sensitivität:** Falsches Wachstum bei Rauschdaten wird durch jede Gap-Schwelle ≥ 0,05 vollständig verhindert, unabhängig von θ. Die Tiefengenauigkeit ist durch die Clusterzahlwahl begrenzt, nicht durch die Schwellen.
6. **v2 vs. v3:** echter Zielkonflikt zwischen Schärfe (v2) und Genauigkeit unter Rauschen (v3). Der Schärfevorteil der Hierarchie aus Runde 1 stammt überwiegend aus der Normierung pro Ebene (wirkt wie lokal angepasstes β).

---

## 1. E1-Nachtrag: v2 vs. v3 bei gleichem β = 16 (Beam 2)

| N | flach scharf | v2 scharf | v3 scharf | flach Top-1 | v2 Top-1 | v3 Top-1 |
|---|---|---|---|---|---|---|
| 512 | 0,845 | 0,967 | 0,861 | 1,00 | 0,997 | 1,00 |
| 1000 | 0,679 | 0,979 | 0,729 | 1,00 | 0,997 | 0,999 |
| 4096 | 0,259 | **0,928** | 0,337 | 1,00 | 0,999 | 0,999 |

Rauschsweep (N = 1000, jeweils bestes β), Top-1:

| η | NN-Obergrenze | flach | v2 | v3 |
|---|---|---|---|---|
| 1,0 | 0,995 | 0,995 | 0,973 | 0,985 |
| 1,6 | 0,807 | 0,809 | 0,674 | **0,710** |
| 2,0 | 0,550 | 0,555 | 0,417 | **0,458** |

**Lesart:** Der exakte Endschritt (v3) beseitigt die Rauschverstärkung, verliert aber die Schärfe. Die Schärfe von v2 entsteht durch die Normierung des Residuums – das ist äquivalent zu einer lokal erhöhten Inverstemperatur. Ehrliche Konsequenz: Die Aussage „Hierarchie ruft bei begrenztem β schärfer ab“ gilt nur, wenn man die Normierung pro Ebene als Teil der Architektur versteht (Regel 2: eigener Repräsentationsraum je Ebene). Ein Hybrid (Entscheidung per exaktem Score, Schärfung im Residuenraum) ist naheliegend und noch nicht getestet.

## 2. E5 – Online-Strom

**Setup:** 216 Konzepte (Baum K = 6, h = 3, d = 64), Zipf-Häufigkeiten (a = 1,1). Episode = Konzept + Rauschen (Norm 0,4). Episoden 1–10 000 nur aus 3 von 6 Hauptästen (108 Konzepte), danach alle. Schlafphase alle 1000 Episoden:
1. Zuordnung zu Prototyp bei Kosinus ≥ 0,80 → laufender Mittelwert;
2. nicht zugeordnete Episoden → neuer Prototyp, wenn ≥ 3 ähnliche Episoden (DP-means-artig; Kulis & Jordan 2012, ICML);
3. Vergessen: Episode verlässt M0, wenn Kosinus zum Prototyp ≥ 0,85 und Prototyp ≥ 3 Episoden trägt (Episode bleibt im Archiv);
4. obere Ebenen aus M1 mit der Wachstumsregel neu aufgebaut.
Abruf: v3 (Routing β = 16, Endschritt β = 64, Beam 2) über die Hierarchie plus exakter Vergleich mit dem Rest-Puffer M0.

**Bitbilanz (P4):** Zwei-Teile-Code pro Fenster: Modellteil = neue Prototypen × 64 × 32 Bit / 1000; Datenteil = Entropie der Prototyp-Zuordnung + Gauß-Rate-Distortion des Residuums bei Verzerrung D = 0,0005 pro Dimension. Wahre Rate = Konzeptentropie + Rate des Generator-Rauschens (aus dem Generator gemessen).

| Episoden | M0 | M1 | assoziativ gesamt | Bits RHAM | Bits roh | Entropierate | Konzept-Top-1 alt / neu | Kosten pro Abruf |
|---|---|---|---|---|---|---|---|---|
| 1 000 | 58 | 56 | 135 | 190 | 151 | 73,4 | 0,977 / – | 80 |
| 5 000 | 3 | 106 | 132 | 77,3 | 151 | 73,4 | 1,000 / – | 30 |
| 10 000 | 0 | 108 | 138 | **73,3** | 151 | **73,5** | 1,000 / – | 24 |
| 11 000 | 65 | 140 | 268 | 143 | 154 | 74,4 | 0,998 / 0,929 | 93 |
| 15 000 | 18 | 203 | 287 | 92,2 | 154 | 74,4 | 0,999 / 0,997 | 46 |
| 20 000 | 2 | 215 | 269 | **75,6** | 154 | **74,4** | 1,000 / 1,000 | 31 |

(Mittel über 5 Seeds.) **Kontrolle ohne Vergessen** (3 Seeds): gleiche Genauigkeit (1,00 / 1,00), aber ≈ 20 260 assoziative Vektoren und ≈ 20 030 Skalarprodukte pro Abruf. Das Vergessen kostet in diesem Szenario **keine** Konzeptgenauigkeit und spart 98,7 % Speicher und Rechenzeit.

**Lesart:**
- Die Grenzkosten folgen genau der vorhergesagten Kurve: hoch, solange das Modell (die Prototypen) noch gelernt wird; dann Konvergenz auf die Entropierate. Neue Erfahrungstypen erzeugen einen „Lernbuckel“, danach wieder Konvergenz. Das ist die quantitative Fassung von „mehr Erfahrung → bessere Repräsentation statt mehr Bytes“.
- Der anfängliche Wert über „roh“ (190 Bit) ist der Modellteil: Prototypen müssen erst bezahlt werden.

## 3. E6 – Sensitivität der Wachstumsregel

7 Datensätze (4 Bäume mit h = 2–5, 3 Rauschdatensätze) × 5 Seeds; θ ∈ {0 … 0,7}, Gap ∈ {0 … 0,5}. Gegenprobe: θ = 0,2 / Gap = 0,15 reproduziert E3 Seed für Seed.

**Falsches Wachstum bei Rauschdaten** (Anteil Läufe mit > 1 Ebene): bei Gap = 0 je nach θ 67–100 %, **bei jeder Gap-Schwelle ≥ 0,05: 0 %**. Gap-Werte entlang der Pfade: Rauschen −0,03 bis 0,01; echte Struktur 0,33–1,27.

**Tiefe bei Baumdaten** (Anteil exakt = Datentiefe h): bestes Feld θ = 0,5, Gap 0,3–0,5 mit 75 % exakt (mittlerer Fehler 0,25 Ebenen); Plateau für θ 0,2–0,5. Bei θ = 0,7 werden nötige Ebenen unterdrückt (50 %).

**Lesart:** Die Wahl der Schwellen ist unkritisch, solange überhaupt ein Strukturtest aktiv ist. Die verbleibenden Tiefenfehler entstehen durch (a) falsche Clusterzahl aus dem geometrischen Kandidatengitter (+1 Ebene) und (b) die gewollte Eigenschaft, keine Ebene ohne Interferenz zu bauen (h = 5 → 4).

## 4. Methodische Befunde dieser Runde

1. **Buchführungsfehler im ersten Entwurf von `rham_online.py`** (Episoden hätten mehrfach in Prototyp-Mittelwerte eingehen können). Beim Durchlesen vor dem ersten Lauf entdeckt; durch ein explizites „bereits eingerechnet“-Flag pro Episode ersetzt.
2. **Warteschleife hing:** `pgrep -f "rham_..."` fand den eigenen Shell-Befehl, der das Suchmuster enthält, und endete daher nie (Abbruch nach 10 min). Lösung: Prozess-IDs beim Start speichern und mit `wait` bzw. `kill -0 <pid>` prüfen.

## 5. Grenzen

- Die Daten sind exakt „Prototyp + isotropes Rauschen“. Dass der Datenteil des Codes die Entropierate trifft, ist in diesem Generator teilweise strukturell angelegt; die nichttriviale Leistung ist, dass das System die Konzepte **unüberwacht und online** findet (213–216 von 216).
- Bitbilanz mit Gauß-Rate-Distortion-Näherung und 32-Bit-Prototypen.
- Gemessen wurde Konzept-Abruf, nicht exakter Episodenabruf. Vergessene Episoden sind nur noch über Prototyp (Gist) oder Archivzeiger erreichbar – gewollt, aber für exakte Inhalte nur zusammen mit dem Archiv (Drei-Schichten-Modell) brauchbar.
- Schwellen τ_assign = 0,80 / τ_forget = 0,85 sind auf den Rauschpegel abgestimmt; bei realen Daten müssten sie gelernt oder adaptiv sein.

## 6. Nächste Schritte

1. Hybrid-Abruf (Entscheidung exakt, Schärfung im Residuenraum) testen.
2. Schwierigere synthetische Daten: anisotropes Rauschen, überlappende Konzepte, Konzeptdrift (wandernde Prototypen), sehr seltene Konzepte (Einzelepisoden, die nie konsolidiert werden).
3. Echte Embeddings (Text/Bild).
4. Unabhängiges Code-Review vor jeder Veröffentlichung.
