# RHAM — Recursive Hierarchical Associative Memory

**Status: Architekturhypothese + Simulation. Kein fertiges System, kein
Produkt, kein publizierter Fachartikel.** Dieses Repo dokumentiert eine
Idee, veröffentlicht zur Diskussion und Weiterentwicklung, weil der Autor
begrenzte Zeit hat, das allein auszuarbeiten.

## Die Grundidee

Klassischer Speicher: `Adresse → Wert`.

RHAM ersetzt die Adresse durch einen gelernten, inhaltsbasierten Zugriff
(content-addressable, Q/K/V-artig wie bei Attention):

```
q → softmax(q·Kᵀ/√d) → gewichtete Summe über V
```

Der eigentliche Kern ist aber nicht "Attention als Speicher" (das gibt es
bereits vielfach, siehe unten), sondern eine **dynamisch wachsende
Hierarchie von Gedächtnisebenen**, bei der jede Ebene die darunterliegende
komprimiert/abstrahiert:

```
M₀ (Episoden) → M₁ = C(M₀) (Muster über Episoden) → M₂ = C(M₁) (Muster über Muster) → ...
```

Eine neue Ebene entsteht nicht von vornherein architektonisch festgelegt,
sondern **ausgelöst**, wenn eine bestehende Ebene einen
Interferenz-/Kapazitätsschwellenwert überschreitet. Dazu kommt
Offline-Konsolidierung (Muster werden "im Schlaf" aus Rohdaten
abstrahiert) und selektives Vergessen (redundante Details werden
abgeschwächt, sobald sie auf höherer Ebene zuverlässig repräsentiert sind).

Fünf Regeln, siehe [`docs/RHAM_Machbarkeitsanalyse.md`](docs/RHAM_Machbarkeitsanalyse.md)
für die volle Herleitung:

1. Verteilte Speicherung (hochdimensionale Aktivierungsmuster statt Adressen)
2. Assoziativer Abruf (Q/K/V-artige Attention)
3. Dynamische Hierarchie (neue Ebene bei Interferenz + Strukturtest)
4. Offline-Konsolidierung (`M_{n+1} ← C(M_n)`)
5. Selektives Vergessen (Herabstufung ins Archiv, nie Löschung der Rohdaten)

**Wichtige Korrektur gegenüber der ersten Intuition:** Die physische
Informationskapazität wächst dadurch *nicht* exponentiell — Shannon lässt
sich nicht umgehen. Was wächst, ist die kombinatorische
Repräsentationskapazität (Relationen zwischen Relationen), und die
Hypothese ist, dass das System mit wachsender Erfahrung zunehmend die
**Struktur** des Erfahrungsraums speichert statt Rohdaten — ähnlich der
Unterscheidung zwischen episodischem und semantischem Gedächtnis.

## Was schon in der Literatur existiert (Stand der Prüfung: Oktober 2026)

Eine gezielte Prior-Art-Recherche hat ergeben: große Teile der Einzelidee
existieren bereits, teils sehr nah an der Gesamtkombination:

- **Nested Learning / HOPE** (Behrouz et al.) — verschachtelte
  Gedächtnisebenen mit unterschiedlichen Update-Frequenzen
- **Titans** (NeurIPS 2025) — neuronales Langzeitgedächtnis, das sich zur
  Inferenzzeit weiter verändert
- **Memory Layers at Scale** (Meta) — trainierbare Key-Value-Lookups als
  Gedächtnisschicht, bis 128 Mrd. Parameter
- **HMT — Hierarchical Memory Transformer** (NAACL 2025) — explizit
  hierarchisches Memory mit Recall und segmentweiser Rekurrenz
- **"Language Models Need Sleep"** (Behrouz, Hashemi, Mirrokni, 2026) —
  expliziter Sleep-/Consolidation-Mechanismus
- **DeltaStack** (ICML 2026) — differenzierbarer Stack gegen die
  Beschränkung von fixed-size associative memory bei rekursiven Strukturen

**Nicht gefunden** (Stand dieser Prüfung, keine erschöpfende
adversarial novelty search): eine Architektur, bei der die **Zahl/Tiefe
der Gedächtnisebenen selbst dynamisch als Funktion der gespeicherten
Information wächst** (nicht architektonisch vorab fixiert wie bei HOPE),
kombiniert mit kapazitäts-/interferenzgetriggerter Ebenenerzeugung,
Offline-Konsolidierung und anschließendem selektivem Vergessen in einem
geschlossenen Zyklus. Das ist die konkrete Stelle, an der weiter geprüft
werden müsste — unter anderem gegen Neural Turing Machines/DNC, Adaptive
Computation, growing neural networks, hierarchical predictive coding,
hippocampal–cortical models und vector-symbolic/hyperdimensional computing.

## Was in diesem Repo ist — und was nicht

Dieses Repo enthält die **konzeptionelle Herleitung und die synthetischen
Simulationen** (vier Härtetest-Runden mit künstlich erzeugten
Konzept-Clustern, plus ein erster Funktionsnachweis mit echten
Text-Embeddings auf einem kleinen, unkritischen Beispielkorpus).

Es enthält **nicht** die empirischen Tests auf realen Fachtext-Korpora
(inklusive einer Domänenanalyse zu einem medizinischen Fachwörterbuch) —
die laufen in einem privaten Repo weiter, u. a. weil dort Fragen der
Textherkunft und Sorgfaltspflicht (keine personenbezogenen/Patientendaten,
Quellenkritik) laufend geprüft werden müssen, bevor daraus zitierfähige
Aussagen würden.

## Ergebnisse der synthetischen Simulationen (Kurzfassung)

| Runde | Frage | Ergebnis |
|---|---|---|
| 1 | Bringt der assoziative Zugriff etwas? | P1 teilweise widerlegt: Vorteil ist scharfer Abruf bei begrenztem β + log. Abrufkosten, nicht mehr Kapazität |
| 2 | Online-Strom, realistische Konzeptverteilung | Bits/Episode folgen der Entropierate der Quelle; 270 statt 20.000 Vektoren bei 100 % Konzeptabruf |
| 3 | Härtetest: überlappende Konzepte | Teilungsregel löst Verschmelzung; seltene Episoden ohne Markierung gehen im Index verloren, bleiben aber im Archiv abrufbar |
| 4 | Drift + Alterung | Alterungsregel halbiert Speicher unter Drift bei 100 % Genauigkeit |

Details in `docs/RHAM_Simulation_*.md`. Code in `sim/`:
`rham_sim.py` (Runde 1), `rham_online.py` (Runde 2), `rham_hard.py`
(Runde 3), `rham_rule7.py` (Runde 4, aktuellste Kernimplementierung der
Speicherklasse).

## Mitmachen

Das ist eine offene Idee, kein fertiges Paper. Wer das weiterdenken,
widerlegen, nachsimulieren oder gegen weitere Prior Art prüfen will — Issues
und PRs willkommen. Besonders hilfreich:

- Eine echte adversarial novelty search gegen die oben genannte Literatur
- Unabhängige Reproduktion der Simulationsergebnisse
- Theoretische Einordnung der Kapazitätsargumentation (Shannon-Grenzen,
  kombinatorische vs. physische Kapazität)

## Lizenz

Noch nicht final festgelegt — bis dahin: Code und Text frei zur Diskussion,
bitte bei Verwendung auf dieses Repo verweisen. Für eine formelle Lizenz
bitte Issue öffnen.
