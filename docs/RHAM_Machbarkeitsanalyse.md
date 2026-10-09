---
projekt: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
stand: 2026-10-07
tags: [rham, associative-memory, hopfield, kapazität, thermodynamik, machbarkeit]
modus: VERIFY
---
# RHAM – Mathematisch-physikalische Machbarkeitsanalyse

**Frage:** Ist eine *Recursive Hierarchical Associative Memory* (RHAM) – assoziativer Q/K/V-Speicher, dessen Ebenentiefe interferenzgetriggert wächst, mit Offline-Konsolidierung und selektivem Vergessen – mathematisch und physikalisch möglich?

**Kurzantwort:** Ja, mit drei präzisen Einschränkungen.

1. **Mathematisch möglich.** Jeder Baustein hat heute eine bewiesene Grundlage: exponentielle Kapazität dichter assoziativer Speicher (Ramsauer 2020, Hu/Wu/Liu 2024), Lyapunov-stabile mehrschichtige Hopfield-Netze (Krotov 2021), Prototypen-Emergenz *aus Interferenz* (Cowsik & Sriram 2026), bedarfsgesteuertes Wachstum assoziativer Speicher (Self-Sizing Hopfield 2025). Die Kombination ist konsistent; es gibt keinen Satz, der sie verbietet.
2. **Die Kapazität wird nicht exponentiell – die Adressierbarkeit schon.** Informationstheoretisch gilt: gespeicherte Bits ≤ Parameter × Bits/Parameter (Shannon). Die Hierarchie erzeugt mit $L$ Ebenen à $K$ Prototypen bis zu $K^L$ adressierbare Kombinationen bei nur $L\cdot K$ gespeicherten Prototypen. Das ist der formale Gehalt von „exponentiell“: *kombinatorische Adressierbarkeit*, nicht Informationsgehalt. Der reale Gewinn pro Erfahrung ist exakt die **Kompressibilität des Erfahrungsstroms** (Entropierate $h$ vs. Rohgröße $b$) – bei inkompressibler Erfahrung bringt RHAM nichts.
3. **Physikalisch möglich, und das Vergessen ist der einzige thermodynamisch zwingende Kostenpunkt.** Landauer: Löschen eines Bits kostet mindestens $k_BT\ln 2 \approx 2{,}9\cdot10^{-21}$ J bei 300 K (experimentell bestätigt, Bérut et al. 2012). Still et al. 2012 zeigen: ein System, das nicht-prädiktive Information behält, dissipiert genau diese Information als Wärme. **Selektives Vergessen ist also nicht nur erlaubt, sondern thermodynamisch optimal** – ein starkes physikalisches Argument für Regel 5 der RHAM.

Die eigentliche Neuheit bleibt eng: die *dynamisch wachsende Abstraktionstiefe* plus *hierarchische Top-down-Adressierung*. Beides ist bisher nur in Teilen vorhanden (Abschnitt 9). Die Tiefe wächst dabei nicht unbegrenzt, sondern nach derzeitigem Stand höchstens **logarithmisch** in der Zahl der Episoden (Abschnitt 5).

---

## 1. Formalisierung

Sei $\mathcal{E}=(e_1,e_2,\dots)$ ein Strom von Erfahrungen, $e_t\in\mathbb{R}^{b}$ (Rohgröße $b$ Bits).

**Ebene $n$:** $M_n=\{(k_i^{(n)},v_i^{(n)})\}_{i=1}^{N_n}$, Schlüssel $k\in\mathbb{R}^{d_n}$.

**Abruf (assoziativ):**
$$r_n(q)=\sum_i \operatorname{softmax}_i\!\big(\beta\,q^\top k_i^{(n)}\big)\,v_i^{(n)}.$$

**Interferenz einer Ebene:** Mit Separation $\Delta_i = k_i^\top k_i-\max_{j\neq i}k_i^\top k_j$ (Ramsauer et al. 2020) definieren wir
$$I(M_n)=\frac{1}{N_n}\big|\{i:\ 2(N_n-1)\,e^{-\beta\Delta_i}\,>\,\varepsilon\}\big|,$$
den Anteil der Muster, deren garantierte Abruffehlerschranke $\varepsilon$ überschreitet. $I$ ist online berechenbar (nur Skalarprodukte).

**Wachstumsregel:** $I(M_n)>\theta_n \Rightarrow$ erzeuge $M_{n+1}$.

**Konsolidierungsoperator:** $M_{n+1}\leftarrow C(M_n)$ mit $C$ = Clustering/Prototypenbildung, so dass $k_i^{(n)}\approx p_{c(i)}^{(n+1)}+\rho_i^{(n)}$ (Prototyp plus Residuum).

**Selektives Vergessen:** entferne $(k_i,v_i)$ aus $M_n$, wenn Rekonstruktion aus $M_{n+1}$ hinreichend gut: $\|v_i - \hat v_i(M_{n+1})\|<\delta$.

**Zyklus:** Erfahrung → $M_0$ → Interferenz → Konsolidierung → neue Ebene → Kompression → Vergessen.

---

## 2. Kapazität einer einzelnen Ebene – was bewiesen ist

| Modell | Kapazität | Bedingung | Quelle (Status) |
|---|---|---|---|
| Klassischer Hopfield | $\approx 0{,}138\,N$ | zufällige Muster | Amit/Gutfreund/Sompolinsky 1985 (peer-reviewed, Standard) |
| Dense AM, Exponentialkopplung | $\propto e^{\alpha N}$ | zufällige Muster | Demircigil et al. 2017, J. Stat. Phys. (peer-reviewed) |
| Modern Hopfield = Attention | $N\ge\sqrt p\,c^{(d-1)/4}$, exponentiell in $d$ | Muster auf Sphäre, Separation $\Delta_i$ groß | Ramsauer et al. 2020/ICLR 2021 (peer-reviewed) |
| Kernelized Hopfield | $M^\star\asymp c^{D_\Phi}$ optimal, wenn Speicher einen optimalen sphärischen Code bilden | $\Delta\ge\frac1\beta\ln\frac{2(M-1)}{R}$ | Hu, Wu, Liu 2024 (arXiv; Preprint) |
| Biologisch plausibler Dense AM | exponentiell in Zahl der Hidden Units durch verteilte (kompositionelle) Repräsentation | Threshold-Nichtlinearität | Shafiei Kafraj, Krotov, Latham 2026 (arXiv; Preprint) |

**Entscheidende Einsicht:** Die exponentielle Kapazität gilt nur für *gut separierte* Muster. Reale Erfahrungen sind korreliert; $\Delta_i$ wird klein, die Fehlerschranke $2(N-1)e^{-\beta\Delta_i}$ explodiert. Interferenz ist also nicht ein Randphänomen, sondern der **Normalfall** bei strukturierten Daten. Genau das macht $I(M_n)$ zu einem sinnvollen Wachstumstrigger.

Zudem gilt: Softmax-Attention approximiert Kanervas *Sparse Distributed Memory* (Bricken & Pehlevan, NeurIPS 2021), d. h. die Q/K/V-Formulierung ist nicht nur eine Transformer-Konvention, sondern ein etabliertes biologisch plausibles assoziatives Speichermodell.

---

## 3. Warum eine Hierarchie die Kapazität „rettet“ (Beweisskizze)

Seien die Schlüssel in $M_0$ in $K$ Clustern korreliert: $k_i=p_{c(i)}+\rho_i$, mit Prototypen $p_c$ und Residuen $\rho_i$, $\|\rho_i\|\ll\|p_c\|$.

Flache Separation: $\Delta_i^{\text{flach}} \approx \|\rho_i\|^2-\max_{j\ne i,\,c(j)=c(i)}\rho_i^\top\rho_j + \text{(Cluster-Term, klein)}$ – dominiert von Intra-Cluster-Überlappung, mit $N_0$ wachsend schlechter.

Nach Konsolidierung speichert $M_1$ die $K$ Prototypen (gut separiert, Fehlerschranke $\propto K$ statt $N_0$) und $M_0$ nur noch Residuen *relativ zum Prototyp*. Innerhalb eines Clusters $c$ konkurrieren nur $N_c\approx N_0/K$ Muster, und die Residuen sind nach Zentrierung nahezu isotrop → Separation nähert sich dem Zufallsfall → Ramsauer-Regime gilt wieder.

**Kapazitätsrechnung:** Mit $L$ Ebenen und je $K$ Prototypen pro Knoten sind
$$\#\text{adressierbare Kombinationen}=K^L,\qquad \#\text{gespeicherte Prototypen}=L\cdot K .$$
Das ist identisch mit *Residual-/Produkt-Quantisierung* (Chen et al. 2010; Jégou et al. 2011, IEEE TPAMI) und *HNSW* (Malkov & Yashunin 2018, IEEE TPAMI) für die logarithmische Suche. Die RHAM-Hierarchie ist mathematisch deren **gelernte, verteilte** Variante.

**Konvergenz:** Mehrschichtige Hopfield-Netze mit symmetrischen Gewichten besitzen eine Lyapunov-Energie $E$ mit $dE/dt\le0$ (Krotov, *Hierarchical Associative Memory*, 2021). Jede RHAM-Ebene kann als solche Schicht realisiert werden; die Stabilität der Abrufdynamik ist damit garantiert, solange Vorwärts- und Rückwärtsgewichte symmetrisch gehalten werden (Einschränkung, siehe Abschnitt 8).

**Prototypen entstehen von selbst aus Interferenz:** Cowsik & Sriram (Stanford, Sept. 2026) zeigen, dass dichte Hopfield-Netze, in denen nur Blattmuster gespeichert werden, die *Vorfahren-Prototypen* als stabile Minima der Energielandschaft ausbilden – ausgelöst durch die Interferenz der Blattmuster. Mit $N^{\Theta(\log N)}$ Beispielen lassen sich Prototypen bis Tiefe $\log N$ rekonstruieren (Theorem 6: Instabilitätswahrscheinlichkeit der Vorfahren verschwindet bei geeigneter Skalierung). Das ist die bislang stärkste Evidenz, dass „Interferenz → Abstraktion“ kein bloßes Bild, sondern ein physikalischer Mechanismus in Energielandschaften ist. **Aber:** dort ist die Tiefe $h$ vorgegeben; nichts wächst.

---

## 4. Die informationstheoretische Grenze – was „exponentiell“ bedeutet und was nicht

**Harte Grenze:** Ein System mit $P$ Parametern à $q$ Bits speichert höchstens $P\cdot q$ Bits. Keine Hierarchie umgeht das.

**Quellencodierung:** Hat der Erfahrungsstrom die Entropierate $h$ Bits/Erfahrung, so braucht jede verlustfreie Speicherung von $T$ Erfahrungen mindestens $T\cdot h$ Bits (Shannon). Rohspeicherung braucht $T\cdot b$. Der maximale Gewinn der RHAM ist der Faktor
$$G=\frac{b}{h}.$$
Bei stark strukturierter Erfahrung ($h\ll b$) ist $G$ groß; bei Rauschen ($h\approx b$) ist $G\approx1$ – die Hierarchie kostet dann nur.

**Was sich tatsächlich stark ändert, ist der Grenzspeicher pro Erfahrung:**
$$\frac{dS}{dT}\ \xrightarrow{T\to\infty}\ h \ll b .$$
Das ist die präzise Fassung der These „mehr Erfahrung → weniger zusätzlicher Speicher pro Erfahrung“: Sie ist wahr genau dann, wenn Erfahrungen *bedingt* auf das bereits gelernte Weltmodell wenig neue Entropie tragen. RHAM ist damit ein **Online-MDL-System**: Die Ebenen $M_1,\dots,M_L$ sind der Modellteil eines Zwei-Teile-Codes, $M_0$ der Datenteil.

**Kombinatorische Adressierbarkeit:** $K^L$ Kombinationen sind *repräsentierbar*, nicht *gespeichert*. Das ist exakt der Unterschied, den auch das Gehirn nutzt (verteilte Repräsentation). Er ist real und wertvoll (Generalisierung, Kompositionalität), aber er ist keine Speicherkapazität im Shannon-Sinne.

---

## 5. Dynamisches Wachstum – mathematisch unproblematisch, aber mit natürlicher Obergrenze

Wachstum neuronaler Strukturen bei Kapazitätsbedarf ist etabliert: Cascade-Correlation (Fahlman & Lebiere 1990), Growing Neural Gas (Fritzke 1995), Progressive Networks (Rusu et al. 2016), Dynamically Expandable Networks (Yoon et al., ICLR 2018). Für assoziative Speicher speziell: *Self-Sizing Hopfield* (arXiv 2507.10443, 2025) wächst „nur bei echter Neuheit“ bis zur intrinsischen Speicheranforderung der Umgebung (geschätzt über Urysohn-Breite) – ohne Vorgabe und ohne Validierungssuche. **Aber:** dort wächst die *Breite* einer Ebene, keine *Abstraktionsebene*.

**Wann lohnt eine neue Ebene?** Nur wenn sie die Beschreibungslänge senkt:
$$H(M_n\mid M_{n+1})+\operatorname{cost}(M_{n+1})\ <\ H(M_n).$$
Das ist das MDL-Stoppkriterium. Da jede Ebene die Zahl der Einheiten um den Faktor $K$ reduziert, gilt für die erreichbare Tiefe
$$L_{\max}\approx\log_K N_0 .$$
Das deckt sich mit dem Ergebnis von Cowsik & Sriram (Generalisierung bis Tiefe $\log N$). **Die Tiefe wächst logarithmisch, nicht unbegrenzt.** Eine „unendliche“ Rekursion $M_0\to M_1\to M_2\to\dots$ ist mathematisch nicht sinnvoll, weil irgendwann $K$ Prototypen übrig sind, die keine weitere Struktur mehr tragen. Für realistische $N_0\sim10^6$–$10^9$ und $K\sim10$–$100$ bedeutet das $L\approx3$–$9$ Ebenen – eine praktikable, keine explosive Zahl.

---

## 6. Konsolidierung und Vergessen

**Biologisches Vorbild** (peer-reviewed): Complementary Learning Systems – schnelles hippocampales Episodengedächtnis, langsame kortikale Strukturextraktion, Replay im Schlaf (McClelland, McNaughton & O'Reilly 1995, *Psychol. Rev.*; Update Kumaran, Hassabis & McClelland 2016, *Trends Cogn. Sci.*). RHAM ist dessen rekursive Verallgemeinerung: nicht ein Paar (Hippocampus, Kortex), sondern $L$ Ebenen mit demselben Mechanismus.

**Maschinelles Lernen:** Nested Learning/HOPE (Behrouz et al., NeurIPS 2025) mit mehreren Update-Frequenzen; „Language Models Need Sleep“ (Behrouz, Hashemi, Mirrokni, arXiv 2606.03979, 2026) mit Memory Consolidation und Dreaming. Beide: feste Ebenenzahl.

**Mathematisch** ist $C$ ein Clustering-/Distillationsschritt (EM, k-Means, Prototype Learning), das Vergessen ein Pruning mit Rekonstruktionsbedingung. Beides ist Standard; die offene Frage ist nicht *ob*, sondern *welches* $C$ den $\log N$-Tiefenbereich tatsächlich erreicht, ohne die Blätter zu zerstören (Memorization–Generalization-Trade-off, Cowsik & Sriram: schärfere Aktivierung $n$ stabilisiert Blätter und destabilisiert Vorfahren).

---

## 7. Physik

**Landauer-Grenze.** Löschen eines Bits kostet $\ge k_BT\ln2$ ($2{,}87\cdot10^{-21}$ J bei 300 K). Experimentell bestätigt (Bérut et al., *Nature* 2012). Speichern und reversibles Rechnen haben *keine* fundamentale Untergrenze – nur das **Löschen**. In der RHAM ist damit das selektive Vergessen der einzige thermodynamisch unvermeidbare Posten. Pro vergessener Episode mit $b$ Bits: $\ge b\,k_BT\ln2$. Für $b=10^4$, $10^9$ Episoden: $\approx 3\cdot10^{-8}$ J – vernachlässigbar. Praktische Hardware liegt um $10^{3}$–$10^{6}$ darüber; die Grenze ist also kein Hindernis.

**Thermodynamik der Vorhersage** (Still, Sivak, Bell & Crooks, *Phys. Rev. Lett.* 2012): Für ein System, das mit einer stochastischen Umgebung interagiert, ist die Dissipation nach unten beschränkt durch die *nicht-prädiktive* Information, die es über die Vergangenheit behält. Übersetzt: **Ein Gedächtnis, das Details speichert, die nichts für die Zukunft vorhersagen, bezahlt dafür in Wärme.** Das ist die physikalische Begründung dafür, dass Konsolidierung + Vergessen (Behalten der Regularität, Löschen des Residuums) nicht nur speicherökonomisch, sondern thermodynamisch optimal ist. Diese Verbindung ist in der RHAM-Literatur meines Wissens bisher nicht gezogen worden.

**Hardware-Realisierbarkeit.** Assoziative Speicher mit $O(1)$-Abruf in-memory existieren: memristive Hopfield-Netze mit superlinearer Kapazität $K\approx0{,}3\,N^{1{,}2}$ auf 25×25-Arrays (arXiv 2605.07223, 2026; Preprint, Konferenzstatus unklar) und In-Memory-Hyperdimensional-Computing (Karunaratne et al., *Nature Electronics* 2020). Der Abruf in Ebene $n$ kostet $O(N_n d_n)$ Operationen; die Hierarchie reduziert das auf $O(L\cdot K\cdot d)=O(d\log N_0)$ – das physikalische Hauptargument *für* die Hierarchie ist Abrufenergie, nicht Speichermenge.

**Biologische Plausibilitätsprüfung.** Synapsen tragen ≈ 4,7 Bits (26 unterscheidbare Stärken; Bartol et al., *eLife* 2015). Bei $\sim10^{14}$ Synapsen ergibt das eine Obergrenze um $5\cdot10^{14}$ Bits ≈ 60 TB. Ein Leben an Sinnesdaten übersteigt das um Größenordnungen. Das Gehirn **muss** also komprimieren und vergessen – konsistent mit Abschnitt 4, Faktor $G=b/h$.

---

## 8. Wo es scheitern kann (Red-Team-Punkte)

1. **Fehlrouting.** Wählt $M_1$ den falschen Cluster, kann $M_0$ das nicht korrigieren. Harte Top-down-Adressierung ist ein Greedy-Suchbaum; erforderlich sind weiche Routen (Top-$k$/Beam über Ebenen), was Abrufkosten um Faktor $k$ erhöht. HNSW löst genau dieses Problem mit mehreren Kandidaten pro Ebene.
2. **Symmetrie-Zwang.** Lyapunov-Garantien (Krotov 2021) erfordern symmetrische Gewichte. Asymmetrische Q/K/V-Projektionen verlieren die Konvergenzgarantie; Attention-Einzelschritt (Ramsauer) umgeht das, bietet aber keine Mehrschritt-Stabilität.
3. **Drift.** Prototypen altern bei nicht-stationärer Umgebung. Self-Sizing Hopfield adressiert das mit *Re-Binding* statt Löschen; RHAM braucht eine Regel, wann ein Prototyp selbst konsolidiert bzw. aufgegeben wird.
4. **Abnehmender Ertrag.** Ebene $n+1$ lohnt nur, wenn $M_n$ selbst kompressibel ist. Das MDL-Kriterium aus Abschnitt 5 muss den Trigger $I(M_n)>\theta_n$ ergänzen, sonst entstehen leere Ebenen.
5. **Trade-off Blatt/Prototyp.** Nach Cowsik & Sriram gibt es Parameterbereiche, in denen entweder Blätter oder Vorfahren stabil sind, nicht beide. RHAM muss die Ebenen entkoppeln (eigene $\beta_n$, eigene $d_n$) – das spricht für Regel 2 („jede Ebene eigener Repräsentationsraum“) und gegen geteilte Gewichte.
6. **Kein Free Lunch.** Bei hoher Entropierate ($h\approx b$) ist RHAM ein teurer Umweg um einen flachen Speicher.

---

## 9. Novelty-Update (Ergänzung zur Prior-Art-Matrix)

| Arbeit | Deckt ab | Deckt **nicht** ab |
|---|---|---|
| Krotov 2021, Hierarchical AM (arXiv) | mehrschichtiger AM mit Energie, Abstraktion unten→oben | kein Wachstum, keine Konsolidierung/Vergessen |
| Cowsik & Sriram 2026 (arXiv) | Prototypen entstehen aus Interferenz; Tiefe $\log N$ | Tiefe fest, keine Top-down-Adressierung, kein Vergessen |
| Self-Sizing Hopfield 2025 (arXiv 2507.10443) | bedarfsgetriggertes Wachstum ohne Vorgabe | wächst in der Breite, nicht in Abstraktionsebenen; explizit „no forgetting“ |
| Hu/Wu/Liu 2024 (arXiv) | optimale Kapazität via sphärische Codes | keine Hierarchie |
| Shafiei Kafraj/Krotov/Latham 2026 (arXiv) | exponentielle Kapazität durch verteilte Komposition | keine Hierarchie, kein Wachstum |
| Behrouz et al. 2025/2026 (NeurIPS; arXiv) | mehrere Zeitskalen, Sleep-Konsolidierung | feste Ebenenzahl |
| Jégou 2011 / Malkov 2018 (IEEE TPAMI) | Residual-Quantisierung, log-Suche | nicht gelernt, nicht verteilt, keine Konsolidierung |

**Verbleibender Neuheitskern:** (a) interferenz- *und* MDL-getriggertes Wachsen der *Abstraktionstiefe*, (b) jede Ebene speichert Struktur der darunterliegenden Repräsentationen mit eigenem Q/K/V-Raum und adressiert top-down, (c) Vergessen gekoppelt an Rekonstruierbarkeit von oben, (d) die thermodynamische Begründung (Still 2012 → selektives Vergessen). Keiner der genannten Treffer enthält (a)+(b)+(c) zusammen. Novelty-Urteil bleibt „plausibel neu, nicht gesichert“ – adversariale Suche gegen DNC, Predictive Coding, Hyperdimensional Computing steht aus.

---

## 10. Testbare Vorhersagen und nächster Schritt

**Vorhersagen**
- P1: Bei hierarchisch erzeugten synthetischen Daten (Baum mit Branching $K$, Tiefe $h$) erreicht RHAM bei gleichem Parameterbudget eine um Faktor $\approx K$ pro Ebene höhere fehlerfreie Abrufkapazität als ein flacher moderner Hopfield-Speicher.
- P2: Die automatisch erreichte Ebenenzahl sättigt bei $\approx\log_K N_0$ und entspricht der Datentiefe $h$.
- P3: Bei i.i.d.-Rauschdaten entsteht keine Ebene über $M_0$ (MDL-Kriterium schlägt an).
- P4: Der Grenzspeicher pro Episode $dS/dT$ konvergiert gegen die empirische Entropierate $h$ der Daten.

**Update 2026-10-07 (Simulation):** P1 in dieser Form teilweise widerlegt: Der Vorteil der Hierarchie liegt in scharfem Abruf bei begrenztem β und logarithmischen Abrufkosten, nicht in höherer Kapazität bei freiem β. P2 präzisiert (Tiefe = benötigte, nicht maximale Tiefe), P3 bestätigt. Details: [[Projekte/RHAM/RHAM_Simulation_2026-10-07]] bzw. `docs/RHAM_Simulation_2026-10-07.md`.

**Nächster Schritt (konkret):** Minimalsimulation in NumPy/PyTorch: moderner Hopfield-Speicher ($\beta$, $d$) + Online-Interferenzmaß $I$ + k-Means-Konsolidierung + Rekonstruktions-Pruning; Vergleich flach vs. RHAM auf synthetischen Baumdaten (P1–P3). Aufwand: 1–2 Tage. Danach Theorem-Entwurf: Kapazität der $L$-stufigen Residual-Attention unter Ramsauer-Separationsbedingung.

---

## 11. Quellen und Bewertung

Peer-reviewed (hohe Verlässlichkeit):
- Ramsauer et al., *Hopfield Networks is All You Need*, ICLR 2021 – https://arxiv.org/abs/2008.02217 (Theorem 3, Fehlerschranke, Attention-Äquivalenz)
- Bricken & Pehlevan, *Attention Approximates Sparse Distributed Memory*, NeurIPS 2021 – https://proceedings.neurips.cc/paper/2021/hash/8171ac2c5544a5cb54ac0f38bf477af4-Abstract.html
- Demircigil et al., *On a model of associative memory with huge storage capacity*, J. Stat. Phys. 2017 – https://arxiv.org/abs/1702.01929
- Amit, Gutfreund, Sompolinsky, Phys. Rev. Lett. 55 (1985) 1530 – Kapazität 0,138 N
- Bérut et al., *Experimental verification of Landauer's principle*, Nature 483 (2012) 187
- Still, Sivak, Bell, Crooks, *Thermodynamics of prediction*, Phys. Rev. Lett. 109 (2012) 120604 – https://arxiv.org/abs/1203.3271
- Bartol et al., *Nanoconnectomic upper bound on the variability of synaptic plasticity*, eLife 2015 – https://elifesciences.org/articles/10778
- McClelland, McNaughton, O'Reilly, Psychol. Rev. 102 (1995) 419; Kumaran, Hassabis, McClelland, Trends Cogn. Sci. 20 (2016) 512
- Yoon et al., *Lifelong Learning with Dynamically Expandable Networks*, ICLR 2018 – https://arxiv.org/abs/1708.01547
- Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011; Malkov & Yashunin, HNSW, IEEE TPAMI 2018
- Behrouz et al., *Nested Learning*, NeurIPS 2025 – https://arxiv.org/abs/2512.24695
- Karunaratne et al., *In-memory hyperdimensional computing*, Nature Electronics 2020 – https://arxiv.org/abs/1906.01548

Preprints (inhaltlich geprüft, nicht begutachtet – mit Vorsicht):
- Krotov, *Hierarchical Associative Memory*, arXiv 2107.06446 (2021)
- Hu, Wu, Liu, *Provably Optimal Memory Capacity for Modern Hopfield Models*, arXiv 2410.23126 (2024)
- Cowsik & Sriram, *Hierarchical Prototype Emergence in Modern Hopfield Models*, arXiv 2609.12079 (Sept. 2026)
- *Associative Memory for Non-Stationary Environments: A Self-Sizing Generalization of Hopfield Networks*, arXiv 2507.10443 (2025)
- Shafiei Kafraj, Krotov, Latham, *A Biologically Plausible Dense Associative Memory with Exponential Capacity*, arXiv 2601.00984 (2026)
- Behrouz, Hashemi, Mirrokni, *Language Models Need Sleep*, arXiv 2606.03979 (2026)
- *Hardware-aware Hopfield Network with a Nonlinear Memristor Array*, arXiv 2605.07223 (2026)

Nicht verifizierbar in dieser Sitzung (Volltext nicht abrufbar): arXiv 2609.02195 „Memory as an Energy Landscape“ – nicht verwendet.
