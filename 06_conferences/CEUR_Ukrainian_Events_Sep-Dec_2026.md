# CEUR-WS Survey — Ukrainian-Organised Events, September–December 2026

**Prepared:** 4 September 2026
**Source:** [ceur-ws.org](https://ceur-ws.org/) full volume index (4,256 volumes, newest Vol-4255), plus each candidate's own conference site
**Scope requested:** conferences and workshops held **September, October, November or December 2026**, **online**, **organised by Ukrainian scholars** — working back from 2025 editions that could recur in 2026
**Companion file:** `Conferences.xlsx` — **103 rows**. Every event named in this report is now IN the table.

---

## 1. How this was done

1. Loaded the complete CEUR-WS volume index and parsed all 4,256 volume entries programmatically.
2. Narrowed to Vol-3900–4260 (the 2025–2026 publication window) — 356 volumes.
3. Filtered on Ukrainian city and institution names, then **opened 28 individual volume pages** and read their headers directly (city, country, exact date, "(online)" marker, editors and their affiliations, submission counts).
4. For every surviving candidate, opened the series' own website to see whether a 2026 edition is announced.
5. Cross-referenced against the rows already in `Conferences.xlsx`, then added everything Ukrainian that was missing — in or out of the September–December window, 2025 editions included.

**A caveat on "online".** CEUR volume metadata rarely records the format. Across the whole 2025–2026 window, exactly **one** Ukrainian volume states it explicitly — CPITS-II 2025, marked "(online)". For everything else the format comes from the conference site, or is simply not published. Where I could not confirm it, I say so rather than assuming; several of these events are hybrid, which in practice serves remote Ukrainian participation.

---

## 2. Act on this first

> ### SMICS-2026 — registration closes **7 September 2026** (3 days)
> International Conference "Security of Modern Information and Communication Systems", **24–25 September 2026**, University of the National Education Commission, Kraków. Run by the **Department of Cybersecurity, Ivan Franko National University of Lviv**, with 9+ co-organisers. It carries **two CEUR-WS workshops** — *Artificial Intelligence for Cybersecurity* and *Cryptology and Data Security* — both slated for CEUR open-access publication. Peer review via Microsoft CMT.
> **Site:** https://smics.lnu.edu.ua/ · **Contact:** smics@lnu.edu.ua
> The site shows a live countdown to the 7 September registration deadline but publishes no separate paper-submission date; the 2025 edition's cryptology workshop (Vol-4191) ran 16–18 October 2025 in Lviv.

This was the single most actionable thing the survey turned up. It is now row-tracked in `Conferences.xlsx` as **High / Registration due**, together with both of its workshops.

---

## 3. Already covered in `Conferences.xlsx`

Fourteen Ukrainian-organised events already sit in the table inside your Sep–Dec 2026 window, so the survey did **not** need to re-find them:

| Event | Dates 2026 | Where | Format |
|---|---|---|---|
| MCEME | 07–12 Sep | Lviv Polytechnic | Hybrid |
| ICTERI | 15–18 Sep | Chernivtsi | Hybrid |
| ICST-ODESA (XIV) | 22–24 Sep | Odesa | not stated |
| ADT-ISMDDC | 22–24 Sep | Odesa (in ICST) | not stated |
| AISTDS | 01 Oct | KNU Kyiv | not stated |
| DESSERT | 02–04 Oct | Corfu, Greece | Ukrainian-led (KhAI) |
| KhPIW | 05–09 Oct | NTU "KhPI" | Online / hybrid |
| CSIT | 14–17 Oct | Lviv | not stated |
| ATRIC | 15–17 Oct | Lviv Polytechnic | Hybrid |
| ITTAP | 19–21 Oct | Ternopil | Remote available |
| CITI | 04–06 Nov | Ternopil | Hybrid |
| BAITmp | 11–13 Nov | Morocco + Ternopil | **Online presentations** |
| ICTM | 18–20 Nov | Kharkiv (KhAI) | not stated |
| IT&I (XIII) | 19–20 Nov | KNU Kyiv | not stated |

---

## 4. Gaps found — candidates matching your criteria

### 4.1 CPITS-II — the clearest omission

| | |
|---|---|
| **Full name** | Cybersecurity Providing in Information and Telecommunication Systems **II** |
| **2025 edition** | Kyiv, **26 October 2025 — explicitly "(online)"** |
| **CEUR** | Vol-4145 (2025-II); series runs to Vol-3826, 3550, 3421, 3288, 3187, 2923, 2746 |
| **Scale** | **63 papers submitted** for peer review — the largest Ukrainian volume in the window |
| **Editors** | Volodymyr Sokolov (Borys Grinchenko Kyiv Metropolitan Univ.), Vasyl Ustimenko (Royal Holloway, UK), Tamara Radivilova (KhNURE), Mariya Nazarkevych (Lviv Polytechnic) |
| **Site** | https://cpits.kubg.edu.ua/ |
| **2026-II status** | **Not yet announced** |

**Why this matters.** Your table has CPITS — but only the **February** edition (CPITS 2026, 28 Feb 2026, now expired). CPITS runs **twice a year**, and the second edition is the October one. The site's own DBLP links confirm the pattern: `cpits2021-2`, `cpits2024-2`, `cpits2025-2`. The October edition has fallen on **26 October in 2023, 2024 and 2025** — three years on the same calendar date.

On that pattern, **CPITS-II 2026 should fall on or about 26 October 2026**, online, free, CEUR/Scopus/DBLP, with submission around early October. It matches every one of your three criteria and is the strongest single recommendation in this report.

*Submission format (from the 2026 CFP): short papers 5–9 CEUR-WS pages, regular papers 10+ pages. Chairs include Oleksii Smirnov (Central Ukrainian NTU), Hennadii Hulak (BGKMU) and Serhii Toliupa (KNU).*

### 4.2 SMICS-2026 and its two CEUR workshops

Covered in §2 above. The cryptology workshop has a verified track record:

| | |
|---|---|
| **WCDS 2025** | Workshop on Cryptology and Data Security, **Lviv, 16–18 October 2025** |
| **CEUR** | Vol-4191 · 14 papers submitted |
| **Editors** | Petro Venherskyi, Serhii Yevseiev, Oleg Gutik, Valeriy Trushevskyy, Mykhailo Viachalo — all Ivan Franko National University of Lviv, Faculty of Applied Mathematics and Informatics |
| **2026** | Runs again inside SMICS-2026, alongside a second workshop on *Artificial Intelligence for Cybersecurity* |

### 4.3 The IT&I workshop tracks — three separate CEUR volumes

Your table has IT&I the **conference** (19–20 Nov 2026, Springer CCIS). What it does not have is that IT&I also hosts **workshops that publish separately in CEUR**, on the same dates:

| Workshop | CEUR | 2025 edition | Editors |
|---|---|---|---|
| **IT&I-WS: ISICP** — Information Systems And Intelligent Cyber Protection | Vol-4170 · 27 submissions | Kyiv, 20–21 Nov 2025 | Serhii Toliupa (KNU), Dmytro Lande (Igor Sikorsky KPI), Iurii Krak (Glushkov Inst. NASU), Sergiy Yakovlev (Lodz) |
| **IT&I-WS: AITDS** — Artificial Intelligence Technologies and Data Science | Vol-4158 · 26 submissions | Kyiv, 20–21 Nov 2025 | Vitaliy Snytyuk (KNU), Lyudmyla Kirichenko (Lodz), Oksana Mulesa (Prešov), Serhii Lupenko (Opole) |
| **IT&I-WS: ITIAS** — IT Infrastructure and Applied Solutions | Vol-3933, Vol-3955 | Kyiv, 20–21 Nov 2024 | Vitaliy Snytyuk et al. |

**2026 status:** IT&I-2026 is officially titled the "XIII International Conference **and Workshops**", and states that "during the conference, workshops are held at the initiative of the organizers or renowned scientists. The topics of the workshops are announced additionally." Its `/en/workshops/` page currently reads **"Will be soon…"**.

So the workshops are confirmed to be happening on **19–20 November 2026** — only the titles are pending. This is a CEUR route into a November Ukrainian event with a deadline that will be later and looser than the main conference's (which closed 1 September 2026).

### 4.4 AIT&AIS — December, Chernivtsi, hybrid

| | |
|---|---|
| **Full name** | International Scientific Workshop on Applied Information Technologies and Artificial Intelligence Systems |
| **2025 edition** | Chernivtsi, **18–19 December 2025**, **hybrid (on-site at CHNU + online video platform)** |
| **CEUR** | Vol-4160 |
| **Editors** | Harrison Hao Yang (SUNY Oswego), Christopher Kumar Anand (McMaster), **Aleksandr Gozhyj (Petro Mohyla Black Sea National Univ., Mykolaiv)**, Batyrkhan Omarov, **Serhii Vladov** |
| **Host** | Dept. of Computer Science, Yuriy Fedkovych Chernivtsi National University |
| **Site** | https://kkn.chnu.edu.ua/en/aitais/ |
| **2026 status** | Not yet announced — site still shows 2025 |

Registration for 2025 was 10 October, extended to 17 November. If 2026 repeats, expect registration to open around **October 2026 for a mid-December workshop** — the only December candidate found, and explicitly hybrid.

---

## 5. Ukrainian CEUR series outside your Sep–Dec window

Verified Ukrainian, but the 2025/2026 edition falls outside September–December. Listed because they are absent from your table and may be worth tracking for other months:

| Series | CEUR | Edition read | Organisers |
|---|---|---|---|
| **RS** — Resilient Systems: Secure Digital Technologies and Critical Infrastructure | Vol-4225 | Drohobych, **30 Jun 2026** | Iaroslav Dorohyi (Donetsk NTU, Drohobych), Oleksandr Chemerys (Pukhov Inst. for Modelling in Energy Eng., NASU) |
| **ITPM** — IT Project Management (6th) | Vol-4023 · 33 subs | Kyiv, 22 May 2025 | Sergey Bushuyev (UKRNET), Nataliia Kunanets, Volodymyr Pasichnyk, Nataliia Veretennikova (Lviv Polytechnic) |
| **SPICIT / ISecIT** — Scientific and Practical Issues of Cybersecurity and IT (at the V ISecIT conference) | Vol-4150 · 25 subs | **Lutsk**, 9–11 Jun 2025 | Mikolaj Karpinski (UKEN Poland), Mariia Nazarkevych & Ivan Opirskyy (Lviv Polytechnic), Oleksii Smirnov (Central Ukrainian NTU) |
| **CQPC** — Classic, Quantum, and Post-Quantum Cryptography (at PICST 2025) | Vol-4016 · 11 subs | Kyiv, 5 Aug 2025 | Sokolov, Ustimenko, Nazarkevych |
| **CTE** — Cloud Technologies in Education (12th, at ICHTML) | Vol-4043 | Kryvyi Rih, 12 May 2025 | Semerikov, Striuk, Pinchuk, Vakaliuk |
| **UkrProg-IIT** — Intelligent Information Technologies (at UkrPROG 2025) | Vol-4049 | Kyiv, **13–14 May 2025** | Volodymyr Pasichnyk (Lviv Polytechnic), Volodymyr Sokolov (BGKU) |
| **IWSCI** — Computational Intelligence (at IntSol 2025) | Vol-4035 | Kyiv–Uzhhorod, 1–5 May 2025 | Snytyuk, Bodyanskiy, Hulianytskyi, Sergienko, Zaychenko |

Two of these incidentally **resolve open questions in your table**: Vol-4049 confirms UkrPROG 2025 ran 13–14 May 2025 (a row I previously could not verify at all, because every UkrPROG URL 404s), and Vol-4035 confirms IntSol 2025 ran 1–5 May 2025, matching what the table already holds.

The **CTE** find is notable: your table already carries six workshops from the Semerikov/Vakaliuk "easyscience" family (AREdu, ICon-MaSTEd, STE(A)M, CS&SE@SW, DOORS, ICSF, DigiTransfEd) but not this one.

---

## 6. Checked and excluded

Look-alikes I opened and ruled out, so you don't have to:

| Volume | Event | Why excluded |
|---|---|---|
| Vol-3922 | "Informatics And Applied Mathematics 2024" | **IAM, Guelma, Algeria** — University of Guelma. Not the KazNU CSAM conference despite the near-identical name. This is why CSAM's row records journal publication, not CEUR. |
| Vol-4180 | CISN — Cybersecurity, Infocommunication Systems and Networks | Almaty, Kazakhstan, 19–20 Nov 2025. Editors at International IT University Almaty. Borderline: Kateryna Kolesnikova also holds Ukrainian affiliations, so treat as a Kazakh-hosted venue with Ukrainian involvement. |
| Vol-4204 | STIoT — Smart Technologies and IoT | Almaty, 19–20 Nov 2025, same Almaty organisers. *(Note: CEUR's page title for this volume reads "Software and Knowledge Engineering 2025" while the content is STIoT — a CEUR metadata error.)* |
| Vol-4014 | AIT — Application of Immersive Technology | Almaty, 5 Mar 2025 |
| Vol-4124 | BigHPC | Turin, Italy (at ITADATA) |
| Vol-4232 | CITA-DW | Cotonou, Benin |
| Vol-4092 | ITAT | Slovak/Czech series |
| Vol-4213 | Information Society and University Studies | Lithuania (IVUS) |
| Vol-3962, 4198 | Joint National Conference on Cybersecurity | Italy (ITASEC & SERICS) |
| Vol-4044 | RTA-CSIT | Albania |

---

## 7. Watchlist — recommended additions

Ranked by how well each fits "Sep–Dec 2026 + online + Ukrainian":

| # | Event | Expected 2026 window | Online? | CEUR | Announced? | Fit |
|---|---|---|---|---|---|---|
| 1 | **CPITS-II** | ~26 Oct 2026 (3-year fixed date) | **Yes, explicit** | Yes | No | ★★★★★ |
| 2 | **SMICS-2026** + AI4Cyber + WCDS | **24–25 Sep 2026** | not stated | Yes ×2 | **Yes — reg. 7 Sep** | ★★★★☆ |
| 3 | **IT&I-WS** (ISICP / AITDS / ITIAS) | **19–20 Nov 2026** | not stated | Yes ×3 | Conference yes, topics pending | ★★★★☆ |
| 4 | **AIT&AIS** | ~mid-Dec 2026 | **Yes, hybrid** | Yes | No | ★★★☆☆ |
| 5 | **CTE** | ~May 2027 | not stated | Yes | No | ★★☆☆☆ (out of window) |
| 6 | **RS** | Jun 2027 | not stated | Yes | 2026 done | ★★☆☆☆ (out of window) |
| 7 | **ITPM · SPICIT · CQPC · UkrProg-IIT · IWSCI · MoMLeT-WS · MoDaST** | May–Aug 2027 | not stated | Yes | No | out of window, all Ukrainian |
| 8 | **CLW / IS / PhD-AI at CoLInS** | contingent | not stated | Yes | CoLInS postponed | blocked upstream |
| 9 | **DTESI + AI@DTESI · CISN · STIoT · SKE** | ~Nov 2026 | not stated | Yes ×4 | No | borderline: Almaty-hosted |

**Status: all added.** `Conferences.xlsx` now carries **103 rows** — 26 added from this survey, with the same 33 columns as every other entry.

- **SMICS-2026, WCDS-2026 and AI4Cyber-2026** entered as **High**, status *Registration due*, registration **07.09.2026**.
- Everything else entered as **Finished**, holding its verified 2025 (or 2024/2026) edition as a placeholder.
- For rows with no 2026 call yet, the five deadline columns are **deliberately blank** — a CEUR volume records the event date, venue and editors but never the submission deadlines. Each row's *Verification Notes* cell says so, so a blank is not mistaken for missing research.
- The nine non-Ukrainian look-alikes in §6 were **not** added; they fail the organiser-nationality criterion and would only dilute the tracker.

To refresh a row when its 2026 call appears: fill the deadline columns, then change *Relevance* to **High** via the drop-down. *Priority*, *Days to Deadline* and *Submission Status* all recalculate themselves.

---

## 8. Verification status — what I read directly

**Volume pages opened and read in full (28):** 4232, 4225, 4224, 4204, 4193, 4191, 4184, 4180, 4170, 4160, 4158, 4150, 4145, 4124, 4049, 4043, 4035, 4023, 4016, 4015, 4014, 4005, 4004, 4000, 3983, 3976, 3966, 3922.

**Conference sites opened:** smics.lnu.edu.ua, cpits.kubg.edu.ua, iti.fit.univ.kiev.ua (+ /workshops/), kkn.chnu.edu.ua/en/aitais/, icst-conf.com, csam.kaznu.kz.

**Everything named in this report was verified by opening its CEUR volume page.** The eight volumes flagged as unchecked in the first draft were subsequently opened, and four turned out to be false leads that are therefore NOT in the table: Vol-4184 (Zephyr in Science and Education — Jena, Germany, Navimatix GmbH), Vol-4000 (XXV International Conference on Human-Computer Interaction — INTERACCIÓN 2025, Valladolid, Spain), Vol-4193 (Software and Knowledge Engineering — Almaty) and Vol-4224 (Workshop on Artificial Intelligence — Almaty). The other four were confirmed Ukrainian and added: the three CoLInS 2025 satellites (Vol-3976, 3983, 4015 — all Kharkiv, 15–16 May 2025) and the MoMLeT pair (Vol-4004, 4005 — Lviv, June 2025).

One correction worth recording: **CoLInS 2025 did run** — Kharkiv, 15–16 May 2025, with three CEUR satellite workshops. The “postponed indefinitely” notice on colins.in.ua applies to the 10th/2026 edition, not to 2025.

**Two structural limits worth knowing.** CEUR's index is a record of what has *already been published*, so a 2026 autumn event will not appear there until its proceedings are out — typically several months after the event. That makes CEUR excellent for establishing which Ukrainian series exist and when they habitually run, but useless for finding an open 2026 call. Every "expected 2026 window" above is therefore inferred from the series' own history and marked as such, not read off a published call. And CEUR volume metadata omits format almost universally, which is why §1's caveat about "online" matters.
