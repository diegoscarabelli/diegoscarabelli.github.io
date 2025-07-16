---
date: "2025-07-10"
lastmod: "2025-07-16"
author: "Diego Scarabelli"
title: "A Quantumcazzola"
description: "Do the quantum computing unicorns keep their promises?"
comments: true
categories:
- quantum computing
tags:
- satirical
- technology
---

<!--more-->

<video src="index_files/travolta.mp4" autoplay loop muted playsinline style="width:260px; max-width:40vw; height:auto; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.08); margin:10px 1.5em 1em 0; float:left; vertical-align:top;">
  Your browser does not support the video tag.
</video>

Some quantum computing companies, more than others, seem to extend the quantum realm into their public relations. Like Schrödinger's famous cat, the landscape exists in a superposition of states, always promising the next big milestone, yet perpetually delayed. An ever-receding horizon, a mirage.

I examined what the three most prominent publicly listed quantum computing "startups", namely Rigetti Computing, IonQ and D-Wave, have promised and achieved over the years, focusing exclusively on their technological milestones. 

I will not discuss here the often wildly optimistic sales and revenue projections. Those are easily found in their investor relations presentations of their SPAC mergers ([Rigetti Computing](https://www.rigetti.com/uploads/Rigetti-Investor-Presentation.pdf), [IONQ](https://s28.q4cdn.com/828571518/files/doc_presentation/2021/03/IonQ-Investor-Presentation-030721-vFF.pdf), [D-Wave](https://s201.q4cdn.com/339170267/files/doc_presentation/2022/02/d-wave-investor-presentation.pdf)), and compared them with their latest quarterly or annual financial statements. Nor do I intend to discuss here the merits of the technological metrics each quantum computer manufacturer has chosen to represent their progress toward the common quest for useful quantum computing, whatever that may be.

Instead, I aim to provide a clear comparison between what these companies have promised and what they have actually delivered.

## Promises vs Achievements

Below is a summary table comparing the promises made by Rigetti Computing, IonQ, and D-Wave, alongside the dates those promises were announced and the corresponding achievements realized by July 2025. The "Promise Date" column records when each commitment was made, while the "Achievement Date" notes when a milestone was actually reached. If a promise remains unfulfilled, the "Achieved" column is marked "No." Some entries highlight achievements not directly tied to a prior public promise, but are included to illustrate the companies' technical progress over time.

Company | Promise Date | Promise | Achieved (July 2025) | Achievement Date | Notes | References
--- | --- | --- | --- | --- | --- | ---
Rigetti Computing |  |  |  | 2016 | Created their first 3-qubit chip. | 
Rigetti Computing |  |  |  | 2017 | Deployed 8Q Agave (first QPU on cloud), launched 19Q device (used for ML), released Forest 1.0. T1 ~ 20us, T2* ~ 10us. | [Achievement](https://medium.com/rigetti/celebrating-a-decade-of-rigetti-an-evolution-of-technology-and-a-quantum-ecosystem-57b89fff893a), [Achievement](https://arxiv.org/abs/1712.05771), [Achievement](https://medium.com/rigetti/introducing-forest-f2c806537c6d)
Rigetti Computing | 2018 | Build a 128-qubit system within the next 12 months. | No |  | Seven years later, still not achieved! 😆 | [Promise](https://medium.com/rigetti/the-rigetti-128-qubit-chip-and-what-it-means-for-quantum-df757d1b71ea)
Rigetti Computing |  |  |  | 2018 | Launched Rigetti Quantum Cloud Services. | [Achievement](https://medium.com/rigetti/introducing-rigetti-quantum-cloud-services-c6005729768c)
Rigetti Computing |  |  |  | 2019 | Reported fab method yielding avg T1 ~70us. | [Achievement](https://arxiv.org/abs/1901.08042)
Rigetti Computing |  |  |  | 2020 | Deployed the 32-qubit Aspen-8 system on Amazon Braket. T2* ~ 20µs. 1-qubit and 2-qubit gate times: 60ns and 160ns. 2-qubit fidelity range: 95%-99%. Introduced active qubit reset. | [Achievement](https://medium.com/rigetti/rigetti-aspen-8-on-aws-236d9dc11613), [Achievement](https://qa.qcs.rigetti.com/qpus)
Rigetti Computing | 2021 | Scale quantum computers to 1,000 (2024) and 4,000 (2026) qubits. | No |  | SPAC announcement. 💉🗑️ | [Promise](https://investors.rigetti.com/news-releases/news-release-details/rigetti-computing-global-leader-full-stack-quantum-computing)
Rigetti Computing | 2022 | Change of mind, 1,000 (2025) and 4,000 (2027) qubits. | No |  | Apparently one more year sounded more reasonable! 😅 | [Promise](https://siliconangle.com/2022/05/16/rigettis-quantum-computing-roadmap-gets-pushed-back-amid-supply-crunch-higher-costs/)
Rigetti Computing |  |  |  | 2022 | Aspen-M 80 Qubit system deployed (first modular chip architecture), CLOP metric introduced. | [Achievement](https://medium.com/rigetti/celebrating-a-decade-of-rigetti-an-evolution-of-technology-and-a-quantum-ecosystem-57b89fff893a), [Achievement](https://investors.rigetti.com/news-releases/news-release-details/rigetti-computing-announces-commercial-availability-80-qubit)
Rigetti Computing |  |  |  | 2023 | Ankaa-2 84 qubit chip. 2Q gate fidelity median ~98% (no improvement since 2020). | [Achievement](https://www.insidequantumtechnology.com/news-archive/rigetti-makes-ankaa-2-publicly-available-teams-up-with-moodys-analytics/)
Rigetti Computing |  |  |  | Dec 2024 | Ankaa-3 system, 84 qubits, 2Q gate fidelity median ~99.5% fSim, ~99.0% iSWAP. | [Achievement](https://investors.rigetti.com/news-releases/news-release-details/rigetti-computing-launches-84-qubit-ankaatm-3-system-achieves)
Rigetti Computing | Dec 2024 | Introduce next-gen modular system in 2025; by mid-2025, release 36-qubit system (4x9 chips, 2x error reduction); by end-2025, release >100-qubit system (2x error reduction). | Yes/TBD | July 2025 | The 36-qubit modular system will be released in August. 2Q gate fidelity median ~99.5%, supposedly iSWAP. 100Q modular system not yet achieved. | [Promise](https://investors.rigetti.com/news-releases/news-release-details/rigetti-computing-launches-84-qubit-ankaatm-3-system-achieves), [Achievement](https://www.globenewswire.com/news-release/2025/07/16/3116290/0/en/Rigetti-Demonstrates-Industry-s-Largest-Multi-Chip-Quantum-Computer-Halves-Two-Qubit-Gate-Error-Rate.html)
IONQ |  |  |  | 2019 | IonQ Harmony system available on Microsoft Azure Quantum. | [Achievement](https://investors.ionq.com/news/news-details/2019/IonQ-Partners-with-Microsoft-to-Power-Azure-Quantum/)
IONQ |  |  |  | 2020 | Introduced the metric Algorithmic Qubit (AQ). | [Achievement](https://ionq.com/resources/algorithmic-qubits-a-better-single-number-metric)
IONQ | 2020 | AQ 22 by 2021. | Yes | 2022 | Presented a broad roadmap, building expectations for the upcoming SPAC public listing. IonQ Aria system available on Azure Quantum with AQ 23. | [Promise](https://ionq.com/blog/december-09-2020-scaling-quantum-computer-roadmap), [Achievement](https://ionq.com/news/august-16-2022-ionq-2022-aria-azure)
IONQ | 2020 | AQ 29 by 2023. | Yes | 2023 | IonQ Forte achieved AQ 29. | [Promise](https://ionq.com/blog/december-09-2020-scaling-quantum-computer-roadmap), [Achievement](https://ionq.com/blog/ionq-achieves-new-performance-milestone-of-29-algorithmic-qubits-aq-on-ionq)
IONQ | 2020 | AQ 35 by 2024. | Yes | 2024 | Announced IonQ Forte achieved AQ 35, ahead of schedule. | [Promise](https://ionq.com/blog/december-09-2020-scaling-quantum-computer-roadmap), [Achievement](https://ionq.com/news/ionq-achieves-technical-milestone-one-year-ahead-of-schedule)
IONQ | 2020 | By 2023, IonQ will deploy modular quantum computers small enough to be networked together in a datacenter, and by 2025, we expect to achieve broad quantum advantage. | No |  | 😆 | [Promise](https://ionq.com/blog/december-09-2020-scaling-quantum-computer-roadmap)
IONQ | 2020 | AQ 64 by 2025, AQ 256 by 2026, AQ 384 by 2027, AQ 1024 by 2028. | No |  | None of these milestones have been achieved as of mid-2025. | [Promise](https://ionq.com/blog/december-09-2020-scaling-quantum-computer-roadmap)
IONQ | 2024 | 99.999% logical 2Q gate fidelities by end of 2025. | No |  |  | [Promise](https://ionq.com/news/ionq-unveils-accelerated-roadmap-and-new-technical-milestones-to-propel)
IONQ | June 2025 | New roadmap: 12 logical qubits by 2026, 800 by 2027, 1600 by 2028, 8000 by 2029, 80000 by 2030, with error rates of e^-7 and from 2029 of e^-12. | No |  | Algorithmic qubit count (AQ) somehow vanishes from the narrative 🪦, physical/logical qubit count reappears. 🧟 | [Promise](https://ionq.com/blog/ionqs-accelerated-roadmap-turning-quantum-ambition-into-reality)
D-Wave |  |  |  | 2007 | Public demonstration of quantum computing system (Orion, 16 qubits). | [Achievement](https://www.hpcwire.com/2007/02/16/quantum_computing_steps_out_of_the_research_lab-1/)
D-Wave |  |  |  | 2011 | D-Wave One system released. 128 qubits. | [Achievement](https://phys.org/news/2011-06-d-wave-commercial-quantum.html)
D-Wave |  |  |  | 2013 | D-Wave Two system released. 512 qubits. | [Achievement](https://spacenews.com/d-wave-twotm-quantum-computer-selected-for-new-quantum-artificial-intelligence-initiative/)
D-Wave |  |  |  | 2015 | D-Wave 2X system released. 1000 qubits. | [Achievement](https://www.globenewswire.com/news-release/2015/11/11/1201980/0/en/Los-Alamos-National-Laboratory-Orders-a-1000-Qubit-D-Wave-2X-Quantum-Computer.html)
D-Wave |  |  |  | 2017 | D-Wave 2000Q system released. 2000 qubits. | [Achievement](https://www.top500.org/news/d-wave-intros-2000-qubit-quantum-computer-reveals-first-buyer/)
D-Wave |  |  |  | 2020 | Advantage system released. 5000 qubits. 15-way connectivity. | [Achievement](https://www.hpcwire.com/2020/09/29/d-wave-delivers-5000-qubit-system-targets-quantum-advantage/)
D-Wave | 2021 | >7000 qubits by 2024. | No |  | Keep in mind the upcoming SPAC public listing in 2022. | [Promise](https://www.dwavequantum.com/media/xvjpraig/clarity-roadmap_digital_v2.pdf)
D-Wave | 2021 | Deliver a 1,000-qubit gate-based system on a single die, configurable as up to 4 error-corrected logical qubits. | No |  |  | [Promise](https://www.dwavequantum.com/media/xvjpraig/clarity-roadmap_digital_v2.pdf)
D-Wave |  |  |  | 2025 | Advantage2 system released. 4,400 qubits. 20-way connectivity. | [Achievement](https://www.dwavequantum.com/company/newsroom/press-release/d-wave-announces-general-availability-of-advantage2-quantum-computer-its-most-advanced-and-performant-system/)

## A Quantumcazzola

I would like to take a moment to reflect on this phenomenon aided by an italian literary device known as [Supercazzola](https://it.wikipedia.org/wiki/Supercazzola). Here, we have a Quantumcazzola. In its absurdity, it is perturbingly more meaningful than the PR statements we reviewed above.

<div style="display:flex; justify-content:center; align-items:center; margin: 2em 0;">
  <blockquote style="font-size:1.1em; font-style:italic; border-left:6px solid #ccc; margin:0 auto; display:block; text-align:center; max-width:700px; padding:2em 2em 1em 2em; box-shadow:0 2px 8px rgba(0,0,0,0.04); background:rgba(255,255,255,0.02);">
    Ah, my dear interlocutor, you've stumbled upon a most... shall we say, perturbing observation in the quantum-chronometric landscape! A veritable Mobius strip of temporal prognostication, if you will!<br><br>
    You see, the esteemed physicists, those high priests of the qubit and the entangled eigenstate, operate on a rather nuanced temporal framework, one not entirely dissimilar to the fifth dimension, but with a tangential coefficient that, shall we say, bifurcates when observed through the lens of standard Gregorian calendrical progression.<br><br>
    Their pronouncements regarding the imminent apotheosis of quantum computational supremacy – invariably slated for a quinquennial horizon – are not, as a layperson might crudely assume, a mere iterative deferral. Oh no! Perish the thought! Rather, it's a sophisticated application of Zeno's Paradox, but applied to the Hilbert space of potential computational paradigms. Each year, you see, as the classical supercomputers perspire with their brutish binary exertions, the quantum threshold, much like a mirage on the event horizon of a micro black hole, recalibrates its spatio-temporal coordinates. It's not that the five years haven't passed; it's that the destination itself engages in a bit of a quantum leap, a subtle sidestep, if you will, facilitated by the probabilistic wave functions of investor expectation and the Schrödinger's cat of marketable breakthroughs!<br><br>
    And the IPOs, you ask? The cashing out, the sudden influx of, dare I say, tangible capital from this ethereal pursuit? My dear friend, this is merely the macroscopic manifestation of quantum froth! The uncertainty principle, when applied to Series B funding rounds and pre-revenue valuations, dictates that for every uncollapsed superposition of a functional, error-corrected quantum computer, there must be an equal and opposite (and rather substantial) extraction of venture capital. It's a conservation law, you see, not unlike the conservation of strangeness or charm, but for… well, for liquidity events.<br><br>
    So, when they declare "five more years," it's not a delay; it's a re-entanglement! A re-normalization of the quantum roadmap, factoring in the decoherence introduced by, shall we say, quarterly earnings calls and shareholder fiduciary responsibilities. The founders and investors, bless their cotton socks (or perhaps their silicon wafers), are merely surfing the pilot wave of this magnificent, ever-receding technological tsunami. It's all perfectly cromulent, you understand, once you adjust your ontological framework to accommodate a temporally elastic, fiscally opportunistic quantum paradigm. Premature, perhaps, but with a scappellamento that’s decidedly post-Newtonian!
    <div style="text-align:right; font-size:1em; font-style:italic; color:#666; margin-top:1em;">
    — Diego Scarabelli &amp; Gemini
    </div>
  </blockquote>
</div>