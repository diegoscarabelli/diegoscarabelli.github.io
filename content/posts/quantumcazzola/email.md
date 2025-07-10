Coming to your question, I did look briefly some time ago at the promises in terms of technological figures of merits (qubit count, gate fidelity, architecture, connectivity, etc). However, I did not write much. So I took your question and did a little research. Below are my preliminary findings. I separate the actual milestones achieved from the promises.

Achievements

1. 2016: Created their first 3-qubit chip.  
2. 2017: Deployed the 8Q Agave device on "cloud" and launched the 19Q device. 8Q Agave was the first QPU deployed on their cloud. 19Q device was used to solve an unsupervised machine learning problem. ([source\_1](https://medium.com/rigetti/celebrating-a-decade-of-rigetti-an-evolution-of-technology-and-a-quantum-ecosystem-57b89fff893a), [source\_2](https://arxiv.org/abs/1712.05771)). These are the two chips I developed the fab process for. I also remember bringing up the 19Q chips. Released Forest 1.0, a quantum computing platform for algorithm development. ([source\_3](https://medium.com/rigetti/introducing-forest-f2c806537c6d))   
   T1 \~ 20us, T2\* \~ 10us  
3. 2018: Launched Rigetti Quantum Cloud Services ([source\_4](https://medium.com/rigetti/introducing-rigetti-quantum-cloud-services-c6005729768c))  
4. 2019 report fab method which yielded avg T1 \~ 70us. I worked on this fab method. ([source\_5](https://arxiv.org/abs/1901.08042))  
5. 2020: 32Q Aspen-8 system deployed on Amazon Braket. ([source\_6](https://medium.com/rigetti/rigetti-aspen-8-on-aws-236d9dc11613), [source\_7](https://qa.qcs.rigetti.com/qpus))  
   \- T2\* \~ 20µs  
   \- 1Q and 2Q gate times: 60ns and 160ns  
   \- 2Q fidelity range 95%-99%   
6. \- active reset (I worked on it)  
   The 32Q chip is the last one I worked on before leaving in late 2018, therefore it took almost 2 years for them to actually deploy it.   
7. 2022 (post SPAC): Aspen-M System: Deployed Aspen-M 80 Qubit system, the first device using modular chip architecture ([source\_8](https://medium.com/rigetti/celebrating-a-decade-of-rigetti-an-evolution-of-technology-and-a-quantum-ecosystem-57b89fff893a), [source\_9](https://investors.rigetti.com/news-releases/news-release-details/rigetti-computing-announces-commercial-availability-80-qubit)). They introduce the CLOP metric (circuit layer operation per sec).  
8. 2023: Ankaa-2 84 qubit chip ([source\_10](https://www.insidequantumtechnology.com/news-archive/rigetti-makes-ankaa-2-publicly-available-teams-up-with-moodys-analytics/)).  
   2Q gate \~ 98% median (basically no improvement from 2020 on this figure of merit).  
9. Dec 2024: Ankaa-3 system, 84 qubits.([source\_11](https://investors.rigetti.com/news-releases/news-release-details/rigetti-computing-launches-84-qubit-ankaatm-3-system-achieves))  
   2Q gate fidelity median \~ 99.5%

Promises

1. 2018: "we will build a 128-qubit system during the next 12 months" ([source\_12](https://medium.com/rigetti/the-rigetti-128-qubit-chip-and-what-it-means-for-quantum-df757d1b71ea)). 6 years later, it still has to happen.  
2. 2021 (during the SPAC, to pump it): "Rigetti expects to scale its quantum computers from 80 qubits in 2021, to 1,000 qubits in 2024, and to 4,000 qubits in 2026" ([source\_13](https://investors.rigetti.com/news-releases/news-release-details/rigetti-computing-global-leader-full-stack-quantum-computing))  
3. 2022: pushes back the previous promise to 2025 and 2027 respectively. ([source\_14](https://siliconangle.com/2022/05/16/rigettis-quantum-computing-roadmap-gets-pushed-back-amid-supply-crunch-higher-costs/))  
4. Dec 2024: "Rigetti plans to introduce the next generation of its modular system architecture, while continuing to increase fidelities, in 2025\. By mid-year 2025, the Company expects to release a 36-qubit system based on four 9-qubit chips tiled together, with a targeted 2x reduction in error rates from the current level. By the end of 2025, the Company expects to release a system with over 100 qubits with a targeted 2x reduction in error rates from the current level." ([source\_11](https://investors.rigetti.com/news-releases/news-release-details/rigetti-computing-launches-84-qubit-ankaatm-3-system-achieves))  
5. 

