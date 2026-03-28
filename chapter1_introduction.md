# Chapter 1: Introduction

## 1.1 Problem Domain

Proteins rarely function in isolation. Nearly every biological process — from DNA replication and signal transduction to immune response — depends on proteins physically binding to one another to form functional complexes [1]. Understanding which proteins interact, and through which structural interfaces, is therefore one of the central problems in molecular biology. A complete map of protein–protein interactions (PPIs) for a given organism would be of considerable value: it could reveal new drug targets, explain disease mechanisms, and clarify how cellular machinery is assembled and regulated [1, 2].

Despite its importance, the protein interaction landscape remains largely uncharted. The human proteome alone contains approximately 20,000 proteins, giving rise to roughly 2 × 10⁸ possible pairwise interactions. Experimental techniques for detecting PPIs, such as yeast two-hybrid (Y2H) screening and affinity-purification mass spectrometry (AP-MS), can identify interactions at scale, but they are noisy, biased towards certain interaction types, and do not cover the full proteome [1, 3]. High-throughput screens typically achieve only partial coverage: the most comprehensive human interactome studies to date have mapped fewer than 10% of estimated interactions, and the overlap between independent experimental datasets is often surprisingly low, reflecting both biological complexity and technical limitations [3]. Exhaustively testing the full space of pairwise interactions experimentally is simply not feasible with current technology.

Computational methods offer a way to narrow this search space. Over the past two decades, a range of approaches have been developed, including sequence-based co-evolution analysis, machine-learning classifiers trained on known interactions, and physics-based docking simulations [4, 5]. Each comes with significant trade-offs. Co-evolution methods require deep multiple sequence alignments that are unavailable for many protein families. Docking is computationally expensive and demands high-quality input structures. Supervised learning approaches depend heavily on the coverage and quality of their training data, and risk learning dataset-specific biases rather than generalisable interaction signals [5].

A fundamental shift occurred in 2021 with the release of AlphaFold2 [6]. AlphaFold2 predicts protein tertiary structure from sequence alone with near-experimental accuracy, and the AlphaFold Protein Structure Database (AFDB) has since been expanded to cover over 200 million predicted structures across hundreds of organisms [7]. This unprecedented availability of reliable structural data creates a new opportunity: the possibility of inferring protein interactions at proteome scale using structural information alone, without requiring expensive experimental assays or deep evolutionary signals.

The principle that underlies this opportunity is structural homology transfer. If a domain in protein X is structurally very similar to a domain in protein Y, and protein Y is known to interact with protein Z through that domain, then protein X may also interact with Z (or proteins structurally similar to Z) via an analogous binding interface. This principle is well established for transferring functional annotations between proteins [8], but its systematic application to large-scale PPI prediction remains relatively underexplored. This project investigates whether domain-level structural homology, captured through fast structural search, can serve as a useful signal for predicting protein–protein interactions at proteome scale.


## 1.2 Benefits and Use Cases

A scalable, structure-based method for predicting PPIs would have broad applications across biology and medicine. In drug discovery, identifying novel interaction partners for a protein of interest can reveal previously unknown targets for therapeutic intervention. Many diseases arise from the disruption or dysregulation of specific protein interactions — for example, oncogenic signalling often involves aberrant binding between kinases and their substrates [2]. A computational pipeline that can rapidly screen for candidate interaction partners would allow researchers to prioritise which interactions to validate experimentally, substantially reducing the time and cost of target identification.

Beyond drug discovery, PPI prediction supports the broader goal of mapping complete cellular interaction networks. Such maps are essential for systems biology, where understanding how perturbations propagate through a network can explain complex phenotypes, predict drug side effects, and guide the design of combination therapies [2]. Current experimental interactome maps are incomplete and organism-specific; a computational approach that generalises across organisms via structural similarity could help fill gaps in species where experimental data is sparse.

The approach taken in this project is particularly well suited to these use cases because it operates at the domain level rather than the whole-protein level. Many proteins are multi-domain, and different domains within the same protein can mediate distinct interactions. By decomposing proteins into their constituent domains and searching for structural homologs of specific interface domains, the pipeline can make more fine-grained predictions than whole-protein methods, and can naturally handle the common biological scenario where a single protein participates in multiple, structurally distinct interactions.

Furthermore, the infrastructure developed in this project — including a SQLite metadata index and a filtered embedding iterator — enables queries to be scoped by organism and structural family. This makes the system practical for targeted biological investigations, such as searching for all candidate interaction partners of a particular human kinase domain, rather than requiring a full proteome-wide screen every time.


## 1.3 Project Aim

The overarching aim of this project is to investigate whether domain-level structural homology, as captured by Merizo-search and the TED domain pair list, provides a useful signal for predicting protein–protein interactions at proteome scale. In doing so, the project seeks to extend Merizo-search — a fast domain segmentation and structural similarity search tool developed at UCL — with a novel application it was not originally designed for: using structural search results to infer biological interactions between proteins.

Two complementary aims support this central investigation:

- To develop practical infrastructure (filtering, metadata indexing, and efficient iteration) that makes structural homology search usable for targeted biological queries over a database of more than 200 million domains.

- To critically assess the strengths and limitations of a purely structure-based approach to PPI prediction, identifying where complementary signals such as co-expression or co-evolution data would be needed to improve prediction quality.


## 1.4 Project Objectives

Following the distinction drawn in the UCL project guidelines, the aims above describe the overall purpose of the project, while the following objectives are specific, measurable deliverables:

1. **Implement a PPI prediction pipeline** based on domain structural homology search, using Merizo-search and the TED pair list as the underlying infrastructure. The pipeline should accept a query protein, decompose it into domains, identify known domain–domain contacts from the TED dataset, and use Foldclass to search for structural homologs of the interface domain. Proteins containing high-scoring homologs are returned as candidate interaction partners.

2. **Build a searchable, filtered database** derived from the TED pair list, enabling queries to be scoped by organism (taxonomy ID) and structural classification (CATH family). This involves constructing a SQLite metadata index storing per-domain information (taxonomy ID, CATH classification, model confidence, domain density) and implementing a filtered embedding iterator that loads only the relevant subset of embeddings into memory during search.

3. **Evaluate the approach against the Zhang et al. benchmark** [1], reporting area under the weighted precision–recall curve (AUCPR) and precision at recall thresholds of 0.1, 0.2, and 0.5 across a range of TM-score cutoffs. The benchmark comprises 3,000 high-confidence positive PPI pairs and 30,000 random negative pairs drawn from the intersection of STRING, BioGRID, and UniProt.

4. **Analyse the limitations of the method**, particularly failure cases where structural homology alone is insufficient to predict biologically meaningful interactions — for example, paralogs with different interaction partners, convergent fold evolution, and intrinsically disordered interfaces.


## 1.5 Project Approach

This project extends Merizo-search [9], a structural domain search tool that combines two components. The first is **Merizo** [10], a deep-learning model that segments a protein structure into its constituent domains using invariant point attention. The second is **Foldclass** [9], a structural comparison engine that encodes each domain as a fixed-length neural-network embedding using an equivariant graph neural network (EGNN). Given a query domain, Foldclass searches a precomputed database of embeddings via cosine similarity to rapidly retrieve structurally similar candidates, which are then re-ranked using TM-align [11] for precise structural alignment scoring.

The core idea of the project is as follows. Consider a protein containing two domains, Domain A and Domain B, where Domain B serves as the interface domain responsible for mediating a binding interaction. If a structural search identifies another protein whose domain B' is a close structural homolog of Domain B, then that protein becomes a candidate for interacting with partners of Domain A via the same binding interface. More formally, if Domain B is known to interact with Domain A, and B' is a structural homolog of B, then B' is hypothesised to also interact with A (or structural homologs of A). This hypothesis — structural homology transfer applied to domain–domain contacts — forms the foundation of the proposed pipeline.

Domain pairing information, that is, which domains are predicted to form interaction interfaces, is derived from the TED (The Encyclopaedia of Domains) dataset [7]. TED decomposes AlphaFold-predicted structures into domains and identifies intra-protein domain–domain contacts. Each such contact pair is treated as a potential interaction template: Domain B (the interface domain) is used as the search query, and proteins containing high-scoring structural homologs of Domain B are returned as predicted interaction partners. To enable this search at scale, the TED pair list — comprising over 200 million domain entries — was transformed into a Foldclass-compatible indexed database.

Beyond the core search pipeline, several features were implemented to improve practical usability. A metadata index was constructed using SQLite, storing per-domain information including taxonomy ID, CATH fold classification, model confidence, and domain density. This index supports query-time filtering, allowing results to be restricted to a specific organism or structural family. A filtered iterator was also implemented to load only the relevant subset of embeddings into memory during search, reducing both runtime and memory consumption when operating on targeted subsets of the database.

The system was evaluated against the benchmark of Zhang et al. [1] (*Science*, 2025), which provides 3,000 high-confidence positive PPI pairs and 30,000 random negative pairs. Evaluation follows a weighted precision–recall framework with a positive weight of 0.01, reflecting the approximately 1:1,000 signal-to-noise ratio expected in proteome-wide screening. A sweep of TM-score thresholds was conducted to characterise the trade-off between precision and recall, and a critical analysis identifies the principal failure modes of the approach.


## 1.6 Report Structure

The remainder of this report is organised as follows:

**Chapter 2: Background and Related Work** provides the biological and computational context for this project. It introduces protein–protein interactions and their significance, surveys existing computational approaches to PPI prediction and their limitations, and describes the enabling technologies underpinning this work: AlphaFold2, the TED dataset, domain-based protein representation through CATH classification, and the Merizo-search pipeline. The chapter concludes with a discussion of structural homology transfer as an established technique in function annotation and its gap in PPI prediction, and introduces the evaluation framework used in this project.

**Chapter 3: System Design and Implementation** describes the architecture of the PPI prediction pipeline in detail. It covers the construction of the Foldclass-compatible database from the TED pair list, the design of the search pipeline (domain extraction, embedding search, TM-score re-ranking), the SQLite metadata index schema, the filtered embedding iterator, and the end-to-end workflow. The chapter also discusses key design decisions and the alternatives that were considered.

**Chapter 4: Evaluation Methodology** presents the evaluation strategy. It describes the Zhang et al. benchmark, explains why standard precision–recall is insufficient for proteome-wide screening, introduces the weighted precision–recall framework, and details the metrics and TM-score threshold sweep design used in the evaluation.

**Chapter 5: Results and Analysis** reports and analyses the experimental results. It presents benchmark performance at the default TM threshold, the threshold sweep results, and case studies of where the method succeeds and fails. It includes a critical failure analysis examining cases where structural homology is insufficient, validates the filtering infrastructure, and compares performance against a baseline.

**Chapter 6: Conclusions** summarises the project's contributions, addresses each objective from this chapter, critically evaluates whether structural homology alone is sufficient for PPI prediction, discusses limitations and future work, and offers reflections on the project.


## References (for this chapter)

[1] J. Zhang, I. R. Humphreys, J. Pei *et al.*, "Predicting protein-protein interactions in the human proteome," *Science*, 2025.

[2] M. Vidal, M. E. Cusick, and A.-L. Barabási, "Interactome networks and human disease," *Cell*, vol. 144, no. 6, pp. 986–998, 2011.

[3] T. Rolland *et al.*, "A proteome-scale map of the human interactome network," *Cell*, vol. 159, no. 5, pp. 1212–1226, 2014.

[4] D. Szklarczyk *et al.*, "The STRING database in 2023," *Nucleic Acids Research*, vol. 51, pp. D483–D489, 2023.

[5] Q. C. Zhang *et al.*, "Structure-based prediction of protein-protein interactions on a genome-wide scale," *Nature*, vol. 490, pp. 556–560, 2012.

[6] J. Jumper, R. Evans, A. Pritzel *et al.*, "Highly accurate protein structure prediction with AlphaFold," *Nature*, vol. 596, pp. 583–589, 2021.

[7] A. M. Lau *et al.*, "Exploring structural diversity across the protein universe with the encyclopedia of domains," *Science*, 2024.

[8] C. A. Orengo *et al.*, "CATH — a hierarchic classification of protein domain structures," *Structure*, vol. 5, no. 8, pp. 1093–1108, 1997.

[9] S. M. Kandathil, A. M. Lau, D. W. A. Sheridan, and D. T. Jones, "Foldclass and merizo-search: scalable structural similarity search for single- and multi-domain proteins using geometric learning," *Bioinformatics*, vol. 41, no. 5, p. btaf277, 2025.

[10] A. M. Lau *et al.*, "Merizo: a rapid and accurate protein domain segmentation method using invariant point attention," *Nature Communications*, vol. 14, p. 8445, 2023.

[11] Y. Zhang and J. Skolnick, "TM-align: a protein structure alignment algorithm based on the TM-score," *Nucleic Acids Research*, vol. 33, no. 7, pp. 2302–2309, 2005.
