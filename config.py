#!/usr/bin/env python3
"""
Configuration file for KNOT GNN druggability prediction
Updated for new data structure
"""

import torch
from pathlib import Path

# ===============================================================================
# PATHS
# ===============================================================================

# Data paths
DATA_DIR = Path('/home/zihend1/Genesis/Data/TargetIdentification/data')
FEATURES_FILE = DATA_DIR / 'gene_features.tsv'
LABELS_FILE = DATA_DIR / 'gene_labels.tsv'
OUTPUT_DIR = Path('./results')
CHECKPOINT_DIR = Path('./checkpoints')

# Edge data paths
EDGE_DIR = DATA_DIR / 'edge'
EDGE_PATHS = {
    'regnet': EDGE_DIR / 'human.source',
    'trrust': EDGE_DIR / 'trrust_rawdata.human.tsv',
    'coexp': EDGE_DIR / 'coexpression_edges_99p.tsv',
    'ppi': EDGE_DIR / 'ppi_symbol_links.tsv'
}

# ===============================================================================
# FEATURE CATEGORIES
# ===============================================================================

# DepMap features (21 features)
DEPMAP_FEATURES = [
    'crispr_dependency_mean', 'crispr_dependency_std', 'crispr_dependency_min',
    'crispr_dependency_max', 'crispr_dependency_median', 'crispr_dependency_strong_count',
    'cnv_wgs_mean', 'cnv_wgs_std', 'cnv_wgs_min', 'cnv_wgs_max', 'cnv_wgs_median',
    'expression_mean', 'expression_std', 'expression_min', 'expression_max',
    'expression_median', 'expression_high_prop',
    'damaging_mutation_sum', 'damaging_high_af_count', 'hotspot_mutation_sum',
    'hotspot_high_af_count'
]

# Non-DepMap features by category (482 features total)
NON_DEPMAP_FEATURES = {
    'ExAC': [
        'GeneSize', 'ExAC_mean_rd', 'ExAC_gc_content', 'ExAC_complexity', 'ExAC_cds_len',
        'ExAC_gene_length', 'ExAC_num_targ', 'ExAC_segdups', 'ExAC_dip', 'ExAC_del',
        'ExAC_dup', 'ExAC_del.sing', 'ExAC_dup.sing', 'ExAC_del.sing.score',
        'ExAC_dup.sing.score', 'ExAC_del.score', 'ExAC_dup.score', 'ExAC_cnv.score',
        'ExAC_flag'
    ],
    'Mouse_Genes': [
        'MouseGenes_Essential', 'MouseGenes_Non-essential',
        'MouseGenes_Overexpressed_in_the_brain', 'MouseGenes_GC%'
    ],
    'Genic_Intolerance': [
        'GenicIntolerance_RVIS', 'GenicIntolerance_LoF_FDR_ExAC', 'GenicIntolerance_RVIS_ExAC',
        'GenicIntolerance_geneCov_ExACv2', 'GenicIntolerance_RVIS_ExACv2', 'GenicIntolerance_MTR_ExACv2'
    ],
    'GnomAD': [
        'GnomAD_obs_lof', 'GnomAD_exp_lof', 'GnomAD_oe_lof', 'GnomAD_oe_lof_lower',
        'GnomAD_oe_lof_upper', 'GnomAD_obs_mis', 'GnomAD_exp_mis', 'GnomAD_oe_mis',
        'GnomAD_oe_mis_lower', 'GnomAD_oe_mis_upper', 'GnomAD_lof_z', 'GnomAD_mis_z',
        'GnomAD_pLI', 'GnomAD_pRec', 'GnomAD_pNull'
    ],
    'GWAS': [
        'GWAS_hits', 'GWAS_max_P_VALUE', 'GWAS_min_P_VALUE', 'GWAS_max_OR',
        'GWAS_min_OR', 'GWAS_tissue_trait_flag'
    ],
    'MGI': ['MGI_essential_gene'],
    'Pharos': [
        'monoclonalCount', 'antibodyCount', 'ppiCount', 'hpm_prot_spec',
        'hpm_gene_spec', 'hpa_RNA_spec', 'hpa_prot_spec', 'uniprot_seq_len'
    ],
    'DGIdb': ['DGIdb_interaction_types'],
    'Reactome': ['Re_L1_seed_genes_overlap', 'Re_L2_seed_genes_overlap'],
    'InWeb': [
        'in_web_L1_experimental_seed_genes_overlap', 'in_web_L2_experimental_seed_genes_overlap',
        'in_web_L1_inferred_seed_genes_overlap', 'in_web_L2_inferred_seed_genes_overlap'
    ],
    'CTDbase': [
        # CTD affects features (20)
        'ctd_affects^abundance', 'ctd_affects^acetylation', 'ctd_affects^activity', 'ctd_affects^binding',
        'ctd_affects^chemical synthesis', 'ctd_affects^cleavage', 'ctd_affects^cotreatment',
        'ctd_affects^expression', 'ctd_affects^folding', 'ctd_affects^localization',
        'ctd_affects^metabolic processing', 'ctd_affects^methylation', 'ctd_affects^phosphorylation',
        'ctd_affects^reaction', 'ctd_affects^response to substance', 'ctd_affects^secretion',
        'ctd_affects^splicing', 'ctd_affects^sumoylation', 'ctd_affects^transport', 'ctd_affects^uptake',
        
        # CTD decreases features (18)
        'ctd_decreases^ADP-ribosylation', 'ctd_decreases^abundance', 'ctd_decreases^acetylation',
        'ctd_decreases^activity', 'ctd_decreases^chemical synthesis', 'ctd_decreases^cleavage',
        'ctd_decreases^degradation', 'ctd_decreases^expression', 'ctd_decreases^localization',
        'ctd_decreases^metabolic processing', 'ctd_decreases^methylation', 'ctd_decreases^phosphorylation',
        'ctd_decreases^reaction', 'ctd_decreases^response to substance', 'ctd_decreases^secretion',
        'ctd_decreases^stability', 'ctd_decreases^sumoylation', 'ctd_decreases^uptake',
        
        # CTD increases features (27)
        'ctd_increases^ADP-ribosylation', 'ctd_increases^O-linked glycosylation', 'ctd_increases^abundance',
        'ctd_increases^acetylation', 'ctd_increases^activity', 'ctd_increases^chemical synthesis',
        'ctd_increases^cleavage', 'ctd_increases^degradation', 'ctd_increases^export',
        'ctd_increases^expression', 'ctd_increases^hydrolysis', 'ctd_increases^hydroxylation',
        'ctd_increases^import', 'ctd_increases^localization', 'ctd_increases^metabolic processing',
        'ctd_increases^methylation', 'ctd_increases^mutagenesis', 'ctd_increases^oxidation',
        'ctd_increases^phosphorylation', 'ctd_increases^reaction', 'ctd_increases^reduction',
        'ctd_increases^response to substance', 'ctd_increases^secretion', 'ctd_increases^stability',
        'ctd_increases^sumoylation', 'ctd_increases^transport', 'ctd_increases^ubiquitination',
        'ctd_increases^uptake',
        
        # CTD interaction counts (2)
        'ctd_uniqueInteractions', 'ctd_otherInteractionsCount',
        
        # CTD pathway features (240)
        'ctd_ARMS-mediated activation', 'ctd_Activation of HOX genes during differentiation',
        'ctd_Activation of anterior HOX genes in hindbrain development during early embryogenesis',
        'ctd_Adaptive Immune System', 'ctd_Adrenergic signaling in cardiomyocytes',
        'ctd_Alcoholism', 'ctd_Alzheimer\'s disease',
        'ctd_Antigen processing: Ubiquitination & Proteasome degradation',
        'ctd_Apelin signaling pathway', 'ctd_Apoptosis', 'ctd_Asparagine N-linked glycosylation',
        'ctd_Autophagy - animal', 'ctd_Axon guidance', 'ctd_Beta-catenin independent WNT signaling',
        'ctd_Biological oxidations', 'ctd_Breast cancer', 'ctd_C-type lectin receptors (CLRs)',
        'ctd_Calcium signaling pathway', 'ctd_Cap-dependent Translation Initiation',
        'ctd_Cardiac conduction', 'ctd_Cell Cycle', 'ctd_Cell Cycle Checkpoints',
        'ctd_Cell Cycle, Mitotic', 'ctd_Cell adhesion molecules (CAMs)', 'ctd_Cell cycle',
        'ctd_Cell surface interactions at the vascular wall', 'ctd_Cell-Cell communication',
        'ctd_Cellular Senescence', 'ctd_Cellular responses to stress', 'ctd_Chemokine signaling pathway',
        'ctd_Chromatin modifying enzymes', 'ctd_Chromatin organization', 'ctd_Cilium Assembly',
        'ctd_Class A/1 (Rhodopsin-like receptors)', 'ctd_Class I MHC mediated antigen processing & presentation',
        'ctd_Clathrin-mediated endocytosis', 'ctd_Cytokine Signaling in Immune system',
        'ctd_Cytokine-cytokine receptor interaction', 'ctd_DAP12 interactions', 'ctd_DAP12 signaling',
        'ctd_DNA Double-Strand Break Repair', 'ctd_DNA Repair', 'ctd_Deubiquitination',
        'ctd_Developmental Biology', 'ctd_Disease', 'ctd_Diseases of signal transduction',
        'ctd_Dopaminergic synapse', 'ctd_Downstream signal transduction',
        'ctd_Downstream signaling events of B Cell Receptor (BCR)', 'ctd_ER to Golgi Anterograde Transport',
        'ctd_Endocytosis', 'ctd_Epigenetic regulation of gene expression', 'ctd_Epstein-Barr virus infection',
        'ctd_Eukaryotic Translation Initiation', 'ctd_Extracellular matrix organization',
        'ctd_FCERI mediated MAPK activation', 'ctd_Factors involved in megakaryocyte development and platelet production',
        'ctd_Fatty acid, triacylglycerol, and ketone body metabolism', 'ctd_Fc epsilon receptor (FCERI) signaling',
        'ctd_Fluid shear stress and atherosclerosis', 'ctd_Focal adhesion', 'ctd_FoxO signaling pathway',
        'ctd_Frs2-mediated activation', 'ctd_G alpha (i) signalling events', 'ctd_G alpha (q) signalling events',
        'ctd_G alpha (s) signalling events', 'ctd_G2/M Checkpoints', 'ctd_G2/M Transition',
        'ctd_GAB1 signalosome', 'ctd_GPCR downstream signaling', 'ctd_GPCR ligand binding',
        'ctd_GRB2 events in EGFR signaling', 'ctd_Gastrin-CREB signalling pathway via PKC and MAPK',
        'ctd_Gene Expression', 'ctd_Gene Silencing by RNA', 'ctd_Generic Transcription Pathway',
        'ctd_Glycerophospholipid biosynthesis', 'ctd_Glycosaminoglycan metabolism', 'ctd_HATs acetylate histones',
        'ctd_HDR through Homologous Recombination (HR) or Single Strand Annealing (SSA)',
        'ctd_HIV Infection', 'ctd_HIV Life Cycle', 'ctd_HTLV-I infection', 'ctd_Hemostasis',
        'ctd_Hepatitis B', 'ctd_Hepatitis C', 'ctd_Herpes simplex infection', 'ctd_Hippo signaling pathway',
        'ctd_Homology Directed Repair', 'ctd_Host Interactions of HIV factors', 'ctd_Huntington\'s disease',
        'ctd_IGF1R signaling cascade', 'ctd_IRS-mediated signalling', 'ctd_IRS-related events triggered by IGF1R',
        'ctd_Immune System', 'ctd_Immunoregulatory interactions between a Lymphoid and a non-Lymphoid cell',
        'ctd_Infectious disease', 'ctd_Influenza A', 'ctd_Influenza Infection', 'ctd_Influenza Life Cycle',
        'ctd_Influenza Viral RNA Transcription and Replication', 'ctd_Innate Immune System',
        'ctd_Insulin receptor signalling cascade', 'ctd_Insulin signaling pathway', 'ctd_Interferon Signaling',
        'ctd_Interleukin receptor SHC signaling', 'ctd_Interleukin-2 signaling', 'ctd_Interleukin-3, 5 and GM-CSF signaling',
        'ctd_Intra-Golgi and retrograde Golgi-to-ER traffic', 'ctd_Ion channel transport', 'ctd_Jak-STAT signaling pathway',
        'ctd_Keratinization', 'ctd_Late Phase of HIV Life Cycle', 'ctd_M Phase', 'ctd_MAPK family signaling cascades',
        'ctd_MAPK signaling pathway', 'ctd_MAPK1/MAPK3 signaling', 'ctd_Major pathway of rRNA processing in the nucleolus and cytosol',
        'ctd_Measles', 'ctd_Membrane Trafficking', 'ctd_Metabolic pathways', 'ctd_Metabolism',
        'ctd_Metabolism of amino acids and derivatives', 'ctd_Metabolism of carbohydrates',
        'ctd_Metabolism of lipids and lipoproteins', 'ctd_Metabolism of proteins', 'ctd_Metabolism of vitamins and cofactors',
        'ctd_MicroRNAs in cancer', 'ctd_Mitotic Anaphase', 'ctd_Mitotic G1-G1/S phases', 'ctd_Mitotic G2-G2/M phases',
        'ctd_Mitotic Metaphase and Anaphase', 'ctd_Mitotic Prophase', 'ctd_Muscle contraction',
        'ctd_NCAM signaling for neurite out-growth', 'ctd_NGF signalling via TRKA from the plasma membrane',
        'ctd_NOD-like receptor signaling pathway', 'ctd_Natural killer cell mediated cytotoxicity',
        'ctd_Neuroactive ligand-receptor interaction', 'ctd_Neuronal System',
        'ctd_Neurotransmitter Receptor Binding And Downstream Transmission In The  Postsynaptic Cell',
        'ctd_Neutrophil degranulation', 'ctd_Non-alcoholic fatty liver disease (NAFLD)', 'ctd_Olfactory Signaling Pathway',
        'ctd_Olfactory transduction', 'ctd_Oocyte meiosis', 'ctd_Organelle biogenesis and maintenance',
        'ctd_Osteoclast differentiation', 'ctd_Oxidative Stress Induced Senescence', 'ctd_Oxidative phosphorylation',
        'ctd_Oxytocin signaling pathway', 'ctd_PI3K-Akt signaling pathway', 'ctd_PI3K/AKT activation',
        'ctd_PIP3 activates AKT signaling', 'ctd_PPARA activates gene expression', 'ctd_Parkinson\'s disease',
        'ctd_Pathways in cancer', 'ctd_Peptide ligand-binding receptors', 'ctd_Phagosome',
        'ctd_Phospholipase D signaling pathway', 'ctd_Phospholipid metabolism', 'ctd_Platelet activation, signaling and aggregation',
        'ctd_Platelet degranulation', 'ctd_Post-translational protein modification', 'ctd_Processing of Capped Intron-Containing Pre-mRNA',
        'ctd_Programmed Cell Death', 'ctd_Prolonged ERK activation events', 'ctd_Protein processing in endoplasmic reticulum',
        'ctd_Proteoglycans in cancer', 'ctd_Purine metabolism', 'ctd_RAF/MAP kinase cascade', 'ctd_RET signaling',
        'ctd_RHO GTPase Effectors', 'ctd_RNA Polymerase I, RNA Polymerase III, and Mitochondrial Transcription',
        'ctd_RNA Polymerase II Transcription', 'ctd_RNA transport', 'ctd_Rap1 signaling pathway',
        'ctd_Ras signaling pathway', 'ctd_Regulation of TP53 Activity', 'ctd_Regulation of actin cytoskeleton',
        'ctd_Regulation of lipid metabolism by Peroxisome proliferator-activated receptor alpha (PPARalpha)',
        'ctd_Respiratory electron transport, ATP synthesis by chemiosmotic coupling, and heat production by uncoupling proteins.',
        'ctd_Response to elevated platelet cytosolic Ca2+', 'ctd_Rho GTPase cycle', 'ctd_Ribosome',
        'ctd_Role of LAT2/NTAL/LAB on calcium mobilization', 'ctd_S Phase', 'ctd_SHC1 events in EGFR signaling',
        'ctd_SLC-mediated transmembrane transport', 'ctd_SOS-mediated signalling', 'ctd_Separation of Sister Chromatids',
        'ctd_Signal Transduction', 'ctd_Signaling by EGFR', 'ctd_Signaling by GPCR', 'ctd_Signaling by Hedgehog',
        'ctd_Signaling by Insulin receptor', 'ctd_Signaling by Interleukins', 'ctd_Signaling by Leptin',
        'ctd_Signaling by PDGF', 'ctd_Signaling by Rho GTPases', 'ctd_Signaling by SCF-KIT',
        'ctd_Signaling by Type 1 Insulin-like Growth Factor 1 Receptor (IGF1R)', 'ctd_Signaling by VEGF',
        'ctd_Signaling by Wnt', 'ctd_Signaling by the B Cell Receptor (BCR)', 'ctd_Signaling pathways regulating pluripotency of stem cells',
        'ctd_Signalling by NGF', 'ctd_Signalling to ERKs', 'ctd_Signalling to RAS', 'ctd_Signalling to p38 via RIT and RIN',
        'ctd_Spliceosome', 'ctd_Systemic lupus erythematosus', 'ctd_TCF dependent signaling in response to WNT',
        'ctd_The citric acid (TCA) cycle and respiratory electron transport', 'ctd_Tight junction',
        'ctd_Toll Like Receptor 4 (TLR4) Cascade', 'ctd_Toll-Like Receptors Cascades', 'ctd_Transcriptional Regulation by TP53',
        'ctd_Transcriptional misregulation in cancer', 'ctd_Translation', 'ctd_Transmembrane transport of small molecules',
        'ctd_Transmission across Chemical Synapses', 'ctd_Transport to the Golgi and subsequent modification',
        'ctd_Tuberculosis', 'ctd_Ub-specific processing proteases', 'ctd_Ubiquitin mediated proteolysis',
        'ctd_VEGFA-VEGFR2 Pathway', 'ctd_VEGFR2 mediated cell proliferation', 'ctd_Vesicle-mediated transport',
        'ctd_Viral carcinogenesis', 'ctd_Wnt signaling pathway', 'ctd_cAMP signaling pathway',
        'ctd_cGMP-PKG signaling pathway', 'ctd_mRNA Splicing', 'ctd_mRNA Splicing - Major Pathway',
        'ctd_mTOR signaling pathway', 'ctd_rRNA processing', 'ctd_rRNA processing in the nucleus and cytosol',
        'ctd_otherPathways'
    ],
    'STRING_db': [
        'string_db_L1_protein_seed_genes_overlap', 'string_db_L2_protein_seed_genes_overlap',
        'string_db_L1_physical_seed_genes_overlap', 'string_db_L2_physical_seed_genes_overlap',
        'string_db_L1_protein_seed_genes_overlap_weighted_score', 'string_db_L2_protein_sseed_genes_overlap_weighted_score',
        'string_db_L1_protein_sseed_genes_overlap_hmean_score', 'string_db_L1_physical_seed_genes_overlap_weighted_score',
        'string_db_L2_physical_sseed_genes_overlap_weighted_score', 'string_db_L1_physical_sseed_genes_overlap_hmean_score'
    ],
    'InterPro': [
        # InterPro domains (31)
        'IPR_d_AAA+ ATPase domain', 'IPR_d_Ankyrin repeat-containing domain', 'IPR_d_BTB/POZ domain',
        'IPR_d_C2 domain', 'IPR_d_Cadherin-like', 'IPR_d_EF-hand domain', 'IPR_d_EGF-like calcium-binding domain',
        'IPR_d_EGF-like domain', 'IPR_d_Fibronectin type III', 'IPR_d_GPCR, rhodopsin-like, 7TM',
        'IPR_d_Homeobox domain', 'IPR_d_Immunoglobulin I-set', 'IPR_d_Immunoglobulin V-set domain',
        'IPR_d_Immunoglobulin subtype', 'IPR_d_Immunoglobulin subtype 2', 'IPR_d_Immunoglobulin-like domain',
        'IPR_d_Krueppel-associated box', 'IPR_d_Myc-type, basic helix-loop-helix (bHLH) domain',
        'IPR_d_PDZ domain', 'IPR_d_Pleckstrin homology domain', 'IPR_d_Protein kinase domain',
        'IPR_d_RNA recognition motif domain', 'IPR_d_SH2 domain', 'IPR_d_SH3 domain',
        'IPR_d_Serine proteases, trypsin domain', 'IPR_d_Serine-threonine/tyrosine-protein kinase, catalytic domain',
        'IPR_d_Small GTP-binding protein domain', 'IPR_d_Tetratricopeptide repeat-containing domain',
        'IPR_d_WD40-repeat-containing domain', 'IPR_d_Zinc finger C2H2-type', 'IPR_d_Zinc finger, RING-type',
        
        # InterPro families (20)
        'IPR_f_Actin family', 'IPR_f_BTB-kelch protein', 'IPR_f_Cytochrome P450', 'IPR_f_Cytochrome P450, E-class, group I',
        'IPR_f_G protein-coupled receptor, rhodopsin-like', 'IPR_f_GPCR, family 2, secretin-like',
        'IPR_f_Keratin-associated protein', 'IPR_f_Major facilitator superfamily', 'IPR_f_Major facilitator,  sugar transporter-like',
        'IPR_f_Neurotransmitter-gated ion-channel', 'IPR_f_Nuclear hormone receptor', 'IPR_f_Olfactory receptor',
        'IPR_f_P-type ATPase', 'IPR_f_PMP-22/EMP/MP20/Claudin superfamily', 'IPR_f_Peptidase S1A, chymotrypsin family',
        'IPR_f_Serpin family', 'IPR_f_Short-chain dehydrogenase/reductase SDR', 'IPR_f_Small GTPase',
        'IPR_f_Small GTPase superfamily, Ras-type', 'IPR_f_Transforming growth factor-beta-related',
        
        # InterPro superfamilies (46)
        'IPR_sf_Alpha/Beta hydrolase fold', 'IPR_sf_Ankyrin repeat-containing domain superfamily',
        'IPR_sf_Armadillo-like helical', 'IPR_sf_Armadillo-type fold', 'IPR_sf_C-type lectin fold',
        'IPR_sf_C-type lectin-like/link domain superfamily', 'IPR_sf_C2 domain superfamily', 'IPR_sf_Cadherin-like superfamily',
        'IPR_sf_Concanavalin A-like lectin/glucanase domain superfamily', 'IPR_sf_EF-hand domain pair',
        'IPR_sf_Fibronectin type III superfamily', 'IPR_sf_Growth factor receptor cysteine-rich domain superfamily',
        'IPR_sf_Helix-loop-helix DNA-binding domain superfamily', 'IPR_sf_Homeobox-like domain superfamily',
        'IPR_sf_Immunoglobulin E-set', 'IPR_sf_Immunoglobulin-like domain superfamily', 'IPR_sf_Immunoglobulin-like fold',
        'IPR_sf_KRAB domain superfamily', 'IPR_sf_Leucine-rich repeat domain superfamily', 'IPR_sf_MFS transporter superfamily',
        'IPR_sf_NAD(P)-binding domain superfamily', 'IPR_sf_Nucleotide-binding alpha-beta plait domain superfamily',
        'IPR_sf_P-loop containing nucleoside triphosphate hydrolase', 'IPR_sf_PDZ superfamily',
        'IPR_sf_PH-like domain superfamily', 'IPR_sf_Papain-like cysteine peptidase superfamily',
        'IPR_sf_Peptidase S1, PA clan', 'IPR_sf_Peptidase S1, PA clan, chymotrypsin-like fold',
        'IPR_sf_Protein kinase-like domain superfamily', 'IPR_sf_RNA-binding domain superfamily',
        'IPR_sf_S-adenosyl-L-methionine-dependent methyltransferase', 'IPR_sf_SH2 domain superfamily',
        'IPR_sf_SH3-like domain superfamily', 'IPR_sf_SKP1/BTB/POZ domain superfamily',
        'IPR_sf_Sterile alpha motif/pointed domain superfamily', 'IPR_sf_Tetratricopeptide-like helical domain superfamily',
        'IPR_sf_Thioredoxin-like superfamily', 'IPR_sf_Ubiquitin-like domain superfamily',
        'IPR_sf_WD40-repeat-containing domain superfamily', 'IPR_sf_WD40/YVTN repeat-like-containing domain superfamily',
        'IPR_sf_Winged helix DNA-binding domain superfamily', 'IPR_sf_Winged helix-like DNA-binding domain superfamily',
        'IPR_sf_Zinc finger C2H2 superfamily', 'IPR_sf_Zinc finger, FYVE/PHD-type',
        'IPR_sf_Zinc finger, RING/FYVE/PHD-type', 'IPR_sf_von Willebrand factor A-like domain superfamily'
    ],
    'OMIM': ['OMIM_uniq_diseases'],
    'GTEx': ['GTEx_spec']
}

# Get all non-DepMap features as flat list
ALL_NON_DEPMAP_FEATURES = []
for features in NON_DEPMAP_FEATURES.values():
    ALL_NON_DEPMAP_FEATURES.extend(features)

# Feature selection options
FEATURE_CONFIGS = {
    'depmap_only': DEPMAP_FEATURES,
    'non_depmap_only': ALL_NON_DEPMAP_FEATURES,
    'all_features': DEPMAP_FEATURES + ALL_NON_DEPMAP_FEATURES,
    'depmap_plus_pharos': DEPMAP_FEATURES + NON_DEPMAP_FEATURES['Pharos'],
    'depmap_plus_gnomad': DEPMAP_FEATURES + NON_DEPMAP_FEATURES['GnomAD']
}

# ===============================================================================
# DRUGGABILITY TASKS
# ===============================================================================

DRUGGABILITY_TASKS = {

    # ============================================================
    # PHAROS-based tasks (Disease-agnostic)
    # ============================================================
    'pharos_tclin_vs_others': {
        'label_col': 'task_pharos_tclin_vs_others',
        'display_name': 'Clinical Targets (Tclin)',
        'description': 'Identify clinically validated FDA-approved drug targets (Tclin) versus all other genes',
        'task_type': 'binary'
    },

    'pharos_tclin_tchem_vs_others': {
        'label_col': 'task_pharos_tclin_tchem_vs_others',
        'display_name': 'Clinical & Chemical Targets (Tclin+Tchem)',
        'description': 'Identify targets with clinical or strong chemical evidence (Tclin or Tchem) versus others',
        'task_type': 'binary'
    },

    # ============================================================
    # Triage assessment tasks (Disease-agnostic)
    # ============================================================
    'triage_tier1_vs_others': {
        'label_col': 'task_triage_tier1_vs_others',
        'display_name': 'Top-Tier Targets (Tier 1)',
        'description': 'Identify highest-confidence druggable targets assessed as Tier 1',
        'task_type': 'binary'
    },

    'triage_tier12_vs_others': {
        'label_col': 'task_triage_tier12_vs_others',
        'display_name': 'High-Confidence Targets (Tier 1–2)',
        'description': 'Identify high-confidence druggable targets assessed as Tier 1 or Tier 2',
        'task_type': 'binary'
    },

    # ============================================================
    # Cancer druggability tasks (Domain-specific)
    # ============================================================
    'cancer_relevant_targets': {
        'label_col': 'task_cancer_druggability',
        'display_name': 'Cancer-Relevant Targets',
        'description': 'Predict druggable targets curated as relevant to cancer biology',
        'task_type': 'binary'
    },

    'cancer_type_specific_targets': {
        'label_col': 'task_cancer_type_specific_target_prioritization',
        'display_name': 'Cancer-Type-Specific Targets',
        'description': 'Predict druggable targets specific to individual cancer types',
        'task_type': 'binary'
    },

    'pan_cancer_targets': {
        'label_col': 'task_pan_cancer_target_prioritization',
        'display_name': 'Pan-Cancer Targets',
        'description': 'Predict druggable targets recurrently implicated across multiple cancer types',
        'task_type': 'binary'
    },

    'pan_cancer_T1_targets': {
        'label_col': 'task_T1_targets_only',
        'display_name': 'Tier 1 Cancer Targets',
        'description': 'Identify cancer drug targets with approved therapeutic evidence (Tier 1)',
        'task_type': 'binary'
    },

    'pan_cancer_T12_targets': {
        'label_col': 'task_T1_T2_targets',
        'display_name': 'Tier 1–2 Cancer Targets',
        'description': 'Identify approved or repurposed cancer drug targets (Tier 1–2)',
        'task_type': 'binary'
    },

    'pan_cancer_T123_targets': {
        'label_col': 'task_T1_T2_T3_targets',
        'display_name': 'Tier 1–3 Cancer Targets',
        'description': 'Identify approved or investigational cancer drug targets (Tier 1–3)',
        'task_type': 'binary'
    },

    # ============================================================
    # Drug modality-specific tasks (Domain-specific)
    # ============================================================
    'sm_bucket1_vs_others': {
        'label_col': 'task_sm_bucket1_vs_others',
        'display_name': 'Small Molecule Targets (Bucket 1)',
        'description': 'Identify targets of approved small-molecule drugs (SM Bucket 1)',
        'task_type': 'binary'
    },

    'sm_bucket123_vs_others': {
        'label_col': 'task_sm_bucket123_vs_others',
        'display_name': 'Small Molecule Targets (Bucket 1–3)',
        'description': 'Identify targets of approved or clinical-stage small-molecule drugs (SM Bucket 1–3)',
        'task_type': 'binary'
    },

    'ab_bucket1_vs_others': {
        'label_col': 'task_ab_bucket1_vs_others',
        'display_name': 'Antibody Targets (Bucket 1)',
        'description': 'Identify targets of approved antibody therapeutics (AB Bucket 1)',
        'task_type': 'binary'
    },

    'ab_bucket123_vs_others': {
        'label_col': 'task_ab_bucket123_vs_others',
        'display_name': 'Antibody Targets (Bucket 1–3)',
        'description': 'Identify targets of approved or clinical-stage antibody therapeutics (AB Bucket 1–3)',
        'task_type': 'binary'
    },

    'protac_bucket1234_vs_others': {
        'label_col': 'task_protac_bucket1234_vs_others',
        'display_name': 'PROTAC Targets (Bucket 1–4)',
        'description': 'Identify targets supported by literature-curated PROTAC evidence (Bucket 1–4)',
        'task_type': 'binary'
    }
}
# Task order for display
TASK_ORDER = [
    'tclin_vs_others',
    'tclin_tchem_vs_others',
    'tier1_vs_others',
    'tier12_vs_others',
    'cancer_druggability',
    'ab_bucket1_vs_others',
    'ab_bucket123_vs_others',
    'sm_bucket1_vs_others',
    'sm_bucket123_vs_others',
    'protac_bucket1234_vs_others'
]

# ===============================================================================
# MODEL HYPERPARAMETERS
# ===============================================================================

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model architecture
HIDDEN_CHANNELS = 512
NUM_LAYERS = 3
NUM_HEADS = 8
DROPOUT = 0.2

# Training
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 256

# Data splits
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Graph construction
MAX_NEIGHBORS_PER_NODE = 10
NEIGHBOR_SAMPLING = [25, 20, 15]  # For each layer

# Edge type configurations
EDGE_CONFIGS = {
    'all': ['regnet', 'trrust', 'coexp', 'ppi'],
    'regulatory': ['regnet', 'trrust'],
    'functional': ['coexp', 'ppi'],
    'ppi_only': ['ppi'],
    'coexp_only': ['coexp'],
    'reg_ppi': ['regnet', 'ppi'],
    'reg_coexp': ['regnet', 'coexp']
}

# Default configuration
DEFAULT_EDGE_CONFIG = 'all'
DEFAULT_FEATURE_CONFIG = 'all_features'
DEFAULT_TASK = 'tier12_vs_others'
DEFAULT_SEED = 42