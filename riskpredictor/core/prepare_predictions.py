"""

Risk prediction pre-calculation pipeline:
     0. Prepare a plink-formatted validation and LD reference genotype file.

     1. Parse relevant summary statistics into a HDF5 file(s)
         - Use parse_BMI_HEIGHT or similar function.
         - It identifies the set of SNPs used by
             a) the summary statistics
             b) 1K genomes
             c) the LD-reference (and validation) genotypes
             d) A HonestGene individual genotype (e.g. 23andme v4)

     2. Coordinate the summary statistics and LD-reference (and validation) genotypes, to create a LDpred prediction data file
         - Use coordinate_LDpred_data or similar
     3. Train prediction SNP weights. (Use ldpred: https://bitbucket.org/bjarni_vilhjalmsson/ldpred)
         - Run LDpred using the coordinated file.
     4. Validate prediction using validation data
         - Run LDpred to get validation.
     5. Coordinate SNP weights directions with a HonestGene individual genotype (e.g. 23andme v4) and estimate cov. weights
         - missing weights for HG IG SNPs will be set to 0.
         - Use the gen_weight_files function.

When the resulting SNP weight files for each trait are available, we can use them to obtain the polygenic scores for each
trait relatively easily using

"""



import logging
from plinkio import plinkfile
import itertools as it
import h5py
import scipy as sp
from scipy import stats
import pandas
import random
import gzip
log = logging.getLogger(__name__)

ambig_nts = set([('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')])
opp_strand_dict = {'A':'T', 'G':'C', 'T':'A', 'C':'G'}
valid_nts = set(['A','T','C','G'])



def parse_BMI_HEIGHT():

    bmi_file = '/home/bjarni/TheHonestGene/faststorage/prediction_data/SNP_gwas_mc_merge_nogc.tbl.uniq'
    height_file = '/home/bjarni/TheHonestGene/faststorage/prediction_data/GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt'
    KGpath = '/home/bjarni/TheHonestGene/faststorage/1Kgenomes/'
    comb_hdf5_file = '/home/bjarni/TheHonestGene/faststorage/prediction_data/HEIGHT_BMI.hdf5'
    bimfile = '/home/bjarni/TheHonestGene/faststorage/prediction_data/wayf_407.bim'  #This is the LD reference and validation data SNP file.
    hg_indiv_genot_file = '/home/bjarni/TheHonestGene/faststorage/prediction_data/imputed.hdf5'  #the individual genotype example
    ig_h5f = h5py.File(hg_indiv_genot_file)
    ok_sids = []
    for chrom in ig_h5f.keys():
        ok_sids.extend(ig_h5f[chrom]['sids'][...])
    parse_sum_stats(height_file,comb_hdf5_file,'height',KGpath, bimfile=bimfile, ok_sids=ok_sids)
    parse_sum_stats(bmi_file,comb_hdf5_file,'BMI',KGpath,bimfile=bimfile, ok_sids=ok_sids)


def coordinate_BMI_HEIGHT():
    ss_hdf5_file = '/home/bjarni/TheHonestGene/faststorage/prediction_data/HEIGHT_BMI.hdf5'
    coord_height_file = '/home/bjarni/TheHonestGene/faststorage/prediction_data/LDpred_coord_WAYF407_HEIGHT.hdf5'
    coord_bmi_file = '/home/bjarni/TheHonestGene/faststorage/prediction_data/LDpred_coord_WAYF407_BMI.hdf5'
    ld_ref_prefix = '/home/bjarni/TheHonestGene/faststorage/prediction_data/wayf_407'  #This is the LD reference and validation data SNP file.

    coordinate_LDpred_data(genotype_file=ld_ref_prefix,
                           coord_hdf5_file=coord_height_file,
                           ss_file=ss_hdf5_file,
                           ss_id = 'height',
                           min_maf =0.01)

    coordinate_LDpred_data(genotype_file=ld_ref_prefix,
                           coord_hdf5_file=coord_bmi_file,
                           ss_file=ss_hdf5_file,
                           ss_id = 'BMI',
                           min_maf =0.01)


def _parse_plink_snps_(genotype_file, snp_indices):
    plinkf = plinkfile.PlinkFile(genotype_file)
    samples = plinkf.get_samples()
    num_individs = len(samples)
    num_snps = len(snp_indices)
    raw_snps = sp.empty((num_snps,num_individs),dtype='int8')
    #If these indices are not in order then we place them in the right place while parsing SNPs.
    snp_order = sp.argsort(snp_indices)
    ordered_snp_indices = list(snp_indices[snp_order])
    ordered_snp_indices.reverse()
    print('Iterating over file to load SNPs')
    snp_i = 0
    next_i = ordered_snp_indices.pop()
    line_i = 0
    max_i = ordered_snp_indices[0]
    while line_i <= max_i:
        if line_i < next_i:
            plinkf.next()
        elif line_i==next_i:
            line = plinkf.next()
            snp = sp.array(line, dtype='int8')
            bin_counts = line.allele_counts()
            if bin_counts[-1]>0:
                mode_v = sp.argmax(bin_counts[:2])
                snp[snp==3] = mode_v
            s_i = snp_order[snp_i]
            raw_snps[s_i]=snp
            if line_i < max_i:
                next_i = ordered_snp_indices.pop()
            snp_i+=1
        line_i +=1
    plinkf.close()
    assert snp_i==len(raw_snps), 'Failed to parse SNPs?'
    num_indivs = len(raw_snps[0])
    freqs = sp.sum(raw_snps,1, dtype='float32')/(2*float(num_indivs))
    return raw_snps, freqs


#-----------------------------Code for parsing summary statistics (of various formats) ----------------------------------------------
lc_2_cap_map = {'a':'A', 'c':'C', 'g':'G', 't':'T'}


def get_sid_pos_map(sids, KGenomes_prefix):
    h5fn = '%ssnps.hdf5'%(KGenomes_prefix)
    h5f = h5py.File(h5fn,'r')
    sid_map = {}
    for chrom_i in range(1,23):
        cg = h5f['chrom_%d' % chrom_i]
        sids_1k = cg['sids'][...]
        sids_filter_1k = sp.in1d(sids_1k, sp.array(sids))
        common_sids = sids_1k[sids_filter_1k]
        common_positions = cg['positions'][sids_filter_1k]
        eur_mafs = cg['eur_mafs'][sids_filter_1k]
        for sid,pos,eur_maf in it.izip(common_sids,common_positions,eur_mafs):
            sid_map[sid]={'pos':pos, 'chrom':chrom_i, 'eur_maf':eur_maf}
    return sid_map


def parse_sum_stats(filename,
                    comb_hdf5_file,
                    ss_id,
                    KGpath,
                    bimfile=None,
                    ok_sids=None):
    """
    """
    headers = {'SSGAC1':['MarkerName', 'Effect_Allele', 'Other_Allele', 'EAF', 'Beta', 'SE', 'Pvalue'],
           'SSGAC2':['MarkerName', 'Effect_Allele', 'Other_Allele', 'EAF', 'OR', 'SE', 'Pvalue'],
           'CHIC':['SNP', 'CHR', 'BP', 'A1', 'A2', 'FREQ_A1', 'EFFECT_A1', 'SE', 'P'],
           'GCAN':['chromosome', 'position', 'SNP', 'reference_allele', 'other_allele', 'eaf', 'OR',
                             'OR_se', 'OR_95L', 'OR_95U', 'z', 'p_sanger', '_-log10_p-value', 'q_statistic',
                             'q_p-value', 'i2', 'n_studies', 'n_samples', 'effects'],
           'TESLOVICH':['MarkerName', 'Allele1', 'Allele2', 'Weight', 'GC.Zscore', 'GC.Pvalue', 'Overall', 'Direction'],
           'GIANT1':['MarkerName', 'Allele1', 'Allele2', 'FreqAllele1HapMapCEU', 'b', 'se', 'p', 'N'],
           'GIANT1b':['MarkerName', 'Allele1', 'Allele2', 'Freq.Allele1.HapMapCEU', 'b', 'SE', 'p', 'N'],
           'GIANT1c':['MarkerName', 'Chr', 'Pos', 'Allele1', 'Allele2', 'FreqAllele1HapMapCEU', 'b', 'se', 'p', 'N'],
           'GIANT2':['SNP', 'A1', 'A2', 'Freq1.Hapmap', 'b', 'se', 'p', 'N'],
           'MAGIC':['snp', 'effect_allele', 'other_allele', 'maf', 'effect', 'stderr', 'pvalue'],
           'CARDIoGRAM':['SNP', 'chr_pos_(b36)', 'reference_allele', 'other_allele', 'ref_allele_frequency', 'pvalue', 'het_pvalue', 'log_odds', 'log_odds_se', 'N_case', 'N_control', 'model'],
           'DIAGRAM':['SNP', 'CHROMOSOME', 'POSITION', 'RISK_ALLELE', 'OTHER_ALLELE', 'P_VALUE', 'OR', 'OR_95L', 'OR_95U', 'N_CASES', 'N_CONTROLS'],
           'TAG':['CHR', 'SNP', 'BP', 'A1', 'A2', 'FRQ_A', 'FRQ_U', 'INFO', 'OR', 'SE', 'P'],
           'CD':['CHR', 'SNP', 'BP', 'A1', 'A2', 'FRQ_A_5956', 'FRQ_U_14927', 'INFO', 'OR', 'SE', 'P', 'Direction', 'HetISqt', 'HetPVa'],
           'UC':['CHR', 'SNP', 'BP', 'A1', 'A2', 'FRQ_A_6968', 'FRQ_U_20464', 'INFO', 'OR', 'SE', 'P', 'Direction', 'HetISqt', 'HetPVa'],
           'GEFOS':['chromosome', 'position', 'rs_number', 'reference_allele', 'other_allele', 'eaf', 'beta', 'se', 'beta_95L', 'beta_95U', 'z', 'p-value', '_-log10_p-value', 'q_statistic', 'q_p-value', 'i2', 'n_studies', 'n_samples', 'effects'],
           'RA':['SNPID','Chr','Position(hg19)','A1','A2','OR(A1)','OR_95%CIlow','OR_95%CIup','P-val'],
           'ASTHMA':['Chr', 'rs', 'position', 'Allele_1', 'Allele_2', 'freq_all_1_min', 'freq_all_1_max', 'OR_fix', 'ORl_fix', 'ORu_fix', 'P_fix'],
           'ICBP': ['ID', 'Analysis', 'ID', 'SNP', 'ID', 'P-value', 'Rank', 'Plot', 'data', 'Chr', 'ID', 'Chr', 'Position', 'Submitted', 'SNP', 'ID', 'ss2rs', 'rs2genome', 'Allele1', 'Allele2', 'Minor', 'allele', 'pHWE', 'Call', 'Rate', 'Effect', 'SE', 'R-Squared', 'Coded', 'Allele', 'Sample', 'size', 'Bin', 'ID']
           }

    h5f = h5py.File(comb_hdf5_file)
    if bimfile!=None:
        print('Parsing SNP list')
        valid_sids = []
        print('Parsing .bim file: %s'%bimfile)
        with open(bimfile) as f:
            for line in f:
                l = line.split()
                valid_sids.append(l[1])
        print('Found %d SNPs in .bim file'%len(valid_sids))
        valid_sids = sp.array(valid_sids)
    chrom_dict = {}

    if ok_sids !=None:
        valid_sids_filter = sp.in1d(valid_sids, sp.array(ok_sids))
        valid_sids = valid_sids[valid_sids_filter]
        print('Retained %d SNPs after removing SNPs not in HG indiv. genotype.'%len(valid_sids))


    print('Parsing SNP rsIDs from summary statistics.')
    sids = []
    with open(filename) as f:

        line = f.next()
        header = line.split()
        if header==['hg19chrc', 'snpid', 'a1', 'a2', 'bp', 'info', 'or', 'se', 'p', 'ngt'] or header==headers['TAG'] or header==headers['CD'] or header==headers['UC'] or header==headers['ASTHMA']:
            for line in f:
                l = line.split()
                sids.append(l[1])
        elif header==['Chromosome', 'Position', 'MarkerName', 'Effect_allele', 'Non_Effect_allele', 'Beta', 'SE', 'Pvalue'] or header==headers['GCAN'] or header==headers['GEFOS'] or header==headers['ICBP']:
#             line_count =0
            for line in f:
#                 line_count +=1
#                 if line_count<100000:
                l = line.split()
                sids.append(l[2])
#                 elif random.random()<0.1:
#                     l = line.split()
#                     sids.append(l[2])

        else:
            for line in f:
                l = line.split()
                sids.append(l[0])

    if bimfile!=None:
        valid_sids_filter = sp.in1d(valid_sids, sp.array(sids))
        sids = valid_sids[valid_sids_filter]

    print('Retrieving 1K genomes positions.')
    sid_map = get_sid_pos_map(sids,KGpath)
    assert len(sid_map)>0, 'WTF?'

    print('Parsing the file: %s' % filename)
    with open(filename) as f:
        line = f.next()
        header = line.split()
        line_i = 0
        if header== ['snpid', 'hg18chr', 'bp', 'a1', 'a2', 'or', 'se', 'pval', 'info', 'ngt', 'CEUaf']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[],'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[7])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = -sp.log(float(l[5]))
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)

        elif header==['hg19chrc', 'snpid', 'a1', 'a2', 'bp', 'info', 'or', 'se', 'p', 'ngt']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[1]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[],'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[8])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = -sp.log(float(l[6]))
                    if random.random()>0.5:
                        nt = [l[2], l[3]]
                    else:
                        nt = [l[3], l[2]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)

        elif header== ['snpid', 'hg18chr', 'bp', 'a1', 'a2', 'zscore', 'pval', 'CEUmaf']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[],'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[6])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = float(l[5])
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)

        elif header==['SNP', 'CHR', 'BP', 'A1', 'A2', 'OR', 'SE', 'P', 'INFO', 'EUR_FRQ']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[],'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[7])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = -sp.log(float(l[5]))
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)

        elif header==['Chromosome', 'Position', 'MarkerName', 'Effect_allele', 'Non_Effect_allele', 'Beta', 'SE', 'Pvalue']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[2]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[],'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[7])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = float(l[5])
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)

        elif header ==headers['SSGAC1']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[],'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[6])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = float(l[4])
                    if random.random()>0.5:
                        nt = [l[1], l[2]]
                    else:
                        nt = [l[2], l[1]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)
        elif header ==headers['SSGAC2']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[],'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[6])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = sp.log(float(l[4]))
                    if random.random()>0.5:
                        nt = [l[1], l[2]]
                    else:
                        nt = [l[2], l[1]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)
        elif header ==headers['CHIC']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[],'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[8])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = float(l[8])
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)

        elif header==headers['GCAN']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[2]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[11])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = -sp.log(float(l[6]))
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)

        elif header==headers['TESLOVICH']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[5])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = float(l[4])
                    if random.random()>0.5:
                        nt = [lc_2_cap_map[l[1]], lc_2_cap_map[l[2]]]
                    else:
                        nt = [lc_2_cap_map[l[2]], lc_2_cap_map[l[1]]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = int(float(l[3]))
                    chrom_dict[chrom]['weights'].append(weight)
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                if line_i%100000==0:
                    print(line_i)

        elif header==headers['GIANT1'] or header==headers['GIANT1b'] or header==headers['GIANT2']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[6])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = float(l[4])
                    if random.random()>0.5:
                        nt = [l[1], l[2]]
                    else:
                        nt = [l[2], l[1]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = int(float(l[7]))
                    chrom_dict[chrom]['weights'].append(weight)
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                if line_i%100000==0:
                    print(line_i)
        elif header==headers['GIANT1c']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[8])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = float(l[6])
                    chrom_dict[chrom]['raw_betas'].append(raw_beta)
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = int(float(l[9]))
                    chrom_dict[chrom]['weights'].append(weight)
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                if line_i%100000==0:
                    print(line_i)
        elif header==headers['MAGIC']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[6])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = float(l[4])
                    if random.random()>0.5:
                        nt = [lc_2_cap_map[l[1]], lc_2_cap_map[l[2]]]
                    else:
                        nt = [lc_2_cap_map[l[2]], lc_2_cap_map[l[1]]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)
        elif header==headers['CARDIoGRAM']:

            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[5])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = float(l[7])
                    if random.random()>0.5:
                        nt = [l[2], l[3]]
                    else:
                        nt = [l[3], l[2]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = float(l[9]) +float(l[10])
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)
        elif header==headers['DIAGRAM']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[5])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = sp.log(float(l[6]))
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = float(l[9]) +float(l[10])
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)
        elif header==headers['TAG']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[1]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[10])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = float(l[8])
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)
        elif header==headers['CD'] or header==headers['UC']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[1]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[10])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = sp.log(float(l[8]))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta)
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)
        elif header==headers['GEFOS']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[2]
                d = sid_map.get(sid,None)
                if d is not None:
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[11])
                    chrom_dict[chrom]['ps'].append(pval)
                    raw_beta = float(l[6])
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
#                     weight = (z/beta)**2
#                     weight = float(l[16])  # Number of studies used.
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)
#           'RA':['SNPID','Chr','Position(hg19)','A1','A2','OR(A1)','OR_95%CIlow','OR_95%CIup','P-val'],
#           'ASTHMA':['Chr', 'rs', 'position', 'Allele_1', 'Allele_2', 'freq_all_1_min', 'freq_all_1_max', 'OR_fix', 'ORl_fix', 'ORu_fix', 'P_fix'],
        elif header==headers['RA']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[0]
                d = sid_map.get(sid,None)
                if d is not None:
                    raw_beta = sp.log(float(l[5]))
                    if raw_beta==0:
                        continue
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    pval = float(l[8])
                    chrom_dict[chrom]['ps'].append(pval)
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)
        elif header==headers['ASTHMA']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[1]
                d = sid_map.get(sid,None)
                if d is not None:
                    raw_beta = sp.log(float(l[7]))
                    pval = float(l[10])
                    if raw_beta==0 or pval == 0:
                        continue
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    chrom_dict[chrom]['ps'].append(pval)
                    if random.random()>0.5:
                        nt = [l[3], l[4]]
                    else:
                        nt = [l[4], l[3]]
                        raw_beta = -raw_beta

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sp.sign(raw_beta) * stats.norm.ppf(pval/2.0)
                    weight = z**2/((raw_beta**2)*2*eur_maf*(1-eur_maf))
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(raw_beta/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)
        elif header==headers['ICBP']:
            for line in f:
                line_i +=1
                l = line.split()
                sid = l[2]
                d = sid_map.get(sid,None)
                coded_allele = l[16]
                if d is not None and coded_allele in valid_nts:
#                     raw_beta = sp.log(float(l[7]))
                    pval = float(l[3])
                    if pval == 0:
                        continue
                    pos = d['pos']
                    chrom = d['chrom']
                    eur_maf = d['eur_maf']
                    if not chrom in chrom_dict.keys():
                        chrom_dict[chrom] = {'ps':[], 'betas':[], 'nts': [], 'sids': [],
                                             'positions': [], 'eur_maf':[], 'weights':[], 'raw_betas':[]}
                    chrom_dict[chrom]['sids'].append(sid)
                    chrom_dict[chrom]['positions'].append(pos)
                    chrom_dict[chrom]['eur_maf'].append(eur_maf)
                    chrom_dict[chrom]['ps'].append(pval)
#                     if random.random()>0.5:
                    nt = [l[11], l[12]]
                    sign = 1
#                     else:
#                         sign = -1
                    if coded_allele==nt[1] or opp_strand_dict[coded_allele]==nt[1]:
                        nt = [l[12], l[11]]
                        sign = -1#*sign
#                     else:
#                         assert coded_allele==nt[0] or opp_strand_dict[coded_allele]==nt[0]

                    chrom_dict[chrom]['nts'].append(nt)
                    z = sign * stats.norm.ppf(pval/2.0)
                    weight = float(l[17])
                    chrom_dict[chrom]['betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['raw_betas'].append(z/sp.sqrt(weight))
                    chrom_dict[chrom]['weights'].append(weight)
                if line_i%100000==0:
                    print(line_i)

        else:
            raise Exception('Wrong or unknown file format')

        assert sp.all(sp.isreal(chrom_dict[1]['betas'])), 'WTF?'

    print('SS file loaded, now sorting and storing in HDF5 file.')
    assert not ss_id in h5f.keys(), 'Summary stats with this name are already in the HDF5 file?'
    ssg = h5f.create_group(ss_id)
    num_snps = 0
    for chrom in chrom_dict.keys():
        print('Parsed summary stats for %d SNPs on chromosome %d'%(len(chrom_dict[chrom]['positions']),chrom))
        sl = zip(chrom_dict[chrom]['positions'], chrom_dict[chrom]['sids'], chrom_dict[chrom]['nts'],
                 chrom_dict[chrom]['ps'], chrom_dict[chrom]['betas'], chrom_dict[chrom]['raw_betas'],
                 chrom_dict[chrom]['eur_maf'], chrom_dict[chrom]['weights'])
        sl.sort()
        ps = []
        nts = []
        sids = []
        positions = []
        betas = []
        raw_betas = []
        eur_mafs = []
        weights = []
        prev_pos = -1
        for pos, sid, nt, p, beta, raw_beta, eur_maf, weight in sl:
            if pos == prev_pos:
                print('duplicated position %d' % pos)
                continue
            else:
                prev_pos = pos
            ps.append(p)
            nts.append(nt)
            sids.append(sid)
            positions.append(pos)
            betas.append(beta)
            raw_betas.append(raw_beta)
            eur_mafs.append(eur_maf)
            weights.append(weight)
        g = ssg.create_group('chrom_%d' % chrom)
        g.create_dataset('ps', data=sp.array(ps))
        g.create_dataset('nts', data=nts)
        g.create_dataset('sids', data=sids)
        g.create_dataset('eur_mafs', data=eur_mafs)
        g.create_dataset('positions', data=positions)
        g.create_dataset('betas', data=betas)
        g.create_dataset('raw_betas', data=raw_betas)
        g.create_dataset('weights', data=weights)
        num_snps +=len(sids)
        h5f.flush()

    print('In all, %d SNPs parsed from summary statistics file.'%num_snps)



#----------------------------------------- Code for coordinating the datasets summary statistics (of various formats) ---------------------------------------------


def _get_chrom_dict_(loci, chromosomes):
    chr_dict = {}
    for chrom in chromosomes:
        chr_str = 'chrom_%d'%chrom
        chr_dict[chr_str] = {'sids':[],'snp_indices':[],'positions':[], 'nts':[]}

    for i, l in enumerate(loci):
        chrom = l.chromosome
        pos = l.bp_position
        chr_str = 'chrom_%d'%chrom
        chr_dict[chr_str]['sids'].append(l.name)
#         chr_dict[chr_str]['sids'].append('%d_%d'%(chrom,pos))
        chr_dict[chr_str]['snp_indices'].append(i)
        chr_dict[chr_str]['positions'].append(pos)
        chr_dict[chr_str]['nts'].append([l.allele1,l.allele2])

    print('Genotype dictionary filled')
    return chr_dict


def coordinate_LDpred_data(genotype_file=None,
                           coord_hdf5_file=None,
                           ss_file=None,
                           ss_id=None,
                           genetic_map_dir=None,
                           check_mafs=False,
                           min_maf=0.01):
    """
    Coordinates the genotypes across 2 data sets:
        a) the summary statistics
        b) the LD-reference panel

    Assumes plink BED files.  Imputes missing genotypes.
    """

    plinkf = plinkfile.PlinkFile(genotype_file)
    samples = plinkf.get_samples()
    num_individs = len(samples)
#        num_individs = len(gf['chrom_1']['snps'][:, 0])
#     Y = sp.array(gf['indivs']['phenotype'][...] == 'Case', dtype='int8')
    Y = [s.phenotype for s in samples]
    fids = [s.fid for s in samples]
    iids = [s.iid for s in samples]
    unique_phens = sp.unique(Y)
    if len(unique_phens)==1:
        print('Unable to find phenotype values.')
        has_phenotype=False
    elif len(unique_phens)==2:
        cc_bins = sp.bincount(Y)
        assert len(cc_bins)==2, 'Problems with loading phenotype'
        print('Loaded %d controls and %d cases'%(cc_bins[0], cc_bins[1]))
        has_phenotype=True
    else:
        print('Found quantitative phenotype values')
        has_phenotype=True
    risk_scores = sp.zeros(num_individs)
    rb_risk_scores = sp.zeros(num_individs)
    num_common_snps = 0
    corr_list = []
    rb_corr_list = []

    coord_h5f = h5py.File(coord_hdf5_file)
    if has_phenotype:
        coord_h5f.create_dataset('y', data=Y)

    coord_h5f.create_dataset('fids', data=fids)
    coord_h5f.create_dataset('iids', data=iids)
    ss_h5f = h5py.File(ss_file)
    ssf = ss_h5f[ss_id]
    cord_data_g = coord_h5f.create_group('cord_data')

    #Figure out chromosomes and positions by looking at SNPs.
    loci = plinkf.get_loci()
    plinkf.close()
    gf_chromosomes = [l.chromosome for l in loci]

    chromosomes = sp.unique(gf_chromosomes)
    chromosomes.sort()
    chr_dict = _get_chrom_dict_(loci, chromosomes)

    tot_num_non_matching_nts = 0
    for chrom in chromosomes:
        chr_str = 'chrom_%d'%chrom
        print('Working on chromsome: %s'%chr_str)

        chrom_d = chr_dict[chr_str]
        try:
            ssg = ssf['chrom_%d' % chrom]

        except Exception as err_str:
            print(err_str)
            print('Did not find chromsome in SS dataset.')
            print('Continuing.')
            continue

        g_sids = chrom_d['sids']
        g_sid_set = set(g_sids)
        assert len(g_sid_set) == len(g_sids), 'Some duplicates?'
        ss_sids = ssg['sids'][...]
        ss_sid_set = set(ss_sids)
        assert len(ss_sid_set) == len(ss_sids), 'Some duplicates?'

        #Figure out filters:
        g_filter = sp.in1d(g_sids,ss_sids)
        ss_filter = sp.in1d(ss_sids,g_sids)

        #Order by SNP IDs
        g_order = sp.argsort(g_sids)
        ss_order = sp.argsort(ss_sids)

        g_indices = []
        for g_i in g_order:
            if g_filter[g_i]:
                g_indices.append(g_i)

        ss_indices = []
        for ss_i in ss_order:
            if ss_filter[ss_i]:
                ss_indices.append(ss_i)

        g_nts = chrom_d['nts']
        snp_indices = chrom_d['snp_indices']
        ss_nts = ssg['nts'][...]
        betas = ssg['betas'][...]
        log_odds =  ssg['raw_betas'][...]
        assert not sp.any(sp.isnan(betas)), 'WTF?'
        assert not sp.any(sp.isinf(betas)), 'WTF?'

        num_non_matching_nts = 0
        num_ambig_nts = 0
        ok_nts = []
        print('Found %d SNPs present in both datasets'%(len(g_indices)))

        if 'eur_mafs' in ssg.keys():
            ss_freqs = ssg['eur_mafs'][...]
#             ss_freqs_list=[]

        ok_indices = {'g':[], 'ss':[]}
        for g_i, ss_i in it.izip(g_indices, ss_indices):

            #Is the nucleotide ambiguous?
            #g_nt = [recode_dict[g_nts[g_i][0]],recode_dict[g_nts[g_i][1]]
            g_nt = [g_nts[g_i][0],g_nts[g_i][1]]
            if tuple(g_nt) in ambig_nts:
                num_ambig_nts +=1
                tot_num_non_matching_nts += 1
                continue

            #First check if nucleotide is sane?
            if (not g_nt[0] in valid_nts) or (not g_nt[1] in valid_nts):
                num_non_matching_nts += 1
                tot_num_non_matching_nts += 1
                continue

            ss_nt = ss_nts[ss_i]
            #Are the nucleotides the same?
            flip_nts = False
            os_g_nt = sp.array([opp_strand_dict[g_nt[0]], opp_strand_dict[g_nt[1]]])
            if not (sp.all(g_nt == ss_nt) or sp.all(os_g_nt == ss_nt)):
                # Opposite strand nucleotides
                flip_nts = (g_nt[1] == ss_nt[0] and g_nt[0] == ss_nt[1]) or (os_g_nt[1] == ss_nt[0] and os_g_nt[0] == ss_nt[1])
                if flip_nts:
                    betas[ss_i] = -betas[ss_i]
                    log_odds[ss_i] = -log_odds[ss_i]
                    if 'eur_mafs' in ssg.keys():
                        ss_freqs[ss_i] = 1-ss_freqs[ss_i]
                else:
#                     print "Nucleotides don't match after all?: g_sid=%s, ss_sid=%s, g_i=%d, ss_i=%d, g_nt=%s, ss_nt=%s" % \
#                         (g_sids[g_i], ss_sids[ss_i], g_i, ss_i, str(g_nt), str(ss_nt))
                    num_non_matching_nts += 1
                    tot_num_non_matching_nts += 1

                    continue

            # everything seems ok.
            ok_indices['g'].append(g_i)
            ok_indices['ss'].append(ss_i)
            ok_nts.append(g_nt)

        print('%d SNPs were excluded due to ambiguous nucleotides.' % num_ambig_nts)
        print('%d SNPs were excluded due to non-matching nucleotides.' % num_non_matching_nts)

        #Resorting by position
        positions = sp.array(chrom_d['positions'])[ok_indices['g']]
        order = sp.argsort(positions)
        ok_indices['g'] = list(sp.array(ok_indices['g'])[order])
        ok_indices['ss'] = list(sp.array(ok_indices['ss'])[order])
        positions = positions[order]

        #Parse SNPs
        snp_indices = sp.array(chrom_d['snp_indices'])
        snp_indices = snp_indices[ok_indices['g']] #Pinpoint where the SNPs are in the file.
        raw_snps, freqs = _parse_plink_snps_(genotype_file, snp_indices)
        print('raw_snps.shape=', raw_snps.shape)

        snp_stds = sp.sqrt(2*freqs*(1-freqs)) #sp.std(raw_snps, 1)
        snp_means = freqs*2 #sp.mean(raw_snps, 1)

        betas = betas[ok_indices['ss']]
        log_odds = log_odds[ok_indices['ss']]
        ps = ssg['ps'][...][ok_indices['ss']]
        nts = sp.array(ok_nts)[order]
        sids = ssg['sids'][...][ok_indices['ss']]

        #Check SNP frequencies..
        if check_mafs and 'eur_mafs' in ssg.keys():
            ss_freqs = ss_freqs[ok_indices['ss']]
            freq_discrepancy_snp = sp.absolute(ss_freqs-(1-freqs))>0.15
            if sp.any(freq_discrepancy_snp):
                print('Warning: %d SNPs appear to have high frequency discrepancy between summary statistics and validation sample'%sp.sum(freq_discrepancy_snp))
                print(freqs[freq_discrepancy_snp])
                print(ss_freqs[freq_discrepancy_snp])

                #Filter freq_discrepancy_snps
                ok_freq_snps = sp.negative(freq_discrepancy_snp)
                raw_snps = raw_snps[ok_freq_snps]
                snp_stds = snp_stds[ok_freq_snps]
                snp_means = snp_means[ok_freq_snps]
                freqs = freqs[ok_freq_snps]
                ps = ps[ok_freq_snps]
                positions = positions[ok_freq_snps]
                nts = nts[ok_freq_snps]
                sids = sids[ok_freq_snps]
                betas = betas[ok_freq_snps]
                log_odds = log_odds[ok_freq_snps]


        #Filter minor allele frequency SNPs.
        maf_filter = (freqs>min_maf)*(freqs<(1-min_maf))
        maf_filter_sum = sp.sum(maf_filter)
        n_snps = len(maf_filter)
        assert maf_filter_sum<=n_snps, "WTF?"
        if sp.sum(maf_filter)<n_snps:
            raw_snps = raw_snps[maf_filter]
            snp_stds = snp_stds[maf_filter]
            snp_means = snp_means[maf_filter]
            freqs = freqs[maf_filter]
            ps = ps[maf_filter]
            positions = positions[maf_filter]
            nts = nts[maf_filter]
            sids = sids[maf_filter]
            betas = betas[maf_filter]
            log_odds = log_odds[maf_filter]


            print('%d SNPs with MAF < %0.3f were filtered'%(n_snps-maf_filter_sum,min_maf))

        print('%d SNPs were retained on chromosome %d.' % (maf_filter_sum, chrom))

        rb_prs = sp.dot(sp.transpose(raw_snps), log_odds)
        if has_phenotype:
            print('Normalizing SNPs')
            snp_means.shape = (len(raw_snps),1)
            snp_stds.shape = (len(raw_snps),1)
            snps = (raw_snps - snp_means) / snp_stds
            assert snps.shape==raw_snps.shape, 'Aha!'
            snp_stds = snp_stds.flatten()
            snp_means = snp_means.flatten()
            prs = sp.dot(sp.transpose(snps), betas)
            corr = sp.corrcoef(Y, prs)[0, 1]
            corr_list.append(corr)
            print('PRS correlation for chromosome %d was %0.4f' % (chrom, corr))
            rb_corr = sp.corrcoef(Y, rb_prs)[0, 1]
            rb_corr_list.append(rb_corr)
            print('Raw effect sizes PRS correlation for chromosome %d was %0.4f' % (chrom, rb_corr))

        sid_set = set(sids)
        if genetic_map_dir is not None:
            genetic_map = []
            with gzip.open(genetic_map_dir+'chr%d.interpolated_genetic_map.gz'%chrom) as f:
                for line in f:
                    l = line.split()
                    if l[0] in sid_set:
                        genetic_map.append(l[0])

        print('Now storing coordinated data to HDF5 file.')
        ofg = cord_data_g.create_group('chrom_%d' % chrom)
        ofg.create_dataset('raw_snps_ref', data=raw_snps, compression='lzf')
        ofg.create_dataset('snp_stds_ref', data=snp_stds)
        ofg.create_dataset('snp_means_ref', data=snp_means)
        ofg.create_dataset('freqs_ref', data=freqs)
        ofg.create_dataset('ps', data=ps)
        ofg.create_dataset('positions', data=positions)
        ofg.create_dataset('nts', data=nts)
        ofg.create_dataset('sids', data=sids)
        if genetic_map_dir is not None:
            ofg.create_dataset('genetic_map', data=genetic_map)
#         print 'Sum of squared effect sizes:', sp.sum(betas ** 2)
#         print 'Sum of squared log odds:', sp.sum(log_odds ** 2)
        ofg.create_dataset('betas', data=betas)
        ofg.create_dataset('log_odds', data=log_odds)
        ofg.create_dataset('log_odds_prs', data=rb_prs)
        if has_phenotype:
            risk_scores += prs
        rb_risk_scores += rb_prs
        num_common_snps += len(betas)

    if has_phenotype:
        # Now calculate the prediction r^2
        corr = sp.corrcoef(Y, risk_scores)[0, 1]
        rb_corr = sp.corrcoef(Y, rb_risk_scores)[0, 1]
        print('PRS R2 prediction accuracy for the whole genome was %0.4f (corr=%0.4f)' % (corr ** 2,corr))
        print('Log-odds (effects) PRS R2 prediction accuracy for the whole genome was %0.4f (corr=%0.4f)' % (rb_corr ** 2, rb_corr))
    print('There were %d SNPs in common' % num_common_snps)
    print('In all, %d SNPs were excluded due to nucleotide issues.' % tot_num_non_matching_nts)
    print('Done coordinating genotypes and summary statistics datasets.')



def coord_snp_weights_file(indiv_genot, SNP_weights_file, out_SNP_weights_h5file):
    """
    Coordinate SNPs (make sure that directions are ok), and store coordinated SNP weights to file.
    """
    snp_nt_dict = {}
    h5_ig = h5py.File(indiv_genot)
    for chrom_i in range(1,23):
        chr_str = 'Chr%d'%chrom_i
        for sid, nts in it.izip(h5_ig[chr_str]['sids'][...],h5_ig[chr_str]['nts'][...]):
            snp_nt_dict[sid] = nts
    print('Found %d SNPs in individual genotype'%len(snp_nt_dict))
    """
    SNP weight file format
    chrom    pos    sid    nt1    nt2    raw_beta     ldpred_beta
    chrom_1    798959    rs11240777    A    G    1.8529e-06    -6.0638e-05
    chrom_1    1005806    rs3934834    T    C    2.6001e-05    -6.8804e-04

    """
    snp_weights_data = pandas.read_table(SNP_weights_file,delim_whitespace=True)
    chrom_dict = {}
    for chrom_i in range(1,23):
        chr_str = 'Chr%d'%chrom_i
        chrom_dict[chr_str] = {'sids_dict':{}}
    for index, row in snp_weights_data.iterrows():
        if index%100000==0:
            print(index)
        chr_str = 'Chr%d'%int(row['chrom'][6:])
        d = chrom_dict[chr_str]
        sid = row['sid']
#         pos = row['pos']
        ldpred_beta = row['ldpred_beta']
        nts = snp_nt_dict.get(sid,None)
        assert nts is not None, 'Arrrgh, SNP weight coordination failed!'
        nt1 = row['nt1']
        nt2 = row['nt2']
        nt1_os = opp_strand_dict[nt1]
        nt2_os = opp_strand_dict[nt2]
        if sp.all([nt2,nt1]==nts) or sp.all([nt2_os,nt1]==nts) or sp.all([nt2,nt1_os]==nts) or sp.all([nt2_os,nt1_os]==nts):
            ldpred_beta = -ldpred_beta
        elif not (sp.all([nt1,nt2]==nts) or sp.all([nt1_os,nt2]==nts) or sp.all([nt1,nt2_os]==nts) or sp.all([nt1_os,nt2_os]==nts)):
            raise Exception('Somethings wrong with the nucelotides.')
        d[sid] = {'ldpred_beta':ldpred_beta}

    #Now fill new coordinated file, using the same SNP order as the individual genotype!
    oh5f = h5py.File(out_SNP_weights_h5file)
    num_snps_found = 0
    num_snps = 0
    for chrom_i in range(1,23):
        chr_str = 'Chr%d'%chrom_i
        print(chr_str)
        d = chrom_dict[chr_str]
        positions = []
        ldpred_betas = []
        sids = []
        nts_list = []
        for sid, pos, nts in it.izip(h5_ig[chr_str]['sids'][...],h5_ig[chr_str]['positions'][...],h5_ig[chr_str]['nts'][...]):
            num_snps +=1
            sid_dict = d.get(sid,None)
            if sid_dict is None:
                ldpred_betas.append(0)
            else:
                num_snps_found +=1
                ldpred_betas.append(sid_dict['ldpred_beta'])
            sids.append(sid)
            positions.append(pos)
            nts_list.append(nts)

        chr_g = oh5f.create_group(chr_str)
        chr_g.create_dataset('sids',data=sids)
        chr_g.create_dataset('nts',data=nts_list)
        chr_g.create_dataset('positions',data=positions)
        chr_g.create_dataset('ldpred_betas',data=ldpred_betas)
    print(num_snps, num_snps_found)
    h5_ig.close()
    oh5f.close()

