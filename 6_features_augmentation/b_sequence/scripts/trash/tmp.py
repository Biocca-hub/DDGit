# https://biopython.org/docs/dev/api/Bio.SeqUtils.ProtParam.html

from Bio.SeqUtils.ProtParam import ProteinAnalysis

puta = 'MAENLLDGPPNPKRAKLSSPGFSANDNTDFGSLFDLENDLPDELIPNGELSLLNSGNLVPDAASKHKQLSELLRGGSGSSINPGIGNVSASSPVQQGLGGQAQGQPNSTNMASLGAMGKSPLNQGDSSTPNLPKQAASTSGPTPPASQALNPQAQKQVGLVTSSPATSQTGPGICMNANFNQTHPGLLNSNSGHSLMNQAQQGQAQVMNGSLGAAGRGRGAGMPYPAPAMQGATSSVLAETLTQVSPQMAGHAGLNTAQAGGMTKMGMTGTTSPFGQPFSQTGGQQMGATGVNPQLASKQSMVNSLPAFPTDIKNTSVTTVPNMSQLQTSVGIVPTQAIATGPTADPEKRKLIQQQLVLLLHAHKCQRREQANGEVRACSLPHCRTMKNVLNHMTHCQAGKACQVAHCASSRQIISHWKNCTRHDCPVCLPLKNASDKRNQQTILGSPASGIQNTIGSVGAGQQNATSLSNPNPIDPSSMQRAYAALGLPYMNQPQTQLQPQVPGQQPAQPPAHQQMRTLNALGNNPMSIPAGGITTDQQPPNLISESALPTSLGATNPLMNDGSNSGNIGSLSTIPTAAPPSSTGVRKGWHEHVTQDLRSHLVHKLVQAIFPTPDPAALKDRRMENLVAYAKKVEGDMYESANSRDEYYHLLAEKIYKIQKELEEKRRSRLHKQGILGNQPALPASGAQPPVIPPAQSVRPPNGPLPLPVNRMQVSQGMNSFNPMSLGNVQLPQAPMGPRAASPMNHSVQMNSMASVPGMAISPSRMPQPPNMMGTHANNIMAQAPTQNQFLPQNQFPSSSGAMSVNSVGMGQPAAQAGVSQGQVPGAALPNPLNMLAPQASQLPCPPVTQSPLHPTPPPASTAAGMPSLQHPTAPGMTPPQPAAPTQPSTPVSSGQTPTPTPGSVPSAAQTQSTPTVQAAAQAQVTPQPQTPVQPPSVATPQSSQQQPTPVHTQPPGTPLSQAAASIDNRVPTPSSVTSAETSSQQPGPDVPMLEMKTEVQTDDAEPEPTESKGEPRSEMMEEDLQGSSQVKEETDTTEQKSEPMEVEEKKPEVKVEAKEEEENSSNDTASQSTSPSQPRKKIFKPEELRQALMPTLEALYRQDPESLPFRQPVDPQLLGIPDYFDIVKNPMDLSTIKRKLDTGQYQEPWQYVDDVWLMFNNAWLYNRKTSRVYKFCSKLAEVFEQEIDPVMQSLGYCCGRKYEFSPQTLCCYGKQLCTIPRDAAYYSYQNRYHFCEKCFTEIQGENVTLGDDPSQPQTTISKDQFEKKKNDTLDPEPFVDCKECGRKMHQICVLHYDIIWPSGFVCDNCLKKTGRPRKENKFSAKRLQTTRLGNHLEDRVNKFLRRQNHPEAGEVFVRVVASSDKTVEVKPGMKSRFVDSGEMSESFPYRTKALFAFEEIDGVDVCFFGMHVQNTALIAPHQIQGRVYISYLDSIHFFRPRCLRTAVYHEILIGYLEYVKKLGYVTGHIWACPPSEGDDYIFHCHPPDQKIPKPKRLQEWYKKMLDKAFAERIINDYKDIFKQANEDRLTSAKELPYFEGDFWPNVLEESIKELEQEEEERKKEESTAASETPEGSQGDSKNAKKKNNKKTNKNKSSISRANKKKPSMPNVSNDLSQKLYATMEKHKEVFFVIHLHAGPVISTQPPIVDPDPLLSCDLMDGRDAFLTLARDKHWEFSSLRRSKWSTLCMLVELHTQGQDRFVYTCNECKHHVETRWHCTVCEDYDLCINCYNTKSHTHKMVKWGLGLDDEGSSQGEPQSKSPQESRRLSIQRCIQSLVHACQCRNANCSLPSCQKMKRVVQHTKGCKRKTNGGCPVCKQLIALCCYHAKHCQENKCPVPFCLNIKHKLRQQQIQHRLQQAQLMRRRMATMNTRNVPQQSLPSPTSAPPGTPTQQPSTPQTPQPPAQPQPSPVNMSPAGFPNVARTQPPTIVSAGKPTNQVPAPPPPAQPPPAAVEAARQIEREAQQQQHLYRANINNGMPPGRAGMGTPGSQMTPVGLNVPRPNQVSGPVMSSMPPGQWQQAPIPQQQPMPGMPRPVMSMQAQAAVAGPRMPNVQPPRSISPSALQDLLRTLKSPSSPQQQQQVLNILKSNPQLMAAFIKQRTAKYVANQPGMQPQPGLQSQPGMQPQPGMHQQPSLQNLNAMQAGVPRPGVPPPQPAMGGLNPQGQALNIMNPGHNPNMTNMNPQYREMVRRQLLQHQQQQQQQQQQQQQQQNSASLAGGMAGHSQFQQPQGPGGYAPAMQQQRMQQHLPIQGSSMGQMAAPMGQLGQMGQPGLGADSTPNIQQALQQRILQQQQMKQQIGSPGQPNPMSPQQHMLSGQPQASHLPGQQIATSLSNQVRSPAPVQSPRPQSQPPHSSPSPRIQPQPSPHHVSPQTGSPHPGLAVTMASSMDQGHLGNPEQSAMLPQLNTPNRSALSSELSLVGDTTGDTLEKFVEGL'
print(puta[98])
"""
Class containing methods for protein analysis.

The constructor takes two arguments. The first is the protein sequence as a string or a Seq object.

The second argument is optional. If set to True, the weight of the amino acids will be calculated using their monoisotopic mass 
(the weight of the most abundant isotopes for each element), instead of the average molecular mass (the averaged weight of all stable isotopes for each element). If set to false (the default value) or left out, the IUPAC average molecular mass will be used for the calculation.

__init__(prot_sequence, monoisotopic=False)
Initialize the class.
"""

"""
count_amino_acids()
    Count standard amino acids, return a dict. Counts the number times each amino acid is in the protein sequence. Returns a dictionary 

    {AminoAcid:Number}

    The return value is cached in self.amino_acids_content. It is not recalculated upon subsequent calls.
"""

'''X = ProteinAnalysis("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGT"
                    "RDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPSEEC"
                    "LFLERLEENHYNTYTSKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILF"
                    "LPLPV")'''

# print(X.count_amino_acids())

# {'A': 6, 'C': 3, 'D': 5, 'E': 12, 'F': 6, 'G': 14, 'H': 5, 'I': 5, 'K': 12, 'L': 18, 'M': 2, 'N': 7, 'P': 8, 'Q': 6, 'R': 6, 'S': 10, 'T': 13, 'V': 5, 'W': 1, 'Y': 8}

# print(X.count_amino_acids()['C']) # 3

"""
propertyamino_acids_percent
    Get the amino acid content in percentages.

    The same as count_amino_acids, but returns the number as a percentage of the sequence length. Returns a dictionary of 
    
    {AminoAcid:percentage} 
    
    where the percentage is in the range 0-100.
"""

# non si può chiamare come X.amino_acids_percent
# print(X.amino_acids_percent) # {'A': 3.9473684210526314, 'C': 1.9736842105263157, 'D': 3.289473684210526, 'E': 7.894736842105263, ...}

"""
molecular_weight()
    Calculate MW from Protein sequence.
"""

# print(X.molecular_weight()) # 17103.1617

"""
aromaticity()
    Calculate the aromaticity according to Lobry, 1994.
    Calculates the aromaticity value of a protein according to Lobry, 1994. 
    It is simply the relative frequency of Phe+Trp+Tyr.
"""

# print(X.aromaticity()) # 0.09868421052631579

"""
instability_index()
    Calculate the instability index according to Guruprasad et al 1990.
    Implementation of the method of Guruprasad et al. 1990 to test a protein for stability. 
    Any value above 40 means the protein is unstable (has a short half life).
    See: Guruprasad K., Reddy B.V.B., Pandit M.W. Protein Engineering 4:155-161(1990).
"""

# print(X.instability_index()) # 41.980263157894726

"""
flexibility()
    Calculate the flexibility according to Vihinen, 1994.
    No argument to change window size because parameters are specific for a window=9. 
    The parameters used are optimized for determining the flexibility.
"""

# print(X.flexibility()) # List of values (sequence length - 9)

"""
gravy(scale='KyteDoolitle')
    Calculate the GRAVY (Grand Average of Hydropathy) according to Kyte and Doolitle, 1982.
    Utilizes the given Hydrophobicity scale, by default uses the original proposed by Kyte and Doolittle (KyteDoolitle). 
    Other options are: Aboderin, AbrahamLeo, Argos, BlackMould, BullBreese, Casari, Cid, Cowan3.4, Cowan7.5, Eisenberg, 
                       Engelman, Fasman, Fauchere, GoldSack, Guy, Jones, Juretic, Kidera, Miyazawa, Parker,Ponnuswamy, Rose, 
                       Roseman, Sweet, Tanford, Wilson and Zimmerman.
    New scales can be added in ProtParamData.
"""

# print(X.gravy()) # one value

"""
protein_scale(param_dict, window, edge=1.0)
    Compute a profile by any amino acid scale.
    An amino acid scale is defined by a numerical value assigned to each type of amino acid. The most frequently used scales are 
    the hydrophobicity or hydrophilicity scales and the secondary structure conformational parameters scales, but many other scales exist 
    which are based on different chemical and physical properties of the amino acids. You can set several parameters that control 
    the computation of a scale profile, such as the window size and the window edge relative weight value.

    WindowSize: The window size is the length of the interval to use for the profile computation. For a window size n, 
    we use the i-(n-1)/2 neighboring residues on each side to compute the score for residue i. 
    The score for residue i is the sum of the scaled values for these amino acids, optionally weighted according to their position in the window.
    
    Edge: The central amino acid of the window always has a weight of 1. By default, the amino acids at the remaining window positions 
    have the same weight, but you can make the residue at the center of the window have a larger weight than the others by setting the edge 
    value for the residues at the beginning and end of the interval to a value between 0 and 1. For instance, for Edge=0.4 and a window size 
    of 5 the weights will be: 0.4, 0.7, 1.0, 0.7, 0.4.
    
    The method returns a list of values which can be plotted to view the change along a protein sequence. Many scales exist. 
    Just add your favorites to the ProtParamData modules.

    Similar to expasy`s ProtScale: http://www.expasy.org/cgi-bin/protscale.pl
"""

# print(X.protein_scale())

"""
isoelectric_point()
    Calculate the isoelectric point.
    Uses the module IsoelectricPoint to calculate the pI of a protein.

charge_at_pH(pH)
    Calculate the charge of a protein at given pH.

secondary_structure_fraction()
    Calculate fraction of helix, turn and sheet.
    Returns a list of the fraction of amino acids which tend to be in Helix, Turn or Sheet, according to Haimov and Srebnik, 2016; 
    Hutchinson and Thornton, 1994; and Kim and Berg, 1993, respectively.
    
    Amino acids in helix: E, M, A, L, K. Amino acids in turn: N, P, G, S, D. Amino acids in sheet: V, I, Y, F, W, L, T.
    
    Note that, prior to v1.82, this method wrongly returned (Sheet, Turn, Helix) while claiming to return (Helix, Turn, Sheet).
    
    Returns a tuple of three floats (Helix, Turn, Sheet).

molar_extinction_coefficient()
    Calculate the molar extinction coefficient.
    Calculates the molar extinction coefficient assuming cysteines (reduced) and cystines residues (Cys-Cys-bond)"""