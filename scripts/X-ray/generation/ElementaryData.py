#Translation table for atomic numbers to element names and vice versa
#Note that the NIST database provides data up to atomic number 92 (= Uranium)
#Last column contains material densities #TODO: Fill in last column manually
ElementaryData = [
   (0,  "Void",             "X",    0),
   (1,  "Hydrogen",         "H",    8.375E-05),
   (2,	"Helium",           "He",   1.663E-04),
   (3,	"Lithium",          "Li",   0),
   (4,	"Beryllium",        "Be",   0),
   (5,	"Boron",            "B",    0),
   (6,	"Carbon",           "C",    0),
   (7,	"Nitrogen",         "N",    0),
   (8,	"Oxygen",           "O",    0),
   (9,	"Fluorine",         "F",    0),
   (10,	"Neon",             "Ne",   0),
   (11,	"Sodium",           "Na",   0),
   (12,	"Magnesium",        "Mg",   0),
   (13,	"Aluminium",         "Al",   0),
   (14,	"Silicon",          "Si",   0),
   (15,	"Phosphorus",       "P",    0),
   (16,	"Sulfur",           "S",    0),
   (17,	"Chlorine",         "Cl",   0),
   (18,	"Argon",            "Ar",   0),
   (19,	"Potassium",        "K",    0),
   (20,	"Calcium",          "Ca",   0),
   (21,	"Scandium",         "Sc",   0),
   (22,	"Titanium",         "Ti",   0),
   (23,	"Vanadium",         "V",    0),
   (24,	"Chromium",         "Cr",   0),
   (25,	"Manganese",        "Mn",   0),
   (26,	"Iron",             "Fe",   7.874E+00),
   (27,	"Cobalt",           "Co",   0),
   (28,	"Nickel",           "Ni",   0),
   (29,	"Copper",           "Cu",   0),
   (30,	"Zinc",             "Zn",   0),
   (31, "Gallium",          "Ga",   0),
   (32,	"Germanium",        "Ge",   0),
   (33,	"Arsenic",          "As",   0),
   (34,	"Selenium",     	"Se",   0),
   (35,	"Bromine",	        "Br",   0),
   (36,	"Krypton",	        "Kr",   0),
   (37,	"Rubidium",	        "Rb",   0),
   (38,	"Strontium",	    "Sr",   0),
   (39,	"Yttrium",	        "Y",    0),
   (40,	"Zirconium",	    "Zr",   0),
   (41,	"Niobium",	        "Nb",   0),
   (42,	"Molybdenum",	    "Mo",   0),
   (43,	"Technetium",	    "Tc",   0),
   (44,	"Ruthenium",	    "Ru",   0),
   (45,	"Rhodium",	        "Rh",   0),
   (46,	"Palladium",	    "Pd",   0),
   (47,	"Silver",	        "Ag",   0),
   (48,	"Cadmium",	        "Cd",   0),
   (49,	"Indium",	        "In",   0),
   (50,	"Tin",	            "Sn",   0),
   (51,	"Antimony",	        "Sb",   0),
   (52,	"Tellurium",	    "Te",   0),
   (53,	"Iodine",	        "I",    0),
   (54,	"Xenon",	        "Xe",   0),
   (55,	"Cesium",	        "Cs",   0),
   (56,	"Barium",	        "Ba",   0),
   (57,	"Lanthanum",	    "La",   0),
   (58,	"Cerium",	        "Ce",   0),
   (59,	"Praseodymium",	    "Pr",   0),
   (60,	"Neodymium",	    "Nd",   0),
   (61,	"Promethium",       "Pm",   0),
   (62,	"Samarium",         "Sm",   0),
   (63,	"Europium",         "Eu",   0),
   (64,	"Gadolinium",       "Gd",   0),
   (65,	"Terbium",          "Tb",   0),
   (66,	"Dysprosium",	    "Dy",   0),
   (67,	"Holmium",	        "Ho",   0),
   (68,	"Erbium",	        "Er",   0),
   (69,	"Thulium",	        "Tm",   0),
   (70,	"Ytterbium",        "Yb",   0),
   (71,	"Lutetium",         "Lu",   0),
   (72,	"Hafnium",	        "Hf",   0),
   (73,	"Tantalum",	        "Ta",   0),
   (74,	"Tungsten",	        "W",    0),
   (75,	"Rhenium",	        "Re",   0),
   (76,	"Osmium",	        "Os",   0),
   (77,	"Iridium",	        "Ir",   0),
   (78,	"Platinum",	        "Pt",   0),
   (79,	"Gold",	            "Au",   1.932E+01),
   (80,	"Mercury",	        "Hg",   0),
   (81,	"Thallium",	        "Tl",   0),
   (82,	"Lead",	            "Pb",   0),
   (83,	"Bismuth",	        "Bi",   0),
   (84,	"Polonium",	        "Po",   0),
   (85,	"Astatine",	        "At",   0),
   (86,	"Radon",	        "Rn",   0),
   (87,	"Francium",	        "Fr",   0),
   (88,	"Radium",	        "Ra",   0),
   (89,	"Actinium",	        "Ac",   0),
   (90,	"Thorium",	        "Th",   0),
   (91,	"Protactinium",	    "Pa",   0),
   (92,	"Uranium",	        "U",    0),
   (93,	"Neptunium",	    "Np",   0),
   (94,	"Plutonium",	    "Pu",   0),
   (95,	"Americium",	    "Am",   0),
   (96,	"Curium",	        "Cm",   0),
   (97,	"Berkelium",    	"Bk",   0),
   (98,	"Californium",  	"Cf",   0),
   (99,	"Einsteinium",  	"Es",   0),
   (100,"Fermium",	        "Fm",   0),
   (101,"Mendelevium",	    "Md",   0),
   (102,"Nobelium",     	"No",   0),
   (103,"Lawrencium",	    "Lr",   0),
   (104,"Rutherfordium",    "Rf",   0),
   (105,"Dubnium",	        "Db",   0),
   (106,"Seaborgium",   	"Sg",   0),
   (107,"Bohrium",	        "Bh",   0),
   (108,"Hassium",	        "Hs",   0),
   (109,"Meitnerium",   	"Mt",   0),
   (110,"Darmstadtium", 	"Ds",   0),
   (111,"Roentgenium"	    "Rg",   0),
   (112,"Ununbium",     	"Uub",  0),
   (113,"Ununtrium",	    "Uut",  0),
   (114,"Ununquadium",  	"Uuq",  0),
   (115,"Ununpentium",	    "Uup",  0),
   (116,"Ununhexium",	    "Uuh",  0),
   (117,"Ununseptium",	    "Uus",  0),
   (118,"Ununoctium",	    "Uuo",  0),
   (119,"Bone",             "",     0),
   (120,"Tissue",           "",     0),
]
