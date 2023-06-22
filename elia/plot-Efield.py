# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

# author: Elia Stocco
# email : stocco@fhi-berlin.mpg.de

import argparse
import matplotlib.pyplot as plt
#from ipi.engine.ensembles import ElectricField
#from ipi.utils.io.inputs.io_xml import xml_parse_file
import xml.etree.ElementTree as ET
import ast 
from functions import convert

def prepare_parser():
    """set up the script input parameters"""

    parser = argparse.ArgumentParser(description="Plot the electric field E(t) into a pdf file.")

    parser.add_argument(
        "-i", "--input", action="store", type=str,
        help="input file", default="input.xml"
    )
    parser.add_argument(
        "-o", "--output", action="store", type=str,
        help="output file ", default="Efield.pdf"
    )
       
    options = parser.parse_args()

    return options

def main():
    """main routine"""

    # prepare/read input arguments
    print("\tReading script input arguments")
    options = prepare_parser()

    print("\tReading json file")
    # # Open the JSON file and load the data
    # with open(options.input) as f:
    #     info = json.load(f)

    data = ET.parse(options.input).getroot()

    ensemble = None
    for element in data.iter():
        if element.tag == "ensemble":
            ensemble = element
            break

    data     = {}
    keys     = ["Eamp",          "Efreq",    "Ephase",   "Epeak","Esigma"]
    families = ["electric-field","frequency","undefined","time", "time"  ]
    
    for key,family in zip(keys,families):

        data[key] = { "value" : None }
        
        element = ensemble.find(key)

        if element is not None:
            value = ast.literal_eval(element.text)
            unit = element.attrib["units"]
            if unit is None :
                unit = "atomic_unit"

            data[key]["value"] = convert(value,family,unit,"atomic_unit")


    

    # fig, ax = plt.subplots(figsize=(10,6))

    # ax.plot(t,E)

    # plt.tight_layout()

    # print("\tSaving plot to {:s}".format(options.output))
    # plt.savefig(options.output)

    print("\n\tJob done :)\n")

if __name__ == "__main__":
    main()