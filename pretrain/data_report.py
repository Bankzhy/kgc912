from dataset import KGCodeDataset, init_dataset
import argparse
from args import add_args
def report_dataset():
    struct_type= [
        "control_dependency",
        "data_dependency",
    ]

    syntax_type = [
        'type_of',
        "has_method",
        "has_property",
        "assignment"
    ]

    basic_concept_type = [
        "related_concept"
    ]

    expand_concept_type = [
        "relatedto",
        "formof",
        "isa",
        "partof",
        "hasa",
        "usedfor",
        "capableof",
        "atlocation",
        "causes",
        "hassubevent",
        "hasfirstsubevent",
        "haslastsubevent",
        "hasprerequisite",
        "hasproperty",
        "motivatedbygoal",
        "obstructedby",
        "desires",
        "createdby",
        "synonym",
        "antonym",
        "distinctfrom",
        "derivedfrom",
        "symbolof",
        "definedas",
        "mannerof",
        "locatednear",
        "hascontext",
        "similarto",
        "etymologicallyrelatedto",
        "etymologicallyderivedfrom",
        "causesdesire",
        "madeof",
        "receivesaction",
        "externalurl"
    ]

    struct_num = 0
    syntax_num = 0
    basic_concept_num = 0
    expand_concept_num = 0

    expand_sample_num = 0
    no_expand_sample_num = 0

    # file = r"C:\worksapce\research\kgc912\pretrain\kg_data\data.json"
    #
    # with open(file, encoding='ISO-8859-1') as f:
    #     lines = f.readlines()
    #     print("loading dataset:")
    #     for line in tqdm(lines):
    #         # print(line)
    #         data = json.loads(line.strip())
    #         code = data["code"]
    #         doc = data["doc"]
    #         kg = data["kg"]
    #
    #         for edges in kg:
    #             if edges["type"] in struct_type:
    #                 struct_num += 1
    #             elif edges["type"] in syntax_type:
    #                 syntax_num += 1
    #             elif edges["type"] in basic_concept_type:
    #                 basic_concept_num += 1
    #             else:
    #                 expand_concept_num += 1

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])
    add_args(parser)
    main_args = parser.parse_args()

    dataset = init_dataset(args=main_args)

    for rel in dataset.structures:
        st_l = rel.split(dataset.KG_SEP_TOKEN)
        for st in st_l:
            child_l = st.split(dataset.spliter)
            if len(child_l) < 3:
                continue
            if child_l[1] in struct_type:
                struct_num += 1
            elif child_l[1] in syntax_type:
                syntax_num += 1

    for rel in dataset.nls:
        nls_l = rel.split(",")
        have_expand = False
        for nls in nls_l:
            child_l = nls.split(dataset.spliter)
            if len(child_l) < 3:
                basic_concept_num += 1
            else:
                if child_l[1] in expand_concept_type:
                    expand_concept_num += 1
                    have_expand = True
                else:
                    basic_concept_num += 1
        if have_expand:
            expand_sample_num += 1
        else:
            no_expand_sample_num += 1

    print("struct num:", str(struct_num))
    print("syntax num:", str(syntax_num))
    print("basic concept num:", str(basic_concept_num))
    print("expand concept num:", str(expand_concept_num))
    print("expand sample num:", str(expand_sample_num))
    print("no expand sample num:", str(no_expand_sample_num))

if __name__ == '__main__':
    report_dataset()