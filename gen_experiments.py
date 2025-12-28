import os
import json
import shutil
import pandas as pd

from sklearn.model_selection import StratifiedKFold


class Experiment:
    def __init__(self, experiment_name):
        self.name = experiment_name

    def gen_cluster_expr(self, trn: list, val: list, tst: list):
        expr = {"stage": "cluster"}
        expr['trn_set'] = {
            "root": "./dataset/2023-Thyroid",
            "cases": trn,
            "num_cases": len(trn),
        } 
        expr['val_set'] = {
            "root": "./dataset/2023-Thyroid",
            "cases": val,
            "num_cases": len(val),
        }
        # Test set is optional, external test set can be used
        if len(tst) == 0:
            expr['tst_set'] = {
                "root": "./dataset/2023-Thyroid",
                "cases": val,
                "num_cases": len(val),
            }
        else:
            expr['tst_set'] = {
                "root": "./dataset/2023-Thyroid",
                "cases": tst,
                "num_cases": len(tst),
            }

        self.cluster_expr = expr   


    def gen_inst_expr(self, trn: list, val: list, tst: list):
        expr = {"stage": "inst"}
        expr['trn_set'] = {
            "root": "./dataset/2023-Thyroid",
            "cases": trn,
            "num_cases": len(trn),
        } 
        expr['val_set'] = {
            "root": "./dataset/2023-Thyroid",
            "cases": val,
            "num_cases": len(val),
        }
        # Test set is optional, external test set can be used
        if len(tst) == 0:
            expr['tst_set'] = {
                "root": "./dataset/2023-Thyroid",
                "cases": val,
                "num_cases": len(val),
            }
        else:
            expr['tst_set'] = {
                "root": "./dataset/2023-Thyroid",
                "cases": tst,
                "num_cases": len(tst),
            }
        self.inst_expr = expr   

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)


def get_label_by_cases(df, phenotype, cases):
    labels = []
    for case in cases:
        label = df[df['ID'] == case][phenotype].values[0]
        labels.append(label)
    return labels


def main():
    # load raw dataset
    root_dir = './dataset/2023-Thyroid'
    df = pd.read_csv(os.path.join(root_dir, 'clinical.csv'))

    with open(os.path.join(root_dir, 'ann_all_filter.txt')) as f:
        cluster_cases = [line.strip() for line in f.readlines()]
        cluster_labels = get_label_by_cases(df, 'RAS', cluster_cases)
    
    with open(os.path.join(root_dir, 'internal_cluster_filter.txt')) as f:
        inst_cases_internal = [line.strip() for line in f.readlines()]
        inst_labels_internal = get_label_by_cases(df, 'RAS', inst_cases_internal)

    with open(os.path.join(root_dir, 'external_cluster_filter.txt')) as f:
        inst_cases_external = [line.strip() for line in f.readlines()]
        inst_labels_external = get_label_by_cases(df, 'RAS', inst_cases_external)

    skf = StratifiedKFold(n_splits=5, random_state=4242, shuffle=True)
    print(len(cluster_cases), len(cluster_labels))
    cluster_splits = list(skf.split(cluster_cases, cluster_labels))
    inst_splits = list(skf.split(inst_cases_internal, inst_labels_internal))

    for i, ((ctrn_idx, cval_idx), (itrn_idx, ival_idx)) in enumerate(zip(cluster_splits, inst_splits)):
        experiment = Experiment(f"expr_kFold_{i}")
        print(f"generate >> {experiment.name}", f"cluster: {len(ctrn_idx)} / {len(cval_idx)}", f"inst: {len(itrn_idx)} /{len(ival_idx)} / {len(inst_cases_external)}")
        # cluster split
        ctrn = [cluster_cases[idx] for idx in ctrn_idx]
        cval = [cluster_cases[idx] for idx in cval_idx]
        ctst = []
        experiment.gen_cluster_expr(ctrn, cval, ctst)
        # inst split
        itrn = [inst_cases_internal[idx] for idx in itrn_idx]
        ival = [inst_cases_internal[idx] for idx in ival_idx]
        itst = inst_cases_external
        experiment.gen_inst_expr(itrn, ival, itst)
        if os.path.exists(f"./experiments/{experiment.name}"):
            print(f"experiment {experiment.name} already exists, remove")
            shutil.rmtree(f"./experiments/{experiment.name}")
        os.makedirs(f"./experiments/{experiment.name}", exist_ok=True)
        experiment.save(f"./experiments/{experiment.name}/config.json")


if __name__ == '__main__':
    main()
