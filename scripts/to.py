import json
import pandas as pd

if __name__ == "__main__":
    src_dir = '../dataset/2023-Thyroid'
    df = pd.read_csv(f"{src_dir}/clinical.csv")
    for i in range(5):
        expr = f'./expr_kFold_{i}'
        iroot = f'{expr}/json'
        oroot = f'{expr}'
        
        for phenotype in ['BRAF']:
            for sp in ['trn', 'val', 'tst']:
                with open(f'{iroot}/all_{phenotype}_{sp}.json', 'r') as f:
                    recs = json.load(f)
                cases = recs['cases']
                recs = [df[df['ID'] == case].values.tolist()[0] for case in cases]
                with open(f'{oroot}/clinical_{sp}.csv', 'w') as f:
                    f.write('ID,Gender,Age,Diagnosis,BRAF,TERT,RAS\n')
                    for rec in recs:
                        f.write('{},{},{},{},{},{},{}\n'.format(*rec)) 