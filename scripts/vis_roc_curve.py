import os
import glob
import json
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt


config = {
    "font.family":'Times New Roman',
    'font.size':24,
    'axes.facecolor': 'white',
    'xtick.direction':'in',
    'ytick.direction':'in',
}
rcParams.update(config)


if __name__ == "__main__":
    for i in range(4):
        expr = f'./expr_kFold_{i}'
        iroot = f'{expr}/json'
        oroot = f'{expr}/vis4'


        if not os.path.exists(oroot):
            os.makedirs(oroot)


        for phenotype in ['BRAF', 'RAS']:
        # for phenotype in ['TERT']:
            if  phenotype == "Diagnosis":
                nb_cls = 3
            else:
                nb_cls = 1

            cls = f"cls_{nb_cls}"

            plt.figure(figsize=(18,14))
            gfile = os.path.join(oroot, f"all_{phenotype}_roc.pdf")
            for c, sp in zip(['r', 'g', 'b' ],['trn', 'val', 'tst']):
                jfile = os.path.join(iroot, f"all_{phenotype}_{sp}.json")
                with open(jfile, 'r') as f:
                    recs = json.load(f)
                info = recs[cls]
                label = sp if sp != 'tst' else 'external'
                plt.plot(info['fpr'], info['tpr'], c, 
                         label = f"ROC_AUC={info['roc-auc']:0.3f}, CI={info['ROC_CI'][0]:0.3f}-{info['ROC_CI'][1]:0.3f} [{label}]")
            plt.legend(loc = 'lower right', fontsize=24)
            plt.plot([0, 1], [0, 1],'k--')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            # plt.xaxis.set_ticks(np.arange(0., 1., 0.1))
            # plt.yaxis.set_ticks(np.arange(0., 1., 0.1))
            plt.xticks(np.arange(0., 1.1, 0.1), fontproperties='Times New Roman', size=24)
            plt.yticks(np.arange(0., 1.1, 0.1), fontproperties='Times New Roman', size=24)
            plt.ylabel('True Positive Rate', fontproperties='Times New Roman', size=32)
            plt.xlabel('False Positive Rate', fontproperties='Times New Roman', size=32)
            plt.savefig(gfile, format="pdf")
            plt.close()
            # plt.show()
            
            plt.figure(figsize=(18,14))
            gfile = os.path.join(oroot, f"all_{phenotype}_pr.pdf")
            for c, sp in zip(['r', 'g', 'b' ],['trn', 'val', 'tst']):
                jfile = os.path.join(iroot, f"all_{phenotype}_{sp}.json")
                with open(jfile, 'r') as f:
                    recs = json.load(f)
                info = recs[cls]
                label = sp if sp != 'tst' else 'external'
                plt.plot(info['rec'], info['pre'], c, 
                         label = f"PR_AUC={info['pr-auc']:0.3f}, [{label}]")
            plt.legend(loc = 'lower right', fontsize=24)
            plt.plot([0, 1], [1, 0],'k--')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            # plt.xaxis.set_ticks(np.arange(0., 1., 0.1))
            # plt.yaxis.set_ticks(np.arange(0., 1., 0.1))
            plt.xticks(np.arange(0., 1.1, 0.1), fontproperties='Times New Roman', size=24)
            plt.yticks(np.arange(0., 1.1, 0.1), fontproperties='Times New Roman', size=24)
            plt.ylabel('Precision', fontproperties='Times New Roman', size=32)
            plt.xlabel('Recall', fontproperties='Times New Roman', size=32)
            plt.savefig(gfile, format="pdf")
            plt.close()
            # plt.show()