#! /bin/bash

ptypes="BRAF RAS" 
classes="all PTC AUS"
iter=10000
for i in {0..4}; do 
    for phenotype in $ptypes; do
        echo "Running on $phenotype";
        python geneInstCls/train.py \
            --experiment expr_kFold_$i \
            --trigger iter \
            --interval 100 \
            --terminal $iter \
            --phenotype $phenotype &&

        echo "infering on $phenotype";
        python geneInstCls/infer.py \
            --experiment expr_kFold_$i \
            --phenotype $phenotype
    done
        for cls in $classes; do
            echo "Ensembling on $phenotype @ $cls ";
            python geneEnsemble/test.py \
                --experiment expr_kFold_$i \
                --cls $cls \
                --phenotype $phenotype
        done
done

echo END
