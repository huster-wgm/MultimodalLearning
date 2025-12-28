#! /bin/bash
iter=2500;
for i in {0..4}; do 
    python regionCls/train.py \
        --experiment expr_kFold_$i \
        --trigger iter \
        --interval 50 \
        --terminal $iter &&
    python regionCls/infer.py \
        --experiment expr_kFold_$i ;
done
echo END
