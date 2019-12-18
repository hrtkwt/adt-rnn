


feature() {
    python feature.py m
    python feature.py acd
    python feature.py md
}

train() {
    python train.py K_m
    python train.py K_acd
    python train.py K_md
}

eval() {
    python eval_all.py J_m 7
    python eval_all.py J_acd 8
    python eval_all.py J_md 7
}


train