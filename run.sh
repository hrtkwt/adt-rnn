feature() {
    python feature.py m
    python feature.py acd
    python feature.py md
    python feature.py m_md
    python feature.py m_acd
}

train() {
    python train.py L_m base_1025
    python train.py L_acd base_1025
    python train.py L_md base_1025
    python train.py L_m_md base_2050
    python train.py L_m_acd base_2050
}

eval() {
    python eval_all.py K_m 6
    python eval_all.py K_acd 5
    python eval_all.py K_md 5
}


if [ "$1" = "feature" ]; then
    feature
elif [ "$1" = "train" ]; then
    train
elif [ "$1" = "eval" ]; then
    eval
fi
