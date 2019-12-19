feature() {
    python feature.py m
    python feature.py acd
    python feature.py md
    python feature.py m_md
    python feature.py m_acd
}

train() {
    python train.py M_m base_1025
    python train.py M_acd base_1025
    python train.py M_md base_1025
    python train.py M_m_md base_2050
    python train.py M_m_acd base_2050
}

eval() {
    python eval_all.py L_m base_1025 4
    python eval_all.py L_acd base_1025 6
    python eval_all.py L_md base_1025 3
    python eval_all.py L_m_md base_2050 2
    python eval_all.py L_m_acd base_2050 3
}


if [ "$1" = "feature" ]; then
    feature
elif [ "$1" = "train" ]; then
    train
elif [ "$1" = "eval" ]; then
    eval
fi
