run1() {
    python train.py m_10_2
    python train.py m_md_10_2
    python train.py m_acd_10_2
}

run2() {
    python train.py F_m
    python train.py F_acd
    python train.py F_gamma
}

run3() {
    python feature.py m
    python feature.py acd
    python feature.py gamma
}

run2