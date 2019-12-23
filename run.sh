feature() {
    prefix="T"
    seed="7777"
    python feature.py "$prefix" m "$seed"
    python feature.py "$prefix" acd "$seed"
    python feature.py "$prefix" md "$seed"
    python feature.py "$prefix" m_md "$seed"
    python feature.py "$prefix" m_acd "$seed"
}

train() {
    prefix="T"
    python train.py "${prefix}_m" base_1025
    python train.py "${prefix}_acd" base_1025
    python train.py "${prefix}_md" base_1025
    python train.py "${prefix}_m_md" base_2050
    python train.py "${prefix}_m_acd" base_2050
}

eval() {
    prefix="T"
    python eval.py "${prefix}_m" base_1025 48
    python eval.py "${prefix}_acd" base_1025 45
    python eval.py "${prefix}_md" base_1025 21
    python eval.py "${prefix}_m_md" base_2050 29
    python eval.py "${prefix}_m_acd" base_2050 33
}


if [ "$1" = "feature" ]; then
    feature
elif [ "$1" = "train" ]; then
    train
elif [ "$1" = "eval" ]; then
    eval
fi
