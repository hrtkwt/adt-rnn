feature() {
    prefix="P"
    seed="389098"
    python feature.py "$prefix" m "$seed"
    python feature.py "$prefix" acd "$seed"
    python feature.py "$prefix" md "$seed"
    python feature.py "$prefix" m_md "$seed"
    python feature.py "$prefix" m_acd "$seed"
}

train() {
    prefix="P"
    python train.py "${prefix}_m" base_1025
    python train.py "${prefix}_acd" base_1025
    python train.py "${prefix}_md" base_1025
    python train.py "${prefix}_m_md" base_2050
    python train.py "${prefix}_m_acd" base_2050
}

eval() {
    prefix="P"
    python eval.py "${prefix}_m" base_1025 4
    python eval.py "${prefix}_acd" base_1025 6
    python eval.py "${prefix}_md" base_1025 3
    python eval.py "${prefix}_m_md" base_2050 2
    python eval.py "${prefix}_m_acd" base_2050 3
}


if [ "$1" = "feature" ]; then
    feature
elif [ "$1" = "train" ]; then
    train
elif [ "$1" = "eval" ]; then
    eval
fi
