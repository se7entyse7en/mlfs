if [[ "$1" == "static_checks" ]]; then
    echo "Running static checks..."
    pycodestyle --statistics "`pwd`"
    isort -rc -c -q
elif [[ "$1" == "models_demo" ]]; then
    echo "Running models..."
    echo "Running K-Nearest Neighbors with k=3 and distance=L1..."
    python -m mlfs.supervised.knn run_simple --k 3 --distance L1
    echo "Running K-Nearest Neighbors with k=3 and distance=L2..."
    python -m mlfs.supervised.knn run_simple --k 3 --distance L2
    echo "Running K-Fold Cross validation for K-Nearest Neighbors with k=3, distance=L2 and 5 folds..."
    python -m mlfs.supervised.knn run_k_fold --k 3 --distance L2 --folds_n 5
fi
