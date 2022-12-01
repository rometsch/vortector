if [[ ! -d .venv ]]; then
    echo "Setting up new python virtual env"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install .
fi

source .venv/bin/activate