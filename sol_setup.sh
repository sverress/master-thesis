module load Python/3.8.6-GCCcore-10.2.0
pip install -r requirements.txt
export PYTHONPATH="$PWD"
printf "\n-------------------- Python setup complete --------------------\n\n"
git status