#!/bin/bash
#SBATCH --job-name=verify_contact
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=/home/ycliang/predicators/logs/verify_contact_%j.out
set -euo pipefail
cd /home/ycliang/predicators
export OPENBLAS_NUM_THREADS=1
for s in 0 1 2; do
  echo "########## seed $s ##########"
  python scripts/verify_balloons_contact.py --seed $s || echo "SEED $s FAILED"
done
echo "ALL CONTACT VERIFICATION DONE"
