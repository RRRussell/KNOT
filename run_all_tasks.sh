#!/bin/bash
#SBATCH --job-name=knot_all_tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=400000M
#SBATCH --time=30-00:00
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy

# ============================================
# KNOT GNN - Run All Tasks (Evaluation & Inference)
# ============================================

# Generate timestamp for logs
ts=$(date +'%Y%m%d_%H%M%S')

# Create directories
log_dir="./logs"
results_dir="./results"
mkdir -p "$log_dir"
mkdir -p "$results_dir/evaluation"
mkdir -p "$results_dir/inference"

# Unbuffer Python output
export PYTHONUNBUFFERED=1

# Define timestamp function
timestamp() {
  while IFS= read -r line; do
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
  done
}

# Define all tasks
TASKS=(
    "strong_vs_weak_evidence"
    "approved_targets"
    "known_druggable"
    "high_confidence"
    "cancer_druggability"
    "sm_target"
    "sm_cancer"
    "ab_target"
    "ab_cancer"
    "protac"
)

# Main log file
main_log="$log_dir/all_tasks_${ts}.log"

{
    echo "=========================================="
    echo "KNOT GNN - ALL TASKS EXECUTION"
    echo "=========================================="
    echo "Job started on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Running 10 tasks in both evaluation and inference modes"
    echo ""
    
    # ============================================
    # EVALUATION MODE FOR ALL TASKS
    # ============================================
    
    echo "=========================================="
    echo "PHASE 1: EVALUATION MODE"
    echo "=========================================="
    
    for i in "${!TASKS[@]}"; do
        task="${TASKS[$i]}"
        task_num=$((i + 1))
        
        echo ""
        echo "----------------------------------------"
        echo "[$task_num/10] Running EVALUATION for: $task"
        echo "----------------------------------------"
        
        python main.py \
            --task "$task" \
            --mode evaluation \
            --edge-config all \
            --epochs 200 \
            --patience 30 \
            --seed 42 \
            --save-dir "$results_dir/evaluation/${task}"
        
        if [ $? -eq 0 ]; then
            echo "✓ Evaluation completed for $task"
        else
            echo "✗ Evaluation failed for $task"
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "EVALUATION PHASE COMPLETE"
    echo "=========================================="
    echo ""
    
    # ============================================
    # INFERENCE MODE FOR ALL TASKS
    # ============================================
    
    echo "=========================================="
    echo "PHASE 2: INFERENCE MODE"
    echo "=========================================="
    
    for i in "${!TASKS[@]}"; do
        task="${TASKS[$i]}"
        task_num=$((i + 1))
        
        echo ""
        echo "----------------------------------------"
        echo "[$task_num/10] Running INFERENCE for: $task"
        echo "----------------------------------------"
        
        python main.py \
            --task "$task" \
            --mode inference \
            --edge-config all \
            --epochs 200 \
            --seed 42 \
            --save-dir "$results_dir/inference/${task}"
        
        if [ $? -eq 0 ]; then
            echo "✓ Inference completed for $task"
        else
            echo "✗ Inference failed for $task"
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "INFERENCE PHASE COMPLETE"
    echo "=========================================="
    
    # ============================================
    # SUMMARY
    # ============================================
    
    echo ""
    echo "=========================================="
    echo "EXECUTION SUMMARY"
    echo "=========================================="
    echo "Job finished at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "Results saved in:"
    echo "  - Evaluation: $results_dir/evaluation/"
    echo "  - Inference: $results_dir/inference/"
    echo ""
    echo "Checking output files..."
    
    # Check evaluation results
    echo ""
    echo "Evaluation Results:"
    for task in "${TASKS[@]}"; do
        eval_files=$(find "$results_dir/evaluation/${task}" -name "eval_results_*.json" 2>/dev/null | wc -l)
        if [ "$eval_files" -gt 0 ]; then
            echo "  ✓ $task: $eval_files result file(s)"
        else
            echo "  ✗ $task: No result files found"
        fi
    done
    
    # Check inference results
    echo ""
    echo "Inference Results:"
    for task in "${TASKS[@]}"; do
        rank_files=$(find "$results_dir/inference/${task}" -name "gene_ranking_*.csv" 2>/dev/null | wc -l)
        if [ "$rank_files" -gt 0 ]; then
            echo "  ✓ $task: $rank_files ranking file(s)"
        else
            echo "  ✗ $task: No ranking files found"
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "ALL TASKS COMPLETE!"
    echo "=========================================="
    
} 2>&1 | timestamp | tee -a "$main_log"

echo "Full log saved to: $main_log"